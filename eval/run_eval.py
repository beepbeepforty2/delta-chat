"""One-command eval: run the delta engine against every labeled pair in
the dataset, score against ground truth, print a scorecard, and write
eval/results/{timestamp}.json (diffed against the previous run).

Usage: python -m eval.run_eval --dataset eval/datasets/v0 [--levels L0,L2]
       [--skip-chat] [--skip-baseline]

Chat and the llm_direct baseline both make live LLM calls (order of 70-90
across the full dataset), so they're the slow part of a run and each has a
--skip flag for fast iteration on the deterministic engine alone.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.chat_eval import (
    ChatEvalItem,
    run_chat_for_pair,
    run_judge_on_items,
    score_correctness_aggregate,
    score_groundedness_aggregate,
    score_refusal_accuracy,
    validate_judge,
)
from eval.calibration import bucket_calibration
from eval.chat_eval import load_qa as load_qa_jsonl
from eval.metrics import prf1, score_pair
from src.chat.chat import ChatAnswer
from src.cli import _resolve_with_pid, compute_deltas
from src.delta.precheck import check_same_document
from src.observability.tracer import Tracer

DEFAULT_LEVELS = ["L0", "L2"]


def load_manifest(dataset_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (dataset_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]


def load_gt_deltas(pair_dir: Path) -> list[dict]:
    return json.loads((pair_dir / "gt" / "deltas.json").read_text())


def run_pair_at_level(pair_dir: Path, level: str):
    doc_a = _resolve_with_pid("A", str(pair_dir / "a" / f"{level}.pdf"))
    doc_b = _resolve_with_pid("B", str(pair_dir / "b" / f"{level}.pdf"))
    # Not persisted: an eval run exercises dozens of pair x level
    # combinations, and writing a trace file for each would flood
    # traces/ with little debugging value over the scorecard itself.
    # make run's single-request tracing is what's meant for that.
    tracer = Tracer()
    return compute_deltas(doc_a, doc_b, tracer)


def aggregate(pair_results: list[dict]) -> dict:
    tp = sum(r["overall"]["tp"] for r in pair_results)
    fp = sum(r["overall"]["fp"] for r in pair_results)
    fn = sum(r["overall"]["fn"] for r in pair_results)
    overall = prf1(tp, fp, fn)

    by_kind = {}
    for kind in ("add", "remove", "modify", "move"):
        ktp = sum(r["by_kind"][kind]["tp"] for r in pair_results)
        kfp = sum(r["by_kind"][kind]["fp"] for r in pair_results)
        kfn = sum(r["by_kind"][kind]["fn"] for r in pair_results)
        by_kind[kind] = prf1(ktp, kfp, kfn)

    null_tp = sum(r["semantic_null_detection"]["tp"] for r in pair_results)
    null_fp = sum(r["semantic_null_detection"]["fp"] for r in pair_results)
    null_fn = sum(r["semantic_null_detection"]["fn"] for r in pair_results)
    semantic_null_detection = prf1(null_tp, null_fp, null_fn)

    # .get(...) default: raster recall (src/delta/raster_recall.py) is
    # opt-in and predates this key in some test fixtures -- 0 is always
    # correct when the pass didn't run.
    n_unclassified_visual_change = sum(r.get("n_unclassified_visual_change", 0) for r in pair_results)

    primary_recalls = [r["primary_recall"] for r in pair_results]
    cascade_recalls = [r["cascade_recall"] for r in pair_results if r["cascade_recall"] is not None]
    null_rates = [r["semantic_null_emission_rate"] for r in pair_results
                  if r["semantic_null_emission_rate"] is not None]

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "overall": overall, "by_kind": by_kind,
        "avg_primary_recall": avg(primary_recalls),
        "avg_cascade_recall": avg(cascade_recalls),
        "avg_semantic_null_emission_rate": avg(null_rates),
        "semantic_null_detection": semantic_null_detection,
        "n_unclassified_visual_change": n_unclassified_visual_change,
    }


def eval_edited_pairs(dataset_dir: Path, manifest: list[dict], levels: list[str]) -> dict:
    by_level: dict[str, list[dict]] = {level: [] for level in levels}
    # Accumulated across all pairs at a level for eval/calibration.py --
    # popped off each pair's score dict before it's stored (Delta objects
    # aren't JSON-serializable, and results get written to
    # eval/results/{timestamp}.json).
    calibration_pool: dict[str, dict[str, list]] = {level: {"matched": [], "fp": []} for level in levels}
    for row in manifest:
        if row["kind"] != "edited":
            continue
        pair_dir = dataset_dir / "pairs" / row["pair_id"]
        gt = load_gt_deltas(pair_dir)
        for level in levels:
            deltas = run_pair_at_level(pair_dir, level)
            score = score_pair(deltas, gt)
            calibration_pool[level]["matched"].extend(score.pop("matched_deltas"))
            calibration_pool[level]["fp"].extend(score.pop("false_positive_deltas"))
            by_level[level].append({"pair_id": row["pair_id"], **score})
    return {
        level: {
            "pairs": rows,
            "aggregate": aggregate(rows),
            "calibration": bucket_calibration(calibration_pool[level]["matched"], calibration_pool[level]["fp"]),
        }
        for level, rows in by_level.items()
    }


def eval_null_pairs(dataset_dir: Path, manifest: list[dict]) -> dict:
    out = {}
    for row in manifest:
        if row["kind"] not in ("null_ident", "null_prod", "null_reword"):
            continue
        pair_dir = dataset_dir / "pairs" / row["pair_id"]
        gt = load_gt_deltas(pair_dir)
        deltas = run_pair_at_level(pair_dir, "L0")
        score = score_pair(deltas, gt)
        out[row["pair_id"]] = {
            "kind": row["kind"],
            "n_deltas_emitted": len(deltas),
            "hard_false_positives": score["overall"]["fp"],
            "semantic_null_emission_rate": score["semantic_null_emission_rate"],
        }
    return out


def eval_not_a_pair(dataset_dir: Path, manifest: list[dict]) -> dict:
    out = {}
    for row in manifest:
        if row["kind"] != "not_a_pair":
            continue
        pair_dir = dataset_dir / "pairs" / row["pair_id"]
        doc_a = _resolve_with_pid("A", str(pair_dir / "a" / "L0.pdf"))
        doc_b = _resolve_with_pid("B", str(pair_dir / "b" / "L0.pdf"))
        precheck = check_same_document(doc_a, doc_b)
        out[row["pair_id"]] = {"correctly_refused": not precheck.is_pair, "reason": precheck.reason}
    return out


def eval_chat(dataset_dir: Path, manifest: list[dict]) -> dict:
    """Runs chat over every pair's qa.jsonl. not_a_pair rows short-circuit
    at precheck exactly like cmd_chat does (never build an index) -- their
    QA (always a single refuse-expected probe) is scored as a refusal
    without ever calling chat.answer."""
    all_items = []
    for row in manifest:
        pair_dir = dataset_dir / "pairs" / row["pair_id"]
        qa_items = load_qa_jsonl(pair_dir)
        if not qa_items:
            continue
        if row["kind"] == "not_a_pair":
            for qa in qa_items:
                refused = ChatAnswer(qa["q"], "REFUSED: not a pair", refused=True,
                                      reason="precheck: not a pair", citations=[], retrieved_ids=[])
                all_items.append(ChatEvalItem(row["pair_id"], qa["q"], qa["expected_behavior"],
                                               qa["a"], refused, None))
            continue

        doc_a = _resolve_with_pid("A", str(pair_dir / "a" / "L0.pdf"))
        doc_b = _resolve_with_pid("B", str(pair_dir / "b" / "L0.pdf"))
        precheck = check_same_document(doc_a, doc_b)
        if not precheck.is_pair:
            continue
        tracer = Tracer()
        deltas = compute_deltas(doc_a, doc_b, tracer)
        tracer.finish()
        all_items.extend(run_chat_for_pair(doc_a, doc_b, deltas, qa_items, row["pair_id"]))

    run_judge_on_items(all_items)
    judge_validation = validate_judge()

    from src.chat.llm import judge_is_same_backend

    return {
        "n_questions": len(all_items),
        "refusal_accuracy": score_refusal_accuracy(all_items),
        "groundedness": score_groundedness_aggregate(all_items),
        "correctness": score_correctness_aggregate(all_items),
        "judge_validation": judge_validation,
        # CLAUDE.md eval requirement's own methodological caveat, surfaced
        # in the scorecard itself (not just docs, which go stale) --
        # True until a JUDGE_LLM_*/JUDGE_MODEL credential is configured.
        "judge_is_same_backend_as_chat": judge_is_same_backend(),
        "items": [
            {"pair_id": it.pair_id, "question": it.question,
             "expected_behavior": it.expected_behavior, "refused": it.result.refused,
             "correct": it.correct, "judge_reason": it.judge_reason}
            for it in all_items
        ],
    }


def eval_llm_direct_baseline(dataset_dir: Path, manifest: list[dict]) -> dict:
    from eval.baselines.llm_direct import run_llm_direct_baseline
    return run_llm_direct_baseline(dataset_dir, manifest, n_runs=3)


def run_eval(dataset_dir: Path, levels: list[str], skip_chat: bool = False,
             skip_baseline: bool = False) -> dict:
    manifest = load_manifest(dataset_dir)
    return {
        "dataset": str(dataset_dir),
        "levels": levels,
        "edited": eval_edited_pairs(dataset_dir, manifest, levels),
        "null_pairs": eval_null_pairs(dataset_dir, manifest),
        "not_a_pair": eval_not_a_pair(dataset_dir, manifest),
        "chat": None if skip_chat else eval_chat(dataset_dir, manifest),
        "llm_direct_baseline": None if skip_baseline else eval_llm_direct_baseline(dataset_dir, manifest),
    }


def _fmt_prf1(p: dict) -> str:
    return f"P={p['precision']:.2f} R={p['recall']:.2f} F1={p['f1']:.2f} (tp={p['tp']} fp={p['fp']} fn={p['fn']})"


def print_scorecard(results: dict, diff: dict | None = None) -> None:
    print(f"=== delta-chat eval scorecard -- {results['dataset']} ===\n")

    print("-- edited pairs, by format level --")
    for level, data in results["edited"].items():
        agg = data["aggregate"]
        print(f"  [{level}] overall: {_fmt_prf1(agg['overall'])}")
        for kind, p in agg["by_kind"].items():
            print(f"         {kind:8s} {_fmt_prf1(p)}")
        print(f"         avg primary_recall={agg['avg_primary_recall']}  "
              f"avg cascade_recall={agg['avg_cascade_recall']}  "
              f"avg semantic_null_emission_rate={agg['avg_semantic_null_emission_rate']}")
        print(f"         semantic_null flag detection (engine's own flag vs GT): "
              f"{_fmt_prf1(agg['semantic_null_detection'])}")
        print(f"         unclassified_visual_change (raster recall net, opt-in, "
              f"not scored via P/R/F1): {agg['n_unclassified_visual_change']}")
        print(f"         confidence calibration:")
        for b in data["calibration"]:
            print(f"           {b['band']:12s} precision={b['precision']} (tp={b['tp']} fp={b['fp']} n={b['n']})")
    print()

    print("-- null pairs (any delta is a false positive) --")
    for pair_id, r in results["null_pairs"].items():
        flag = "OK" if r["hard_false_positives"] == 0 else "FAIL"
        extra = f" semantic_null_rate={r['semantic_null_emission_rate']}" if r["semantic_null_emission_rate"] is not None else ""
        print(f"  {pair_id:20s} [{r['kind']:12s}] hard_fp={r['hard_false_positives']} [{flag}]{extra}")
    print()

    print("-- not-a-pair refusal --")
    for pair_id, r in results["not_a_pair"].items():
        flag = "OK" if r["correctly_refused"] else "FAIL"
        print(f"  {pair_id:20s} refused={r['correctly_refused']} [{flag}] ({r['reason']})")
    print()

    print("-- chat correctness / groundedness / refusal accuracy --")
    if results["chat"] is None:
        print("  skipped (--skip-chat)")
    else:
        c = results["chat"]
        print(f"  n_questions={c['n_questions']}")
        ra = c["refusal_accuracy"]
        print(f"  refusal accuracy: {ra['accuracy']} ({ra['n_correct']}/{ra['n']})")
        g = c["groundedness"]
        print(f"  groundedness: fraction_fully_supported={g['fraction_fully_supported']} "
              f"citation_support_rate={g['citation_support_rate']} "
              f"(n_answered_with_citations={g['n_answered_with_citations']})")
        co = c["correctness"]
        caveat = " *** SAME BACKEND AS CHAT -- self-judging risk, treat as an upper bound ***" \
            if c["judge_is_same_backend_as_chat"] else " (independent backend from chat)"
        print(f"  correctness (LLM-judge): accuracy={co['accuracy']} "
              f"(n_judged={co['n_judged']}, n_unparseable={co['n_unparseable']}){caveat}")
        jv = c["judge_validation"]
        print(f"  judge validation vs {jv['n']} hand-checked answers: "
              f"agreement={jv['agreement_rate']} ({jv['n_agree']}/{jv['n']}) "
              f"-- n={jv['n']} is small enough that a perfect score is not strong evidence on its own")
    print()

    print("-- llm_direct baseline (3x @ temperature=0, same metrics path) --")
    if results["llm_direct_baseline"] is None:
        print("  skipped (--skip-baseline)")
    else:
        b = results["llm_direct_baseline"]
        for pair_id, r in b["pairs"].items():
            print(f"  {pair_id:16s} F1 mean={r['f1_mean']:.4f} stdev={r['f1_stdev']:.4f}  "
                  f"n_deltas mean={r['n_deltas_mean']} stdev={r['n_deltas_stdev']}")
        print(f"  aggregate F1 mean={b['aggregate_f1_mean']}  "
              f"within-pair F1 stdev (mean)={b['aggregate_f1_stdev_within_pair_mean']}")
    print()

    if diff:
        print("-- vs previous run --")
        for level in results["edited"]:
            prev = diff.get("edited", {}).get(level, {}).get("aggregate", {}).get("overall", {}).get("f1")
            cur = results["edited"][level]["aggregate"]["overall"]["f1"]
            if prev is not None:
                delta = round(cur - prev, 4)
                arrow = "+" if delta >= 0 else ""
                print(f"  [{level}] overall F1: {prev:.4f} -> {cur:.4f} ({arrow}{delta})")
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/datasets/v0")
    ap.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    ap.add_argument("--results-dir", default="eval/results")
    ap.add_argument("--skip-chat", action="store_true",
                     help="skip chat eval (live LLM calls; slow) for fast iteration on the engine alone")
    ap.add_argument("--skip-baseline", action="store_true",
                     help="skip the llm_direct baseline (live LLM calls; slow)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    levels = args.levels.split(",")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    prior_files = sorted(results_dir.glob("*.json"))
    prior = json.loads(prior_files[-1].read_text()) if prior_files else None

    t0 = time.time()
    results = run_eval(dataset_dir, levels, skip_chat=args.skip_chat, skip_baseline=args.skip_baseline)
    results["elapsed_s"] = round(time.time() - t0, 2)

    print_scorecard(results, diff=prior)
    print(f"(elapsed {results['elapsed_s']}s)")

    out_path = results_dir / f"{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
