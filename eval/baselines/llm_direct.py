"""Baseline arm: does the LLM need our deterministic alignment engine at
all, or can it just be asked to diff two documents directly? CLAUDE.md's
eval requirements name this explicitly: "both PDFs to the LLM with a
delta-schema prompt, parsed into the same Delta type, scored through the
identical metrics path; run 3x at temperature 0 and report output variance
(the determinism argument, measured)."

This sends TEXT, not raw PDF bytes/images. A live spike against this
project's configured provider (z.ai/GLM, Anthropic-compatible API) showed
it does not reliably process image/document content blocks: asked to read
a specific equipment tag and flow-rate value clearly printed on a rendered
page, it returned confident, plausible-looking, and completely wrong values
for both -- hallucinating a different tag and a different unit system
rather than reading the page or reporting an error. That's a provider/proxy
reliability problem, not a finding about whether an LLM *could* do this
task, so it would be dishonest to let it silently determine this baseline's
score. Instead each document's element text is extracted deterministically
via the SAME ingest adapters the real engine uses, zone-annotated the same
way chat's retrieval chunks are (src/chat/retrieval.py:build_chunks), and
handed to the LLM as plain text -- the LLM's job is purely the diff/match/
describe step, on the same input granularity the real engine works from.
Swapping to raw PDF bytes is a one-line change (_document_text below) if
run against a provider with verified vision support.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Callable, Optional

from src.canonical.model import CanonicalDocument
from src.chat.llm import MODEL, get_client
from src.delta.model import Delta

SYSTEM_PROMPT = """You are comparing two revisions (A and B) of a P&ID \
(piping & instrumentation diagram) engineering drawing. You will be given \
the extracted text content of both revisions, sheet by sheet, each line \
tagged with its border-grid zone (e.g. "F-7") and element type.

Identify every discrete change between revision A and revision B and \
output them as a JSON array, one object per change, each with EXACTLY \
these fields:
  "kind": one of "add" | "remove" | "modify" | "move"
  "element_type": a short label for what changed, e.g. "note", \
"instrument", "line_tag", "datasheet_row", "title_field", "rev_row"
  "sheet": integer sheet number the change is on
  "zone_a": the zone of the element in revision A, or null if it did not \
exist in A (kind="add")
  "zone_b": the zone of the element in revision B, or null if it does not \
exist in B (kind="remove")
  "description": a one-sentence human-readable description of the change
  "field_changes": an object of {"field_name": {"from": old, "to": new}} \
if the change is a specific field-level value change, else {}

Do not report cosmetic differences that carry no information (identical \
text re-rendered, whitespace-only differences). Output ONLY the JSON \
array -- no markdown fences, no other text."""


def _document_text(doc: CanonicalDocument, label: str) -> str:
    lines = [f"=== Revision {label} ==="]
    for sheet in doc.sheets:
        lines.append(f"-- sheet {sheet.number} --")
        for el in sheet.elements:
            if el.type == "geometry" or not el.content.strip():
                continue
            zone = el.zone or "-"
            lines.append(f"[{zone}] {el.type}: {el.content}")
    return "\n".join(lines)


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_deltas(raw_text: str) -> list[Delta]:
    match = _JSON_ARRAY_RE.search(raw_text)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    deltas = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "kind" not in item or "sheet" not in item:
            continue
        try:
            deltas.append(Delta(
                did=f"llm{i:04d}",
                kind=item["kind"],
                element_type=item.get("element_type", "unknown"),
                id_a=None, id_b=None,
                sheet=int(item["sheet"]),
                zone_a=item.get("zone_a"),
                zone_b=item.get("zone_b"),
                field_changes=item.get("field_changes") or {},
                confidence=1.0,
                description=item.get("description"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return deltas


def _default_call_llm(system: str, user: str, temperature: Optional[float]) -> str:
    client = get_client()
    kwargs = {"model": MODEL, "max_tokens": 4096, "system": system,
              "messages": [{"role": "user", "content": user}]}
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        resp = client.messages.create(**kwargs)
    except Exception:
        # temperature is best-effort -- some Anthropic-compatible proxies
        # reject the param outright; retry once without it rather than
        # failing the whole run over a param the provider doesn't support.
        kwargs.pop("temperature", None)
        resp = client.messages.create(**kwargs)
    return next((b.text for b in resp.content if b.type == "text"), "")


def run_llm_direct(doc_a: CanonicalDocument, doc_b: CanonicalDocument,
                    temperature: Optional[float] = 0.0,
                    call_llm: Optional[Callable[[str, str, Optional[float]], str]] = None) -> list[Delta]:
    """One call: both documents' extracted text -> a Delta list. call_llm is
    injectable (same pattern as src/chat/chat.py:answer) so tests don't need
    a live API call."""
    llm_call = call_llm or _default_call_llm
    user_message = _document_text(doc_a, "A") + "\n\n" + _document_text(doc_b, "B")
    text = llm_call(SYSTEM_PROMPT, user_message, temperature)
    return _parse_deltas(text)


def run_variance_trial(doc_a: CanonicalDocument, doc_b: CanonicalDocument,
                        gt: list[dict], n_runs: int = 3) -> dict:
    """Run the baseline n_runs times at temperature=0 and score each run
    through the identical eval.metrics path the real engine uses. Reports
    per-run scores plus stdev across runs -- the actual, measured
    determinism argument for building a deterministic engine instead of
    just asking the LLM, rather than an assumed one."""
    from eval.metrics import score_pair

    runs = []
    for _ in range(n_runs):
        deltas = run_llm_direct(doc_a, doc_b, temperature=0.0)
        score = score_pair(deltas, gt)
        runs.append({"n_deltas": len(deltas), "overall": score["overall"]})

    f1s = [r["overall"]["f1"] for r in runs]
    n_deltas = [r["n_deltas"] for r in runs]
    return {
        "runs": runs,
        "f1_mean": round(statistics.mean(f1s), 4),
        "f1_stdev": round(statistics.stdev(f1s), 4) if len(f1s) > 1 else 0.0,
        "n_deltas_mean": round(statistics.mean(n_deltas), 2),
        "n_deltas_stdev": round(statistics.stdev(n_deltas), 4) if len(n_deltas) > 1 else 0.0,
    }


def run_llm_direct_baseline(dataset_dir: Path, manifest: list[dict], n_runs: int = 3) -> dict:
    """Baseline over every 'edited' pair in the dataset, L0 only (this arm
    tests the LLM's diffing ability, not format robustness -- that's
    already covered by the main engine's L0/L2 comparison)."""
    from src.cli import _resolve_with_pid

    out = {}
    for row in manifest:
        if row["kind"] != "edited":
            continue
        pair_dir = dataset_dir / "pairs" / row["pair_id"]
        doc_a = _resolve_with_pid("A", str(pair_dir / "a" / "L0.pdf"))
        doc_b = _resolve_with_pid("B", str(pair_dir / "b" / "L0.pdf"))
        gt = json.loads((pair_dir / "gt" / "deltas.json").read_text())
        out[row["pair_id"]] = run_variance_trial(doc_a, doc_b, gt, n_runs=n_runs)

    all_f1 = [r["f1_mean"] for r in out.values()]
    return {
        "pairs": out,
        "aggregate_f1_mean": round(statistics.mean(all_f1), 4) if all_f1 else None,
        "aggregate_f1_stdev_within_pair_mean": (
            round(statistics.mean([r["f1_stdev"] for r in out.values()]), 4) if out else None
        ),
    }


def main() -> int:
    import argparse
    from eval.run_eval import load_manifest

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval/datasets/v0")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    manifest = load_manifest(dataset_dir)
    result = run_llm_direct_baseline(dataset_dir, manifest, n_runs=args.runs)

    print(f"=== llm_direct baseline -- {dataset_dir} ({args.runs} runs/pair) ===\n")
    for pair_id, r in result["pairs"].items():
        print(f"  {pair_id:16s} F1 mean={r['f1_mean']:.4f} stdev={r['f1_stdev']:.4f}  "
              f"n_deltas mean={r['n_deltas_mean']} stdev={r['n_deltas_stdev']}")
    print(f"\naggregate F1 mean={result['aggregate_f1_mean']}  "
          f"within-pair F1 stdev (mean)={result['aggregate_f1_stdev_within_pair_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
