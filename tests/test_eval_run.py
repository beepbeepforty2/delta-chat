import io
import pathlib
from contextlib import redirect_stdout

import pytest

from eval.run_eval import aggregate, print_scorecard, run_eval, eval_not_a_pair, eval_null_pairs, load_manifest

DATASET_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0"


def _skip_if_missing():
    if not (DATASET_DIR / "manifest.jsonl").exists():
        pytest.skip("run `make dataset` first")


SCORE_A = {
    "overall": {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "tp": 2, "fp": 0, "fn": 2},
    "by_kind": {
        "add": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0},
        "remove": {"precision": 1.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 1},
        "modify": {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "tp": 1, "fp": 0, "fn": 1},
        "move": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0},
    },
    "primary_recall": 0.5, "cascade_recall": 1.0, "semantic_null_emission_rate": 0.0,
    "semantic_null_detection": {"precision": 1.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 1},
    "n_gt_real": 4, "n_gt_semantic_null": 1, "n_pred": 2,
}
SCORE_B = {
    "overall": {"precision": 0.5, "recall": 1.0, "f1": 0.6667, "tp": 2, "fp": 2, "fn": 0},
    "by_kind": {
        "add": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0},
        "remove": {"precision": 0.0, "recall": 1.0, "f1": 0.0, "tp": 1, "fp": 2, "fn": 0},
        "modify": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0},
        "move": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0},
    },
    "primary_recall": 1.0, "cascade_recall": 1.0, "semantic_null_emission_rate": None,
    "semantic_null_detection": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0},
    "n_gt_real": 2, "n_gt_semantic_null": 0, "n_pred": 4,
}


def test_aggregate_sums_tp_fp_fn():
    agg = aggregate([SCORE_A, SCORE_B])
    assert agg["overall"]["tp"] == 4
    assert agg["overall"]["fp"] == 2
    assert agg["overall"]["fn"] == 2


def test_aggregate_by_kind_sums_correctly():
    agg = aggregate([SCORE_A, SCORE_B])
    assert agg["by_kind"]["remove"]["fp"] == 2
    assert agg["by_kind"]["remove"]["fn"] == 1


def test_aggregate_averages_recall_and_null_rate():
    agg = aggregate([SCORE_A, SCORE_B])
    assert agg["avg_primary_recall"] == 0.75
    assert agg["avg_cascade_recall"] == 1.0
    assert agg["avg_semantic_null_emission_rate"] == 0.0


def test_print_scorecard_renders_key_sections():
    results = {
        "dataset": "eval/datasets/v0",
        "edited": {"L0": {"pairs": [{"pair_id": "edited_000", **SCORE_A}], "aggregate": aggregate([SCORE_A]),
                          "calibration": []}},
        "null_pairs": {"null_ident_900": {"kind": "null_ident", "n_deltas_emitted": 0,
                                            "hard_false_positives": 0, "semantic_null_emission_rate": None}},
        "not_a_pair": {"not_a_pair_903": {"correctly_refused": True, "reason": "drawing numbers differ"}},
        "chat": None,
        "llm_direct_baseline": None,
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_scorecard(results)
    out = buf.getvalue()
    assert "edited pairs, by format level" in out
    assert "[L0] overall:" in out
    assert "null_ident_900" in out and "[OK]" in out
    assert "not_a_pair_903" in out
    assert "skipped (--skip-chat)" in out
    assert "skipped (--skip-baseline)" in out


def test_print_scorecard_flags_same_backend_judge_caveat():
    results = {
        "dataset": "d",
        "edited": {"L0": {"pairs": [], "aggregate": aggregate([SCORE_A]), "calibration": []}},
        "null_pairs": {}, "not_a_pair": {},
        "chat": {
            "n_questions": 5,
            "refusal_accuracy": {"accuracy": 0.8, "n_correct": 4, "n": 5},
            "groundedness": {"fraction_fully_supported": 0.9, "citation_support_rate": 0.9,
                              "n_answered_with_citations": 4},
            "correctness": {"accuracy": 0.6, "n_judged": 5, "n_unparseable": 0},
            "judge_validation": {"n": 5, "n_agree": 5, "agreement_rate": 1.0},
            "judge_is_same_backend_as_chat": True,
        },
        "llm_direct_baseline": None,
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_scorecard(results)
    out = buf.getvalue()
    assert "SAME BACKEND AS CHAT" in out
    assert "n=5 is small enough" in out


def test_print_scorecard_shows_diff_vs_previous():
    results = {
        "dataset": "d", "edited": {"L0": {"pairs": [], "aggregate": aggregate([SCORE_A]), "calibration": []}},
        "null_pairs": {}, "not_a_pair": {}, "chat": None, "llm_direct_baseline": None,
    }
    prior = {"edited": {"L0": {"aggregate": aggregate([SCORE_B])}}}
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_scorecard(results, diff=prior)
    assert "vs previous run" in buf.getvalue()


def test_eval_not_a_pair_correctly_flags_refusal():
    _skip_if_missing()
    manifest = load_manifest(DATASET_DIR)
    out = eval_not_a_pair(DATASET_DIR, manifest)
    assert "not_a_pair_903" in out
    assert out["not_a_pair_903"]["correctly_refused"] is True


def test_eval_null_pairs_zero_hard_fp_for_ident_and_prod():
    _skip_if_missing()
    manifest = load_manifest(DATASET_DIR)
    out = eval_null_pairs(DATASET_DIR, manifest)
    for pair_id in ("null_ident_900", "null_prod_901"):
        if pair_id in out:
            assert out[pair_id]["hard_false_positives"] == 0, pair_id


def test_run_eval_end_to_end_native_only():
    """Full pipeline, L0 only (skip L2's OCR cost in the test suite --
    make eval's default already covers L2). Chat and the llm_direct
    baseline both make live LLM calls, so they're skipped here -- pytest
    must never depend on network/API access; both have their own
    fake-call_llm-based test coverage (test_chat_eval.py, test_llm_direct.py)."""
    _skip_if_missing()
    results = run_eval(DATASET_DIR, levels=["L0"], skip_chat=True, skip_baseline=True)
    assert "L0" in results["edited"]
    assert len(results["edited"]["L0"]["pairs"]) == 6  # edited_000..005
    assert results["not_a_pair"]["not_a_pair_903"]["correctly_refused"] is True
    assert results["chat"] is None
    assert results["llm_direct_baseline"] is None
