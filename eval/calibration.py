"""Confidence calibration: DESIGN.md decision #3 defines a confidence
formula (match-cost margin x extraction_conf_a x extraction_conf_b, see
src/delta/classify.py::_confidence) but nothing previously checked whether
it's actually predictive of correctness. Buckets predicted deltas (matched
vs. false-positive, from eval.metrics.score_pair's "matched_deltas"/
"false_positive_deltas") by confidence band and reports precision per band
-- a real reliability check, not a formula nobody validated. A well-
calibrated engine should show precision increasing with confidence band;
this reports the actual measured shape whether or not that holds.
"""
from __future__ import annotations

from src.delta.model import Delta

# Upper bound exclusive except the last band, which is inclusive of 1.0.
BANDS: list[tuple[float, float]] = [(0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0001)]


def _band_label(lo: float, hi: float) -> str:
    hi_disp = "1.0" if hi > 1.0 else str(hi)
    return f"{lo}-{hi_disp}"


def _band_index(confidence: float) -> int:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= confidence < hi:
            return i
    return len(BANDS) - 1  # >= 1.0 (shouldn't happen; confidence is clamped, but stay safe)


def bucket_calibration(matched: list[Delta], false_positives: list[Delta]) -> list[dict]:
    tp_counts = [0] * len(BANDS)
    fp_counts = [0] * len(BANDS)
    for d in matched:
        tp_counts[_band_index(d.confidence)] += 1
    for d in false_positives:
        fp_counts[_band_index(d.confidence)] += 1

    out = []
    for i, (lo, hi) in enumerate(BANDS):
        tp, fp = tp_counts[i], fp_counts[i]
        n = tp + fp
        out.append({
            "band": _band_label(lo, hi), "tp": tp, "fp": fp, "n": n,
            "precision": round(tp / n, 4) if n else None,
        })
    return out
