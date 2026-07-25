"""Delta P/R/F1: match predicted Delta records against GT deltas.json
entries by (kind, sheet, zone), with description-similarity as a tiebreak
-- "zone + type" is the matching CLAUDE.md's eval requirements explicitly
allow as an alternative to full location-IoU matching.

Semantic-null GT entries are excluded from the main P/R/F1 (matching them
would reward re-detecting a change as "real" when it isn't) and reported
in two columns instead: `semantic_null_emission_rate` (how often a NORMAL,
unflagged prediction happens to description-match a null GT entry -- a
proxy for how often the engine would be wrong if it never adjudicated
nulls at all) and `semantic_null_detection` (direct precision/recall of
the engine's own `Delta.semantic_null` flag against GT, now that
src/delta/semantic_null.py exists -- a rule for the DELETED-placeholder
case, opt-in LLM adjudication for reword-equivalence per CLAUDE.md
decision #3c). Both are kept: the first measures the cost of doing
nothing, the second measures whether the actual detector is any good.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from src.delta.model import Delta


def _kind(x) -> str:
    return x.kind if isinstance(x, Delta) else x["kind"]


def _sheet(x) -> int:
    return x.sheet if isinstance(x, Delta) else x["sheet"]


def _zone(x) -> Any:
    if isinstance(x, Delta):
        return x.zone_b or x.zone_a
    return x.get("zone_b") or x.get("zone_a")


def _desc(x) -> str:
    return (x.description if isinstance(x, Delta) else x.get("description")) or ""


def _is_cascade(x) -> bool:
    return x.is_cascade if isinstance(x, Delta) else bool(x.get("is_cascade"))


def _semantic_null(x) -> bool:
    return x.semantic_null if isinstance(x, Delta) else bool(x.get("semantic_null"))


@dataclass
class MatchResult:
    matched: list[tuple[Delta, dict]] = field(default_factory=list)
    false_positives: list[Delta] = field(default_factory=list)   # predicted, no GT match
    false_negatives: list[dict] = field(default_factory=list)    # GT, no prediction matched it


MIN_DESC_SIM = 60  # rapidfuzz.fuzz.ratio floor (0-100): a real gate, not just a tiebreak.
# Calibrated against measured scores, not guessed: this codebase's
# descriptions share templated boilerplate ("X added: ...", "X changed:
# ... -> ..."), which inflates fuzz.ratio even between unrelated content --
# "note added: STRAINER..." vs "note added: something else entirely"
# scores 31, "primary change" vs "cascade change" scores 57. A low floor
# doesn't actually gate anything against this domain's templated text.


def match_deltas(pred: list[Delta], gt: list[dict]) -> MatchResult:
    """Two explicit branches, not one pool with a fallback: a same-zone
    candidate is trusted on zone+kind+sheet agreement alone (description
    similarity only breaks ties among same-zone candidates); a
    no-zone-match candidate must additionally clear MIN_DESC_SIM before
    being accepted, since zone can no longer vouch for the pairing."""
    used_gt: set[int] = set()
    matched: list[tuple[Delta, dict]] = []
    for p in pred:
        candidates = [i for i, g in enumerate(gt)
                      if i not in used_gt and _kind(g) == _kind(p) and _sheet(g) == _sheet(p)]
        same_zone = [i for i in candidates if _zone(gt[i]) == _zone(p)]
        if same_zone:
            best = max(same_zone, key=lambda i: fuzz.ratio(_desc(p), _desc(gt[i])))
            matched.append((p, gt[best]))
            used_gt.add(best)
            continue
        if not candidates:
            continue
        scored = [(fuzz.ratio(_desc(p), _desc(gt[i])), i) for i in candidates]
        best_score, best = max(scored)
        if best_score < MIN_DESC_SIM:
            continue
        matched.append((p, gt[best]))
        used_gt.add(best)

    matched_pred_ids = {id(p) for p, _ in matched}
    false_positives = [p for p in pred if id(p) not in matched_pred_ids]
    false_negatives = [gt[i] for i in range(len(gt)) if i not in used_gt]
    return MatchResult(matched, false_positives, false_negatives)


def prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def score_pair(pred: list[Delta], gt: list[dict]) -> dict:
    """One pair's full scorecard slice: overall + per-kind + primary/cascade
    + semantic-null column, all excluding semantic-null GT entries from the
    "real change" P/R/F1 per the module docstring."""
    real_gt = [g for g in gt if not _semantic_null(g)]
    null_gt = [g for g in gt if _semantic_null(g)]

    # semantic-null column FIRST: a predicted delta that correctly explains
    # a semantic-null GT entry is claimed by this column and removed from
    # the real-change pool below -- otherwise the same known gap (no
    # semantic adjudication) gets double-penalized as both "flagged a
    # null as real" AND "hallucinated an unexplained change".
    null_result = match_deltas(pred, null_gt) if null_gt else MatchResult()
    semantic_null_emission_rate = len(null_result.matched) / len(null_gt) if null_gt else None
    claimed_by_null = {id(p) for p, _ in null_result.matched}

    # Our own semantic_null flag (src/delta/semantic_null.py) also excludes
    # a prediction from the real-change pool: if the engine itself
    # identified a delta as non-substantive, it shouldn't count as a real
    # false positive even when match_deltas' fuzzy description matching
    # doesn't happen to pair it with a GT null entry.
    #
    # unclassified_visual_change (src/delta/raster_diff.py +
    # src/delta/raster_join.py, opt-in) is also excluded: its kind never
    # matches any GT kind by construction
    # (GT only has add/remove/modify/move), so leaving it in would count
    # every single raster-recall hit as an automatic false positive against
    # a P/R/F1 bar it was never meant to be judged by -- it's reported as
    # its own count instead (n_unclassified_visual_change below).
    self_flagged = [p for p in pred if _semantic_null(p)]
    unclassified_visual = [p for p in pred if p.kind == "unclassified_visual_change"]
    pred_real = [p for p in pred if id(p) not in claimed_by_null and not _semantic_null(p)
                 and p.kind != "unclassified_visual_change"]

    # Direct precision/recall of our OWN flag against GT null entries --
    # separate from semantic_null_emission_rate above (which measures
    # whether a NORMAL, unflagged prediction happens to match a null GT
    # entry). This measures whether the flag itself is trustworthy.
    flagged_result = match_deltas(self_flagged, null_gt)
    semantic_null_detection = prf1(
        len(flagged_result.matched), len(flagged_result.false_positives), len(flagged_result.false_negatives),
    )

    result = match_deltas(pred_real, real_gt)
    overall = prf1(len(result.matched), len(result.false_positives), len(result.false_negatives))
    # Exposed for eval/calibration.py: the same matched/false-positive
    # populations overall P/R/F1 is computed from, so a caller can bucket
    # them by Delta.confidence without re-deriving the null-exclusion and
    # matching logic above.
    matched_deltas = [p for p, _ in result.matched]
    false_positive_deltas = list(result.false_positives)

    by_kind = {}
    for kind in ("add", "remove", "modify", "move"):
        gt_k = [g for g in real_gt if _kind(g) == kind]
        pred_k = [p for p in pred_real if p.kind == kind]
        r = match_deltas(pred_k, gt_k)
        by_kind[kind] = prf1(len(r.matched), len(r.false_positives), len(r.false_negatives))

    primary_gt = [g for g in real_gt if not _is_cascade(g)]
    cascade_gt = [g for g in real_gt if _is_cascade(g)]
    r_primary = match_deltas(pred_real, primary_gt)
    r_cascade = match_deltas(pred_real, cascade_gt)
    primary_recall = len(r_primary.matched) / len(primary_gt) if primary_gt else 1.0
    cascade_recall = len(r_cascade.matched) / len(cascade_gt) if cascade_gt else 1.0

    return {
        "overall": overall,
        "by_kind": by_kind,
        "primary_recall": round(primary_recall, 4),
        "cascade_recall": round(cascade_recall, 4),
        "semantic_null_emission_rate": (
            round(semantic_null_emission_rate, 4) if semantic_null_emission_rate is not None else None
        ),
        "semantic_null_detection": semantic_null_detection,
        "n_gt_real": len(real_gt), "n_gt_semantic_null": len(null_gt), "n_pred": len(pred),
        "n_unclassified_visual_change": len(unclassified_visual),
        "matched_deltas": matched_deltas, "false_positive_deltas": false_positive_deltas,
    }
