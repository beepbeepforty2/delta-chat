"""Alignment: sheet matching -> per-sheet bipartite element matching.

CLAUDE.md decision #3: bipartite matching (scipy Hungarian), cost =
w_text*(1-text_sim) + w_spatial*dist. Confidence = match-cost margin
(best vs second-best) x extraction_confidence (the latter applied in
classify.py, where both elements of a matched pair are available).

Design choice: bucket candidates by ElementType before running Hungarian,
rather than one cross-type cost matrix with an explicit type-mismatch
term. Two reasons: (1) cross-type matches are never semantically correct
here (a line_tag never "becomes" a valve_tag), so a hard partition is a
faithful simplification, not a loosening; (2) it keeps each Hungarian
instance small even at real document density (~800 elements/sheet splits
into ~15 type buckets of tens each, not one 800x800 matrix) -- src/
ingest/pdf_native.py's real-sample tests confirmed that density.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from src.canonical.model import CanonicalDocument, CanonicalElement, CanonicalSheet
from src.delta.register import Transform

W_TEXT = float(os.environ.get("DELTA_W_TEXT", "0.6"))
W_SPATIAL = float(os.environ.get("DELTA_W_SPATIAL", "0.4"))
MAX_MATCH_COST = float(os.environ.get("DELTA_MAX_MATCH_COST", "0.55"))

# Type-bucketing assumes an element's type is stable across a revision --
# true for almost everything, but DeleteNoteKeepPlaceholder (see
# eval/datasets/generator/ops.py) reclassifies a note's role to
# "note_deleted" as *part of* the edit: the type itself is the subject of
# that delta. Groups of types that must share a matching bucket so that
# kind of change is still recoverable; anything not listed buckets alone.
TYPE_MATCH_GROUPS = [
    {"note", "note_deleted", "dcn_note"},
]


def match_group(etype: str) -> str:
    for group in TYPE_MATCH_GROUPS:
        if etype in group:
            return "|".join(sorted(group))
    return etype


@dataclass
class MatchedPair:
    a: Optional[CanonicalElement]
    b: Optional[CanonicalElement]
    cost: Optional[float] = None     # None for pure add/remove (no candidate considered)
    margin: Optional[float] = None   # best-cost vs next-best-alternative gap; None if no alternative existed


def match_sheets(doc_a: CanonicalDocument, doc_b: CanonicalDocument):
    """Match sheets by number. Trivial for this project's single-sheet
    documents, but the seam matters for multi-sheet documents (an
    inserted sheet shifts numbering downstream)."""
    by_number_b = {s.number: s for s in doc_b.sheets}
    pairs = []
    for sa in doc_a.sheets:
        pairs.append((sa, by_number_b.pop(sa.number, None)))
    for sb in by_number_b.values():
        pairs.append((None, sb))
    return pairs


def _text_sim(a: CanonicalElement, b: CanonicalElement) -> float:
    if not a.content and not b.content:
        return 1.0
    return fuzz.ratio(a.content, b.content) / 100.0


def _center(el: CanonicalElement) -> tuple[float, float]:
    return ((el.bbox.x0 + el.bbox.x1) / 2, (el.bbox.y0 + el.bbox.y1) / 2)


def _spatial_dist(a: CanonicalElement, b: CanonicalElement, transform: Transform) -> float:
    ax, ay = _center(a)
    bx, by = transform.apply(*_center(b))
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _cost(a: CanonicalElement, b: CanonicalElement, transform: Transform) -> float:
    return W_TEXT * (1.0 - _text_sim(a, b)) + W_SPATIAL * _spatial_dist(a, b, transform)


def _match_bucket(pool_a: list[CanonicalElement], pool_b: list[CanonicalElement],
                   transform: Transform) -> list[MatchedPair]:
    if not pool_a and not pool_b:
        return []
    if not pool_a:
        return [MatchedPair(None, b) for b in pool_b]
    if not pool_b:
        return [MatchedPair(a, None) for a in pool_a]

    cost = np.empty((len(pool_a), len(pool_b)))
    for i, a in enumerate(pool_a):
        for j, b in enumerate(pool_b):
            cost[i, j] = _cost(a, b, transform)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_a, matched_b = set(), set()
    out = []
    for i, j in zip(row_ind, col_ind):
        c = cost[i, j]
        if c > MAX_MATCH_COST:
            continue  # reject: not actually a good match, leave both unmatched

        row = cost[i, :].copy(); row[j] = np.inf
        col = cost[:, j].copy(); col[i] = np.inf
        best_alt = min(row.min(), col.min())
        margin = None if np.isinf(best_alt) else float(best_alt - c)

        out.append(MatchedPair(pool_a[i], pool_b[j], cost=float(c), margin=margin))
        matched_a.add(i)
        matched_b.add(j)

    for i, a in enumerate(pool_a):
        if i not in matched_a:
            out.append(MatchedPair(a, None))
    for j, b in enumerate(pool_b):
        if j not in matched_b:
            out.append(MatchedPair(None, b))
    return out


def match_elements(sheet_a: Optional[CanonicalSheet], sheet_b: Optional[CanonicalSheet],
                    transform: Transform) -> list[MatchedPair]:
    if sheet_a is None:
        return [MatchedPair(None, e) for e in sheet_b.elements]
    if sheet_b is None:
        return [MatchedPair(e, None) for e in sheet_a.elements]

    groups = {match_group(e.type) for e in sheet_a.elements} | {match_group(e.type) for e in sheet_b.elements}
    matches: list[MatchedPair] = []
    for group in sorted(groups):
        pool_a = [e for e in sheet_a.elements if match_group(e.type) == group]
        pool_b = [e for e in sheet_b.elements if match_group(e.type) == group]
        matches.extend(_match_bucket(pool_a, pool_b, transform))
    return matches
