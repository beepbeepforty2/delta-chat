"""Heuristic element-type classification: the adapter sees rendered text +
position, never the GT `role` the generator used to build it. Tiered,
content-first: regex/structural matches (Tier 1) beat region-based fallback
(Tier 2), which beats generic catch-alls (Tier 3).

Every result carries attrs["type_confidence"] (1.0 for an exact structural
match, lower for a region-only guess) and attrs["classification_rule"]
(e.g. "regex:line_tag", "region:title_block", "fallback:text") so eval can
later slice accuracy by which rule fired. This is separate from
CanonicalElement.extraction_confidence, which stays 1.0 for native PDF per
model.py's own convention (extraction != classification).
"""
from __future__ import annotations

import os
import re
from typing import Optional

from .tags import (
    parse_line_tag, parse_instrument, parse_valve_tag, parse_nozzle,
    parse_equipment_tag, parse_title_field, DCN_RE,
)
from .zones import is_zone_label_shaped

DELETED_NOTE_RE = re.compile(r'^\d+(?:-\d+)?\.\s*DELETED\.$')
NUMBERED_NOTE_RE = re.compile(r'^\d+\.\s+.+')
REV_ROW_RE = re.compile(r'^[A-Z]\s{1,2}\d{4}-\d{2}-\d{2}\s{1,2}.+$')

# Region-based fallback rects, normalized [0,1] top-left/y-down
# (x0, y0, x1, y1). A drawing's title-block/datasheet position is a
# template convention, not a physical law -- different vendors put it in
# different corners. Each entry below is a *candidate list*, checked in
# order, one candidate per known template, rather than one hardcoded guess:
#   - synthetic generator (eval/datasets/generator/content.py): bottom-right
#   - real vendor sample (data/samples/, MAN Energy Solutions, see
#     data/samples/PROVENANCE.md): datasheet bottom-left; confirmed by
#     visually inspecting both real 26-KA-901/902 sheets. No separate
#     REV/DRAWN/CHECKED title-block stamp was locatable on those two real
#     exemplars -- documented gap, not extended to TITLE_BLOCK_RECTS.
def _rects(env_name: str, default: list[tuple[float, float, float, float]]):
    raw = os.environ.get(env_name)
    if not raw:
        return default
    return [tuple(float(v) for v in group.split(",")) for group in raw.split(";")]

TITLE_BLOCK_RECTS = _rects("TITLE_BLOCK_RECTS", [
    (0.70, 0.85, 1.0, 1.0),   # synthetic generator
])
DATASHEET_RECTS = _rects("DATASHEET_RECTS", [
    (0.55, 0.55, 1.0, 0.80),  # synthetic generator
    (0.0, 0.68, 0.25, 0.88),  # real vendor sample (MAN Energy Solutions template)
])

# Border rectangle exclusion: c.rect() drawn 6mm inset on an 841x594mm A1
# sheet is not a GT element and must not be emitted.
BORDER_MARGIN_TOL = float(os.environ.get("BORDER_MARGIN_TOL", "0.03"))


def _in_any_rect(x_norm: float, y_norm: float, rects: list[tuple[float, float, float, float]]) -> bool:
    for x0, y0, x1, y1 in rects:
        if x0 <= x_norm <= x1 and y0 <= y_norm <= y1:
            return True
    return False


def _result(etype: str, attrs: dict, confidence: float, rule: str) -> tuple[str, dict]:
    attrs = dict(attrs)
    attrs["type_confidence"] = confidence
    attrs["classification_rule"] = rule
    return etype, attrs


def classify(text: str, x_norm: float, y_norm: float) -> tuple[str, dict]:
    """Classify a logical text run into (ElementType, attrs)."""
    t = text.strip()

    # ---- Tier 1: content-unambiguous structural/regex matches ------------
    if is_zone_label_shaped(t, x_norm, y_norm):
        return _result("zone_label", {}, 1.0, "regex:zone_label")

    parsed = parse_line_tag(t)
    if parsed:
        return _result("line_tag", parsed, 1.0, "regex:line_tag")

    parsed = parse_instrument(t)
    if parsed:
        return _result("instrument", parsed, 1.0, "regex:instrument")

    parsed = parse_valve_tag(t)
    if parsed:
        return _result("valve_tag", parsed, 1.0, "regex:valve_tag")

    parsed = parse_nozzle(t)
    if parsed:
        return _result("nozzle", parsed, 1.0, "regex:nozzle")

    parsed = parse_equipment_tag(t)
    if parsed:
        return _result("equipment_tag", parsed, 1.0, "regex:equipment_tag")

    parsed = parse_title_field(t)
    if parsed:
        return _result("title_field", parsed, 1.0, "regex:title_field")

    if DELETED_NOTE_RE.match(t):
        head = t.split(".", 1)[0]
        attrs = {"deleted": True}
        if "-" in head:  # collapsed range, e.g. "6-7. DELETED." -- no single note_no
            lo, hi = head.split("-")
            attrs["range"] = [int(lo), int(hi)]
        else:
            attrs["note_no"] = int(head)
        return _result("note_deleted", attrs, 1.0, "regex:note_deleted")

    if DCN_RE.search(t) and NUMBERED_NOTE_RE.match(t):
        note_no = int(t.split(".", 1)[0])
        return _result("dcn_note", {"note_no": note_no, "dcns": DCN_RE.findall(t)}, 1.0, "regex:dcn_note")

    if REV_ROW_RE.match(t):
        return _result("rev_row", {}, 1.0, "regex:rev_row")

    if NUMBERED_NOTE_RE.match(t):
        note_no = int(t.split(".", 1)[0])
        return _result("note", {"note_no": note_no}, 1.0, "regex:note")

    # ---- Tier 2: region-based fallback ------------------------------------
    if _in_any_rect(x_norm, y_norm, DATASHEET_RECTS):
        return _result("datasheet_row", {"value": t}, 0.5, "region:datasheet")

    if _in_any_rect(x_norm, y_norm, TITLE_BLOCK_RECTS):
        return _result("title_field", {"field": "unknown", "value": t}, 0.5, "region:title_block")

    # ---- Tier 3: generic catch-alls ---------------------------------------
    if not t:
        return _result("unknown", {}, 0.1, "fallback:empty")
    if " " not in t and len(t) <= 20:
        return _result("unknown", {}, 0.3, "fallback:tag_like")
    return _result("text", {}, 0.3, "fallback:text")


def is_border_rect(bbox_norm: tuple[float, float, float, float]) -> bool:
    """True if a rect-shaped geometry item's bbox matches the sheet-bordering
    rectangle (drawn 6mm inset, not a GT element)."""
    x0, y0, x1, y1 = bbox_norm
    return (x0 <= BORDER_MARGIN_TOL and y0 <= BORDER_MARGIN_TOL and
            x1 >= 1 - BORDER_MARGIN_TOL and y1 >= 1 - BORDER_MARGIN_TOL)


def classify_geometry(kind: str, bbox_norm: tuple[float, float, float, float]) -> Optional[tuple[str, dict]]:
    """kind: "line" | "circle" | "rect", as determined by the adapter from
    page.get_drawings() path items. Returns None for the excluded border
    rect (no GT counterpart -- must not be emitted)."""
    if kind == "rect" and is_border_rect(bbox_norm):
        return None
    x0, y0, x1, y1 = bbox_norm
    if kind == "circle":
        r_norm = ((x1 - x0) + (y1 - y0)) / 4
        return _result("geometry", {"geom_kind": "circle", "r_norm": r_norm}, 1.0, "geom:circle")
    if kind == "line":
        return _result("geometry", {"geom_kind": "line"}, 1.0, "geom:line")
    return _result("geometry", {"geom_kind": kind}, 0.5, "geom:other")
