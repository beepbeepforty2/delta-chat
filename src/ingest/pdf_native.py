"""Native (vector) PDF adapter: fitz extraction -> canonical.

Spike findings that drove this design (see PLAN in git history / CLAUDE.md
Steps #2 for the write-up): fitz's own block/line grouping in
page.get_text("dict") already reassembles most word-fragmented text (the
"alt" producer in eval/datasets/generator/render.py draws one word per
drawString call) into single lines with synthetic " " spans between words.
It does NOT reliably do this across unusually large intra-element gaps
(observed: the generator's double-space rev_row text gets split into three
separate fitz lines under the alt producer, at an ~8.9pt gap on a 7.09pt
font). So extraction here ignores fitz's line/block boundaries entirely and
re-clusters raw spans by (same baseline, small x-gap) instead of trusting
them -- this is the one clustering step that must survive both producer
variants.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import fitz  # PyMuPDF

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalSheet
from src.canonical.classify import classify, classify_geometry
from src.canonical.tags import parse_instrument
from src.canonical.zones import compute_zone
from src.ingest.base import FormatAdapter, element_id

MIN_TEXT_WORDS = int(os.environ.get("PDF_NATIVE_MIN_TEXT_WORDS", "20"))
GAP_MULTIPLIER = float(os.environ.get("PDF_NATIVE_GAP_MULTIPLIER", "3.0"))
Y_TOL_RATIO = float(os.environ.get("PDF_NATIVE_Y_TOL_RATIO", "0.5"))
RASTER_DPI = int(os.environ.get("PDF_NATIVE_RASTER_DPI", "150"))
RASTER_CACHE_DIR = os.environ.get("PDF_NATIVE_RASTER_CACHE_DIR", "raster_cache")
# Real vendor instrument bubbles stack system/function/loop across 3
# separate baselines inside one circle glyph (see _stack_instrument_bubbles
# docstring); padding around the circle's own bbox, as a fraction of the
# circle's own size, when looking for those stacked tokens. 0.6, not a
# small margin: inspecting the real Lift/Export samples directly found the
# "system" (area/unit) label conventionally sits OUTSIDE the bubble to one
# side, at ~0.5x the circle's own width away -- unlike func/loop, which sit
# inside. Tuned empirically against both real samples: the detected count
# plateaus at 0.6-1.0 and only grows marginally even at 2x that, confirming
# this isn't bridging to unrelated nearby bubbles.
INSTRUMENT_BUBBLE_PADDING_RATIO = float(os.environ.get("PDF_NATIVE_INSTRUMENT_BUBBLE_PADDING_RATIO", "0.6"))

_WS_RE = re.compile(r"\s+")
_INSTRUMENT_FUNC_RE = re.compile(r'^[A-Z]{2,4}$')
_INSTRUMENT_SYSTEM_RE = re.compile(r'^\d{2}$')
_INSTRUMENT_LOOP_RE = re.compile(r'^-?\d{3,5}$')


def _extract_spans(page: "fitz.Page") -> list[dict]:
    """Flatten all non-empty text spans, ignoring fitz's own block/line
    grouping (see module docstring for why)."""
    spans = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # skip image blocks
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"] != "":
                    spans.append(span)
    return spans


def _cluster_spans(spans: list[dict]) -> list[list[dict]]:
    """Group spans into logical text runs: same baseline y (within a
    font-size-relative tolerance) and small x-gap (within GAP_MULTIPLIER
    font sizes) to the previous span, sorted left-to-right."""
    ordered = sorted(spans, key=lambda s: (round(s["origin"][1], 1), s["origin"][0]))
    clusters: list[list[dict]] = []
    cur: list[dict] = []
    for s in ordered:
        if not cur:
            cur = [s]
            continue
        prev = cur[-1]
        same_baseline = abs(s["origin"][1] - prev["origin"][1]) <= Y_TOL_RATIO * prev["size"]
        gap = s["bbox"][0] - prev["bbox"][2]
        # gap must be non-negative (strictly left-to-right, non-overlapping):
        # two unrelated spans that are merely close in y but far/overlapping
        # in x are not a text run, regardless of the y-distance check above.
        close_enough = 0 <= gap <= GAP_MULTIPLIER * prev["size"]
        if same_baseline and close_enough:
            cur.append(s)
        else:
            clusters.append(cur)
            cur = [s]
    if cur:
        clusters.append(cur)
    return clusters


def _cluster_text(cluster: list[dict]) -> str:
    """Join a cluster's span texts. Spans fitz already grouped on one line
    include synthetic " " spans between words (concatenate as-is); spans
    bridged across a fitz line boundary by our own clustering have no such
    separator, so insert one when there's a real gap. Whitespace is then
    normalized -- exact run-length (e.g. double spaces) is not preserved,
    a documented, accepted loss (see plan open question on whitespace
    fidelity)."""
    parts = []
    prev_end = None
    for s in cluster:
        if prev_end is not None and s["bbox"][0] - prev_end > 0.5:
            parts.append(" ")
        parts.append(s["text"])
        prev_end = s["bbox"][2]
    return _WS_RE.sub(" ", "".join(parts)).strip()


def _cluster_bbox(cluster: list[dict], page_w: float, page_h: float) -> BBox:
    x0 = min(s["bbox"][0] for s in cluster)
    y0 = min(s["bbox"][1] for s in cluster)
    x1 = max(s["bbox"][2] for s in cluster)
    y1 = max(s["bbox"][3] for s in cluster)
    return BBox(x0 / page_w, y0 / page_h, x1 / page_w, y1 / page_h)


def _drawing_kind(items: list[tuple]) -> str:
    types = [it[0] for it in items]
    if "re" in types:
        return "rect"
    if types and all(t == "c" for t in types):
        return "circle"
    if types and all(t == "l" for t in types):
        return "line"
    return "other"


def _text_elements(page: "fitz.Page", sheet_no: int) -> list[CanonicalElement]:
    page_w, page_h = page.rect.width, page.rect.height
    out = []
    for cluster in _cluster_spans(_extract_spans(page)):
        text = _cluster_text(cluster)
        if not text:
            continue
        bbox = _cluster_bbox(cluster, page_w, page_h)
        # anchor = first span's baseline-left origin, the direct analog of
        # the generator's Element.anchor (drawString's x,y) -- zone/type
        # must be computed from this point, not the bbox centroid, or long
        # strings drift into the wrong zone column.
        ax, ay = cluster[0]["origin"]
        ax_norm, ay_norm = ax / page_w, ay / page_h
        etype, attrs = classify(text, ax_norm, ay_norm)
        zone = compute_zone(ax_norm, ay_norm)
        eid = element_id(sheet_no, etype, text, bbox)
        out.append(CanonicalElement(
            id=eid, type=etype, content=text, bbox=bbox, sheet=sheet_no,
            zone=zone, extraction_confidence=1.0, attrs=attrs,
        ))
    return out


def _geometry_elements(page: "fitz.Page", sheet_no: int) -> list[CanonicalElement]:
    page_w, page_h = page.rect.width, page.rect.height
    out = []
    for drawing in page.get_drawings():
        items = drawing["items"]
        kind = _drawing_kind(items)
        rect = drawing.get("rect")
        if rect is None:
            continue
        # NB: do not skip on rect.is_empty -- a perfectly horizontal or
        # vertical geom_line has a zero-height/width bbox by construction
        # and fitz reports that as "empty", but it is a real element.
        bbox_norm = (rect.x0 / page_w, rect.y0 / page_h, rect.x1 / page_w, rect.y1 / page_h)
        result = classify_geometry(kind, bbox_norm)
        if result is None:  # excluded (sheet border rect)
            continue
        etype, attrs = result
        bbox = BBox(*bbox_norm)
        # anchor: line -> start point p1 (matches geom_line.anchor in the
        # GT model); circle -> bbox center (matches geom_circle.anchor,
        # which the generator passes as the circle's own center).
        if kind == "line" and items and items[0][0] == "l":
            p1 = items[0][1]
            ax_norm, ay_norm = p1.x / page_w, p1.y / page_h
        else:
            ax_norm = (bbox.x0 + bbox.x1) / 2
            ay_norm = (bbox.y0 + bbox.y1) / 2
        zone = compute_zone(ax_norm, ay_norm)
        eid = element_id(sheet_no, etype, "", bbox)
        out.append(CanonicalElement(
            id=eid, type=etype, content="", bbox=bbox, sheet=sheet_no,
            zone=zone, extraction_confidence=1.0, attrs=attrs,
        ))
    return out


def _is_orphan_token(el: CanonicalElement) -> bool:
    """A tier-3 catch-all classification (see classify.py) -- a short,
    unstructured token that didn't match any known tag/field shape on its
    own. The only kind of element _stack_instrument_bubbles is allowed to
    consume; anything already meaningfully classified is left untouched."""
    return el.type == "unknown" and el.attrs.get("classification_rule", "").startswith("fallback:")


def _stack_instrument_bubbles(text_elements: list[CanonicalElement],
                               circle_elements: list[CanonicalElement],
                               sheet_no: int) -> list[CanonicalElement]:
    """Second pass over already-clustered text: real vendor instrument
    bubbles stack system/function/loop text across up to 3 separate
    baselines inside one circle glyph (e.g. "26" / "PI" / "9055" on
    distinct lines), unlike the synthetic generator's single-line
    "FUNC LOOP SYSTEM" format INSTRUMENT_RE expects. Same-baseline
    clustering correctly keeps these apart as distinct text runs -- this
    is a real composition-format gap, not a clustering bug -- so recovery
    needs a position-gated second pass, not a looser same-baseline
    tolerance.

    Gated on real circle geometry (not a free-floating vertical-stacking
    heuristic): only orphan tokens (see _is_orphan_token) whose center
    falls inside -- or just outside, within INSTRUMENT_BUBBLE_PADDING_RATIO
    -- an actual circle glyph's own bbox are ever considered, so this can
    never accidentally merge unrelated stray tokens elsewhere on a dense
    real sheet. A circle "adopts" at most one token per shape (one
    func-shaped, one system-shaped, one loop-shaped); reassembles them in
    parse_instrument's expected order (real vendor stacking order differs
    from that order); and only replaces the 3 orphans with one instrument
    element if the reassembled string actually parses."""
    if not circle_elements:
        return text_elements

    orphans = {el.id: el for el in text_elements if _is_orphan_token(el)}
    if not orphans:
        return text_elements

    consumed_ids: set = set()
    new_elements: list[CanonicalElement] = []

    for circle in circle_elements:
        cx0, cy0, cx1, cy1 = circle.bbox.x0, circle.bbox.y0, circle.bbox.x1, circle.bbox.y1
        pad_x = (cx1 - cx0) * INSTRUMENT_BUBBLE_PADDING_RATIO
        pad_y = (cy1 - cy0) * INSTRUMENT_BUBBLE_PADDING_RATIO
        px0, py0, px1, py1 = cx0 - pad_x, cy0 - pad_y, cx1 + pad_x, cy1 + pad_y

        candidates = []
        for el in orphans.values():
            if el.id in consumed_ids:
                continue
            ex = (el.bbox.x0 + el.bbox.x1) / 2
            ey = (el.bbox.y0 + el.bbox.y1) / 2
            if px0 <= ex <= px1 and py0 <= ey <= py1:
                candidates.append(el)

        func_el = next((el for el in candidates if _INSTRUMENT_FUNC_RE.match(el.content.strip())), None)
        system_el = next((el for el in candidates if _INSTRUMENT_SYSTEM_RE.match(el.content.strip())), None)
        loop_el = next((el for el in candidates if _INSTRUMENT_LOOP_RE.match(el.content.strip())), None)
        if not (func_el and system_el and loop_el):
            continue
        if len({func_el.id, system_el.id, loop_el.id}) != 3:
            continue  # one token can't satisfy two roles at once

        composite = f"{func_el.content.strip()} {loop_el.content.strip()} {system_el.content.strip()}"
        parsed = parse_instrument(composite)
        if not parsed:
            continue

        parts = [func_el, system_el, loop_el]
        bbox = BBox(min(p.bbox.x0 for p in parts), min(p.bbox.y0 for p in parts),
                    max(p.bbox.x1 for p in parts), max(p.bbox.y1 for p in parts))
        eid = element_id(sheet_no, "instrument", composite, bbox)
        new_elements.append(CanonicalElement(
            id=eid, type="instrument", content=composite, bbox=bbox, sheet=sheet_no,
            zone=func_el.zone, extraction_confidence=1.0, attrs=parsed,
        ))
        consumed_ids.update(p.id for p in parts)

    if not new_elements:
        return text_elements
    kept = [el for el in text_elements if el.id not in consumed_ids]
    return kept + new_elements


def _revision_label(elements: list[CanonicalElement]) -> Optional[str]:
    for el in elements:
        if el.type == "title_field" and el.attrs.get("field") == "rev":
            return el.attrs.get("value")
    return None


def _rasterize(doc: "fitz.Document", pid: str) -> dict[int, str]:
    os.makedirs(RASTER_CACHE_DIR, exist_ok=True)
    raster_paths = {}
    for i, page in enumerate(doc):
        sheet_no = i + 1
        out_path = os.path.join(RASTER_CACHE_DIR, f"{pid}_sheet{sheet_no}.png")
        pix = page.get_pixmap(dpi=RASTER_DPI)
        pix.save(out_path)
        raster_paths[sheet_no] = out_path
    return raster_paths


class PdfNativeAdapter(FormatAdapter):
    format_name = "pdf_native"

    def detect(self, path: str) -> bool:
        if not path.lower().endswith(".pdf"):
            return False
        try:
            doc = fitz.open(path)
            if doc.page_count == 0:
                return False
            n_words = len(doc[0].get_text("text").split())
            doc.close()
            return n_words >= MIN_TEXT_WORDS
        except Exception:
            return False

    def ingest(self, pid: str, path: str) -> CanonicalDocument:
        doc = fitz.open(path)
        sheets = []
        all_elements_by_sheet: dict[int, list[CanonicalElement]] = {}
        for i, page in enumerate(doc):
            sheet_no = i + 1
            text_els = _text_elements(page, sheet_no)
            geom_els = _geometry_elements(page, sheet_no)
            circles = [el for el in geom_els if el.attrs.get("geom_kind") == "circle"]
            text_els = _stack_instrument_bubbles(text_els, circles, sheet_no)
            elements = text_els + geom_els
            all_elements_by_sheet[sheet_no] = elements
            sheets.append(CanonicalSheet(
                number=sheet_no, width=page.rect.width, height=page.rect.height,
                elements=elements,
            ))
        raster_paths = _rasterize(doc, pid)
        revision_label = _revision_label(sheets[0].elements) if sheets else None
        doc.close()
        return CanonicalDocument(
            pid=pid, source_format="pdf_native", revision_label=revision_label,
            sheets=sheets, raster_paths=raster_paths,
        )
