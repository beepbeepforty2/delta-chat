"""Markup overlay: renders delta annotations onto each revision's retained
L0 raster (CanonicalDocument.raster_paths) -- README Plan step 7 (bonus).

Draws on the raster, not the original PDF's vector page, so the same code
path works identically for native and scanned inputs: raster_paths is
populated by every FormatAdapter (src/canonical/model.py's own docstring:
"L0 raster page (retained for markup + recall net)"), and
CanonicalElement.bbox is already normalized [0,1] against that same
raster's page for both adapters -- denormalizing against the raster
image's own pixel size (opened directly, not CanonicalSheet.width/height,
which holds the *pre-rasterization* page size in points for native and
pixels for scanned -- two different units, neither the raster's actual
resolution) is the one coordinate step needed, no vector-vs-raster
handling required.

Color by kind (the traditional "what changed" markup vocabulary):
  add    -> green, drawn on B's raster only (didn't exist in A)
  remove -> red, drawn on A's raster only (doesn't exist in B)
  modify -> amber, drawn on BOTH A's and B's raster (old value / new value)
  move   -> blue outline only (no fill), drawn on both at old and new position
Cascade members are drawn thinner and unfilled rather than omitted --
seeing *where* a renumbering cascade landed is still useful, just visually
secondary to primary changes (same "primary first" priority report.py's
severity sort already applies). Severity isn't separately encoded here
(see delta_report.md for that); this is the traditional add/remove/
modify/move overlay a reviewer expects from a markup pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement
from src.delta.model import Delta

COLORS: dict[str, tuple[int, int, int]] = {
    "add": (34, 139, 34),      # forest green
    "remove": (200, 30, 30),   # red
    "modify": (222, 145, 0),   # amber
    "move": (30, 100, 220),    # blue
    "unclassified_visual_change": (124, 58, 168),  # violet -- drawn via
    # bbox_a directly (see _collect_visual_change_boxes below), not via
    # the id_a/id_b element lookup _collect_boxes uses for every other
    # kind, since raster_join.py's deltas have neither.
}
PRIMARY_WIDTH = 4
CASCADE_WIDTH = 2
FILL_ALPHA = 55  # out of 255 -- translucent so drawing content underneath stays legible
PAD_PX = 4       # a box drawn exactly on the text bbox reads as clipping the glyphs


def _index_elements(doc: CanonicalDocument) -> dict[str, CanonicalElement]:
    return {el.id: el for sheet in doc.sheets for el in sheet.elements}


def _denormalize(el: CanonicalElement, size: tuple[int, int]) -> tuple[int, int, int, int]:
    return _denormalize_bbox(el.bbox, size)


def _denormalize_bbox(b: BBox, size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    x0, y0, x1, y1 = b.x0 * w - PAD_PX, b.y0 * h - PAD_PX, b.x1 * w + PAD_PX, b.y1 * h + PAD_PX
    return (max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1)))


def _draw_box(draw: ImageDraw.ImageDraw, box, color, width, filled) -> None:
    if filled:
        draw.rectangle(box, outline=color, width=width, fill=(*color, FILL_ALPHA))
    else:
        draw.rectangle(box, outline=color, width=width)


def _draw_dashed_box(draw: ImageDraw.ImageDraw, box, color, width, dash_len=6, gap_len=4) -> None:
    """PIL's ImageDraw has no native dashed-rectangle support -- hand-
    rolled by walking each of the 4 edges in dash/gap increments. Used
    only for unclassified_visual_change: visually distinct (dashed, no
    fill) from every typed add/remove/modify/move box, since this is a
    located-but-not-characterized region, not a confident classification."""
    x0, y0, x1, y1 = box
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for (ex0, ey0), (ex1, ey1) in edges:
        length = max(abs(ex1 - ex0), abs(ey1 - ey0))
        if length == 0:
            continue
        ux, uy = (ex1 - ex0) / length, (ey1 - ey0) / length
        pos = 0.0
        while pos < length:
            seg_end = min(pos + dash_len, length)
            draw.line([(ex0 + ux * pos, ey0 + uy * pos), (ex0 + ux * seg_end, ey0 + uy * seg_end)],
                       fill=color, width=width)
            pos += dash_len + gap_len


LEGEND_ITEMS = [("add", "Added"), ("remove", "Removed"), ("modify", "Modified"), ("move", "Moved"),
                ("unclassified_visual_change", "Unclassified visual change")]


def _draw_legend(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    x, y = 12, 12
    draw.rectangle((x - 4, y - 4, x + 150, y + len(LEGEND_ITEMS) * 20 + 2),
                    fill=(255, 255, 255, 220), outline=(0, 0, 0, 255), width=1)
    for kind, label in LEGEND_ITEMS:
        color = COLORS[kind]
        draw.rectangle((x, y, x + 18, y + 12), outline=color, width=3, fill=(*color, FILL_ALPHA))
        draw.text((x + 26, y - 2), label, fill=(0, 0, 0, 255))
        y += 20


def _collect_boxes(deltas: list[Delta], els_a: dict, els_b: dict) -> tuple[dict, dict]:
    """sheet -> [(CanonicalElement, Delta)] for A and B. Yields the whole
    Delta, not a hand-picked subset of its fields: pdf_annotate.py needs
    kind/is_cascade/description, html_report.py additionally needs
    severity/confidence/semantic_null/zone/id -- passing the full object
    means a new consumer never has to grow this function's return shape
    again, it just reads what it needs off the Delta itself."""
    by_sheet_a: dict[int, list] = {}
    by_sheet_b: dict[int, list] = {}
    for d in deltas:
        el_a = els_a.get(d.id_a) if d.id_a else None
        el_b = els_b.get(d.id_b) if d.id_b else None
        if el_a is not None:
            by_sheet_a.setdefault(d.sheet, []).append((el_a, d))
        if el_b is not None:
            by_sheet_b.setdefault(d.sheet, []).append((el_b, d))
    return by_sheet_a, by_sheet_b


def _collect_visual_change_boxes(deltas: list[Delta]) -> dict[int, list[Delta]]:
    """sheet -> [Delta] for every unclassified_visual_change delta with a
    real bbox_a set (raster_join.py always sets bbox_a == bbox_b to the
    same region, so bbox_a alone is enough). A small parallel function
    to _collect_boxes, not a rewrite of it: these deltas have no
    id_a/id_b to resolve a CanonicalElement through, so they never reach
    _collect_boxes' element-lookup path at all."""
    by_sheet: dict[int, list[Delta]] = {}
    for d in deltas:
        if d.kind == "unclassified_visual_change" and d.bbox_a is not None:
            by_sheet.setdefault(d.sheet, []).append(d)
    return by_sheet


def _annotate(raster_path: str, entries: list, visual_changes: list[Delta], legend: bool) -> Image.Image:
    base = Image.open(raster_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for el, d in entries:
        box = _denormalize(el, base.size)
        width = CASCADE_WIDTH if d.is_cascade else PRIMARY_WIDTH
        filled = not d.is_cascade and d.kind != "move"
        _draw_box(draw, box, COLORS[d.kind], width, filled)
    for d in visual_changes:
        box = _denormalize_bbox(d.bbox_a, base.size)
        _draw_dashed_box(draw, box, COLORS["unclassified_visual_change"], PRIMARY_WIDTH)
        marker_x, marker_y = box[0] + 3, box[1] + 1
        draw.text((marker_x, marker_y), "?", fill=(*COLORS["unclassified_visual_change"], 255))
    composited = Image.alpha_composite(base, overlay)
    if legend:
        _draw_legend(composited)
    return composited.convert("RGB")


def render_markup(doc_a: CanonicalDocument, doc_b: CanonicalDocument,
                   deltas: list[Delta], out_dir: str) -> tuple[dict[int, str], dict[int, str]]:
    """Writes one annotated PNG per sheet per revision to out_dir; returns
    ({sheet: path}, {sheet: path}) for A and B respectively. A sheet with
    no deltas touching it is still written (unannotated) so the output set
    always covers every sheet in the document, not just the changed ones."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    els_a, els_b = _index_elements(doc_a), _index_elements(doc_b)
    boxes_a, boxes_b = _collect_boxes(deltas, els_a, els_b)
    # One region, drawn on both revisions -- raster_join.py's residue has
    # no id_a/id_b, so there's no basis to say the change "belongs" to
    # one side only, unlike a typed add/remove.
    visual_changes = _collect_visual_change_boxes(deltas)

    paths_a: dict[int, str] = {}
    for sheet_no, raster_path in doc_a.raster_paths.items():
        img = _annotate(raster_path, boxes_a.get(sheet_no, []), visual_changes.get(sheet_no, []), legend=True)
        out_path = out / f"markup_a_sheet{sheet_no}.png"
        img.save(out_path)
        paths_a[sheet_no] = str(out_path)

    paths_b: dict[int, str] = {}
    for sheet_no, raster_path in doc_b.raster_paths.items():
        img = _annotate(raster_path, boxes_b.get(sheet_no, []), visual_changes.get(sheet_no, []), legend=True)
        out_path = out / f"markup_b_sheet{sheet_no}.png"
        img.save(out_path)
        paths_b[sheet_no] = str(out_path)

    return paths_a, paths_b
