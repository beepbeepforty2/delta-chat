"""Markup overlay: real PDF annotations, not raster PNGs.

`overlay.py`'s PNG path bakes colored boxes into a flat raster copy --
useful for a quick terminal/chat preview, but a PNG discards toggleability,
carries no structured metadata, never appears in Acrobat's/Bluebeam's
markup/comments list, and permanently rasterizes content that should stay
vector. This module uses `page.add_rect_annot()` to create native "Square"
annotation objects directly on the ORIGINAL PDF pages instead -- real PDF
annotation objects, addressable, toggleable, and carrying the delta's own
description as the annotation's comment content (the actual interop this
exists for: an engineer opens the output in their existing markup tool and
sees delta-chat's findings in the same comments list as their own redlines).

Reuses `overlay.py`'s `_index_elements`/`_collect_boxes` (same delta ->
element -> bbox resolution, same color/kind vocabulary) so the PNG and PDF
markup paths agree on WHAT gets annotated; this module only differs in
WHERE (a real PDF page, not a raster copy) and HOW (a native annotation
object, not baked-in pixels).

Denormalizing bbox against `page.rect.width`/`height`, not raster pixel
size: bbox is normalized [0,1] as a fraction of the page regardless of
what it was normalized against at ingest time (`page.rect` for
`pdf_native.py`; the OCR'd raster image's own pixel size for
`pdf_scanned.py`) -- the fraction-along-width is basis-independent as long
as the OCR'd raster is a full-page, non-cropped render (`get_pixmap()`
already is), so this is correct for both native and scanned inputs with
no format-specific branching.
"""
from __future__ import annotations

from pathlib import Path

import fitz

from src.canonical.model import CanonicalDocument
from src.delta.model import Delta
from src.markup.overlay import COLORS, _collect_boxes, _index_elements

PRIMARY_BORDER_WIDTH = 2.5
CASCADE_BORDER_WIDTH = 1.0
FILL_OPACITY = 0.25  # primary, filled annotations only -- translucent so page content stays legible
MOVE_OPACITY = 0.9   # move is outline-only (no fill), so its own opacity can stay high


def _rgb_float(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return (color[0] / 255, color[1] / 255, color[2] / 255)


def _annotate_page(page: "fitz.Page", entries: list) -> None:
    """entries: [(CanonicalElement, kind, is_cascade, description)], from
    overlay.py::_collect_boxes."""
    w, h = page.rect.width, page.rect.height
    for el, kind, is_cascade, description in entries:
        b = el.bbox
        rect = fitz.Rect(b.x0 * w, b.y0 * h, b.x1 * w, b.y1 * h)
        annot = page.add_rect_annot(rect)
        color = _rgb_float(COLORS[kind])
        filled = not is_cascade and kind != "move"
        if filled:
            annot.set_colors(stroke=color, fill=color)
            annot.set_opacity(FILL_OPACITY)
        else:
            annot.set_colors(stroke=color)
            annot.set_opacity(MOVE_OPACITY)
        annot.set_border(width=CASCADE_BORDER_WIDTH if is_cascade else PRIMARY_BORDER_WIDTH)
        annot.set_info(content=description or f"{kind} ({el.type})", title=f"delta-chat: {kind}")
        annot.update()


def _annotate_document(pdf_path: str, boxes_by_sheet: dict[int, list], out_path: Path) -> str:
    doc = fitz.open(pdf_path)
    for sheet_no, entries in boxes_by_sheet.items():
        page_index = sheet_no - 1  # sheet numbers are 1-indexed, page order matches (see pdf_native.py::ingest)
        if 0 <= page_index < doc.page_count:
            _annotate_page(doc[page_index], entries)
    doc.save(str(out_path), garbage=4, deflate=True)
    doc.close()
    return str(out_path)


def render_pdf_markup(doc_a: CanonicalDocument, doc_b: CanonicalDocument, deltas: list[Delta],
                       path_a: str, path_b: str, out_dir: str) -> tuple[str, str]:
    """Reopens the ORIGINAL PDF files at path_a/path_b (not the raster
    cache) and writes one annotated PDF per revision. A revision with no
    deltas touching a given sheet still gets that sheet copied through
    unannotated, so the output always covers the full document."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    els_a, els_b = _index_elements(doc_a), _index_elements(doc_b)
    boxes_a, boxes_b = _collect_boxes(deltas, els_a, els_b)

    out_a = _annotate_document(path_a, boxes_a, out / "markup_a.pdf")
    out_b = _annotate_document(path_b, boxes_b, out / "markup_b.pdf")
    return out_a, out_b
