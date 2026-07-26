"""The delta payload: one serializer, two consumers.

`Delta` objects carry ids (`id_a`/`id_b`) rather than coordinates, because
the same delta has a different location in each revision and the geometry
lives on the elements. Every UI therefore needs the same resolution pass --
delta -> element -> normalized bbox, with the raster-origin fallback -- plus
the same severity/cascade/semantic-null metadata alongside it.

That pass used to live inside `html_report.py`, private to the offline
report. `src/web/` needs byte-for-byte the same records, so it moved here
rather than being reimplemented: two serializers that agree today diverge
the first time a field is added to `Delta`, and the divergence is silent --
the web UI would simply stop showing something the downloadable report
still shows, with no test failing. `tests/test_markup_payload.py` pins the
two together by asserting the JSON embedded in report.html IS this payload.

The only thing that legitimately differs between the two consumers is how
page images arrive, hence `inline_images`:

  True   the offline report base64-inlines each sheet's L0 raster, because
         it must survive being emailed as a single file with no server.
  False  the web client fetches the original PDFs from the API and renders
         them with pdf.js, so shipping megabytes of base64 it will never
         draw would just slow the first paint.

Box coordinates are normalized [0,1] in both cases (see `overlay.py`'s
docstring) -- deliberately NOT denormalized here, so a consumer can lay them
out as plain CSS percentages against a container of any size, at any zoom,
without knowing the raster's pixel dimensions.
"""
from __future__ import annotations

import base64
from pathlib import Path

from src.canonical.model import CanonicalDocument
from src.delta.model import Delta
from src.delta.severity import SEVERITY_ORDER
from src.markup.overlay import COLORS, _collect_boxes, _index_elements

KIND_LABELS = {
    "add": "Added",
    "remove": "Removed",
    "modify": "Modified",
    "move": "Moved",
    "unclassified_visual_change": "Unclassified visual change",
}


def kind_hex(kind: str) -> str:
    r, g, b = COLORS[kind]
    return f"#{r:02x}{g:02x}{b:02x}"


def kind_color_map() -> dict[str, str]:
    """`overlay.COLORS` as hex, so the PNG overlay, the PDF annotations, the
    offline report and the web UI all colour a given kind identically."""
    return {kind: kind_hex(kind) for kind in COLORS}


def _bbox_list(el) -> list[float]:
    b = el.bbox
    return [b.x0, b.y0, b.x1, b.y1]


def _bbox_obj_list(b) -> list[float]:
    return [b.x0, b.y0, b.x1, b.y1]


def build_delta_records(deltas: list[Delta], boxes_a: dict, boxes_b: dict) -> list[dict]:
    """One record per Delta (primary AND cascade -- consumers filter, this
    doesn't pre-drop). `unclassified_visual_change` deltas have no id_a/id_b
    by construction (raster_join.py finds pixels, not elements), so they
    never resolve through the element-lookup path (boxes_a/boxes_b) -- but
    raster_join.py DOES set bbox_a/bbox_b directly on the Delta itself (the
    region it found), so that's used as a fallback location. Only a Delta
    with genuinely neither (older callers, or a future producer that doesn't
    set them) falls through to a null box, which consumers render as a
    "no exact location -- zone only" note rather than dropping the row."""
    box_a_by_did = {d.did: _bbox_list(el) for entries in boxes_a.values() for el, d in entries}
    box_b_by_did = {d.did: _bbox_list(el) for entries in boxes_b.values() for el, d in entries}
    records = []
    for d in deltas:
        box_a = box_a_by_did.get(d.did) or (_bbox_obj_list(d.bbox_a) if d.bbox_a is not None else None)
        box_b = box_b_by_did.get(d.did) or (_bbox_obj_list(d.bbox_b) if d.bbox_b is not None else None)
        records.append({
            "did": d.did,
            "kind": d.kind,
            "element_type": d.element_type,
            "sheet": d.sheet,
            "zone": d.zone_b or d.zone_a or "?",
            "severity": d.severity or "low",
            "confidence": round(d.confidence, 2),
            "description": d.description or "",
            "is_cascade": d.is_cascade,
            "primary_did": d.primary_did,
            "semantic_null": d.semantic_null,
            "semantic_null_reason": d.semantic_null_reason,
            "visual_change_kind": d.visual_change_kind,
            "box_a": box_a,
            "box_b": box_b,
        })
    return records


def _b64_png(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def build_sheets(doc_a: CanonicalDocument, doc_b: CanonicalDocument,
                 *, inline_images: bool) -> list[dict]:
    """Union of both documents' sheet numbers -- a sheet added or removed
    between revisions exists on one side only, and the UI still has to offer
    a tab for it (showing one pane empty) rather than hiding the fact."""
    sheet_nos = sorted(set(doc_a.raster_paths) | set(doc_b.raster_paths))
    sheets = []
    for n in sheet_nos:
        if inline_images:
            sheets.append({
                "number": n,
                "img_a": _b64_png(doc_a.raster_paths[n]) if n in doc_a.raster_paths else None,
                "img_b": _b64_png(doc_b.raster_paths[n]) if n in doc_b.raster_paths else None,
            })
        else:
            sheets.append({
                "number": n,
                "has_a": n in doc_a.raster_paths,
                "has_b": n in doc_b.raster_paths,
            })
    return sheets


def build_payload(doc_a: CanonicalDocument, doc_b: CanonicalDocument, deltas: list[Delta],
                  pid_a_label: str, pid_b_label: str, *, inline_images: bool) -> dict:
    """The whole UI-facing view of one comparison. pid_a_label/pid_b_label
    are display names only (the CLI passes the --a/--b paths, the web API
    passes the uploaded filenames) -- everything else comes straight off
    doc_a/doc_b/deltas, so no caller can influence a number here.

    `summary` counts PRIMARY deltas only for severity and semantic-null,
    matching `report.py`'s markdown headline: cascade deltas are
    renumbering side-effects of a primary change, and counting them as
    independent findings would inflate every total shown to a reviewer."""
    els_a, els_b = _index_elements(doc_a), _index_elements(doc_b)
    boxes_a, boxes_b = _collect_boxes(deltas, els_a, els_b)
    records = build_delta_records(deltas, boxes_a, boxes_b)
    sheets = build_sheets(doc_a, doc_b, inline_images=inline_images)

    primary = [d for d in deltas if not d.is_cascade]
    severity_counts = {sev: sum(1 for d in primary if d.severity == sev) for sev in SEVERITY_ORDER}

    return {
        "pid_a": pid_a_label,
        "pid_b": pid_b_label,
        "sheets": sheets,
        "deltas": records,
        "summary": {
            "n_primary": len(primary),
            "n_cascade": sum(1 for d in deltas if d.is_cascade),
            "severity_counts": severity_counts,
            "n_semantic_null": sum(1 for d in primary if d.semantic_null),
        },
    }
