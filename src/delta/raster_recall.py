"""Raster recall net: a confidence-gated fallback for content that never
became a CanonicalElement at all. This is a DIFFERENT failure mode from
the L2 (scanned) scorecard's precision collapse: garbage OCR text produces
WRONG classified deltas (a false-positive problem); this targets content
extraction missed ENTIRELY, so there's no CanonicalElement for the
deterministic match/classify pipeline to work with in the first place.
Both are real scanned-input problems; this only addresses the second.

Trigger (should_run_raster_recall): per sheet, mean extraction_confidence
below RASTER_RECALL_CONF_THRESHOLD (real for OCR'd scanned pages --
native pairs' extraction_confidence is always 1.0, so this never fires on
them; a built-in safety property, not a special case) OR geometry-element
count above RASTER_RECALL_GEOMETRY_DENSITY (the "dense geometry" case:
real vendor P&IDs run ~5000 geometry paths/sheet vs. this project's
synthetic dataset's ~80-120, see data/samples/PROVENANCE.md).

Mechanics: warp B's raster into A's pixel frame using register.py's
Transform (already real, already tested against synthetic ground truth),
take absolute grayscale pixel difference, threshold, find connected diff
regions (scipy.ndimage.label), drop regions overlapping an
already-classified Delta's bbox (already explained by the deterministic
pipeline), emit the rest as kind="unclassified_visual_change" with a
fixed low confidence (0.2 -- explicitly low, not a computed number
pretending to be precise) and a generic templated description (no LLM --
stays inside the deterministic-by-default engine).

Opt-in via DELTA_RASTER_RECALL=1 (default off): doesn't change existing
native-pair scorecard numbers; enabled explicitly for scanned-format eval
runs.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage

from src.canonical.model import CanonicalDocument, CanonicalElement
from src.canonical.zones import compute_zone
from src.delta.model import Delta
from src.delta.register import Transform

RASTER_RECALL_CONF_THRESHOLD = float(os.environ.get("RASTER_RECALL_CONF_THRESHOLD", "0.75"))
RASTER_RECALL_GEOMETRY_DENSITY = int(os.environ.get("RASTER_RECALL_GEOMETRY_DENSITY", "2000"))
RASTER_RECALL_DIFF_THRESHOLD = int(os.environ.get("RASTER_RECALL_DIFF_THRESHOLD", "40"))  # 0-255 grayscale
RASTER_RECALL_MIN_AREA_PX = int(os.environ.get("RASTER_RECALL_MIN_AREA_PX", "150"))
RASTER_RECALL_DEDUP_IOU = float(os.environ.get("RASTER_RECALL_DEDUP_IOU", "0.15"))
RASTER_RECALL_CONFIDENCE = 0.2


def should_run_raster_recall(elements: list[CanonicalElement]) -> bool:
    if not elements:
        return False
    mean_conf = sum(e.extraction_confidence for e in elements) / len(elements)
    n_geometry = sum(1 for e in elements if e.type == "geometry")
    return mean_conf < RASTER_RECALL_CONF_THRESHOLD or n_geometry > RASTER_RECALL_GEOMETRY_DENSITY


def _affine_data_b_to_a(transform: Transform, size_a: tuple[int, int], size_b: tuple[int, int]) -> tuple:
    """PIL Image.transform's AFFINE data (a,b,c,d,e,f): for each OUTPUT
    (A-frame) pixel, the INPUT (B raster) pixel it comes from -- i.e. the
    inverse of Transform, which maps normalized B -> A. Derived directly
    (p_B_norm = (1/s) * R^T * (p_A_norm - t)), not approximated; validated
    against a known non-trivial transform in
    tests/test_delta_raster_recall.py rather than trusted on inspection
    alone, the same standard register.py's own transform was held to."""
    w_a, h_a = size_a
    w_b, h_b = size_b
    s, r, tx, ty = transform.scale, transform.rotation, transform.tx, transform.ty
    cos_r, sin_r = math.cos(r), math.sin(r)
    a = (w_b * cos_r) / (s * w_a)
    b = (w_b * sin_r) / (s * h_a)
    c = (w_b / s) * (-cos_r * tx - sin_r * ty)
    d = -(h_b * sin_r) / (s * w_a)
    e = (h_b * cos_r) / (s * h_a)
    f = (h_b / s) * (sin_r * tx - cos_r * ty)
    return (a, b, c, d, e, f)


def _warp_b_into_a_frame(img_b: Image.Image, transform: Transform, size_a: tuple[int, int]) -> Image.Image:
    data = _affine_data_b_to_a(transform, size_a, img_b.size)
    return img_b.transform(size_a, Image.AFFINE, data, resample=Image.BILINEAR, fillcolor=255)


def _diff_regions(raster_path_a: str, raster_path_b: str, transform: Transform) -> list[tuple[float, float, float, float]]:
    """Normalized [0,1] bboxes of connected diff regions between A's
    raster and B's raster warped into A's frame."""
    img_a = Image.open(raster_path_a).convert("L")
    img_b = Image.open(raster_path_b).convert("L")
    warped_b = _warp_b_into_a_frame(img_b, transform, img_a.size)

    arr_a = np.asarray(img_a, dtype=np.int16)
    arr_b = np.asarray(warped_b, dtype=np.int16)
    diff_mask = np.abs(arr_a - arr_b) > RASTER_RECALL_DIFF_THRESHOLD

    labeled, n_regions = ndimage.label(diff_mask)
    if n_regions == 0:
        return []
    w_a, h_a = img_a.size
    regions = []
    for label_id in range(1, n_regions + 1):
        ys, xs = np.where(labeled == label_id)
        if len(xs) < RASTER_RECALL_MIN_AREA_PX:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        regions.append((x0 / w_a, y0 / h_a, x1 / w_a, y1 / h_a))
    return regions


def _iou(box_a: tuple, box_b: tuple) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _classified_bboxes(deltas: list[Delta], sheet: int, els_a: dict, els_b: dict) -> list[tuple]:
    """Existing classified deltas' bboxes on this sheet -- a diff region
    overlapping one of these is already explained, not a recall-net find."""
    boxes = []
    for d in deltas:
        if d.sheet != sheet:
            continue
        el = els_b.get(d.id_b) or els_a.get(d.id_a)
        if el is not None:
            b = el.bbox
            boxes.append((b.x0, b.y0, b.x1, b.y1))
    return boxes


def detect_unclassified_visual_changes(doc_a: CanonicalDocument, doc_b: CanonicalDocument,
                                        deltas: list[Delta], transform: Transform) -> list[Delta]:
    """Opt-in (DELTA_RASTER_RECALL=1); no-op otherwise. Returns NEW deltas
    to append -- does not mutate the input `deltas` list, since it reads
    from it (for dedup) while building the result."""
    if os.environ.get("DELTA_RASTER_RECALL") != "1":
        return []

    els_a = {el.id: el for sh in doc_a.sheets for el in sh.elements}
    els_b = {el.id: el for sh in doc_b.sheets for el in sh.elements}
    by_sheet_a = {sh.number: sh for sh in doc_a.sheets}
    by_sheet_b = {sh.number: sh for sh in doc_b.sheets}

    new_deltas: list[Delta] = []
    counter = 0
    for sheet_no in sorted(set(by_sheet_a) & set(by_sheet_b)):
        sheet_a, sheet_b = by_sheet_a[sheet_no], by_sheet_b[sheet_no]
        if not (should_run_raster_recall(sheet_a.elements) or should_run_raster_recall(sheet_b.elements)):
            continue
        raster_a = doc_a.raster_paths.get(sheet_no)
        raster_b = doc_b.raster_paths.get(sheet_no)
        if not raster_a or not raster_b:
            continue

        regions = _diff_regions(raster_a, raster_b, transform)
        existing_boxes = _classified_bboxes(deltas, sheet_no, els_a, els_b)
        for region in regions:
            if any(_iou(region, box) > RASTER_RECALL_DEDUP_IOU for box in existing_boxes):
                continue
            counter += 1
            x0, y0, x1, y1 = region
            zone = compute_zone((x0 + x1) / 2, (y0 + y1) / 2)
            new_deltas.append(Delta(
                did=f"raster{counter:04d}", kind="unclassified_visual_change",
                element_type="unclassified_visual_change", id_a=None, id_b=None,
                sheet=sheet_no, zone_a=zone, zone_b=zone, field_changes={},
                confidence=RASTER_RECALL_CONFIDENCE,
                description=f"unclassified visual change detected at zone {zone} "
                             f"(raster diff, no matching extracted element)",
            ))
    return new_deltas
