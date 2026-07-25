"""Raster join: for each proposed ChangeRegion (raster_diff.py), decides
whether the symbolic layer already explains it (skip), and if not,
classifies the residue as graphical / geometry / extraction_gap based on
nearby canonical-element text on each side. Never emits a delta the
symbolic layer already accounts for -- this subtraction is what keeps
precision high, and it is the second half of the "raster localizes,
symbolic classifies" principle: raster_diff.py proposes regions with no
knowledge of the symbolic pipeline at all; this module is the only place
that looks at symbolic deltas, and it does so purely to suppress, never
to help itself explain what changed.
"""
from __future__ import annotations

from src.canonical.model import BBox, CanonicalElement
from src.canonical.zones import compute_zone
from src.delta.model import Delta
from src.delta.raster_diff import ChangeRegion, RasterCfg


def _centroid(b: BBox) -> tuple[float, float]:
    return ((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2)


def _contains(b: BBox, pt: tuple[float, float]) -> bool:
    return b.x0 <= pt[0] <= b.x1 and b.y0 <= pt[1] <= b.y1


def _iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a.x1 - a.x0) * (a.y1 - a.y0)
    area_b = (b.x1 - b.x0) * (b.y1 - b.y0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _bboxes_intersect(a: BBox, b: BBox) -> bool:
    return a.x0 < b.x1 and a.x1 > b.x0 and a.y0 < b.y1 and a.y1 > b.y0


def _symbolic_bbox(d: Delta, els_a_by_id: dict, els_b_by_id: dict) -> BBox | None:
    """Same id_a/id_b -> CanonicalElement -> bbox lookup pattern already
    used by markup/overlay.py::_collect_boxes -- no new Delta field is
    needed for symbolic deltas, only for this module's own emissions
    (see model.py's Delta docstring)."""
    el = els_b_by_id.get(d.id_b) or els_a_by_id.get(d.id_a)
    return el.bbox if el else None


def _is_explained(region: ChangeRegion, sheet_deltas: list[Delta],
                   els_a_by_id: dict, els_b_by_id: dict, cfg: RasterCfg) -> bool:
    for d in sheet_deltas:
        sym_bbox = _symbolic_bbox(d, els_a_by_id, els_b_by_id)
        if sym_bbox is None:
            continue
        if _iou(region.bbox, sym_bbox) > cfg.explain_iou:
            return True
        if _contains(region.bbox, _centroid(sym_bbox)) or _contains(sym_bbox, _centroid(region.bbox)):
            return True
    return False


def _text_elements_overlapping(region: ChangeRegion, elements: list[CanonicalElement],
                                cfg: RasterCfg) -> list[CanonicalElement]:
    """Elements near R with real text content, on the same sheet,
    excluding bare "geometry" type. Padded by cfg.tag_proximity_norm
    because a valve tag's extracted bbox (text glyphs only) and its
    symbol glyph (drawn adjacent, not overlapping, per real P&ID drafting
    convention) are spatially close but not necessarily overlapping --
    exact-bbox intersection alone would misclassify the valve-symbol-
    change case as "geometry" instead of "graphical"."""
    pad = cfg.tag_proximity_norm
    rb = region.bbox
    padded = BBox(rb.x0 - pad, rb.y0 - pad, rb.x1 + pad, rb.y1 + pad)
    return [el for el in elements
            if el.sheet == region.sheet and el.type != "geometry" and el.content.strip()
            and _bboxes_intersect(el.bbox, padded)]


def _overlap_fraction(region_bbox: BBox, el_bbox: BBox) -> float:
    """What fraction of region_bbox's own area is covered by el_bbox --
    not IoU (which also penalizes el_bbox being much larger than the
    region): this measures "is the diff region fully accounted for by
    this element," the right question for suppression."""
    ix0, iy0 = max(region_bbox.x0, el_bbox.x0), max(region_bbox.y0, el_bbox.y0)
    ix1, iy1 = min(region_bbox.x1, el_bbox.x1), min(region_bbox.y1, el_bbox.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    region_area = (region_bbox.x1 - region_bbox.x0) * (region_bbox.y1 - region_bbox.y0)
    return inter / region_area if region_area else 0.0


def _candidate_text_elements(region: ChangeRegion, elements: list[CanonicalElement]) -> list[CanonicalElement]:
    """Text-bearing elements on the region's sheet with ANY positive
    overlap against the region's own (unpadded) bbox -- a loose
    pre-filter; _is_text_confirmed_unchanged applies the real threshold
    to the PAIR's combined overlap, not to each side independently (see
    its docstring for why)."""
    return [el for el in elements
            if el.sheet == region.sheet and el.type != "geometry" and el.content.strip()
            and _overlap_fraction(region.bbox, el.bbox) > 0]


def _is_text_confirmed_unchanged(region: ChangeRegion, elements_a: list[CanonicalElement],
                                  elements_b: list[CanonicalElement], cfg: RasterCfg) -> bool:
    """True when the region is directly, substantially covered by an
    element whose extracted content is IDENTICAL on both sides -- strong
    evidence the underlying pixel diff is a rendering artifact (font,
    anti-aliasing, producer variation), not a real change. This is the
    ensemble half of "raster localizes, symbolic classifies": the
    symbolic layer's own confirmation that nothing changed here is used
    to suppress a raster hit, exactly the way a symbolic CHANGE already
    suppresses one in _is_explained -- just the mirror case. Position
    already does the disambiguation work a global content-equality check
    would need: a matching pair must independently overlap THIS region on
    both sides, so an unrelated same-text element elsewhere on the sheet
    can never trigger this.

    Threshold is applied to the PAIR's *average* overlap fraction, not to
    each side independently: a live check against the generator's
    producer-variation null pair (Helvetica vs. Courier -- a monospace
    font renders the same string at a different width than a proportional
    one) found real near-misses where the identical-content element
    covered the region at 0.56 on one side and 0.62 on the other --
    requiring each side to independently clear the same fixed bar
    rejected these for no good reason; the two sides disagreeing on
    exactly how much of the region their own (differently-metriced) font
    covers is itself part of what a font substitution looks like, not
    evidence the match is wrong."""
    if not cfg.enable_text_confirm:
        return False
    cand_a = _candidate_text_elements(region, elements_a)
    cand_b = _candidate_text_elements(region, elements_b)
    for el_a in cand_a:
        for el_b in cand_b:
            if el_a.content != el_b.content:
                continue
            frac_a = _overlap_fraction(region.bbox, el_a.bbox)
            frac_b = _overlap_fraction(region.bbox, el_b.bbox)
            if (frac_a + frac_b) / 2 >= cfg.text_confirm_overlap_frac:
                return True
    return False


def _confidence(region: ChangeRegion, cfg: RasterCfg) -> float:
    """Low, scaled by diff magnitude and region size, hard-capped below
    any symbolic delta's typical confidence range. area_px isn't
    normalized against the sheet's pixel area here (this function
    doesn't receive raster dimensions) -- regions reaching this point
    already survived propose_change_regions' max_area_frac cutoff, so a
    fixed multiple of cfg.min_area_px is used as a size proxy instead."""
    mag = max(0.0, min(1.0, region.mean_diff_magnitude))
    size_score = min(1.0, region.area_px / (cfg.min_area_px * 20))
    score = 0.5 * mag + 0.5 * size_score
    return round(min(cfg.conf_cap, cfg.conf_base + cfg.conf_scale * score), 4)


def _classify(overlap_a: list[CanonicalElement], overlap_b: list[CanonicalElement]) -> tuple[str, dict, str]:
    content_a = {el.content for el in overlap_a}
    content_b = {el.content for el in overlap_b}

    if bool(overlap_a) != bool(overlap_b):
        return _extraction_gap(overlap_a, overlap_b)
    if overlap_a and overlap_b:
        shared = content_a & content_b
        if shared:
            tags = sorted(shared)
            return ("graphical", {"tags": tags},
                    f"graphical change near {', '.join(tags)}; not characterized by text engine")
        # both sides have nearby text, but none of it matches -- a real,
        # unattributed content difference near this region; the honest
        # label is extraction_gap, not a confident "pure geometry" claim.
        return _extraction_gap(overlap_a, overlap_b)
    return ("geometry", {}, "geometry change (no nearby text either side); not characterized by text engine")


def _extraction_gap(overlap_a: list[CanonicalElement], overlap_b: list[CanonicalElement]) -> tuple[str, dict, str]:
    ids = sorted({el.id for el in overlap_a} | {el.id for el in overlap_b})
    return ("extraction_gap", {"candidate_element_ids": ids},
            f"possible extraction gap near element(s) {', '.join(ids)}; not characterized by text engine")


def join_regions_to_symbolic(regions: list[ChangeRegion],
                              elements_a: list[CanonicalElement],
                              elements_b: list[CanonicalElement],
                              symbolic_deltas: list[Delta],
                              cfg: RasterCfg) -> list[Delta]:
    """For each region, decide if the symbolic layer already explains it.
    Emit ONLY unexplained residue as unclassified_visual_change deltas."""
    els_a_by_id = {el.id: el for el in elements_a}
    els_b_by_id = {el.id: el for el in elements_b}
    deltas_by_sheet: dict[int, list[Delta]] = {}
    for d in symbolic_deltas:
        deltas_by_sheet.setdefault(d.sheet, []).append(d)

    residue: list[Delta] = []
    counter = 0
    for region in regions:
        if _is_explained(region, deltas_by_sheet.get(region.sheet, []), els_a_by_id, els_b_by_id, cfg):
            continue
        if _is_text_confirmed_unchanged(region, elements_a, elements_b, cfg):
            continue

        overlap_a = _text_elements_overlapping(region, elements_a, cfg)
        overlap_b = _text_elements_overlapping(region, elements_b, cfg)
        kind_hint, field_changes, desc = _classify(overlap_a, overlap_b)

        cx, cy = _centroid(region.bbox)
        zone = compute_zone(cx, cy)
        counter += 1
        residue.append(Delta(
            did=f"raster{counter:04d}", kind="unclassified_visual_change",
            element_type="unclassified_visual_change", id_a=None, id_b=None,
            sheet=region.sheet, zone_a=zone, zone_b=zone,
            field_changes=field_changes, confidence=_confidence(region, cfg),
            description=desc, bbox_a=region.bbox, bbox_b=region.bbox,
            visual_change_kind=kind_hint,
        ))
    return residue
