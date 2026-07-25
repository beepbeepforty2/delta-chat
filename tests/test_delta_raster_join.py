from src.canonical.model import BBox, CanonicalElement
from src.delta.model import Delta
from src.delta.raster_diff import ChangeRegion, RasterCfg
from src.delta.raster_join import join_regions_to_symbolic


def _el(id_, content, x0, y0, x1, y1, sheet=1, type_="note"):
    return CanonicalElement(id=id_, type=type_, content=content, bbox=BBox(x0, y0, x1, y1),
                             sheet=sheet, zone="A-1", extraction_confidence=1.0)


def _region(x0, y0, x1, y1, sheet=1, area_px=500, mag=0.6):
    return ChangeRegion(sheet=sheet, bbox=BBox(x0, y0, x1, y1), area_px=area_px, mean_diff_magnitude=mag)


def test_region_overlapping_symbolic_delta_bbox_is_skipped():
    el_a = _el("a1", "old", 0.4, 0.4, 0.6, 0.5)
    el_b = _el("b1", "new", 0.4, 0.4, 0.6, 0.5)
    d = Delta("d1", "modify", "note", "a1", "b1", 1, "A-1", "A-1", {"content": ["old", "new"]})
    region = _region(0.41, 0.41, 0.59, 0.49)  # inside/overlapping the delta's bbox

    residue = join_regions_to_symbolic([region], [el_a], [el_b], [d], RasterCfg())
    assert residue == []


def test_region_containing_a_small_symbolic_delta_centroid_is_skipped():
    """Centroid-in-region case: the symbolic bbox is much smaller than
    the region (e.g. a small tag inside a broader diff blob), so IoU
    alone would be low, but the delta's own centroid falls inside the
    region -- still counted as explained."""
    el_a = _el("a1", "old", 0.48, 0.48, 0.50, 0.50)
    el_b = _el("b1", "new", 0.48, 0.48, 0.50, 0.50)
    d = Delta("d1", "modify", "note", "a1", "b1", 1, "A-1", "A-1", {})
    region = _region(0.3, 0.3, 0.7, 0.7, area_px=5000)  # big region containing the small element

    residue = join_regions_to_symbolic([region], [el_a], [el_b], [d], RasterCfg())
    assert residue == []


def test_region_over_identical_text_both_sides_classified_graphical():
    el_a = _el("v1", "26GT9143", 0.40, 0.40, 0.50, 0.42, type_="valve_tag")
    el_b = _el("v1", "26GT9143", 0.40, 0.40, 0.50, 0.42, type_="valve_tag")
    # region sits just left of the tag's own bbox (like a valve glyph
    # drawn adjacent to, not overlapping, its tag) -- within padding
    region = _region(0.38, 0.395, 0.395, 0.415)

    residue = join_regions_to_symbolic([region], [el_a], [el_b], [], RasterCfg())
    assert len(residue) == 1
    d = residue[0]
    assert d.kind == "unclassified_visual_change"
    assert d.visual_change_kind == "graphical"
    assert d.field_changes["tags"] == ["26GT9143"]
    assert d.bbox_a == region.bbox and d.bbox_b == region.bbox
    assert d.id_a is None and d.id_b is None


def test_region_with_text_only_on_a_side_classified_extraction_gap():
    el_a = _el("n1", "26-PDI-9054 HH INITIATE STOP.", 0.40, 0.40, 0.60, 0.42)
    region = _region(0.40, 0.40, 0.60, 0.42)

    residue = join_regions_to_symbolic([region], [el_a], [], [], RasterCfg())
    assert len(residue) == 1
    assert residue[0].visual_change_kind == "extraction_gap"
    assert residue[0].field_changes["candidate_element_ids"] == ["n1"]


def test_region_with_no_text_either_side_classified_geometry():
    region = _region(0.40, 0.40, 0.60, 0.42)
    residue = join_regions_to_symbolic([region], [], [], [], RasterCfg())
    assert len(residue) == 1
    assert residue[0].visual_change_kind == "geometry"
    assert residue[0].field_changes == {}


def test_region_with_nonmatching_text_both_sides_classified_extraction_gap():
    el_a = _el("n1", "old content here", 0.40, 0.40, 0.60, 0.42)
    el_b = _el("n2", "totally different text", 0.40, 0.40, 0.60, 0.42)
    region = _region(0.40, 0.40, 0.60, 0.42)

    residue = join_regions_to_symbolic([region], [el_a], [el_b], [], RasterCfg())
    assert len(residue) == 1
    assert residue[0].visual_change_kind == "extraction_gap"
    assert set(residue[0].field_changes["candidate_element_ids"]) == {"n1", "n2"}


def test_confidence_ordered_by_diff_magnitude_and_area_and_capped():
    cfg = RasterCfg()
    low = _region(0.1, 0.1, 0.2, 0.2, area_px=cfg.min_area_px, mag=0.1)
    high = _region(0.1, 0.1, 0.2, 0.2, area_px=cfg.min_area_px * 30, mag=0.9)

    residue = join_regions_to_symbolic([low, high], [], [], [], cfg)
    conf_low, conf_high = residue[0].confidence, residue[1].confidence
    assert conf_low < conf_high
    assert conf_low < 1.0 and conf_high < 1.0
    assert conf_high <= cfg.conf_cap


def test_tag_proximity_padding_is_load_bearing():
    el_a = _el("v1", "26GT9143", 0.50, 0.50, 0.60, 0.52, type_="valve_tag")
    el_b = _el("v1", "26GT9143", 0.50, 0.50, 0.60, 0.52, type_="valve_tag")
    # region just outside the tag's bbox (mirrors the ~0.01-0.02 gap
    # between a valve glyph and its tag in the synthetic generator)
    region = _region(0.478, 0.495, 0.498, 0.515)

    with_padding = join_regions_to_symbolic([region], [el_a], [el_b], [],
                                             RasterCfg(tag_proximity_norm=0.02))
    without_padding = join_regions_to_symbolic([region], [el_a], [el_b], [],
                                                RasterCfg(tag_proximity_norm=0.0))

    assert with_padding[0].visual_change_kind == "graphical"
    assert without_padding[0].visual_change_kind == "geometry"


def test_multi_sheet_deltas_only_explain_their_own_sheet():
    el_a = _el("a1", "old", 0.4, 0.4, 0.6, 0.5, sheet=1)
    el_b = _el("b1", "new", 0.4, 0.4, 0.6, 0.5, sheet=1)
    d = Delta("d1", "modify", "note", "a1", "b1", 1, "A-1", "A-1", {})
    region_sheet2 = _region(0.41, 0.41, 0.59, 0.49, sheet=2)  # same coords, different sheet

    residue = join_regions_to_symbolic([region_sheet2], [el_a], [el_b], [d], RasterCfg())
    assert len(residue) == 1  # NOT explained -- the symbolic delta is on sheet 1, region on sheet 2
    assert residue[0].sheet == 2
