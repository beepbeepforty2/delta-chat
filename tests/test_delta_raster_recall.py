import pathlib

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalSheet
from src.delta.model import Delta
from src.delta.raster_recall import (
    _classified_bboxes,
    _diff_regions,
    _iou,
    _warp_b_into_a_frame,
    detect_unclassified_visual_changes,
    should_run_raster_recall,
)
from src.delta.register import Transform

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


def _el(id_, type_, conf=1.0, x0=0.1, y0=0.1, x1=0.2, y1=0.2):
    return CanonicalElement(id=id_, type=type_, content="x", bbox=BBox(x0, y0, x1, y1),
                             sheet=1, zone="A-1", extraction_confidence=conf)


# --- should_run_raster_recall -----------------------------------------------

def test_trigger_false_for_empty_elements():
    assert should_run_raster_recall([]) is False


def test_trigger_false_for_high_confidence_native_pair():
    elements = [_el(f"e{i}", "note", conf=1.0) for i in range(10)]
    assert should_run_raster_recall(elements) is False


def test_trigger_true_for_low_mean_confidence():
    elements = [_el(f"e{i}", "note", conf=0.4) for i in range(10)]
    assert should_run_raster_recall(elements) is True


def test_trigger_true_for_dense_geometry():
    elements = [_el(f"e{i}", "geometry", conf=1.0) for i in range(2500)]
    assert should_run_raster_recall(elements) is True


# --- affine warp geometric correctness --------------------------------------

def test_warp_places_marker_at_transform_predicted_position():
    """The core geometric claim: _warp_b_into_a_frame is the correct
    inverse of Transform.apply, not just plausible-looking. Validated the
    same way register.py's own transform was: a known non-trivial
    transform, checked against its predicted output, not trusted on
    inspection of the algebra alone."""
    size = (300, 300)
    img_b = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img_b)
    # A small marker at a known normalized position in B.
    bx, by = 0.3, 0.35
    px, py = bx * size[0], by * size[1]
    draw.rectangle((px - 4, py - 4, px + 4, py + 4), fill=0)

    transform = Transform(scale=1.1, rotation=0.05, tx=0.1, ty=-0.05)
    warped = _warp_b_into_a_frame(img_b, transform, size)

    arr = np.asarray(warped)
    dark = np.where(arr < 128)
    assert len(dark[0]) > 0, "marker must survive the warp"
    actual_y, actual_x = dark[0].mean(), dark[1].mean()
    actual_norm = (actual_x / size[0], actual_y / size[1])

    expected_norm = transform.apply(bx, by)
    assert actual_norm[0] == pytest.approx(expected_norm[0], abs=0.02)
    assert actual_norm[1] == pytest.approx(expected_norm[1], abs=0.02)


def test_warp_identity_transform_is_a_noop():
    size = (100, 100)
    img_b = Image.new("L", size, 255)
    ImageDraw.Draw(img_b).rectangle((40, 40, 60, 60), fill=0)
    warped = _warp_b_into_a_frame(img_b, Transform(), size)
    assert np.array_equal(np.asarray(img_b), np.asarray(warped))


# --- diff region extraction --------------------------------------------------

def test_diff_regions_empty_for_identical_rasters(tmp_path):
    img = Image.new("L", (200, 200), 255)
    ImageDraw.Draw(img).rectangle((50, 50, 70, 70), fill=0)
    path_a, path_b = tmp_path / "a.png", tmp_path / "b.png"
    img.save(path_a)
    img.save(path_b)
    assert _diff_regions(str(path_a), str(path_b), Transform()) == []


def test_diff_regions_finds_added_rectangle(tmp_path):
    img_a = Image.new("L", (200, 200), 255)
    img_b = img_a.copy()
    ImageDraw.Draw(img_b).rectangle((120, 120, 160, 160), fill=0)
    path_a, path_b = tmp_path / "a.png", tmp_path / "b.png"
    img_a.save(path_a)
    img_b.save(path_b)

    regions = _diff_regions(str(path_a), str(path_b), Transform())
    assert len(regions) == 1
    x0, y0, x1, y1 = regions[0]
    assert 0.55 < x0 < 0.65 and 0.55 < y0 < 0.65
    assert 0.75 < x1 < 0.85 and 0.75 < y1 < 0.85


# --- iou / dedup --------------------------------------------------------------

def test_iou_no_overlap_is_zero():
    assert _iou((0, 0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_identical_boxes_is_one():
    assert _iou((0.1, 0.1, 0.3, 0.3), (0.1, 0.1, 0.3, 0.3)) == 1.0


def test_classified_bboxes_filters_by_sheet():
    els_a = {"a1": _el("a1", "note", x0=0.1, y0=0.1, x1=0.2, y1=0.2)}
    els_b = {}
    d_sheet1 = Delta("d1", "remove", "note", "a1", None, 1, "A-1", None, {}, 1.0)
    d_sheet2 = Delta("d2", "remove", "note", "a1", None, 2, "A-1", None, {}, 1.0)
    boxes = _classified_bboxes([d_sheet1, d_sheet2], sheet=1, els_a=els_a, els_b=els_b)
    assert len(boxes) == 1


# --- full pipeline, opt-in behavior ------------------------------------------

def _doc(pid, elements, raster_paths):
    return CanonicalDocument(pid=pid, source_format="pdf_native", revision_label=None,
                              sheets=[CanonicalSheet(number=1, width=1.0, height=1.0, elements=elements)],
                              raster_paths=raster_paths)


def test_detect_returns_empty_when_flag_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("DELTA_RASTER_RECALL", raising=False)
    img = Image.new("L", (100, 100), 255)
    path = tmp_path / "a.png"
    img.save(path)
    doc_a = _doc("A", [_el("a1", "note", conf=0.3)], {1: str(path)})
    doc_b = _doc("B", [_el("b1", "note", conf=0.3)], {1: str(path)})
    result = detect_unclassified_visual_changes(doc_a, doc_b, [], Transform())
    assert result == []


def test_detect_finds_new_region_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("DELTA_RASTER_RECALL", "1")
    img_a = Image.new("L", (200, 200), 255)
    img_b = img_a.copy()
    # 40x40 = 1600px, well above the default RASTER_RECALL_MIN_AREA_PX (150)
    ImageDraw.Draw(img_b).rectangle((120, 120, 160, 160), fill=0)
    path_a, path_b = tmp_path / "a.png", tmp_path / "b.png"
    img_a.save(path_a)
    img_b.save(path_b)

    low_conf_els = [_el(f"e{i}", "note", conf=0.3) for i in range(5)]
    doc_a = _doc("A", low_conf_els, {1: str(path_a)})
    doc_b = _doc("B", low_conf_els, {1: str(path_b)})
    result = detect_unclassified_visual_changes(doc_a, doc_b, [], Transform())

    assert len(result) == 1
    assert result[0].kind == "unclassified_visual_change"
    assert result[0].confidence == 0.2


def test_detect_dedups_against_existing_classified_delta(monkeypatch, tmp_path):
    monkeypatch.setenv("DELTA_RASTER_RECALL", "1")
    img_a = Image.new("L", (200, 200), 255)
    img_b = img_a.copy()
    ImageDraw.Draw(img_b).rectangle((120, 120, 160, 160), fill=0)
    path_a, path_b = tmp_path / "a.png", tmp_path / "b.png"
    img_a.save(path_a)
    img_b.save(path_b)

    low_conf_els = [_el(f"e{i}", "note", conf=0.3) for i in range(5)]
    # An element right on top of the diff region, already linked to a
    # classified delta -- the region should be treated as already explained.
    b_el = _el("b_covers_region", "note", conf=0.3, x0=0.55, y0=0.55, x1=0.85, y1=0.85)
    doc_a = _doc("A", low_conf_els, {1: str(path_a)})
    doc_b = _doc("B", low_conf_els + [b_el], {1: str(path_b)})
    existing = [Delta("d1", "add", "note", None, "b_covers_region", 1, None, "A-1", {}, 1.0)]

    result = detect_unclassified_visual_changes(doc_a, doc_b, existing, Transform())
    assert result == []


def test_raster_recall_never_triggers_on_real_native_pair():
    """Built-in safety property: native pairs' extraction_confidence is
    always 1.0 and geometry density is low, so should_run_raster_recall
    never fires on them."""
    pair_dir = PAIRS_DIR / "edited_000"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")
    from src.cli import _resolve_with_pid

    doc = _resolve_with_pid("A", str(pair_dir / "a" / "L0.pdf"))
    for sheet in doc.sheets:
        assert should_run_raster_recall(sheet.elements) is False
