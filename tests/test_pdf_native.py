"""End-to-end native-PDF adapter tests against a freshly-generated pair.
Mirrors tests/test_generator.py's sys.path pattern for importing the
generator package without requiring `make dataset` to have been run."""
import random
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "eval" / "datasets"))

import pytest

from generator.content import make_sheet
from generator.render import render_pdf, degrade
from src.canonical.model import BBox, CanonicalElement
from src.ingest.pdf_native import PdfNativeAdapter, _stack_instrument_bubbles


@pytest.fixture(scope="module")
def sheet():
    return make_sheet(random.Random(42))


@pytest.fixture(scope="module")
def paths(tmp_path_factory, sheet):
    d = tmp_path_factory.mktemp("pdf_native")
    std = str(d / "std.pdf")
    alt = str(d / "alt.pdf")
    degraded = str(d / "degraded.pdf")
    render_pdf(sheet, std, producer="standard")
    render_pdf(sheet, alt, producer="alt")
    degrade(std, degraded, level=2, seed=1)
    return {"std": std, "alt": alt, "degraded": degraded}


@pytest.fixture(scope="module")
def doc_std(paths):
    return PdfNativeAdapter().ingest("pid_a", paths["std"])


@pytest.fixture(scope="module")
def doc_alt(paths):
    return PdfNativeAdapter().ingest("pid_a", paths["alt"])


def test_detect_true_for_native_pdf(paths):
    a = PdfNativeAdapter()
    assert a.detect(paths["std"]) is True
    assert a.detect(paths["alt"]) is True


def test_detect_false_for_degraded_raster_pdf(paths):
    assert PdfNativeAdapter().detect(paths["degraded"]) is False


def test_one_sheet_correct_dimensions(doc_std, sheet):
    assert len(doc_std.sheets) == 1
    sh = doc_std.sheets[0]
    # generator builds in mm; fitz reports points (mm * 2.834645669)
    assert sh.width == pytest.approx(sheet.width * 2.834645669, rel=1e-3)
    assert sh.height == pytest.approx(sheet.height * 2.834645669, rel=1e-3)


def test_element_count_in_right_ballpark(doc_std, sheet):
    n_gt = len(sheet.elements)
    n_extracted = len(doc_std.sheets[0].elements)
    assert n_gt * 0.7 <= n_extracted <= n_gt * 1.3


def test_line_tag_parsed_matches_gt(doc_std, sheet):
    gt_line = next(e for e in sheet.elements.values() if e.role == "line_tag")
    match = next(e for e in doc_std.sheets[0].elements
                 if e.type == "line_tag" and e.content == gt_line.text)
    for field in ("size", "service", "system", "seq", "pipe_class", "insul"):
        assert match.attrs[field] == gt_line.attrs[field]


def test_instrument_setpoints_match_gt(doc_std, sheet):
    gt_inst = next(e for e in sheet.elements.values()
                    if e.role == "instrument" and "setpoints" in e.attrs)
    match = next(e for e in doc_std.sheets[0].elements
                 if e.type == "instrument" and e.attrs.get("loop") == gt_inst.attrs["loop"])
    assert match.attrs["setpoints"] == gt_inst.attrs["setpoints"]


def test_all_zone_labels_found(doc_std):
    zone_labels = [e for e in doc_std.sheets[0].elements if e.type == "zone_label"]
    assert len(zone_labels) == 44  # 12 cols x2 edges + 10 rows x2 edges


def test_geometry_count_matches_gt(doc_std, sheet):
    # make_sheet() always emits 6 geom_line (all horizontal, dy=0.0 --
    # regression guard: a horizontal/vertical line's fitz rect has zero
    # area and must not be dropped as "empty") + 1 geom_circle = 7.
    #
    # n_extracted is no longer required to equal 7 exactly: every
    # valve_tag now also renders a real vector glyph (render.py's
    # _draw_valve_symbol_pdf, a bowtie +/- a circle) that isn't modeled
    # as a separate geom_line/geom_circle Element in the GT sheet at
    # all -- unmodeled vector art the ingest adapter still picks up as
    # "geometry", exactly matching how a real vendor PDF's actual valve
    # symbols behave (see data/samples/real_pair_valves/PROVENANCE.md,
    # which documents this same interaction on real content). The
    # meaningful regression guard is that nothing GT-modeled ever gets
    # silently dropped -- extraction can only ever find as many or more
    # geometry elements than the model declares, never fewer.
    n_gt_geom = sum(1 for e in sheet.elements.values()
                     if e.role in ("geom_line", "geom_circle"))
    n_extracted = sum(1 for e in doc_std.sheets[0].elements if e.type == "geometry")
    assert n_gt_geom == 7
    assert n_extracted >= n_gt_geom


def test_computed_zone_matches_gt_zone(doc_std, sheet):
    checked = 0
    for gt_el in sheet.elements.values():
        if gt_el.role not in ("line_tag", "instrument", "valve_tag", "nozzle"):
            continue
        match = next((e for e in doc_std.sheets[0].elements if e.content == gt_el.text), None)
        if match is None:
            continue
        assert match.zone == gt_el.zone(sheet), f"{gt_el.eid}: {match.zone} != {gt_el.zone(sheet)}"
        checked += 1
    assert checked >= 5


def test_producer_variants_agree_on_parsed_attrs(doc_std, doc_alt, sheet):
    gt_line = next(e for e in sheet.elements.values() if e.role == "line_tag")
    m_std = next(e for e in doc_std.sheets[0].elements
                 if e.type == "line_tag" and e.attrs.get("seq") == gt_line.attrs["seq"])
    m_alt = next(e for e in doc_alt.sheets[0].elements
                 if e.type == "line_tag" and e.attrs.get("seq") == gt_line.attrs["seq"])
    assert {k: v for k, v in m_std.attrs.items() if k in gt_line.attrs} == \
           {k: v for k, v in m_alt.attrs.items() if k in gt_line.attrs}


def test_no_element_at_border_rect(doc_std):
    for el in doc_std.sheets[0].elements:
        if el.type != "geometry":
            continue
        spans_full_sheet = (el.bbox.x0 < 0.02 and el.bbox.y0 < 0.02 and
                             el.bbox.x1 > 0.98 and el.bbox.y1 > 0.98)
        assert not spans_full_sheet


def test_raster_paths_populated(doc_std):
    assert 1 in doc_std.raster_paths
    assert pathlib.Path(doc_std.raster_paths[1]).exists()


def test_revision_label_extracted(doc_std):
    assert doc_std.revision_label == "A"


def _orphan(id_, content, x0, y0, x1, y1):
    return CanonicalElement(id=id_, type="unknown", content=content, bbox=BBox(x0, y0, x1, y1),
                             sheet=1, zone="A-1", extraction_confidence=1.0,
                             attrs={"classification_rule": "fallback:tag_like"})


def _circle(id_, x0, y0, x1, y1):
    return CanonicalElement(id=id_, type="geometry", content="", bbox=BBox(x0, y0, x1, y1),
                             sheet=1, zone="A-1", extraction_confidence=1.0,
                             attrs={"geom_kind": "circle"})


def test_stacked_bubble_tokens_merge_into_instrument():
    """Real vendor instrument bubbles stack func/loop inside the circle
    and the system/unit label just outside it -- mirrors the exact layout
    found by inspecting data/samples/Lift Gas compressor-P&ID.pdf."""
    circle = _circle("c1", 0.500, 0.500, 0.514, 0.520)
    func = _orphan("t1", "PI", 0.505, 0.505, 0.508, 0.512)
    loop = _orphan("t2", "9055", 0.503, 0.512, 0.512, 0.519)
    system = _orphan("t3", "26", 0.492, 0.503, 0.497, 0.510)
    unrelated = _orphan("t4", "unrelated text", 0.700, 0.700, 0.750, 0.710)

    result = _stack_instrument_bubbles([func, loop, system, unrelated], [circle], sheet_no=1)

    instruments = [e for e in result if e.type == "instrument"]
    assert len(instruments) == 1
    assert instruments[0].attrs == {"func": "PI", "loop": 9055, "system": "26"}
    ids = {e.id for e in result}
    assert "t1" not in ids and "t2" not in ids and "t3" not in ids  # absorbed
    assert "t4" in ids  # untouched


def test_stray_tokens_far_from_any_circle_not_merged():
    """Same 3 token shapes, but nowhere near a circle -- must not merge
    (regression guard: this pass is gated on real bubble geometry, not a
    free-floating vertical-stacking heuristic)."""
    far_circle = _circle("c1", 0.100, 0.100, 0.110, 0.115)
    func = _orphan("t1", "PI", 0.505, 0.505, 0.508, 0.512)
    loop = _orphan("t2", "9055", 0.503, 0.512, 0.512, 0.519)
    system = _orphan("t3", "26", 0.492, 0.503, 0.497, 0.510)

    result = _stack_instrument_bubbles([func, loop, system], [far_circle], sheet_no=1)

    assert not any(e.type == "instrument" for e in result)
    assert {e.id for e in result} == {"t1", "t2", "t3"}


def test_already_classified_element_never_consumed():
    """A short token that already classified as something meaningful
    (not tier-3 fallback) must never be swept into a bubble merge, even
    if positioned exactly right -- only genuinely-unclustered orphans are
    fair game."""
    circle = _circle("c1", 0.500, 0.500, 0.514, 0.520)
    func = _orphan("t1", "PI", 0.505, 0.505, 0.508, 0.512)
    loop = _orphan("t2", "9055", 0.503, 0.512, 0.512, 0.519)
    already_classified = CanonicalElement(
        id="t3", type="zone_label", content="26", bbox=BBox(0.492, 0.503, 0.497, 0.510),
        sheet=1, zone="A-1", extraction_confidence=1.0,
        attrs={"classification_rule": "regex:zone_label"},
    )

    result = _stack_instrument_bubbles([func, loop, already_classified], [circle], sheet_no=1)

    assert not any(e.type == "instrument" for e in result)
    ids = {e.id for e in result}
    assert {"t1", "t2", "t3"} == ids  # nothing absorbed, all left as-is


def test_single_line_synthetic_format_unaffected_by_bubble_stacking(doc_std, sheet):
    """The existing single-line generator format ('PIT 9055 26' already on
    one baseline) must keep working exactly as before -- it's classified
    directly by parse_instrument via the normal per-line path and never
    even reaches _stack_instrument_bubbles as an orphan."""
    gt_inst = next(e for e in sheet.elements.values()
                   if e.role == "instrument" and "setpoints" in e.attrs)
    match = next(e for e in doc_std.sheets[0].elements
                 if e.type == "instrument" and e.attrs.get("loop") == gt_inst.attrs["loop"])
    assert match.attrs["func"] == gt_inst.attrs["func"]
    assert match.attrs["system"] == gt_inst.attrs["system"]
