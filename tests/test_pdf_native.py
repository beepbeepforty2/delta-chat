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
from src.ingest.pdf_native import PdfNativeAdapter


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
