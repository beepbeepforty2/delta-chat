"""Smoke tests against the real vendor P&ID samples in data/samples/ (see
PROVENANCE.md). These are format/density exemplars, not a revision pair --
skip cleanly if the files aren't present (redistribution rights may mean
they're not always checked into a given clone)."""
import pathlib
import time

import pytest

from src.ingest import pdf_native
from src.ingest.pdf_native import PdfNativeAdapter

SAMPLES_DIR = pathlib.Path(__file__).parent.parent / "data" / "samples"
LIFT_PDF = SAMPLES_DIR / "Lift Gas compressor-P&ID.pdf"
EXPORT_PDF = SAMPLES_DIR / "Export Gas Compressor-P&ID (1).pdf"

pytestmark = pytest.mark.skipif(
    not (LIFT_PDF.exists() and EXPORT_PDF.exists()),
    reason="real sample PDFs not present in data/samples/ (see PROVENANCE.md)",
)


@pytest.fixture(autouse=True)
def raster_to_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_native, "RASTER_CACHE_DIR", str(tmp_path))


@pytest.fixture(scope="module", params=[
    (str(LIFT_PDF), "26-KA-901"),
    (str(EXPORT_PDF), "26-KA-902"),
], ids=["lift", "export"])
def sample(request):
    return request.param


def test_detect_true(sample):
    path, _ = sample
    assert PdfNativeAdapter().detect(path) is True


def test_ingest_runs_fast(sample):
    path, _ = sample
    t0 = time.time()
    doc = PdfNativeAdapter().ingest("real_sample", path)
    elapsed = time.time() - t0
    assert elapsed < 10.0, f"ingest took {elapsed:.1f}s, expected sub-10s"
    assert len(doc.sheets) == 1


def test_zone_labels_found(sample):
    path, _ = sample
    doc = PdfNativeAdapter().ingest("real_sample", path)
    zone_labels = [e for e in doc.sheets[0].elements if e.type == "zone_label"]
    assert len(zone_labels) == 44


def test_tag_types_have_real_hits(sample):
    path, _ = sample
    doc = PdfNativeAdapter().ingest("real_sample", path)
    by_type = {}
    for e in doc.sheets[0].elements:
        by_type[e.type] = by_type.get(e.type, 0) + 1
    for etype in ("line_tag", "valve_tag", "nozzle", "note_deleted"):
        assert by_type.get(etype, 0) > 0, f"no {etype} elements found"


def test_instrument_bubbles_detected_on_real_samples(sample):
    """Real instrument bubbles stack system/function/loop across three
    separate baselines (e.g. '26' / 'PI' / '9055' on distinct lines),
    unlike the synthetic generator's single-line 'FUNC LOOP SYS' format
    parse_instrument expects. Same-baseline clustering correctly keeps them
    separate since they are genuinely distinct text runs -- this was a
    real composition-format gap, not a clustering bug -- fixed by
    _stack_instrument_bubbles' position-gated second pass (see
    pdf_native.py)."""
    path, _ = sample
    doc = PdfNativeAdapter().ingest("real_sample", path)
    n_instrument = sum(1 for e in doc.sheets[0].elements if e.type == "instrument")
    assert n_instrument > 0


def test_datasheet_region_recalibration_improved(sample):
    path, _ = sample
    doc = PdfNativeAdapter().ingest("real_sample", path)
    n_datasheet = sum(1 for e in doc.sheets[0].elements if e.type == "datasheet_row")
    assert n_datasheet >= 8


def test_drawing_number_extracted(sample):
    path, drawing_no = sample
    doc = PdfNativeAdapter().ingest("real_sample", path)
    contents = [e.content for e in doc.sheets[0].elements]
    assert any(drawing_no in c for c in contents)
