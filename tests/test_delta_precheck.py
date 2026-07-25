import pathlib

import pytest

from src.delta.precheck import check_same_document
from src.ingest.pdf_native import PdfNativeAdapter

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"
SAMPLES_DIR = pathlib.Path(__file__).parent.parent / "data" / "samples"
LIFT_PDF = SAMPLES_DIR / "Lift Gas compressor-P&ID.pdf"
EXPORT_PDF = SAMPLES_DIR / "Export Gas Compressor-P&ID (1).pdf"


def _ingest(path):
    return PdfNativeAdapter().ingest("doc", str(path))


@pytest.mark.skipif(not (PAIRS_DIR / "edited_000").exists(), reason="run `make dataset` first")
def test_edited_pair_passes_precheck():
    doc_a = _ingest(PAIRS_DIR / "edited_000" / "a" / "L0.pdf")
    doc_b = _ingest(PAIRS_DIR / "edited_000" / "b" / "L0.pdf")
    result = check_same_document(doc_a, doc_b)
    assert result.is_pair is True


@pytest.mark.skipif(not (PAIRS_DIR / "not_a_pair_903").exists(), reason="run `make dataset` first")
def test_not_a_pair_control_is_refused():
    doc_a = _ingest(PAIRS_DIR / "not_a_pair_903" / "a" / "L0.pdf")
    doc_b = _ingest(PAIRS_DIR / "not_a_pair_903" / "b" / "L0.pdf")
    result = check_same_document(doc_a, doc_b)
    assert result.is_pair is False


@pytest.mark.skipif(
    not (LIFT_PDF.exists() and EXPORT_PDF.exists()),
    reason="real sample PDFs not present in data/samples/ (see PROVENANCE.md)",
)
def test_real_sibling_drawings_are_refused():
    # 26-KA-901 vs 26-KA-902: genuinely different equipment, not a revision pair
    doc_a = _ingest(LIFT_PDF)
    doc_b = _ingest(EXPORT_PDF)
    result = check_same_document(doc_a, doc_b)
    assert result.is_pair is False
    assert result.equipment_a != result.equipment_b


@pytest.mark.skipif(not LIFT_PDF.exists(), reason="real sample PDF not present")
def test_real_document_against_itself_passes_precheck():
    doc = _ingest(LIFT_PDF)
    result = check_same_document(doc, doc)
    assert result.is_pair is True
