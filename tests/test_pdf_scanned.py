"""Scanned-PDF adapter tests against the eval dataset's degradation ladder
(L1 clean raster -> L3 skewed/noisy/blurred). No GT-exact-match assertions
here (OCR is inherently lossy) -- loose sanity bounds and cross-level
recall comparison instead, consistent with the honest, documented-not-
hidden approach the rest of this codebase takes toward extraction gaps."""
import pathlib

import pytest

from src.ingest import pdf_scanned as pdf_scanned_mod
from src.ingest.pdf_scanned import PdfScannedAdapter, _ocr_words
from src.ingest.pdf_native import PdfNativeAdapter

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"
PAIR = PAIRS_DIR / "edited_000"


def _skip_if_missing():
    if not PAIR.exists():
        pytest.skip("run `make dataset` first")


@pytest.fixture(scope="module")
def doc_l1():
    _skip_if_missing()
    return PdfScannedAdapter().ingest("scanned_l1", str(PAIR / "a" / "L1.pdf"))


@pytest.fixture(scope="module")
def doc_l3():
    _skip_if_missing()
    return PdfScannedAdapter().ingest("scanned_l3", str(PAIR / "a" / "L3.pdf"))


def test_detect_true_for_degraded_pdfs():
    _skip_if_missing()
    adapter = PdfScannedAdapter()
    for level in ("L1", "L2", "L3"):
        assert adapter.detect(str(PAIR / "a" / f"{level}.pdf")) is True, level


def test_detect_false_for_native_pdf():
    _skip_if_missing()
    assert PdfScannedAdapter().detect(str(PAIR / "a" / "L0.pdf")) is False


def test_native_and_scanned_detect_are_complementary():
    """No gap, no overlap: exactly one of the two adapters claims each
    format level."""
    _skip_if_missing()
    native, scanned = PdfNativeAdapter(), PdfScannedAdapter()
    for level in ("L0", "L1", "L2", "L3"):
        path = str(PAIR / "a" / f"{level}.pdf")
        assert native.detect(path) != scanned.detect(path), level


def test_one_sheet_with_elements(doc_l1):
    assert len(doc_l1.sheets) == 1
    assert len(doc_l1.sheets[0].elements) > 20


def test_source_format_is_pdf_scanned(doc_l1):
    assert doc_l1.source_format == "pdf_scanned"


def test_extraction_confidence_reflects_ocr_not_always_perfect(doc_l1):
    confidences = [e.extraction_confidence for e in doc_l1.sheets[0].elements]
    assert any(c < 1.0 for c in confidences)
    assert all(0.0 <= c <= 1.0 for c in confidences)


def test_tag_types_found_on_clean_raster(doc_l1):
    by_type = {}
    for e in doc_l1.sheets[0].elements:
        by_type[e.type] = by_type.get(e.type, 0) + 1
    for etype in ("note", "line_tag", "valve_tag"):
        assert by_type.get(etype, 0) > 0, f"no {etype} found on L1 (clean raster)"


def test_zone_labels_partially_recovered(doc_l1):
    """Unlike pdf_native's exact 44/44, OCR on small isolated digits is
    genuinely lossy (spike found e.g. '5' misread as 'is)') -- a partial,
    honest lower bound, not the exact GT count."""
    n = sum(1 for e in doc_l1.sheets[0].elements if e.type == "zone_label")
    # Observed on the eval dataset's L1 render: single-character digits at
    # sheet-edge margins are exactly the OCR-hardest case (low information
    # density, easily read as punctuation/garbage -- spike found '5' -> 'is)',
    # '8' -> 'te}'). A real but modest lower bound, not the exact GT count.
    assert n >= 3, f"expected at least a few zone labels recovered, got {n}"
    assert n <= 44


def test_revision_label_extracted(doc_l1):
    assert doc_l1.revision_label in ("A", None)  # None is an acceptable OCR miss, not a crash


def test_recall_does_not_collapse_from_l1_to_l3(doc_l1, doc_l3):
    """L3 adds skew + noise + blur on top of L1. Recall should degrade
    gracefully, not catastrophically -- this is the actual claim CLAUDE.md
    makes about format-level eval (P/R/F1 per format level), checked here
    as a coarse sanity bound rather than the full eval harness."""
    n_l1 = len(doc_l1.sheets[0].elements)
    n_l3 = len(doc_l3.sheets[0].elements)
    assert n_l3 >= n_l1 * 0.5, f"L3 recall collapsed: {n_l3} vs L1's {n_l1}"


def test_no_geometry_elements_extracted(doc_l1):
    """Documented cut: no CV-based line/circle detection from the raster."""
    assert not any(e.type == "geometry" for e in doc_l1.sheets[0].elements)


# --------------------------------------------------------------- OCR robustness


def test_ocr_words_skips_non_numeric_confidence(monkeypatch):
    """Regression: int(data['conf'][i]) used to crash with ValueError when
    tesseract emitted an empty-string confidence (observed on some versions
    for non-text entries). Now such entries are skipped, not fatal.

    Monkeypatches pytesseract.image_to_data to return a controlled dict so the
    test does not depend on a real tesseract run."""
    from PIL import Image

    fake_data = {
        "text": ["good", "", "also", "bad"],
        "conf": ["95", "-1", "80", ""],   # last entry is the malformed one
        "left": [10, 20, 30, 40],
        "top": [10, 20, 30, 40],
        "width": [20, 20, 20, 20],
        "height": [10, 10, 10, 10],
    }
    monkeypatch.setattr(pdf_scanned_mod.pytesseract, "image_to_data",
                        lambda img, output_type=None: fake_data)

    img = Image.new("RGB", (100, 100))
    words = _ocr_words(img)
    # "good" (conf 95) and "also" (conf 80) survive; "" is empty-text (skipped
    # by the empty guard), "-1" conf below threshold (skipped), "" conf is
    # non-numeric (skipped by the new guard, previously crashed).
    texts = [w["text"] for w in words]
    assert "good" in texts
    assert "also" in texts
    assert len(words) == 2


def test_ocr_words_accepts_float_formatted_confidence(monkeypatch):
    """The non-numeric guard above must not swallow *numeric* confidences.

    tesseract/pytesseract builds vary in how they spell conf: an int, "95",
    or "95.0". The guard SKIPS the word on a parse failure, and int("95.0")
    raises ValueError -- so parsing with a bare int() would drop every word
    on a decimal-formatting build, returning an empty OCR result with nothing
    raised. That silent total data loss is a worse failure than the crash the
    guard was added to prevent, which is why parsing goes through float()."""
    from PIL import Image

    fake_data = {
        "text": ["alpha", "beta", "gamma"],
        "conf": ["95.0", 88, "70.5"],   # float-string, real int, fractional
        "left": [10, 20, 30],
        "top": [10, 20, 30],
        "width": [20, 20, 20],
        "height": [10, 10, 10],
    }
    monkeypatch.setattr(pdf_scanned_mod.pytesseract, "image_to_data",
                        lambda img, output_type=None: fake_data)

    words = _ocr_words(Image.new("RGB", (100, 100)))

    assert [w["text"] for w in words] == ["alpha", "beta", "gamma"], (
        "a float-formatted confidence must be parsed, not silently dropped"
    )
    assert [w["conf"] for w in words] == [95, 88, 70]  # truncated toward zero
