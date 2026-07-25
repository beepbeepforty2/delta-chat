import pathlib

import pytest

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalSheet
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


def _tag_el(id_, content, type_="line_tag"):
    return CanonicalElement(id=id_, type=type_, content=content, bbox=BBox(0.1, 0.1, 0.2, 0.11),
                             sheet=1, zone="A-1", extraction_confidence=1.0)


def _doc(pid, tags):
    els = [_tag_el(f"e{i}", t) for i, t in enumerate(tags)]
    return CanonicalDocument(pid=pid, source_format="pdf_native", revision_label=None,
                              sheets=[CanonicalSheet(number=1, width=1.0, height=1.0, elements=els)])


def test_no_titleblock_signal_but_high_tag_overlap_passes():
    """Neither drawno nor equipment_tag extracted on either side, but the
    two documents share the vast majority of their tag identifiers --
    the new tier-3 fallback should treat this as the same document rather
    than proceeding blind."""
    doc_a = _doc("a", ["26-L-1001", "26-L-1002", "26-L-1003", "26-L-1004"])
    doc_b = _doc("b", ["26-L-1001", "26-L-1002", "26-L-1003", "26-L-1099"])
    result = check_same_document(doc_a, doc_b)
    assert result.is_pair is True
    assert "tag-content overlap" in result.reason


def test_no_titleblock_signal_and_low_tag_overlap_is_refused():
    """Neither drawno nor equipment_tag extracted on either side, and the
    tag identifiers barely overlap at all -- likely different documents,
    should now be refused instead of blindly proceeding."""
    doc_a = _doc("a", ["26-L-1001", "26-L-1002", "26-L-1003", "26-L-1004"])
    doc_b = _doc("b", ["44-L-2001", "44-L-2002", "44-L-2003", "44-L-2004"])
    result = check_same_document(doc_a, doc_b)
    assert result.is_pair is False
    assert "tag-content overlap" in result.reason


def test_no_comparable_content_at_all_still_fails_open():
    """The genuinely-blind case (no title-block signal AND no tag content
    on one side either) must still proceed with a warning -- the one
    remaining safety-net gap this fix deliberately leaves in place."""
    doc_a = _doc("a", ["26-L-1001", "26-L-1002"])
    doc_b = _doc("b", [])
    result = check_same_document(doc_a, doc_b)
    assert result.is_pair is True
    assert "proceeding without identity confirmation" in result.reason


def _drawno_el(value):
    """A title_field element carrying a drawno value (possibly empty string)."""
    return CanonicalElement(
        id="drawno", type="title_field", content=str(value),
        bbox=BBox(0.1, 0.1, 0.2, 0.11), sheet=1, zone="A-1",
        extraction_confidence=1.0, attrs={"field": "drawno", "value": value},
    )


def _doc_with_drawno(pid, drawno_value):
    """A doc whose only title-block signal is a (possibly empty) drawno."""
    els = [_drawno_el(drawno_value), _tag_el("e0", "26-L-1001")]
    return CanonicalDocument(pid=pid, source_format="pdf_native", revision_label=None,
                             sheets=[CanonicalSheet(number=1, width=1.0, height=1.0, elements=els)])


def test_empty_string_drawno_on_both_sides_matches_as_pair():
    """Regression: the old ``if drawno_a and drawno_b`` truthiness check treated
    an empty-string drawno (a real extraction artifact from a malformed title
    block) as 'absent', silently downgrading to a weaker tier. With ``is not
    None``, two empty-string drawno values match as equal and the pair is
    accepted at the drawno tier itself."""
    doc_a = _doc_with_drawno("a", "")
    doc_b = _doc_with_drawno("b", "")
    result = check_same_document(doc_a, doc_b)
    assert result.is_pair is True
    assert result.reason == "drawing numbers match"


def test_empty_string_drawno_on_one_side_real_on_other_is_refused():
    """Empty string on one side, real value on the other: they differ, so the
    pair must be refused at the drawno tier (not silently fall through to
    equipment tags or tag-overlap)."""
    doc_a = _doc_with_drawno("a", "")
    doc_b = _doc_with_drawno("b", "DWG-1234")
    result = check_same_document(doc_a, doc_b)
    assert result.is_pair is False
    assert "drawing numbers differ" in result.reason
    assert "''" in result.reason and "DWG-1234" in result.reason
