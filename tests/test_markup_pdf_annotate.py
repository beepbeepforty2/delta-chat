import pathlib

import fitz
import pytest

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalSheet
from src.delta.model import Delta
from src.markup.pdf_annotate import render_pdf_markup

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


def _el(id_, content, x0, y0, x1, y1, sheet=1, zone="A-1", type_="note"):
    return CanonicalElement(id=id_, type=type_, content=content, bbox=BBox(x0, y0, x1, y1),
                             sheet=sheet, zone=zone, extraction_confidence=1.0)


def _doc(pid, elements):
    return CanonicalDocument(pid=pid, source_format="pdf_native", revision_label=None,
                              sheets=[CanonicalSheet(number=1, width=1.0, height=1.0, elements=elements)])


def _blank_pdf(path, width=400, height=300):
    doc = fitz.open()
    doc.new_page(width=width, height=height)
    doc.save(str(path))
    doc.close()
    return str(path)


def _first_page_annots(pdf_path):
    """Keeps the fitz.Document and Page alive by returning them alongside
    the materialized annotation list -- annotation wrapper objects are
    invalidated once their parent Page/Document is garbage collected, a
    real PyMuPDF gotcha (`page.annots()` returning a page object inline,
    with nothing holding a reference to it, gets GC'd before the
    annotations are actually used)."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    return doc, page, list(page.annots())


def test_render_pdf_markup_creates_real_annotation_objects(tmp_path):
    pdf_a = _blank_pdf(tmp_path / "a.pdf")
    pdf_b = _blank_pdf(tmp_path / "b.pdf")
    el_a = _el("a1", "old note", 0.4, 0.4, 0.6, 0.5, type_="note")
    el_b = _el("b1", "new note", 0.4, 0.4, 0.6, 0.5, type_="note")
    doc_a = _doc("A", [el_a])
    doc_b = _doc("B", [el_b])
    deltas = [Delta("d1", "modify", "note", "a1", "b1", 1, "A-1", "A-1",
                     {"content": ["old note", "new note"]}, 1.0, description="note content changed")]

    out_a, out_b = render_pdf_markup(doc_a, doc_b, deltas, pdf_a, pdf_b, str(tmp_path / "out"))

    _doc_a, _page_a, annots_a = _first_page_annots(out_a)
    assert len(annots_a) == 1
    assert annots_a[0].info["content"] == "note content changed"
    assert annots_a[0].info["title"] == "delta-chat: modify"
    assert annots_a[0].type[1] == "Square"

    _doc_b, _page_b, annots_b = _first_page_annots(out_b)
    assert len(annots_b) == 1
    assert annots_b[0].info["content"] == "note content changed"


def test_render_pdf_markup_add_only_on_b(tmp_path):
    pdf_a = _blank_pdf(tmp_path / "a.pdf")
    pdf_b = _blank_pdf(tmp_path / "b.pdf")
    el_b = _el("b1", "new note", 0.4, 0.4, 0.6, 0.5)
    doc_a = _doc("A", [])
    doc_b = _doc("B", [el_b])
    deltas = [Delta("d1", "add", "note", None, "b1", 1, None, "A-1", {}, 1.0, description="note added")]

    out_a, out_b = render_pdf_markup(doc_a, doc_b, deltas, pdf_a, pdf_b, str(tmp_path / "out"))

    _doc_a, _page_a, annots_a = _first_page_annots(out_a)
    assert annots_a == []
    _doc_b, _page_b, annots_b = _first_page_annots(out_b)
    assert len(annots_b) == 1
    assert annots_b[0].info["content"] == "note added"


def test_render_pdf_markup_colors_match_kind(tmp_path):
    pdf_a = _blank_pdf(tmp_path / "a.pdf")
    pdf_b = _blank_pdf(tmp_path / "b.pdf")
    el_b = _el("b1", "new note", 0.4, 0.4, 0.6, 0.5)
    doc_a = _doc("A", [])
    doc_b = _doc("B", [el_b])
    deltas = [Delta("d1", "add", "note", None, "b1", 1, None, "A-1", {}, 1.0, description="note added")]

    _, out_b = render_pdf_markup(doc_a, doc_b, deltas, pdf_a, pdf_b, str(tmp_path / "out"))
    _doc_b, _page_b, annots_b = _first_page_annots(out_b)
    # add == forest green (34, 139, 34) -> normalized float, green channel dominant
    stroke = annots_b[0].colors["stroke"]
    assert stroke[1] > stroke[0] and stroke[1] > stroke[2]


def test_render_pdf_markup_cascade_gets_no_fill(tmp_path):
    pdf_a = _blank_pdf(tmp_path / "a.pdf")
    pdf_b = _blank_pdf(tmp_path / "b.pdf")
    el_a = _el("a1", "old", 0.4, 0.4, 0.6, 0.5)
    el_b = _el("b1", "new", 0.4, 0.4, 0.6, 0.5)
    doc_a = _doc("A", [el_a])
    doc_b = _doc("B", [el_b])
    deltas = [Delta("d1", "modify", "note", "a1", "b1", 1, "A-1", "A-1",
                     {"note_no": [1, 2]}, 1.0, description="renumbered", is_cascade=True)]

    _, out_b = render_pdf_markup(doc_a, doc_b, deltas, pdf_a, pdf_b, str(tmp_path / "out"))
    _doc_b, _page_b, annots_b = _first_page_annots(out_b)
    assert annots_b[0].colors["fill"] == []  # cascade: outline only, no fill


def test_render_pdf_markup_real_pair_end_to_end():
    pair_dir = PAIRS_DIR / "edited_003"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")
    from src.cli import _resolve_with_pid, compute_deltas
    from src.observability.tracer import Tracer

    doc_a = _resolve_with_pid("A", str(pair_dir / "a" / "L0.pdf"))
    doc_b = _resolve_with_pid("B", str(pair_dir / "b" / "L0.pdf"))
    tracer = Tracer()
    deltas = compute_deltas(doc_a, doc_b, tracer)
    tracer.finish()
    assert deltas

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_a, out_b = render_pdf_markup(doc_a, doc_b, deltas, str(pair_dir / "a" / "L0.pdf"),
                                          str(pair_dir / "b" / "L0.pdf"), tmp)
        _doc_b, _page_b, annots_b = _first_page_annots(out_b)
        # every non-cascade delta touching B should produce a real annotation
        # whose content matches that delta's own description
        b_descriptions = {d.description for d in deltas if d.id_b}
        annot_contents = {a.info["content"] for a in annots_b}
        assert annot_contents & b_descriptions
