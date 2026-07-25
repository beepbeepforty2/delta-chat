"""Resource-handle hygiene: fitz Documents and PIL Images opened by the
ingest/markup/generator paths must be closed even when an inner step raises,
not only on the success path.

Strategy: monkeypatch ``fitz.open`` (and ``PIL.Image.open`` where relevant)
with a thin recorder that tracks every handle it returns, then assert each
handle is closed after the call -- on both the success path and the
exception path. The exception path is the whole point: it is what the bare
``doc.close()``-only-on-success code in the ingest adapters used to leak."""
import random
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "eval" / "datasets"))

import pytest

from generator.content import make_sheet
from generator.render import render_pdf, degrade
from src.ingest import pdf_native as pdf_native_mod
from src.markup import overlay as overlay_mod


# --------------------------------------------------------------------- helpers


class _RecordingDoc:
    """Wraps a fitz.Document, recording close() so tests can assert hygiene."""

    def __init__(self, doc):
        self._doc = doc
        self.closed = False

    def __getattr__(self, name):
        return getattr(self._doc, name)

    def __getitem__(self, idx):
        return self._doc[idx]

    def close(self):
        self._doc.close()
        self.closed = True

    def __iter__(self):
        return iter(self._doc)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _install_fitz_recorder(monkeypatch, target_mod):
    """Patch fitz.open (as seen by ``target_mod``) to return _RecordingDoc
    instances. ``target_mod`` does ``import fitz`` and then calls ``fitz.open``
    -- the lookup goes through the fitz module object, so we patch the ``open``
    attribute on that module. Returns a list appended to on every open."""
    import fitz as _fitz
    _real_fitz_open = _fitz.open  # capture before patching

    opened: list[_RecordingDoc] = []
    fitz_mod = target_mod.fitz

    def fake_open(*args, **kwargs):
        rec = _RecordingDoc(_real_fitz_open(*args, **kwargs))
        opened.append(rec)
        return rec

    monkeypatch.setattr(fitz_mod, "open", fake_open)
    return opened


@pytest.fixture(scope="module")
def native_pdf(tmp_path_factory):
    sheet = make_sheet(random.Random(7))
    path = tmp_path_factory.mktemp("hyg") / "native.pdf"
    render_pdf(sheet, str(path), producer="standard")
    return str(path)


# --------------------------------------------------------------------- success


def test_pdf_native_ingest_closes_doc_on_success(native_pdf, monkeypatch):
    opened = _install_fitz_recorder(monkeypatch, pdf_native_mod)
    PdfNativeAdapter = pdf_native_mod.PdfNativeAdapter

    doc = PdfNativeAdapter().ingest("pid_a", native_pdf)
    assert doc.sheets  # sanity: it actually ran
    assert opened, "fitz.open was never called"
    assert all(rec.closed for rec in opened), "a fitz Document was left open on the success path"


def test_pdf_native_detect_closes_doc_on_success(native_pdf, monkeypatch):
    opened = _install_fitz_recorder(monkeypatch, pdf_native_mod)
    assert pdf_native_mod.PdfNativeAdapter().detect(native_pdf) is True
    assert opened and all(rec.closed for rec in opened)


# --------------------------------------------------------------------- failure


def test_pdf_native_ingest_closes_doc_when_inner_step_raises(native_pdf, monkeypatch):
    """The regression: ingest() previously called doc.close() only after every
    inner step returned. If _rasterize (or text/geometry extraction) raised,
    the handle leaked. Force a raise and assert the doc is still closed."""
    opened = _install_fitz_recorder(monkeypatch, pdf_native_mod)

    def boom(*_a, **_kw):
        raise RuntimeError("simulated raster failure")

    # _rasterize runs last inside the try block, after all extraction work.
    monkeypatch.setattr(pdf_native_mod, "_rasterize", boom)

    with pytest.raises(RuntimeError, match="simulated raster failure"):
        pdf_native_mod.PdfNativeAdapter().ingest("pid_a", native_pdf)

    assert opened, "fitz.open was never called"
    assert all(rec.closed for rec in opened), (
        "ingest leaked a fitz Document when an inner step raised -- "
        "the try/finally guard is missing or broken"
    )


def test_pdf_annotate_closes_doc_when_save_raises(tmp_path, monkeypatch):
    """_annotate_document opens the source PDF, mutates pages, then saves+close.
    If save() raised (a real failure mode -- see pdf_annotate.py:48-54), the
    doc used to leak. Force save() to raise and assert closure."""
    from src.markup import pdf_annotate as pdf_annotate_mod

    # Build a tiny one-page PDF to annotate, using the real fitz.open (not
    # the recorder, which we install only for the call under test).
    import fitz as _fitz
    src = _fitz.open()
    src.new_page()
    src_path = tmp_path / "src.pdf"
    src.save(str(src_path))
    src.close()

    opened = _install_fitz_recorder(monkeypatch, pdf_annotate_mod)

    # Make the recorded Document's save() blow up before close() runs.
    def save_boom(*_a, **_kw):
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(_RecordingDoc, "save", save_boom, raising=False)

    with pytest.raises(RuntimeError, match="simulated save failure"):
        pdf_annotate_mod._annotate_document(
            str(src_path), boxes_by_sheet={}, visual_changes_by_sheet={},
            out_path=tmp_path / "out.pdf",
        )
    assert opened and all(rec.closed for rec in opened)


def test_render_degrade_closes_source_doc(tmp_path, monkeypatch):
    """render.degrade() opens the source PDF and (before the fix) never closed
    it. Assert via the recorder that every fitz Document opened during degrade
    (the source reader AND the output writer) is closed by the time it returns.

    We do NOT use ``fitz.TOOLS.mupdf_warnings()`` as the signal here: degrade
    legitimately emits benign MuPDF warnings (image insertion, font handling)
    that have nothing to do with handle leaks, so absence-of-warnings is not a
    reliable hygiene check. The recorder pattern is direct and unambiguous."""
    from generator import render as render_mod

    sheet = make_sheet(random.Random(11))
    src = tmp_path / "src.pdf"
    render_pdf(sheet, str(src), producer="standard")
    out = tmp_path / "deg.pdf"

    opened = _install_fitz_recorder(monkeypatch, render_mod)
    degrade(str(src), str(out), level=2, seed=3)
    assert out.exists()
    assert opened, "degrade never called fitz.open"
    assert all(rec.closed for rec in opened), (
        "degrade left a fitz Document open (source or output) -- "
        "the try/finally guard is missing or broken"
    )


def test_overlay_annotate_does_not_hold_file_open(tmp_path):
    """overlay._annotate() does Image.open(raster_path).convert(...). The lazy
    Image.open holds the file open until GC; after the fix it is closed as
    soon as the convert() copy is made. Assert the raster file can be removed
    immediately after _annotate returns (it could not be while the handle was
    held on Windows / some platforms)."""
    from PIL import Image

    raster = tmp_path / "sheet.png"
    Image.new("RGB", (40, 40), (255, 255, 255)).save(str(raster))

    out = overlay_mod._annotate(str(raster), entries=[], visual_changes=[], legend=False)
    assert out.size == (40, 40)

    # If _annotate still held the file open via Image.open, this unlink would
    # fail on platforms that lock open file handles.
    raster.unlink()
    assert not raster.exists()
