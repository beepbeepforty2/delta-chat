import pathlib

import pytest
from PIL import Image

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalSheet
from src.delta.model import Delta
from src.markup.overlay import COLORS, _denormalize, render_markup

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


def _el(id_, content, x0, y0, x1, y1, sheet=1, zone="A-1"):
    return CanonicalElement(id=id_, type="note", content=content, bbox=BBox(x0, y0, x1, y1),
                             sheet=sheet, zone=zone, extraction_confidence=1.0)


def _doc(pid, elements, raster_paths):
    return CanonicalDocument(pid=pid, source_format="pdf_native", revision_label=None,
                              sheets=[CanonicalSheet(number=1, width=1.0, height=1.0, elements=elements)],
                              raster_paths=raster_paths)


def _blank_png(path, size=(400, 300)):
    Image.new("RGB", size, (255, 255, 255)).save(path)
    return str(path)


def test_denormalize_scales_and_pads():
    el = _el("e1", "x", 0.5, 0.5, 0.6, 0.55)
    box = _denormalize(el, (1000, 1000))
    x0, y0, x1, y1 = box
    assert x0 < 500 and x1 > 600  # padded outward
    assert y0 < 500 and y1 > 550


def test_denormalize_clamps_to_image_bounds():
    el = _el("e1", "x", -0.01, -0.01, 1.01, 1.01)
    box = _denormalize(el, (100, 100))
    assert box == (0, 0, 100, 100)


def test_render_markup_writes_one_png_per_sheet(tmp_path):
    raster_a = _blank_png(tmp_path / "a1.png")
    raster_b = _blank_png(tmp_path / "b1.png")
    el_a = _el("a1", "old note", 0.1, 0.1, 0.3, 0.15)
    el_b = _el("b1", "new note", 0.1, 0.1, 0.3, 0.15)
    doc_a = _doc("A", [el_a], {1: raster_a})
    doc_b = _doc("B", [el_b], {1: raster_b})

    deltas = [Delta("d1", "modify", "note", "a1", "b1", 1, "A-1", "A-1", {"content": ["old", "new"]})]

    out_dir = tmp_path / "out"
    paths_a, paths_b = render_markup(doc_a, doc_b, deltas, str(out_dir))

    assert set(paths_a) == {1}
    assert set(paths_b) == {1}
    assert pathlib.Path(paths_a[1]).exists()
    assert pathlib.Path(paths_b[1]).exists()


def test_render_markup_actually_draws_the_modify_color(tmp_path):
    raster_a = _blank_png(tmp_path / "a1.png", size=(200, 200))
    raster_b = _blank_png(tmp_path / "b1.png", size=(200, 200))
    # Positioned away from the legend's fixed top-left footprint so the
    # sample below reads the delta box, not the always-drawn legend panel.
    el_a = _el("a1", "old note", 0.6, 0.6, 0.8, 0.7)
    el_b = _el("b1", "new note", 0.6, 0.6, 0.8, 0.7)
    doc_a = _doc("A", [el_a], {1: raster_a})
    doc_b = _doc("B", [el_b], {1: raster_b})
    deltas = [Delta("d1", "modify", "note", "a1", "b1", 1, "A-1", "A-1", {"content": ["old", "new"]})]

    paths_a, paths_b = render_markup(doc_a, doc_b, deltas, str(tmp_path / "out"))
    img = Image.open(paths_b[1])
    # sample the box's outline area (top edge of the padded, denormalized box)
    px = img.getpixel((140, 118))
    assert px != (255, 255, 255)  # something was drawn, not left blank


def test_render_markup_add_only_appears_on_b(tmp_path):
    raster_a = _blank_png(tmp_path / "a1.png")
    raster_b = _blank_png(tmp_path / "b1.png")
    # Positioned away from the legend's fixed top-left footprint.
    el_b = _el("b1", "new note", 0.6, 0.6, 0.8, 0.65)
    doc_a = _doc("A", [], {1: raster_a})
    doc_b = _doc("B", [el_b], {1: raster_b})
    deltas = [Delta("d1", "add", "note", None, "b1", 1, None, "A-1", {})]

    paths_a, paths_b = render_markup(doc_a, doc_b, deltas, str(tmp_path / "out"))
    img_a = Image.open(paths_a[1])
    img_b = Image.open(paths_b[1])
    # A's raster gets no *delta* annotation for an "add" (element didn't
    # exist in A) outside the always-drawn legend footprint.
    assert all(img_a.getpixel((x, y)) == (255, 255, 255)
               for x in range(200, 400, 20) for y in range(200, 300, 20))
    assert any(img_b.getpixel((x, y)) != (255, 255, 255)
               for x in range(200, 400, 5) for y in range(150, 250, 5))


def test_render_markup_draws_dashed_box_for_unclassified_visual_change(tmp_path):
    raster_a = _blank_png(tmp_path / "a1.png", size=(200, 200))
    raster_b = _blank_png(tmp_path / "b1.png", size=(200, 200))
    doc_a = _doc("A", [], {1: raster_a})
    doc_b = _doc("B", [], {1: raster_b})
    # Positioned away from the legend's fixed top-left footprint (same
    # pitfall other tests in this file already work around -- the panel
    # is tall enough now (5 legend items) to cover a naive top-left box).
    deltas = [Delta("raster0001", "unclassified_visual_change", "unclassified_visual_change",
                     None, None, 1, "F-6", "F-6", {}, confidence=0.3,
                     description="graphical change; not characterized by text engine",
                     bbox_a=BBox(0.6, 0.6, 0.8, 0.8), bbox_b=BBox(0.6, 0.6, 0.8, 0.8),
                     visual_change_kind="graphical")]

    paths_a, paths_b = render_markup(doc_a, doc_b, deltas, str(tmp_path / "out"))
    img_a = Image.open(paths_a[1])
    img_b = Image.open(paths_b[1])
    # drawn on BOTH sides (no id_a/id_b to say which side "owns" it)
    violet = COLORS["unclassified_visual_change"]
    for img in (img_a, img_b):
        found = any(img.getpixel((x, y))[:3] == violet
                    for x in range(115, 165) for y in range(115, 125))
        assert found, "expected a violet dashed box near the region's top edge"


def test_render_markup_covers_every_sheet_even_without_deltas(tmp_path):
    raster_a = _blank_png(tmp_path / "a1.png")
    raster_b = _blank_png(tmp_path / "b1.png")
    doc_a = _doc("A", [], {1: raster_a})
    doc_b = _doc("B", [], {1: raster_b})
    paths_a, paths_b = render_markup(doc_a, doc_b, [], str(tmp_path / "out"))
    assert set(paths_a) == {1} and set(paths_b) == {1}


def test_render_markup_real_pair_end_to_end(tmp_path):
    """Live check against a real generated pair, not just synthetic
    fixtures -- confirms raster paths, real bbox normalization, and real
    Delta objects from the actual engine all line up."""
    from src.cli import _resolve_with_pid, compute_deltas
    from src.observability.tracer import Tracer

    pair_dir = PAIRS_DIR / "edited_003"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")
    doc_a = _resolve_with_pid("A", str(pair_dir / "a" / "L0.pdf"))
    doc_b = _resolve_with_pid("B", str(pair_dir / "b" / "L0.pdf"))
    tracer = Tracer()
    deltas = compute_deltas(doc_a, doc_b, tracer)
    tracer.finish()
    assert deltas  # sanity: this pair does have real changes

    paths_a, paths_b = render_markup(doc_a, doc_b, deltas, str(tmp_path / "out"))
    assert paths_a and paths_b
    for p in list(paths_a.values()) + list(paths_b.values()):
        assert pathlib.Path(p).exists()
        img = Image.open(p)
        assert img.size[0] > 0 and img.size[1] > 0
