import math
import pathlib

import pytest

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalSheet
from src.delta.register import Transform, register
from src.ingest.pdf_native import PdfNativeAdapter

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


def _doc(elements):
    return CanonicalDocument(pid="X", source_format="pdf_native", revision_label=None,
                              sheets=[CanonicalSheet(number=1, width=1.0, height=1.0, elements=elements)])


def _el(id_, type_, content, x, y, zone=None):
    return CanonicalElement(id=id_, type=type_, content=content,
                             bbox=BBox(x, y, x, y), sheet=1, zone=zone, extraction_confidence=1.0)


def test_register_falls_back_to_identity_with_fewer_than_two_anchors():
    doc_a = _doc([_el("a1", "equipment_tag", "26-KA-902", 0.5, 0.5)])
    doc_b = _doc([_el("b1", "equipment_tag", "26-KA-902", 0.5, 0.5)])
    result = register(doc_a, doc_b)
    assert result == Transform()


def test_register_ignores_non_anchor_types_and_ambiguous_content():
    # "note" isn't an anchor type; the duplicated "B" zone label is ambiguous
    # (appears twice on each side) and must be dropped, not guessed at.
    doc_a = _doc([
        _el("a1", "note", "some note text", 0.5, 0.5),
        _el("a2", "zone_label", "B", 0.1, 0.1),
        _el("a3", "zone_label", "B", 0.9, 0.9),
    ])
    doc_b = _doc([
        _el("b1", "note", "some note text", 0.55, 0.55),
        _el("b2", "zone_label", "B", 0.15, 0.15),
        _el("b3", "zone_label", "B", 0.95, 0.95),
    ])
    result = register(doc_a, doc_b)
    assert result == Transform()  # still fewer than 2 usable anchors


def test_register_recovers_known_similarity_transform():
    known = Transform(scale=1.1, rotation=0.05, tx=0.02, ty=-0.01)
    b_points = [(0.2, 0.3), (0.6, 0.7), (0.4, 0.1)]
    a_points = [known.apply(x, y) for x, y in b_points]

    doc_a = _doc([_el(f"a{i}", "equipment_tag", f"TAG{i}", x, y)
                  for i, (x, y) in enumerate(a_points)])
    doc_b = _doc([_el(f"b{i}", "equipment_tag", f"TAG{i}", x, y)
                  for i, (x, y) in enumerate(b_points)])

    result = register(doc_a, doc_b)
    assert result.scale == pytest.approx(known.scale, abs=1e-6)
    assert result.rotation == pytest.approx(known.rotation, abs=1e-6)
    assert result.tx == pytest.approx(known.tx, abs=1e-6)
    assert result.ty == pytest.approx(known.ty, abs=1e-6)


def test_register_maps_a_held_out_point_correctly():
    """Recovered transform generalizes beyond the anchors themselves --
    the actual thing match_elements needs it for."""
    known = Transform(scale=0.95, rotation=-0.03, tx=0.01, ty=0.02)
    b_anchor_points = [(0.1, 0.1), (0.8, 0.2), (0.3, 0.9)]
    a_anchor_points = [known.apply(x, y) for x, y in b_anchor_points]

    doc_a = _doc([_el(f"a{i}", "title_field", f"FIELD{i}", x, y)
                  for i, (x, y) in enumerate(a_anchor_points)])
    doc_b = _doc([_el(f"b{i}", "title_field", f"FIELD{i}", x, y)
                  for i, (x, y) in enumerate(b_anchor_points)])
    result = register(doc_a, doc_b)

    held_out_b = (0.5, 0.5)
    expected_a = known.apply(*held_out_b)
    actual_a = result.apply(*held_out_b)
    assert actual_a[0] == pytest.approx(expected_a[0], abs=1e-6)
    assert actual_a[1] == pytest.approx(expected_a[1], abs=1e-6)


def test_register_real_native_pair_is_near_identity():
    """Native-native pairs are extracted at exact page dimensions with no
    skew -- registration should correctly converge on (near-)identity,
    not force-fit noise into a spurious transform."""
    pair_dir = PAIRS_DIR / "edited_000"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")
    adapter = PdfNativeAdapter()
    doc_a = adapter.ingest("A", str(pair_dir / "a" / "L0.pdf"))
    doc_b = adapter.ingest("B", str(pair_dir / "b" / "L0.pdf"))
    result = register(doc_a, doc_b)
    assert result.scale == pytest.approx(1.0, abs=0.02)
    assert result.rotation == pytest.approx(0.0, abs=0.02)
    assert abs(result.tx) < 0.02
    assert abs(result.ty) < 0.02
