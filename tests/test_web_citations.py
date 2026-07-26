"""A citation chip that highlights the wrong valve is worse than no chip.
These tests pin the two behaviours that guarantee that: coordinates always
come from the id lookup (never from the model-written sheet/zone in the
marker), and anything unresolvable returns None rather than a guess."""
import pytest

from src.canonical.model import (
    BBox,
    CanonicalDocument,
    CanonicalElement,
    CanonicalSheet,
)
from src.chat.citations import parse_citations
from src.delta.model import Delta
from src.markup.overlay import _collect_boxes, _index_elements
from src.markup.payload import build_delta_records
from src.web.citations import CitationResolver


def _el(id_, content, x0, y0, x1, y1, sheet=1, zone="A-1"):
    return CanonicalElement(id=id_, type="note", content=content, bbox=BBox(x0, y0, x1, y1),
                             sheet=sheet, zone=zone, extraction_confidence=1.0)


def _doc(pid, elements):
    return CanonicalDocument(pid=pid, source_format="pdf_native", revision_label=None,
                              sheets=[CanonicalSheet(number=1, width=1.0, height=1.0,
                                                      elements=elements)],
                              raster_paths={})


def _records(deltas, doc_a, doc_b):
    els_a, els_b = _index_elements(doc_a), _index_elements(doc_b)
    boxes_a, boxes_b = _collect_boxes(deltas, els_a, els_b)
    return build_delta_records(deltas, boxes_a, boxes_b)


@pytest.fixture
def resolver():
    el_a = _el("el_aaa", "old note", 0.1, 0.1, 0.3, 0.15)
    el_b = _el("el_bbb", "new note", 0.1, 0.1, 0.3, 0.15)
    doc_a, doc_b = _doc("A", [el_a]), _doc("B", [el_b])
    deltas = [Delta("delta0001", "modify", "note", "el_aaa", "el_bbb", 1, "A-1", "A-1",
                     {"content": ["old", "new"]},
                     description="note text changed", severity="medium")]
    return CitationResolver(doc_a, doc_b, _records(deltas, doc_a, doc_b))


def test_resolves_an_element_citation_from_revision_a(resolver):
    (c,) = parse_citations("the note [A:1:A-1:el_aaa] was reworded")

    r = resolver.resolve(c)

    assert r["source"] == "A"
    assert r["sheet"] == 1
    assert r["box_a"] == [0.1, 0.1, 0.3, 0.15]
    # Nothing to highlight in the other pane: this element is A's.
    assert r["box_b"] is None
    assert r["description"] == "old note"


def test_resolves_an_element_citation_from_revision_b(resolver):
    (c,) = parse_citations("see [B:1:A-1:el_bbb]")

    r = resolver.resolve(c)

    assert r["source"] == "B"
    assert r["box_b"] == [0.1, 0.1, 0.3, 0.15]
    assert r["box_a"] is None


def test_resolves_a_delta_citation_to_both_panes(resolver):
    (c,) = parse_citations("that change [delta:1:A-1:delta0001] is medium severity")

    r = resolver.resolve(c)

    assert r["source"] == "delta"
    assert r["did"] == "delta0001"
    assert r["box_a"] == [0.1, 0.1, 0.3, 0.15]
    assert r["box_b"] == [0.1, 0.1, 0.3, 0.15]
    assert r["severity"] == "medium"
    assert r["kind"] == "modify"


def test_coordinates_come_from_the_id_not_the_marker(resolver):
    """validate_citations only checks the id (src/chat/citations.py:53), so
    a model can write any sheet/zone it likes into the marker and still
    pass. The resolved location must ignore both."""
    (c,) = parse_citations("[A:97:Z-99:el_aaa]")

    r = resolver.resolve(c)

    assert r["sheet"] == 1                       # from the element, not "97"
    assert r["box_a"] == [0.1, 0.1, 0.3, 0.15]


def test_unknown_element_id_resolves_to_none(resolver):
    (c,) = parse_citations("[A:1:A-1:el_does_not_exist]")

    assert resolver.resolve(c) is None


def test_unknown_delta_id_resolves_to_none(resolver):
    (c,) = parse_citations("[delta:1:A-1:delta9999]")

    assert resolver.resolve(c) is None


def test_unknown_source_label_resolves_to_none(resolver):
    """The citation regex accepts any non-':[]' run as a source, so a model
    can invent one. Unresolvable, not a crash."""
    (c,) = parse_citations("[C:1:A-1:el_aaa]")

    assert resolver.resolve(c) is None


def test_raster_origin_delta_resolves_through_its_own_bbox():
    """unclassified_visual_change deltas have no id_a/id_b -- raster_join
    finds pixels, not elements -- so the only location they have is the
    bbox set directly on the Delta."""
    doc_a, doc_b = _doc("A", []), _doc("B", [])
    deltas = [Delta("delta0001", "unclassified_visual_change", "geometry", None, None, 1,
                     "C-4", "C-4", bbox_a=BBox(0.5, 0.5, 0.6, 0.6),
                     bbox_b=BBox(0.5, 0.5, 0.6, 0.6),
                     visual_change_kind="graphical", severity="low")]
    resolver = CitationResolver(doc_a, doc_b, _records(deltas, doc_a, doc_b))
    (c,) = parse_citations("[delta:1:C-4:delta0001]")

    r = resolver.resolve(c)

    assert r["box_a"] == [0.5, 0.5, 0.6, 0.6]
    assert r["box_b"] == [0.5, 0.5, 0.6, 0.6]


def test_delta_with_no_location_resolves_but_carries_null_boxes():
    """Resolvable as a finding, just not placeable on the sheet. The UI
    shows the row without a jump target rather than hiding the citation."""
    doc_a, doc_b = _doc("A", []), _doc("B", [])
    deltas = [Delta("delta0001", "modify", "note", "gone_a", "gone_b", 1, "A-1", "A-1",
                     description="something changed", severity="low")]
    resolver = CitationResolver(doc_a, doc_b, _records(deltas, doc_a, doc_b))
    (c,) = parse_citations("[delta:1:A-1:delta0001]")

    r = resolver.resolve(c)

    assert r is not None
    assert r["box_a"] is None and r["box_b"] is None
    assert r["description"] == "something changed"


def test_resolve_all_preserves_order_and_keeps_the_raw_marker(resolver):
    """The client substitutes chips into the prose by exact string match on
    `raw`, so it must survive round-tripping."""
    text = "first [A:1:A-1:el_aaa] then [delta:1:A-1:delta0001] then [A:1:A-1:el_nope]"

    out = resolver.resolve_all(parse_citations(text))

    assert [o["raw"] for o in out] == ["[A:1:A-1:el_aaa]",
                                        "[delta:1:A-1:delta0001]",
                                        "[A:1:A-1:el_nope]"]
    assert all(o["raw"] in text for o in out)
    assert out[0]["resolved"] is not None
    assert out[1]["resolved"] is not None
    assert out[2]["resolved"] is None
