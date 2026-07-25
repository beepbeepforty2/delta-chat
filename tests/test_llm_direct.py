import json

from eval.baselines.llm_direct import _document_text, _parse_deltas, run_llm_direct
from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalSheet


def _doc(pid, elements):
    return CanonicalDocument(pid=pid, source_format="pdf_native", revision_label=None,
                              sheets=[CanonicalSheet(number=1, width=1.0, height=1.0, elements=elements)])


def _el(id_, type_, content, zone):
    return CanonicalElement(id=id_, type=type_, content=content,
                             bbox=BBox(0, 0, 0.1, 0.1), sheet=1, zone=zone, extraction_confidence=1.0)


def test_document_text_skips_geometry_and_empty():
    doc = _doc("A", [
        _el("e1", "note", "STRAINER TO BE REMOVED", "B-1"),
        _el("g1", "geometry", "", None),
    ])
    text = _document_text(doc, "A")
    assert "STRAINER TO BE REMOVED" in text
    assert "geometry" not in text
    assert "sheet 1" in text


def test_parse_deltas_extracts_valid_array():
    raw = json.dumps([
        {"kind": "modify", "element_type": "instrument", "sheet": 1,
         "zone_a": "F-7", "zone_b": "F-7", "description": "setpoint changed",
         "field_changes": {"HH": {"from": 150, "to": 214}}},
    ])
    deltas = _parse_deltas(raw)
    assert len(deltas) == 1
    assert deltas[0].kind == "modify"
    assert deltas[0].field_changes == {"HH": {"from": 150, "to": 214}}


def test_parse_deltas_tolerates_surrounding_prose_and_fences():
    raw = "Here is the JSON:\n```json\n" + json.dumps([
        {"kind": "add", "element_type": "note", "sheet": 1, "zone_a": None,
         "zone_b": "B-1", "description": "note added"},
    ]) + "\n```"
    deltas = _parse_deltas(raw)
    assert len(deltas) == 1
    assert deltas[0].kind == "add"


def test_parse_deltas_skips_malformed_items():
    raw = json.dumps([
        {"kind": "add", "sheet": 1},  # missing element_type -- still valid, has default
        {"description": "no kind or sheet at all"},  # invalid, dropped
    ])
    deltas = _parse_deltas(raw)
    assert len(deltas) == 1


def test_parse_deltas_empty_on_garbage():
    assert _parse_deltas("I cannot help with that.") == []


def test_run_llm_direct_uses_injected_call_llm():
    doc_a = _doc("A", [_el("e1", "note", "old text", "B-1")])
    doc_b = _doc("B", [_el("e1", "note", "new text", "B-1")])

    captured = {}

    def fake_call(system, user, temperature):
        captured["system"] = system
        captured["user"] = user
        captured["temperature"] = temperature
        return json.dumps([{"kind": "modify", "element_type": "note", "sheet": 1,
                             "zone_a": "B-1", "zone_b": "B-1", "description": "text changed"}])

    deltas = run_llm_direct(doc_a, doc_b, call_llm=fake_call)
    assert len(deltas) == 1
    assert captured["temperature"] == 0.0
    assert "old text" in captured["user"]
    assert "new text" in captured["user"]
