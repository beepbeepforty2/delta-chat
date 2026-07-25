import pathlib

import pytest

from src.delta.model import Delta
from src.delta.severity import classify_severity
from src.cli import _resolve_with_pid, compute_deltas
from src.observability.tracer import Tracer
from tests._gt_helpers import pairs_with_op

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


def _delta(**kw):
    base = dict(did="d1", kind="modify", element_type="note", id_a="a", id_b="b",
                sheet=1, zone_a="A-1", zone_b="A-1", field_changes={}, is_cascade=False)
    base.update(kw)
    return Delta(**base)


def test_setpoint_change_is_critical():
    d = _delta(element_type="instrument", field_changes={"setpoints": [{"HH": 235}, {"HH": 240}]})
    assert classify_severity(d) == "critical"


def test_pipe_class_change_is_high():
    d = _delta(element_type="line_tag", field_changes={"pipe_class": ["AC21S", "GC11S"]})
    assert classify_severity(d) == "high"


def test_equipment_tag_modify_is_high():
    d = _delta(element_type="equipment_tag", field_changes={"seq": [902, 903]})
    assert classify_severity(d) == "high"


def test_process_element_add_remove_is_high():
    for kind in ("add", "remove"):
        d = _delta(kind=kind, element_type="valve_tag", field_changes={})
        assert classify_severity(d) == "high", kind


def test_datasheet_row_modify_is_medium():
    d = _delta(element_type="datasheet_row", field_changes={"value": ["40210", "78511"]})
    assert classify_severity(d) == "medium"


def test_note_modify_is_low():
    d = _delta(element_type="note", field_changes={"body": ["old text", "new text"]})
    assert classify_severity(d) == "low"


def test_move_is_always_low_even_on_process_element():
    d = _delta(kind="move", element_type="line_tag", field_changes={})
    assert classify_severity(d) == "low"


def test_cascade_member_is_always_low_even_on_setpoint_field():
    """is_cascade wins over every other rule -- a cascade is a
    renumbering side-effect with no independent engineering content,
    regardless of what element type or field it landed on."""
    d = _delta(element_type="instrument", field_changes={"setpoints": [{"HH": 1}, {"HH": 2}]},
               is_cascade=True)
    assert classify_severity(d) == "low"


def test_title_field_modify_is_low():
    d = _delta(element_type="title_field", field_changes={"value": ["REV A", "REV B"]})
    assert classify_severity(d) == "low"


def test_severity_survives_to_dict():
    d = _delta(element_type="instrument", field_changes={"setpoints": [{"HH": 1}, {"HH": 2}]})
    d.severity = "critical"
    assert d.to_dict()["severity"] == "critical"


def _run(pair_id: str, level: str = "L0"):
    pair_dir = PAIRS_DIR / pair_id
    doc_a = _resolve_with_pid("A", str(pair_dir / "a" / f"{level}.pdf"))
    doc_b = _resolve_with_pid("B", str(pair_dir / "b" / f"{level}.pdf"))
    tracer = Tracer()
    deltas = compute_deltas(doc_a, doc_b, tracer)
    tracer.finish()
    return deltas


def test_real_pair_every_delta_gets_a_severity():
    if not (PAIRS_DIR / "edited_000").exists():
        pytest.skip("run `make dataset` first")
    deltas = _run("edited_000")
    assert deltas
    for d in deltas:
        assert d.severity in ("critical", "high", "medium", "low")


def test_real_setpoint_change_pair_is_critical():
    """Whichever pair actually used ChangeSetpoint (looked up from the
    manifest, not hardcoded -- which pair index draws which op shifts
    whenever CONTENT_OPS' length changes) -- confirms the engine's own
    matched/classified output (not a synthetic Delta) gets tagged
    critical on a real instrument setpoint change."""
    pair_ids = pairs_with_op("ChangeSetpoint")
    if not pair_ids or not (PAIRS_DIR / pair_ids[0]).exists():
        pytest.skip("run `make dataset` first")
    deltas = _run(pair_ids[0])
    setpoint_deltas = [d for d in deltas if d.element_type == "instrument"
                        and "setpoints" in d.field_changes]
    assert setpoint_deltas
    assert all(d.severity == "critical" for d in setpoint_deltas)


def test_real_pipe_class_change_pair_is_high():
    """edited_004's GT has a real pipe_class change (AC21S -> GC11S)."""
    if not (PAIRS_DIR / "edited_004").exists():
        pytest.skip("run `make dataset` first")
    deltas = _run("edited_004")
    pipe_class_deltas = [d for d in deltas if d.element_type == "line_tag"
                          and "pipe_class" in d.field_changes]
    assert pipe_class_deltas
    assert all(d.severity == "high" for d in pipe_class_deltas)
