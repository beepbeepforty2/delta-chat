"""Composite-tag parser tests, against literal strings the generator emits."""
from src.canonical.tags import (
    parse_line_tag, parse_instrument, parse_valve_tag, parse_nozzle,
    parse_equipment_tag, parse_title_field,
)


def test_parse_line_tag():
    assert parse_line_tag('4"-PV-26-9048-GC11S-38') == {
        "size": '4"', "service": "PV", "system": "26",
        "seq": "9048", "pipe_class": "GC11S", "insul": "38",
    }


def test_parse_line_tag_rejects_prose():
    assert parse_line_tag("HIGH POINT VENT AND LOW POINT DRAIN.") is None


def test_parse_instrument_with_setpoints():
    assert parse_instrument("PIT 9055 26 SD HH:245 LL:120") == {
        "func": "PIT", "loop": 9055, "system": "26",
        "setpoints": {"HH": 245, "LL": 120},
    }


def test_parse_instrument_without_setpoints():
    assert parse_instrument("TIT 9012 26") == {
        "func": "TIT", "loop": 9012, "system": "26",
    }


def test_parse_instrument_negative_loop_after_renumber():
    # systematic_tag_renumber can push loop numbers below the 4-digit base
    assert parse_instrument("PIT 8971 26")["loop"] == 8971


def test_parse_instrument_rejects_line_tag():
    assert parse_instrument('4"-PV-26-9048-GC11S-38') is None


def test_parse_valve_tag():
    assert parse_valve_tag("26BL9123") == {"system": "26", "body": "BL", "seq": "9123"}


def test_parse_valve_tag_rejects_nozzle():
    assert parse_valve_tag("N4203") is None


def test_parse_nozzle():
    assert parse_nozzle("N4203") == {"noz_no": 4203}


def test_parse_equipment_tag():
    assert parse_equipment_tag("26-KA-905") == {"system": "26", "type": "KA", "seq": 905}


def test_parse_title_field_drawno():
    assert parse_title_field("0D204-PID-26-905-001") == {
        "field": "drawno", "value": "0D204-PID-26-905-001",
    }


def test_parse_title_field_rev():
    assert parse_title_field("REV A") == {"field": "rev", "value": "A"}


def test_parse_title_field_sheet():
    assert parse_title_field("SHEET 1 OF 1") == {
        "field": "sheet", "sheet_no": 1, "of": 1,
    }


def test_parse_title_field_date():
    assert parse_title_field("2026-01-15") == {"field": "date", "value": "2026-01-15"}


def test_parse_title_field_rejects_generic_prose():
    assert parse_title_field("3RD STAGE HP GAS LIFT COMPRESSOR") is None
