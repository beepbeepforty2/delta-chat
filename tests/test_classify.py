from src.canonical.classify import classify, classify_geometry, is_border_rect


def test_classify_zone_label():
    etype, attrs = classify("7", 0.5, 0.005)
    assert etype == "zone_label"
    assert attrs["classification_rule"] == "regex:zone_label"


def test_classify_line_tag():
    etype, attrs = classify('4"-PV-26-9048-GC11S-38', 0.3, 0.3)
    assert etype == "line_tag"
    assert attrs["pipe_class"] == "GC11S"


def test_classify_instrument():
    etype, attrs = classify("PIT 9055 26 SD HH:245 LL:120", 0.3, 0.3)
    assert etype == "instrument"
    assert attrs["setpoints"] == {"HH": 245, "LL": 120}


def test_classify_valve_tag():
    etype, attrs = classify("26BL9123", 0.3, 0.3)
    assert etype == "valve_tag"


def test_classify_nozzle():
    etype, attrs = classify("N4203", 0.3, 0.3)
    assert etype == "nozzle"


def test_classify_equipment_tag():
    etype, attrs = classify("26-KA-905", 0.45, 0.5)
    assert etype == "equipment_tag"


def test_classify_title_field_rev():
    etype, attrs = classify("REV A", 0.98, 0.94)
    assert etype == "title_field"
    assert attrs["field"] == "rev"


def test_classify_deleted_note():
    etype, attrs = classify("15. DELETED.", 0.05, 0.5)
    assert etype == "note_deleted"
    assert attrs["deleted"] is True
    assert attrs["note_no"] == 15


def test_classify_deleted_note_range():
    etype, attrs = classify("6-7. DELETED.", 0.05, 0.5)
    assert etype == "note_deleted"
    assert attrs["range"] == [6, 7]
    assert "note_no" not in attrs


def test_classify_dcn_note():
    etype, attrs = classify("21. THIS P&ID CONTAINS DCN-KP-0273-1.", 0.05, 0.5)
    assert etype == "dcn_note"
    assert attrs["dcns"] == ["DCN-KP-0273-1"]
    assert attrs["note_no"] == 21


def test_classify_rev_row():
    etype, attrs = classify("A  2026-01-15  ISSUED FOR DESIGN", 0.98, 0.92)
    assert etype == "rev_row"


def test_classify_generic_note():
    etype, attrs = classify("3. ATMOSPHERIC VENT.", 0.05, 0.5)
    assert etype == "note"
    assert attrs["note_no"] == 3


def test_classify_datasheet_region_fallback():
    etype, attrs = classify("DUTY kW 776", 0.75, 0.65)
    assert etype == "datasheet_row"
    assert attrs["type_confidence"] < 1.0


def test_classify_datasheet_region_fallback_real_vendor_template():
    # real-vendor (MAN Energy Solutions) datasheet sits bottom-left, not
    # bottom-right like the synthetic generator -- second candidate rect
    etype, attrs = classify("DUTY kW 1835", 0.05, 0.75)
    assert etype == "datasheet_row"
    assert attrs["type_confidence"] < 1.0


def test_classify_title_block_region_fallback():
    etype, attrs = classify("3RD STAGE HP GAS LIFT COMPRESSOR", 0.85, 0.92)
    assert etype == "title_field"
    assert attrs["type_confidence"] < 1.0


def test_classify_unmatched_prose_falls_back_to_text():
    etype, attrs = classify("SOME RANDOM UNMATCHED PROSE HERE", 0.4, 0.4)
    assert etype == "text"


def test_classify_short_token_falls_back_to_unknown():
    etype, attrs = classify("XQ99Z", 0.4, 0.4)
    assert etype == "unknown"


def test_is_border_rect_true_for_full_sheet_inset():
    assert is_border_rect((0.007, 0.010, 0.993, 0.990)) is True


def test_is_border_rect_false_for_small_shape():
    assert is_border_rect((0.4, 0.4, 0.5, 0.5)) is False


def test_classify_geometry_excludes_border():
    assert classify_geometry("rect", (0.007, 0.010, 0.993, 0.990)) is None


def test_classify_geometry_line():
    etype, attrs = classify_geometry("line", (0.1, 0.1, 0.3, 0.1))
    assert etype == "geometry"
    assert attrs["geom_kind"] == "line"


def test_classify_geometry_circle():
    etype, attrs = classify_geometry("circle", (0.4, 0.4, 0.5, 0.5))
    assert etype == "geometry"
    assert attrs["geom_kind"] == "circle"
    assert attrs["r_norm"] > 0
