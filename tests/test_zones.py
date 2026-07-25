"""Zone computation tests. The corner cases and the cross-check against
Sheet.zone_of are load-bearing: they validate the top-left/y-down BBox
convention decision end-to-end."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "eval" / "datasets"))

from src.canonical.zones import compute_zone, is_zone_label_shaped
from generator.model import Sheet


def test_corner_top_left():
    assert compute_zone(0.0, 0.0) == "A-1"


def test_corner_bottom_right():
    assert compute_zone(0.999, 0.999) == "J-12"


def test_zone_center():
    assert compute_zone(0.5, 0.5) == "F-7"


def test_matches_sheet_zone_of_across_grid():
    sh = Sheet(sheet_no=1, of=1, rev="A")
    for gx in range(12):
        for gy in range(10):
            x_mm = (gx + 0.5) / 12 * sh.width
            y_mm = sh.height - (gy + 0.5) / 10 * sh.height  # bottom-left/y-up
            expected = sh.zone_of((x_mm, y_mm))
            x_norm = x_mm / sh.width
            y_norm = (sh.height - y_mm) / sh.height  # top-left/y-down
            assert compute_zone(x_norm, y_norm) == expected


def test_is_zone_label_shaped_column_in_margin():
    assert is_zone_label_shaped("7", 0.5, 0.005) is True


def test_is_zone_label_shaped_row_in_margin():
    assert is_zone_label_shaped("F", 0.995, 0.5) is True


def test_is_zone_label_shaped_rejects_out_of_margin():
    assert is_zone_label_shaped("7", 0.5, 0.5) is False


def test_is_zone_label_shaped_rejects_non_label_text():
    assert is_zone_label_shaped("PIT 9055 26", 0.005, 0.5) is False
