from eval.calibration import bucket_calibration
from src.delta.model import Delta


def _d(confidence):
    return Delta("d1", "add", "note", None, "x", 1, None, "A-1", {}, confidence)


def test_bucket_calibration_buckets_by_confidence():
    matched = [_d(0.95), _d(0.92), _d(0.6)]
    fp = [_d(0.4), _d(0.6)]
    bands = bucket_calibration(matched, fp)
    by_band = {b["band"]: b for b in bands}

    assert by_band["0.9-1.0"]["tp"] == 2
    assert by_band["0.9-1.0"]["fp"] == 0
    assert by_band["0.9-1.0"]["precision"] == 1.0

    assert by_band["0.5-0.75"]["tp"] == 1
    assert by_band["0.5-0.75"]["fp"] == 1
    assert by_band["0.5-0.75"]["precision"] == 0.5

    assert by_band["0.0-0.5"]["tp"] == 0
    assert by_band["0.0-0.5"]["fp"] == 1
    assert by_band["0.0-0.5"]["precision"] == 0.0


def test_bucket_calibration_empty_band_has_no_precision():
    bands = bucket_calibration([], [])
    for b in bands:
        assert b["n"] == 0
        assert b["precision"] is None


def test_bucket_calibration_boundary_value_goes_to_higher_band():
    # 0.75 is the lower (inclusive) bound of the 0.75-0.9 band, not the
    # upper bound of 0.5-0.75.
    bands = bucket_calibration([_d(0.75)], [])
    by_band = {b["band"]: b for b in bands}
    assert by_band["0.75-0.9"]["tp"] == 1
    assert by_band["0.5-0.75"]["tp"] == 0


def test_bucket_calibration_confidence_of_exactly_one():
    bands = bucket_calibration([_d(1.0)], [])
    by_band = {b["band"]: b for b in bands}
    assert by_band["0.9-1.0"]["tp"] == 1
