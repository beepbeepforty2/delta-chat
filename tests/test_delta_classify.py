"""Delta classification tests against eval GT deltas.json.

Reuses the GT-element alignment harness (isolates classify.py logic from
extraction noise) then compares the (kind, id_a, id_b) delta set against
gt/deltas.json's (kind, eid_a, eid_b) set -- the same signal the
generator's own round-trip validator (eval/datasets/generator/generate.py
::validate_roundtrip) uses to check its own GT emission.
"""
import pytest

from src.canonical.model import BBox, CanonicalElement
from src.delta.align import MatchedPair, match_elements
from src.delta.classify import _confidence, classify_matches, _detect_family_offset_cascades
from src.delta.model import Delta
from src.delta.register import Transform

from tests._gt_helpers import PAIRS_DIR, EDITED_PAIRS, gt_sheet, gt_deltas


def _el(conf, id_="e1"):
    return CanonicalElement(id=id_, type="note", content="x", bbox=BBox(0, 0, 0.1, 0.1),
                             sheet=1, zone="A-1", extraction_confidence=conf)


def test_confidence_is_product_not_min_of_both_sides():
    """Regression: an earlier version used min(ext_conf_a, ext_conf_b),
    which agrees with the product whenever both sides are 1.0 (every
    native-native pair) but silently overstates confidence once real
    (<1.0) OCR confidence is involved -- CLAUDE.md decision #3 specifies
    the product."""
    pair = MatchedPair(a=_el(0.5), b=_el(0.8), margin=1.0)
    assert _confidence(pair) == round(0.5 * 0.8, 4)
    assert _confidence(pair) != min(0.5, 0.8)


def test_confidence_native_native_unaffected_by_the_fix():
    pair = MatchedPair(a=_el(1.0), b=_el(1.0), margin=1.0)
    assert _confidence(pair) == 1.0


@pytest.mark.parametrize("pair_id", EDITED_PAIRS)
def test_classify_matches_gt_delta_keys(pair_id):
    pair_dir = PAIRS_DIR / pair_id
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")

    sheet_a = gt_sheet(pair_dir, "a")
    sheet_b = gt_sheet(pair_dir, "b")
    matches = match_elements(sheet_a, sheet_b, Transform())
    deltas = classify_matches(matches, sheet=1)

    gt = gt_deltas(pair_dir)
    gt_keys = {(d["kind"], d["eid_a"], d["eid_b"]) for d in gt}
    pred_keys = {(d.kind, d.id_a, d.id_b) for d in deltas}

    missing = gt_keys - pred_keys
    extra = pred_keys - gt_keys
    # allow small slack: a handful of edge disagreements (e.g. move-threshold
    # tuning) are expected; the bulk must agree
    assert len(missing) <= 1, f"{pair_id}: missing {missing}"
    assert len(extra) <= 1, f"{pair_id}: extra {extra}"


@pytest.mark.parametrize("pair_id", ["edited_000", "edited_002", "edited_005"])
def test_note_insertion_cascade_detected(pair_id):
    """These pairs use InsertNoteWithCascade -- the tail of the notes block
    shifts by a constant +1 offset, which _detect_family_offset_cascades
    should catch generically (see classify.py's module docstring for the
    known gap: cascade members link to an arbitrary group member, not
    necessarily the true root-cause 'add' event)."""
    pair_dir = PAIRS_DIR / pair_id
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")
    sheet_a = gt_sheet(pair_dir, "a")
    sheet_b = gt_sheet(pair_dir, "b")
    matches = match_elements(sheet_a, sheet_b, Transform())
    deltas = classify_matches(matches, sheet=1)
    cascades = [d for d in deltas if d.is_cascade]
    assert len(cascades) >= 3, (
        f"{pair_id}: expected a note-renumbering cascade group, "
        f"found {len(cascades)} cascade deltas"
    )
    primaries = {d.primary_did for d in cascades}
    assert primaries, f"{pair_id}: cascade members have no primary_did set"


def test_family_offset_cascade_synthetic():
    """SystematicTagRenumber shape (not exercised by the currently
    generated seed=42 dataset): N same-type modifies with a constant
    numeric-field offset should group into one primary + cascades."""
    deltas = [
        Delta(f"d{i}", "modify", "instrument", f"a{i}", f"b{i}", 1, "A-1", "A-1",
              {"loop": [9000 + i, 8961 + i]}, 1.0)
        for i in range(5)
    ]
    _detect_family_offset_cascades(deltas)
    cascades = [d for d in deltas if d.is_cascade]
    primaries = [d for d in deltas if not d.is_cascade]
    assert len(cascades) == 4
    assert len(primaries) == 1
    assert all(d.primary_did == primaries[0].did for d in cascades)


def test_below_threshold_family_offset_not_grouped():
    """Fewer than CASCADE_MIN_FAMILY_SIZE members sharing an offset should
    NOT be grouped -- avoids hiding real independent changes behind a
    cascade tag just because two of them happen to share a delta value."""
    deltas = [
        Delta(f"d{i}", "modify", "instrument", f"a{i}", f"b{i}", 1, "A-1", "A-1",
              {"loop": [9000 + i, 8961 + i]}, 1.0)
        for i in range(2)
    ]
    _detect_family_offset_cascades(deltas)
    assert all(not d.is_cascade for d in deltas)


def test_multi_field_delta_not_swept_into_cascade():
    """A delta whose numeric field matches the group offset but which ALSO
    carries an independent content change (e.g. a dcn_note whose note_no
    shifted from renumbering AND whose dcns list gained a real new entry)
    must stay a standalone primary change, not get folded into the
    cascade tag -- it has meaning beyond the renumber."""
    deltas = [
        Delta(f"d{i}", "modify", "note", f"a{i}", f"b{i}", 1, "A-1", "A-1",
              {"note_no": [10 + i, 11 + i]}, 1.0)
        for i in range(3)
    ]
    mixed = Delta("dmix", "modify", "dcn_note", "a99", "b99", 1, "A-1", "A-1",
                   {"note_no": [13, 14], "dcns": [["x"], ["x", "y"]]}, 1.0)
    all_deltas = deltas + [mixed]
    _detect_family_offset_cascades(all_deltas)
    assert not mixed.is_cascade
    assert sum(d.is_cascade for d in deltas) == 2  # 3 pure single-field members -> 1 primary + 2 cascade


def test_unchanged_pair_emits_no_delta():
    from src.canonical.model import BBox, CanonicalElement
    el = CanonicalElement(id="x", type="note", content="1. FOO.", bbox=BBox(0.1, 0.1, 0.2, 0.11),
                           sheet=1, zone="A-1", extraction_confidence=1.0, attrs={"note_no": 1})
    from src.delta.align import MatchedPair
    deltas = classify_matches([MatchedPair(el, el, cost=0.0, margin=1.0)], sheet=1)
    assert deltas == []


def test_producer_jitter_float_noise_not_flagged_as_change():
    """Regression: the 'alt' PDF producer's sub-point coordinate jitter
    (eval/datasets/generator/render.py) shows up as e.g. r_norm differing
    at the 1e-8 scale on an otherwise-identical circle. Exact float
    equality would wrongly flag this on a null_prod (producer-variation)
    pair, which must emit zero deltas."""
    from src.canonical.model import BBox, CanonicalElement
    from src.delta.align import MatchedPair
    a = CanonicalElement(id="g1", type="geometry", content="", bbox=BBox(0.4, 0.4, 0.5, 0.5),
                          sheet=1, zone="F-6", extraction_confidence=1.0,
                          attrs={"geom_kind": "circle", "r_norm": 0.04308841705055533})
    b = CanonicalElement(id="g2", type="geometry", content="", bbox=BBox(0.4, 0.4, 0.5, 0.5),
                          sheet=1, zone="F-6", extraction_confidence=1.0,
                          attrs={"geom_kind": "circle", "r_norm": 0.04308840424921956})
    deltas = classify_matches([MatchedPair(a, b, cost=0.0, margin=1.0)], sheet=1)
    assert deltas == []
