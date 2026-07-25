from src.delta.model import Delta
from eval.metrics import match_deltas, prf1, score_pair

GT_TEMPLATE = {
    "did": "d1", "kind": "add", "role": "note", "eid_a": None, "eid_b": "note27",
    "sheet": 1, "zone_a": None, "zone_b": "A-3", "field_changes": {},
    "description": "note added: STRAINER TO BE REMOVED AFTER COMMISSIONING.",
    "is_cascade": False, "primary_did": None, "semantic_null": False,
}


def gt(**overrides):
    d = dict(GT_TEMPLATE)
    d.update(overrides)
    return d


def test_prf1_perfect():
    assert prf1(5, 0, 0) == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 5, "fp": 0, "fn": 0}


def test_prf1_no_predictions_no_gt():
    assert prf1(0, 0, 0)["precision"] == 1.0
    assert prf1(0, 0, 0)["recall"] == 1.0


def test_match_deltas_exact_match():
    pred = [Delta("p1", "add", "note", None, "x", 1, None, "A-3", {}, 1.0,
                   "note added: STRAINER TO BE REMOVED AFTER COMMISSIONING.")]
    result = match_deltas(pred, [gt()])
    assert len(result.matched) == 1
    assert not result.false_positives
    assert not result.false_negatives


def test_match_deltas_wrong_zone_no_match():
    pred = [Delta("p1", "add", "note", None, "x", 1, None, "Z-9", {}, 1.0, "note added: something else entirely")]
    result = match_deltas(pred, [gt()])
    assert len(result.matched) == 0
    assert len(result.false_positives) == 1
    assert len(result.false_negatives) == 1


def test_match_deltas_prefers_zone_over_description():
    """Two GT candidates share (kind, sheet); zone match should win over a
    candidate with a better description-similarity score but wrong zone."""
    pred = [Delta("p1", "add", "note", None, "x", 1, None, "A-3", {}, 1.0, "totally different text")]
    close_desc_wrong_zone = gt(zone_b="Z-9", description="totally different text")
    right_zone = gt(zone_b="A-3", description="unrelated description")
    result = match_deltas(pred, [close_desc_wrong_zone, right_zone])
    assert len(result.matched) == 1
    assert result.matched[0][1]["zone_b"] == "A-3"


def test_score_pair_excludes_semantic_null_from_overall():
    pred = []  # engine detects nothing
    gts = [gt(kind="modify", description="reworded", semantic_null=True)]
    score = score_pair(pred, gts)
    assert score["overall"]["fn"] == 0  # semantic-null GT excluded, not a missed real change
    assert score["n_gt_semantic_null"] == 1
    assert score["semantic_null_emission_rate"] == 0.0  # correctly did NOT flag it


def test_score_pair_semantic_null_emission_rate_when_engine_flags_it():
    pred = [Delta("p1", "modify", "note", "a", "a", 1, "A-1", "A-1", {"body": ["x", "y"]}, 1.0, "reworded")]
    gts = [gt(kind="modify", zone_a="A-1", zone_b="A-1", eid_a="a", eid_b="a",
               description="reworded", semantic_null=True)]
    score = score_pair(pred, gts)
    assert score["semantic_null_emission_rate"] == 1.0  # engine has no semantic adjudication -- expected
    assert score["overall"]["fp"] == 0  # not counted against the real-change metric either


def test_score_pair_primary_cascade_recall_split():
    primary = gt(kind="modify", did="d1", zone_a="A-1", zone_b="A-1", description="primary change")
    cascade = gt(kind="modify", did="d2", zone_a="B-1", zone_b="B-1", description="cascade change",
                 is_cascade=True, primary_did="d1")
    pred = [Delta("p1", "modify", "note", "x", "x", 1, "A-1", "A-1", {"a": [1, 2]}, 1.0, "primary change")]
    score = score_pair(pred, [primary, cascade])
    assert score["primary_recall"] == 1.0
    assert score["cascade_recall"] == 0.0


def test_score_pair_by_kind_breakdown():
    gts = [gt(kind="add", did="d1"), gt(kind="remove", did="d2", zone_a="B-1", zone_b=None)]
    pred = [Delta("p1", "add", "note", None, "x", 1, None, "A-3", {}, 1.0, GT_TEMPLATE["description"])]
    score = score_pair(pred, gts)
    assert score["by_kind"]["add"]["tp"] == 1
    assert score["by_kind"]["remove"]["fn"] == 1
