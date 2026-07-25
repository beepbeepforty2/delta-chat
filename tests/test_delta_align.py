"""Alignment tests against the eval dataset's ground truth.

Feeds match_elements() the GT L1 gold elements directly (no pdf_native
extraction in the loop) so this isolates alignment accuracy from
extraction accuracy -- the layered-GT testing strategy from the original
design brainstorm. Every pair here already stresses the DELETED-placeholder
ambiguity (make_sheet() always seeds 2-4 identical "DELETED." notes,
distinguished only by position), which is exactly the case a raw
content-only differ gets wrong and the spatial cost term is meant to fix.
"""
import pytest

from src.delta.align import match_elements
from src.delta.register import Transform

from tests._gt_helpers import PAIRS_DIR, EDITED_PAIRS, gt_sheet, gt_correspondence


@pytest.mark.parametrize("pair_id", EDITED_PAIRS)
def test_alignment_matches_gt_correspondence(pair_id):
    pair_dir = PAIRS_DIR / pair_id
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")

    sheet_a = gt_sheet(pair_dir, "a")
    sheet_b = gt_sheet(pair_dir, "b")
    corr = gt_correspondence(pair_dir)

    gt_matched = {(a, b) for a, b in corr["matched"]}
    gt_only_a = set(corr["only_a"])
    gt_only_b = set(corr["only_b"])

    result = match_elements(sheet_a, sheet_b, Transform())
    pred_matched = {(m.a.id, m.b.id) for m in result if m.a and m.b}
    pred_only_a = {m.a.id for m in result if m.a and not m.b}
    pred_only_b = {m.b.id for m in result if m.b and not m.a}

    correct = len(pred_matched & gt_matched)
    precision = correct / len(pred_matched) if pred_matched else 1.0
    recall = correct / len(gt_matched) if gt_matched else 1.0

    assert precision >= 0.95, f"{pair_id}: precision {precision:.2f}, wrong matches: {pred_matched - gt_matched}"
    assert recall >= 0.95, f"{pair_id}: recall {recall:.2f}, missed matches: {gt_matched - pred_matched}"
    assert pred_only_a == gt_only_a, f"{pair_id}: only_a mismatch: pred={pred_only_a} gt={gt_only_a}"
    assert pred_only_b == gt_only_b, f"{pair_id}: only_b mismatch: pred={pred_only_b} gt={gt_only_b}"


def test_null_ident_pair_matches_everything():
    pair_dir = PAIRS_DIR / "null_ident_900"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")
    sheet_a = gt_sheet(pair_dir, "a")
    sheet_b = gt_sheet(pair_dir, "b")
    result = match_elements(sheet_a, sheet_b, Transform())
    assert all(m.a and m.b for m in result)
    assert all(m.a.id == m.b.id for m in result)
