"""Integrity checks for the held-out real-P&ID set (eval/datasets/holdout/).

**These tests deliberately assert NOTHING about detection quality.** No
assertion here may check that a particular delta is found, or that any metric
clears a bar. The moment a holdout is used to gate development it stops being
held out and becomes a training signal -- the exact overfitting the set exists
to detect. Quality is measured only by `make eval-holdout`, reported separately
and never pooled with the seeded set.

What IS asserted: the fixture is present, well-formed, and shaped so the
scorer can actually consume it. A holdout that silently stopped being scored
would be worse than no holdout, because the absence looks identical to a pass
(this happened during construction: GT rows the scorer didn't recognize were
parsed fine and then contributed nothing).
"""
import json
import pathlib

import pytest

HOLDOUT = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "holdout"
MANIFEST = HOLDOUT / "manifest.jsonl"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="held-out set not present (see createrealdatapdf/real-pair-sources.md)",
)

# Mirrors eval/run_eval.py::_RASTER_ONLY_OPS. Duplicated on purpose: if that
# tuple is renamed, this test must fail loudly rather than follow along, since
# the rename would silently drop the holdout from the raster measurement.
RASTER_ONLY_OPS = ("ChangeValveSymbol", "RerouteLine")


def _manifest():
    return [json.loads(ln) for ln in MANIFEST.read_text().splitlines() if ln.strip()]


def test_manifest_rows_point_at_real_pairs():
    rows = _manifest()
    assert rows, "holdout manifest is empty"
    for row in rows:
        pair = HOLDOUT / "pairs" / row["pair_id"]
        assert (pair / "a" / "L0.pdf").exists(), f"{row['pair_id']}: missing a/L0.pdf"
        assert (pair / "b" / "L0.pdf").exists(), f"{row['pair_id']}: missing b/L0.pdf"
        assert (pair / "gt" / "deltas.json").exists(), f"{row['pair_id']}: missing GT"
        assert row.get("held_out") is True, f"{row['pair_id']}: not marked held_out"


def test_gt_rows_carry_the_keys_the_scorer_reads():
    """run_eval._gt_row_found reads row["sheet"] and row["kind"]; a GT that
    omits `sheet` raises KeyError mid-run, and one whose `kind` is outside the
    scorer's vocabulary is silently never matched."""
    for row in _manifest():
        gt = json.loads((HOLDOUT / "pairs" / row["pair_id"] / "gt" / "deltas.json").read_text())
        for d in gt:
            assert "sheet" in d, f"{row['pair_id']}/{d.get('did')}: GT row has no 'sheet'"
            assert d["kind"] in ("add", "remove", "modify", "move"), (
                f"{row['pair_id']}/{d['did']}: kind {d['kind']!r} is outside the "
                f"scorer's vocabulary and would never match"
            )


def test_raster_pairs_are_actually_selected_by_the_raster_measurement():
    """eval_raster_recall_pairs selects on manifest `ops` AND on the GT row
    shape (_is_raster_only_gt_row). A pair failing either is skipped rather
    than reported as zero -- indistinguishable from 'nothing to measure'."""
    raster_rows = [r for r in _manifest()
                   if r["kind"] == "edited"
                   and any(op in r.get("ops", []) for op in RASTER_ONLY_OPS)]
    assert raster_rows, "no holdout pair would be picked up by the raster measurement"

    for row in raster_rows:
        gt = json.loads((HOLDOUT / "pairs" / row["pair_id"] / "gt" / "deltas.json").read_text())
        # same predicate as run_eval._is_raster_only_gt_row
        recognized = [
            d for d in gt
            if (d.get("role") == "valve_tag" and "symbol_type" in (d.get("field_changes") or {}))
            or (d.get("role") == "geom_line"
                and {"dx", "dy"} <= set((d.get("field_changes") or {})))
        ]
        assert recognized, (
            f"{row['pair_id']}: no GT row matches _is_raster_only_gt_row, so this "
            f"pair contributes nothing to the raster-recall number it exists for"
        )


def test_null_control_ground_truth_is_empty():
    """The null pair's whole value is that GT is empty by construction: any
    delta is a false positive. A non-empty GT here would quietly turn the
    calibration number into something else."""
    for row in _manifest():
        if row["kind"] != "null_prod":
            continue
        gt = json.loads((HOLDOUT / "pairs" / row["pair_id"] / "gt" / "deltas.json").read_text())
        assert gt == [], f"{row['pair_id']}: null control must have empty GT, got {len(gt)}"


def test_provenance_records_license_and_source():
    """A held-out set is only usable if its redistribution basis is explicit."""
    for row in _manifest():
        prov = json.loads(
            (HOLDOUT / "pairs" / row["pair_id"] / "gt" / "provenance.json").read_text())
        assert prov.get("license"), f"{row['pair_id']}: provenance has no license"
        assert prov.get("source_name"), f"{row['pair_id']}: provenance has no source_name"


def test_committed_pairs_carry_no_third_party_chrome_text():
    """The base page carried a course copyright line, page number and caption.
    The crop must have removed all of it -- what is committed has to be only
    the public-domain figure."""
    fitz = pytest.importorskip("fitz")
    for row in _manifest():
        for side in ("a", "b"):
            path = HOLDOUT / "pairs" / row["pair_id"] / side / "L0.pdf"
            doc = fitz.open(path)
            try:
                text = " ".join(p.get_text("text") for p in doc).lower()
            finally:
                doc.close()
            for banned in ("copyright", "ludwigson", "page 21", "figure 15"):
                assert banned not in text, (
                    f"{row['pair_id']}/{side}: committed PDF still contains {banned!r}"
                )
