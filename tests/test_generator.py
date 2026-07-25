"""Generator guards as pytest — the round-trip property is the load-bearing one."""
import random
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "eval" / "datasets"))
from generator.model import diff_sheets, Sheet
from generator.content import make_sheet
from generator.generate import apply_script, sample_edit_script
from generator.ops import CONTENT_OPS, OpContext

def test_roundtrip_all_ops_individually():
    for op in CONTENT_OPS:
        rng = random.Random(7)
        a = make_sheet(rng)
        b = a.clone()
        res = op(OpContext(rng), a, b)
        if not res.applied:
            continue
        ref = {(d.kind, d.eid_a, d.eid_b) for d in diff_sheets(a, b)}
        emit = {(d.kind, d.eid_a, d.eid_b) for d in res.deltas}
        assert ref == emit, f"{res.op_name}: ref^emit={ref ^ emit}"

def test_sampled_scripts_roundtrip():
    for seed in range(5):
        rng = random.Random(seed)
        a = make_sheet(rng)
        b, deltas, _ = apply_script(rng, a, sample_edit_script(rng))
        ref = {(d.kind, d.eid_a, d.eid_b) for d in diff_sheets(a, b)}
        emit = {(d.kind, d.eid_a, d.eid_b) for d in deltas}
        assert ref == emit

def test_model_json_roundtrip():
    a = make_sheet(random.Random(3))
    assert Sheet.from_json(a.to_json()).to_json() == a.to_json()

def test_cascades_reference_primary():
    rng = random.Random(11)
    a = make_sheet(rng)
    b, deltas, _ = apply_script(rng, a, sample_edit_script(rng))
    dids = {d.did for d in deltas}
    for d in deltas:
        if d.is_cascade and d.primary_did is not None:
            assert d.primary_did in dids
