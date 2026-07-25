"""Generate labeled document-revision pairs.

Usage:
    python -m generator.generate --out eval/datasets/v0 --n 8 --seed 42

Emits per pair:
    a/model.json, a/sheet.dxf, a/L0.pdf .. a/L3.pdf   (same for b/)
    gt/elements_a.json  gt/elements_b.json            (L1: inventory + anchors + zones)
    gt/correspondence.json                            (L2: eid map incl. unmatched)
    gt/deltas.json                                    (L3: typed deltas, cascades, nulls)
    gt/render_transforms.json
    qa.jsonl                                          (templated Q&A with citation targets)
Top level: manifest.jsonl (pair_id, seed, kind, ops, hashes, canary score).

Pair kinds and mixture:
    edited      revision pair via sampled operator script (default bulk)
    null_ident  byte-identical re-render                (GT: empty delta)
    null_prod   same model, two PDF producers           (GT: empty delta)
    null_reword only semantic-null ops applied          (GT: only semantic_null deltas)
    not_a_pair  two different sheets entirely           (GT: refuse-to-diff)

Locality: edits are drawn toward 1-3 'hot' zones rather than uniformly —
clustered edits destroy local alignment context, matching real revisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .model import Sheet, diff_sheets
from .content import make_sheet
from .ops import CONTENT_OPS, OpContext, bump_revision_and_dcn
from .render import render_pdf, render_dxf, degrade

GENERATOR_VERSION = "0.1.0"


# ---- edit-script sampling -------------------------------------------------

def sample_edit_script(rng: random.Random):
    # heavy-tailed edit count; floor of 2 so every revision exercises
    # alignment under multiple concurrent edits (canary showed 1-op
    # revisions are trivially recovered by string set-difference)
    n = max(2, min(8, 1 + int(rng.paretovariate(1.1))))
    ops = [rng.choice(CONTENT_OPS) for _ in range(n)]
    return ops


def apply_script(rng: random.Random, a: Sheet, ops) -> tuple[Sheet, list, list[str]]:
    b = a.clone()
    ctx = OpContext(rng)
    deltas, applied = [], []
    for op in ops:
        res = op(ctx, a, b)
        if res.applied:
            deltas += res.deltas
            applied.append(res.op_name)
    res = bump_revision_and_dcn(ctx, a, b)
    deltas += res.deltas
    applied.append(res.op_name)
    return b, coalesce(deltas, a, b), applied


def coalesce(deltas, a: Sheet, b: Sheet):
    """Composed ops can cancel (insert renumbers notes down, a later collapse
    shifts them back). GT must describe A->B NET change: drop emitted deltas
    for elements that ended up unchanged, keep one delta per (kind, eids) with
    field endpoints recomputed from the final models, preserving operator
    knowledge (cascade/primary/semantic_null) from the last emission."""
    ref = diff_sheets(a, b)
    by_key = {}
    for d in deltas:                       # last emission wins for tags
        by_key[(d.kind, d.eid_a, d.eid_b)] = d
    out = []
    for r in ref:
        k = (r.kind, r.eid_a, r.eid_b)
        d = by_key.get(k)
        if d is None:                      # ref-only should not happen; keep ref
            out.append(r)
            continue
        if r.kind == "modify":             # recompute endpoints A->B
            d.field_changes = r.field_changes
        d.zone_a, d.zone_b = r.zone_a, r.zone_b
        out.append(d)
    return out


# ---- validation guards ----------------------------------------------------

def validate_roundtrip(a: Sheet, b: Sheet, deltas) -> None:
    """Guard 1: reference model-differ must recover the same changed-eid set
    (cascade/primary/null tags are operator knowledge, set equality is the check)."""
    ref = diff_sheets(a, b)
    ref_set = {(d.kind, d.eid_a, d.eid_b) for d in ref}
    emit_set = {(d.kind, d.eid_a, d.eid_b) for d in deltas}
    missing = ref_set - emit_set
    extra = emit_set - ref_set
    assert not missing and not extra, f"GT emission mismatch: missing={missing} extra={extra}"


def validate_render_fidelity(sheet: Sheet, pdf_path: str) -> None:
    """Guard 2: every model text element must appear in the L0 text layer."""
    import fitz
    doc = fitz.open(pdf_path)
    page_text = doc[0].get_text()
    missing = [el.eid for el in sheet.elements.values()
               if el.text and el.text.split(" ")[0] not in page_text]
    assert not missing, f"render dropped elements: {missing[:5]}"


def canary_score(a: Sheet, b: Sheet, deltas) -> float:
    """Guard 3: dumb baseline = exact set-difference of rendered strings.
    Returns its recall on non-null GT. Too high => set too easy; ~0 => broken."""
    ta = {e.text for e in a.elements.values() if e.text}
    tb = {e.text for e in b.elements.values() if e.text}
    changed_strings = (ta - tb) | (tb - ta)
    real = [d for d in deltas if not d.semantic_null]
    if not real:
        return 1.0
    hit = 0
    for d in real:
        sa = a.elements[d.eid_a].text if d.eid_a and d.eid_a in a.elements else None
        sb = b.elements[d.eid_b].text if d.eid_b and d.eid_b in b.elements else None
        if (sa and sa in changed_strings) or (sb and sb in changed_strings):
            hit += 1
    return hit / len(real)


# ---- QA templating --------------------------------------------------------

def make_qa(a: Sheet, b: Sheet, deltas, kind: str) -> list[dict]:
    qa = []
    real = [d for d in deltas if not d.is_cascade and not d.semantic_null]
    if kind == "not_a_pair":
        return [{"q": "What changed between these two documents?",
                 "expected_behavior": "refuse",
                 "a": "These are different documents, not two revisions of one document.",
                 "citations": ["tf_equip"]}]
    qa.append({"q": "What changed on sheet 1?",
               "expected_behavior": "answer",
               "a": f"{len(real)} primary change(s): " + "; ".join(d.description for d in real[:6]),
               "citations": sorted({d.did for d in real})})
    zones = sorted({d.zone_b or d.zone_a for d in real if (d.zone_b or d.zone_a)})
    if zones:
        z = zones[0]
        in_z = [d for d in real if (d.zone_b or d.zone_a) == z]
        qa.append({"q": f"Did anything change in zone {z}?",
                   "expected_behavior": "answer",
                   "a": "Yes: " + "; ".join(d.description for d in in_z),
                   "citations": [d.did for d in in_z]})
    sp = [d for d in real if any(k.startswith("setpoint") for k in d.field_changes)]
    qa.append({"q": "Did any alarm or trip setpoints change?",
               "expected_behavior": "answer",
               "a": ("Yes: " + "; ".join(d.description for d in sp)) if sp
                    else "No setpoint changes between these revisions.",
               "citations": [d.did for d in sp]})
    # unanswerable probe (content question about a sheet that doesn't exist)
    qa.append({"q": "What changed on sheet 7?",
               "expected_behavior": "refuse",
               "a": "There is no sheet 7; the document has 1 sheet.",
               "citations": []})
    # content question over rev B
    inst = next((e for e in b.elements.values()
                 if e.role == "instrument" and "setpoints" in e.attrs), None)
    if inst:
        qa.append({"q": f"What is the HH setpoint of {inst.attrs['func']}-{inst.attrs['loop']} in the revised document?",
                   "expected_behavior": "answer",
                   "a": str(inst.attrs["setpoints"]["HH"]),
                   "citations": [inst.eid]})
    return qa


# ---- pair emission --------------------------------------------------------

def emit_pair(out: Path, pair_id: str, kind: str, seed: int) -> dict:
    rng = random.Random(seed)
    pd = out / "pairs" / pair_id
    (pd / "gt").mkdir(parents=True, exist_ok=True)
    a = make_sheet(rng)
    ops_applied: list[str] = []
    transforms = {"a": {}, "b": {}}

    if kind == "edited":
        b, deltas, ops_applied = apply_script(rng, a, sample_edit_script(rng))
        validate_roundtrip(a, b, deltas)
    elif kind == "null_ident":
        b, deltas = a.clone(), []
    elif kind == "null_prod":
        b, deltas = a.clone(), []
    elif kind == "null_reword":
        from .ops import reword_note_equivalent, collapse_deleted_range
        ctx = OpContext(rng)
        b = a.clone()
        deltas = []
        for op in (reword_note_equivalent, collapse_deleted_range):
            r = op(ctx, a, b)
            if r.applied:
                deltas += r.deltas
                ops_applied.append(r.op_name)
        validate_roundtrip(a, b, deltas)
    elif kind == "not_a_pair":
        b = make_sheet(random.Random(seed + 10_000))  # sibling drawing, different equipment
        deltas = []
    else:
        raise ValueError(kind)

    for name, sh in (("a", a), ("b", b)):
        d = pd / name
        d.mkdir(exist_ok=True)
        (d / "model.json").write_text(sh.to_json())
        producer = "alt" if (kind == "null_prod" and name == "b") else "standard"
        render_pdf(sh, str(d / "L0.pdf"), producer=producer)
        render_dxf(sh, str(d / "sheet.dxf"))
        validate_render_fidelity(sh, str(d / "L0.pdf"))
        for lvl in (1, 2, 3):
            transforms[name][f"L{lvl}"] = degrade(str(d / "L0.pdf"),
                                                  str(d / f"L{lvl}.pdf"), lvl, seed + lvl)

    # GT emission
    for name, sh in (("a", a), ("b", b)):
        inv = [{"eid": e.eid, "role": e.role, "text": e.text,
                "anchor_mm": e.anchor, "zone": e.zone(sh), "layer": e.layer,
                "attrs": e.attrs}
               for e in sorted(sh.elements.values(), key=lambda x: x.eid)]
        (pd / "gt" / f"elements_{name}.json").write_text(json.dumps(inv, indent=1))
    shared = sorted(set(a.elements) & set(b.elements))
    corr = {"matched": [[e, e] for e in shared],
            "only_a": sorted(set(a.elements) - set(b.elements)),
            "only_b": sorted(set(b.elements) - set(a.elements)),
            "not_a_pair": kind == "not_a_pair"}
    (pd / "gt" / "correspondence.json").write_text(json.dumps(corr, indent=1))
    (pd / "gt" / "deltas.json").write_text(
        json.dumps([d.to_dict() for d in deltas], indent=1))
    (pd / "gt" / "render_transforms.json").write_text(json.dumps(transforms, indent=1))
    qa = make_qa(a, b, deltas, kind)
    (pd / "qa.jsonl").write_text("\n".join(json.dumps(q) for q in qa))

    cscore = canary_score(a, b, deltas) if kind in ("edited", "null_reword") else None
    return {"pair_id": pair_id, "kind": kind, "seed": seed,
            "ops": ops_applied,
            "n_deltas": len(deltas),
            "n_primary": sum(1 for d in deltas if not d.is_cascade),
            "n_cascade": sum(1 for d in deltas if d.is_cascade),
            "n_semantic_null": sum(1 for d in deltas if d.semantic_null),
            "canary_recall": cscore,
            "generator_version": GENERATOR_VERSION}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/datasets/v0")
    ap.add_argument("--n", type=int, default=8, help="number of edited pairs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gen_hash = hashlib.sha256(
        b"".join(sorted(p.read_bytes() for p in Path(__file__).parent.glob("*.py")))
    ).hexdigest()[:12]

    plan = [("edited", i) for i in range(args.n)]
    plan += [("null_ident", 900), ("null_prod", 901),
             ("null_reword", 902), ("not_a_pair", 903)]
    rows = []
    for kind, i in plan:
        pair_id = f"{kind}_{i:03d}"
        row = emit_pair(out, pair_id, kind, args.seed + i)
        row["generator_hash"] = gen_hash
        rows.append(row)
        print(f"  {pair_id:20s} deltas={row['n_deltas']:3d} "
              f"(primary {row['n_primary']}, cascade {row['n_cascade']}, "
              f"null {row['n_semantic_null']}) canary={row['canary_recall']}")
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(f"\nwrote {len(rows)} pairs -> {out}/  (generator {GENERATOR_VERSION} @ {gen_hash})")


if __name__ == "__main__":
    main()
