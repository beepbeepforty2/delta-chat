"""Edit operators. Each applies to the Sheet MODEL and emits GTDeltas,
including cascades (tagged is_cascade + primary_did) and semantic nulls.

Every operator here is grounded in an edit observed in the real pair
(26-KA-901 vs 26-KA-902) or a standard revision practice:
  InsertNoteWithCascade    -> B's note 27 insertion, +1 then -1 offsets
  DeleteNoteKeepPlaceholder-> "15. DELETED." practice
  CollapseDeletedRange     -> "6-7, 12. DELETED." (semantic null)
  SystematicTagRenumber    -> instrument -39 / nozzle -1000 family offsets
  ChangePipeClass          -> GC11S -> FC11S on otherwise-identical tag
  ChangeSetpoint           -> HH 245/LL 120 -> 214/110
  RewordNoteEquivalent     -> note 13 rewording (semantic null)
  ChangeNoteSubstantive    -> note 18 control-philosophy change
  ChangeDatasheetValue     -> duty/flow row changes
  AddDCN + BumpRevision    -> every real revision does both
  MoveAnnotation           -> position-only change
  Reexport                 -> producer-variation null (handled at render)
  ChangeValveSymbol        -> valve glyph swap, tag text unchanged (graphical-
                              only edit; exercises src/delta/raster_diff.py +
                              raster_join.py, invisible to the text pipeline)
  RerouteLine              -> geom_line endpoint moved to a different nozzle
                              (pure geometry, no text element involved at all)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from .model import Sheet, Element, GTDelta, LineTag

EQUIV_REWORDS = {
    "UPSTREAM STRAIGHT RUN MIN. 10xD. DOWNSTREAM STRAIGHT RUN MIN. 5xD.":
        "UPSTREAM STRAIGHT PIPE RUN MIN. 10xD, DOWNSTREAM MIN 5xD.",
    "ATMOSPHERIC VENT.": "VENT TO ATMOSPHERE.",
    "VENT ROUTED TO SAFE LOCATION.": "VENT TO BE ROUTED TO SAFE LOCATION.",
    "MANUAL DRAIN PRIOR TO EACH START-UP. PUSH BUTTON WITH PERMISSIVE FOR START-UP SEQUENCE.":
        "MANUAL DRAIN PRIOR TO START-UP. PUSH BUTTON WITH PERMISSIVE TO START COMPRESSOR.",
    "LUBE OIL RESERVOIR VENT. OIL MIST SEPARATOR INSTALLED OFF-SKID.":
        "LUBE OIL RESERVOIR VENT. OIL MIST SEPERATOR INSTALLED OFF-SKID.",  # real typo pair
}
SUBSTANTIVE_REWRITES = {
    "MAX BACK-PRESSURE 0.005 BARG.": "MAX BACK-PRESSURE 0.05 BARG.",
    "SAFETY CRITICAL HEAT TRACING - HYDRATE MITIGATION (25 DEGC).":
        "SAFETY CRITICAL HEAT TRACING - HYDRATE MITIGATION (30 DEGC).",
    "DIFFERENTIAL PRESSURE BASED ORIFICE TYPE VOLUME FLOW METER.":
        "ULTRASONIC TYPE VOLUME FLOW METER.",
}


class OpContext:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self._n = 0

    def did(self) -> str:
        self._n += 1
        return f"d{self._n:04d}"


@dataclass
class OpResult:
    deltas: list[GTDelta] = field(default_factory=list)
    applied: bool = False
    op_name: str = ""


def _mod_delta(ctx, a: Sheet, b: Sheet, eid, fc, desc, cascade=False, primary=None, null=False):
    ea, eb = a.elements[eid], b.elements[eid]
    return GTDelta(ctx.did(), "modify", ea.role, eid, eid, a.sheet_no,
                   ea.zone(a), eb.zone(b), fc, desc,
                   is_cascade=cascade, primary_did=primary, semantic_null=null)


# ---------------------------------------------------------------------------
def _emit_note_cascade(ctx, a: Sheet, b: Sheet, primary_did: str) -> list[GTDelta]:
    changed = b.renumber_notes()
    out = []
    for eid in changed:
        if eid not in a.elements:
            continue
        old_no = a.elements[eid].attrs["note_no"]
        new_no = b.elements[eid].attrs["note_no"]
        out.append(_mod_delta(ctx, a, b, eid, {"note_no": [old_no, new_no]},
                              f"note renumbered {old_no} -> {new_no} (cascade)",
                              cascade=True, primary=primary_did))
        # references to this note elsewhere on the sheet would shift too;
        # note refs are held in Element.refs and rendered as "NOTE n"
        for oid, oel in b.elements.items():
            if eid in oel.refs and f"NOTE {old_no}" in oel.text:
                oel.text = oel.text.replace(f"NOTE {old_no}", f"NOTE {new_no}")
                out.append(_mod_delta(ctx, a, b, oid, {"note_ref": [old_no, new_no]},
                                      f"note reference updated {old_no}->{new_no} (cascade)",
                                      cascade=True, primary=primary_did))
    return out


def insert_note_with_cascade(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    r = OpResult(op_name="InsertNoteWithCascade")
    pos = ctx.rng.randint(2, max(3, len(b.note_order) - 3))
    eid = f"note_ins_{ctx.rng.randint(100, 999)}"
    y_ref = b.elements[b.note_order[pos]].anchor[1]
    body = ctx.rng.choice([
        "FOR COMPRESSOR, GEAR AND MOTOR VIBRATION MONITORING REFER VENDOR DRAWING.",
        "STRAINER TO BE REMOVED AFTER COMMISSIONING.",
        "TIE-IN POINT TO BE VERIFIED AT SITE.",
    ])
    el = Element(eid, "note", f"0. {body}", (12, y_ref + 3.5),
                 attrs={"note_no": 0, "body": body}, layer="NOTES")
    b.add(el)
    b.note_order.insert(pos, eid)
    did = ctx.did()
    r.deltas.append(GTDelta(did, "add", "note", None, eid, b.sheet_no, None,
                            el.zone(b), {}, f"note inserted at position {pos + 1}: {body[:50]}"))
    r.deltas += _emit_note_cascade(ctx, a, b, did)
    r.applied = True
    return r


def delete_note_keep_placeholder(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    r = OpResult(op_name="DeleteNoteKeepPlaceholder")
    cands = [e for e in b.note_order if b.elements[e].role == "note"]
    if not cands:
        return r
    eid = ctx.rng.choice(cands)
    el = b.elements[eid]
    old_body = el.attrs["body"]
    el.role = "note_deleted"
    el.attrs["body"] = "DELETED."
    el.text = f"{el.attrs['note_no']}. DELETED."
    r.deltas.append(_mod_delta(ctx, a, b, eid, {"body": [old_body, "DELETED."]},
                               f"note {el.attrs['note_no']} deleted (placeholder retained): {old_body[:40]}"))
    r.applied = True
    return r


def collapse_deleted_range(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    """Merge >=2 consecutive DELETED notes into one range element.
    Semantic null: the notes were already deleted in A."""
    r = OpResult(op_name="CollapseDeletedRange")
    runs, cur = [], []
    for eid in b.note_order:
        if b.elements[eid].role == "note_deleted":
            cur.append(eid)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    if not runs:
        return r
    run = ctx.rng.choice(runs)
    nos = [b.elements[e].attrs["note_no"] for e in run]
    keep, rest = run[0], run[1:]
    kel = b.elements[keep]
    kel.text = f"{nos[0]}-{nos[-1]}. DELETED."
    kel.attrs["range"] = [nos[0], nos[-1]]
    did = ctx.did()
    r.deltas.append(GTDelta(did, "modify", "note_deleted", keep, keep, b.sheet_no,
                            a.elements[keep].zone(a), kel.zone(b),
                            {"collapsed": [None, nos]},
                            f"DELETED notes {nos} collapsed into range (no semantic change)",
                            semantic_null=True))
    for eid in rest:
        el = a.elements[eid]
        r.deltas.append(GTDelta(ctx.did(), "remove", "note_deleted", eid, None,
                                a.sheet_no, el.zone(a), None, {},
                                f"placeholder '{el.text}' merged into range",
                                is_cascade=True, primary_did=did, semantic_null=True))
        del b.elements[eid]
        b.note_order.remove(eid)
    r.applied = True
    return r


def systematic_tag_renumber(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    """Constant offset applied to a whole tag family (real: instruments -39,
    nozzles -1000). One primary + N cascades, so scorers can reward grouping."""
    r = OpResult(op_name="SystematicTagRenumber")
    family = ctx.rng.choice(["instrument", "nozzle"])
    offset = ctx.rng.choice([-39, -28, +11, -1000] if family == "nozzle" else [-39, -28, +11])
    members = [e for e in b.elements.values() if e.role == family]
    if len(members) < 3:
        return r
    primary = ctx.did()
    first = True
    for el in sorted(members, key=lambda e: e.eid):
        if family == "instrument":
            old, new = el.attrs["loop"], el.attrs["loop"] + offset
            el.attrs["loop"] = new
        else:
            old, new = el.attrs["noz_no"], el.attrs["noz_no"] + offset
            el.attrs["noz_no"] = new
        el.text = el.text.replace(str(old), str(new))
        d = _mod_delta(ctx, a, b, el.eid,
                       {"loop" if family == "instrument" else "noz_no": [old, new]},
                       f"{family} renumbered {old} -> {new} (family offset {offset:+d})",
                       cascade=not first, primary=None if first else primary)
        if first:
            d.did = primary
            first = False
        r.deltas.append(d)
    r.applied = True
    return r


def change_pipe_class(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    r = OpResult(op_name="ChangePipeClass")
    lines = [e for e in b.elements.values() if e.role == "line_tag"]
    if not lines:
        return r
    el = ctx.rng.choice(lines)
    old = el.attrs["pipe_class"]
    new = ctx.rng.choice([c for c in ["GC11S", "FC11S", "AC21S", "EC11S"] if c != old])
    el.attrs["pipe_class"] = new
    el.text = LineTag(**el.attrs).render()
    r.deltas.append(_mod_delta(ctx, a, b, el.eid, {"pipe_class": [old, new]},
                               f"pipe class {old} -> {new} on {el.text} (material spec change)"))
    r.applied = True
    return r


def change_setpoint(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    r = OpResult(op_name="ChangeSetpoint")
    cands = [e for e in b.elements.values()
             if e.role == "instrument" and "setpoints" in e.attrs]
    if not cands:
        return r
    el = ctx.rng.choice(cands)
    sp = dict(el.attrs["setpoints"])
    key = ctx.rng.choice(list(sp))
    old = sp[key]
    new = old + ctx.rng.choice([-31, -10, +5, +14])
    sp[key] = new
    el.attrs["setpoints"] = sp
    el.text = el.text.replace(f"{key}:{old}", f"{key}:{new}")
    r.deltas.append(_mod_delta(ctx, a, b, el.eid, {f"setpoint_{key}": [old, new]},
                               f"{el.attrs['func']}-{el.attrs['loop']} {key} setpoint {old} -> {new}"))
    r.applied = True
    return r


def reword_note_equivalent(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    r = OpResult(op_name="RewordNoteEquivalent")
    cands = [e for e in b.note_order
             if b.elements[e].attrs.get("body") in EQUIV_REWORDS]
    if not cands:
        return r
    eid = ctx.rng.choice(cands)
    el = b.elements[eid]
    old = el.attrs["body"]
    el.attrs["body"] = EQUIV_REWORDS[old]
    el.text = f"{el.attrs['note_no']}. {el.attrs['body']}"
    r.deltas.append(_mod_delta(ctx, a, b, eid, {"body": [old, el.attrs["body"]]},
                               "note reworded, meaning unchanged", null=True))
    r.applied = True
    return r


def change_note_substantive(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    r = OpResult(op_name="ChangeNoteSubstantive")
    cands = [e for e in b.note_order
             if b.elements[e].attrs.get("body") in SUBSTANTIVE_REWRITES]
    if not cands:
        return r
    eid = ctx.rng.choice(cands)
    el = b.elements[eid]
    old = el.attrs["body"]
    el.attrs["body"] = SUBSTANTIVE_REWRITES[old]
    el.text = f"{el.attrs['note_no']}. {el.attrs['body']}"
    r.deltas.append(_mod_delta(ctx, a, b, eid, {"body": [old, el.attrs["body"]]},
                               "note revised — substantive engineering change"))
    r.applied = True
    return r


def change_datasheet_value(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    r = OpResult(op_name="ChangeDatasheetValue")
    eid = ctx.rng.choice(["ds_duty", "ds_flow"])
    el = b.elements[eid]
    m = re.search(r"(\d+)$", el.text)
    old = int(m.group(1))
    new = int(old * ctx.rng.choice([0.85, 1.1, 1.25]))
    el.text = el.text[: m.start()] + str(new)
    el.attrs["value"] = el.text
    r.deltas.append(_mod_delta(ctx, a, b, eid, {"value": [old, new]},
                               f"datasheet {eid[3:]} {old} -> {new}"))
    r.applied = True
    return r


def move_annotation(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    r = OpResult(op_name="MoveAnnotation")
    cands = [e for e in b.elements.values() if e.role in ("valve_tag", "line_tag", "nozzle")]
    if not cands:
        return r
    el = ctx.rng.choice(cands)
    x, y = el.anchor
    el.anchor = (min(b.width - 60, max(20, x + ctx.rng.uniform(-80, 80))),
                 min(b.height - 20, max(20, y + ctx.rng.uniform(-60, 60))))
    ea = a.elements[el.eid]
    r.deltas.append(GTDelta(ctx.did(), "move", el.role, el.eid, el.eid, a.sheet_no,
                            ea.zone(a), el.zone(b), {},
                            f"'{el.text}' relocated {ea.zone(a)} -> {el.zone(b)} (content unchanged)"))
    r.applied = True
    return r


def add_instrument(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    r = OpResult(op_name="AddInstrument")
    eid = f"inst_new_{ctx.rng.randint(100, 999)}"
    loop = 9700 + ctx.rng.randint(0, 99)
    hh = ctx.rng.choice([140, 160, 245])
    el = Element(eid, "instrument", f"PIT {loop} 26 SD HH:{hh} LL:75",
                 (ctx.rng.uniform(150, b.width - 320), ctx.rng.uniform(150, b.height - 100)),
                 attrs={"func": "PIT", "loop": loop, "system": "26",
                        "setpoints": {"HH": hh, "LL": 75}}, layer="INSTR")
    b.add(el)
    r.deltas.append(GTDelta(ctx.did(), "add", "instrument", None, eid, b.sheet_no,
                            None, el.zone(b), {}, f"instrument PIT-{loop} added"))
    r.applied = True
    return r


def remove_valve(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    r = OpResult(op_name="RemoveValve")
    cands = [e for e in b.elements.values() if e.role == "valve_tag"]
    if not cands:
        return r
    el = ctx.rng.choice(cands)
    ea = a.elements[el.eid]
    r.deltas.append(GTDelta(ctx.did(), "remove", "valve_tag", el.eid, None,
                            a.sheet_no, ea.zone(a), None, {}, f"valve {el.text} removed"))
    del b.elements[el.eid]
    r.applied = True
    return r


def change_valve_symbol(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    """Swaps a valve's drawn glyph (gate <-> globe) WITHOUT touching its
    tag text -- the case the symbolic (text-only) pipeline is
    structurally blind to by construction, since CanonicalElement.content
    never changes. Exists specifically to exercise src/delta/raster_diff.py
    + raster_join.py. attrs["symbol_type"] is a new attribute, deliberately
    NOT attrs["body"] (which IS baked into the rendered tag text -- see
    content.py -- so changing it would defeat the whole point)."""
    r = OpResult(op_name="ChangeValveSymbol")
    cands = [e for e in b.elements.values() if e.role == "valve_tag"]
    if not cands:
        return r
    el = ctx.rng.choice(cands)
    old = el.attrs.get("symbol_type", "gate")
    new = "globe" if old == "gate" else "gate"
    el.attrs["symbol_type"] = new
    r.deltas.append(_mod_delta(ctx, a, b, el.eid, {"symbol_type": [old, new]},
                                f"valve symbol changed {old} -> {new} on {el.text} (tag unchanged)"))
    r.applied = True
    return r


def reroute_line(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    """Moves a geom_line's endpoint to a different nozzle -- a purely
    graphical/geometric edit with no text-bearing element involved at
    all, exercising raster_join.py's "geometry" classification (as
    opposed to change_valve_symbol's "graphical" one)."""
    r = OpResult(op_name="RerouteLine")
    lines = [e for e in b.elements.values() if e.role == "geom_line"]
    nozzles = [e for e in b.elements.values() if e.role == "nozzle"]
    if not lines or not nozzles:
        return r
    el = ctx.rng.choice(lines)
    noz = ctx.rng.choice(nozzles)
    old_dx, old_dy = el.attrs["dx"], el.attrs["dy"]
    new_dx, new_dy = noz.anchor[0] - el.anchor[0], noz.anchor[1] - el.anchor[1]
    el.attrs["dx"], el.attrs["dy"] = new_dx, new_dy
    r.deltas.append(_mod_delta(ctx, a, b, el.eid,
                                {"dx": [old_dx, new_dx], "dy": [old_dy, new_dy]},
                                f"line rerouted to nozzle {noz.text}"))
    r.applied = True
    return r


def bump_revision_and_dcn(ctx: OpContext, a: Sheet, b: Sheet) -> OpResult:
    """Always applied last. Real change, never the interesting one."""
    r = OpResult(op_name="BumpRevisionAndDCN")
    old_rev, new_rev = b.rev, chr(ord(b.rev) + 1)
    b.rev = new_rev
    tf = b.elements["tf_rev"]
    tf.text = f"REV {new_rev}"
    r.deltas.append(_mod_delta(ctx, a, b, "tf_rev", {"rev": [old_rev, new_rev]},
                               f"revision {old_rev} -> {new_rev}"))
    row = Element(f"rev_{new_rev.lower()}", "rev_row",
                  f"{new_rev}  2026-06-30  REVISED AS PER DCN",
                  (b.elements["rev_a"].anchor[0], b.elements["rev_a"].anchor[1] + 5),
                  attrs={"rev": new_rev}, layer="TITLE")
    b.add(row)
    r.deltas.append(GTDelta(ctx.did(), "add", "rev_row", None, row.eid, b.sheet_no,
                            None, row.zone(b), {}, f"revision table row {new_rev} added"))
    dcn_eids = [e for e in b.note_order if b.elements[e].role == "dcn_note"]
    if dcn_eids:
        el = b.elements[dcn_eids[-1]]
        new_dcn = f"DCN-KP-{ctx.rng.randint(800, 1300):04d}-1"
        el.attrs["dcns"] = el.attrs["dcns"] + [new_dcn]
        el.text = el.text.rstrip(".") + f"/{new_dcn.split('-', 2)[2]}."
        r.deltas.append(_mod_delta(ctx, a, b, el.eid,
                                   {"dcns": [el.attrs["dcns"][:-1], el.attrs["dcns"]]},
                                   f"DCN record appended: {new_dcn}"))
    r.applied = True
    return r


CONTENT_OPS = [
    insert_note_with_cascade, delete_note_keep_placeholder, collapse_deleted_range,
    systematic_tag_renumber, change_pipe_class, change_setpoint,
    reword_note_equivalent, change_note_substantive, change_datasheet_value,
    move_annotation, add_instrument, remove_valve,
    change_valve_symbol, reroute_line,
]
