"""Synthesize a plausible P&ID-flavoured sheet, structured like the two real
exemplars (26-KA-901 / 26-KA-902): zone grid, title block, equipment
datasheet, numbered notes with DELETED placeholders, instrument bubbles
with setpoints, composite line tags, valve tags, nozzles, and simple
process geometry. Content is plausible-not-valid engineering.
"""
from __future__ import annotations

import random
from .model import Sheet, Element, LineTag

SERVICES = ["PV", "VF", "WC", "AI", "GI", "DC", "DO", "VA", "PL"]
PIPE_CLASSES = ["GC11S", "FC11S", "AC21S", "AS20S", "GS20S", "EC11S", "GD20S", "AC21", "AS20"]
INSTR_FUNCS = ["PIT", "TIT", "PDIT", "FIT", "LIT", "PSE"]
NOTE_POOL = [
    "HIGH POINT VENT AND LOW POINT DRAIN TO BE FINALIZED BY PIPING AS PER PIPING LAYOUT.",
    "OIL CHANGE BY USING TEMPORARY ARRANGEMENT WITH HOSES.",
    "FLAME ARRESTER INCLUDES FLAME FILTER AND LOCATED AT END OF PIPE.",
    "MANUAL DRAIN PRIOR TO EACH START-UP. PUSH BUTTON WITH PERMISSIVE FOR START-UP SEQUENCE.",
    "VENT ROUTED TO SAFE LOCATION.",
    "SECONDARY SEAL GAS AND SEPARATION GAS.",
    "INSULATION THICKNESS TO BE TAKEN INTO CONSIDERATION ON TW SELECTION.",
    "DIFFERENTIAL PRESSURE BASED ORIFICE TYPE VOLUME FLOW METER.",
    "FROM SEAL GAS SYSTEM RUPTURE DISCS.",
    "MAX BACK-PRESSURE 0.005 BARG.",
    "MAX BACK-PRESSURE 1.5 BARG.",
    "BOTTOM ENTRY FOR SUCTION AND DISCHARGE NOZZLES.",
    "SAFETY CRITICAL HEAT TRACING - HYDRATE MITIGATION (25 DEGC).",
    "LL SET POINT IS OVERRIDEN AT START-UP.",
    "CHEMICAL CLEANING (CIP); BLEED/DRAIN CONNECTION.",
    "UPSTREAM STRAIGHT RUN MIN. 10xD. DOWNSTREAM STRAIGHT RUN MIN. 5xD.",
    "ATMOSPHERIC VENT.",
    "LUBE OIL RESERVOIR VENT. OIL MIST SEPARATOR INSTALLED OFF-SKID.",
    "PRIMARY SEAL GAS IS TAKEN DOWNSTREAM LAST COMPRESSING STAGE. SKID INTERNAL DESIGN.",
    "PROVISION FOR FUTURE HOT GAS BYPASS LINE.",
]


def _instrument(sh: Sheet, rng: random.Random, eid: str, func: str, loop: int,
                system: str, x: float, y: float, with_sd: bool):
    attrs = {"func": func, "loop": loop, "system": system}
    text = f"{func} {loop} {system}"
    if with_sd:
        hh = rng.choice([135, 140, 150, 160, 209, 214, 235, 245])
        ll = rng.choice([50, 75, 110, 120, 125])
        attrs["setpoints"] = {"HH": hh, "LL": ll}
        text += f" SD HH:{hh} LL:{ll}"
    sh.add(Element(eid, "instrument", text, (x, y), attrs=attrs, layer="INSTR"))


def make_sheet(rng: random.Random, sheet_no: int = 1, of: int = 1) -> Sheet:
    sh = Sheet(sheet_no=sheet_no, of=of, rev="A")
    system = "26"
    equip_seq = rng.randint(901, 909)
    equip = f"{system}-KA-{equip_seq}"
    sh.meta["equipment"] = equip
    sh.meta["service"] = rng.choice(
        ["3RD STAGE HP GAS LIFT COMPRESSOR", "3RD STAGE HP GAS EXPORT COMPRESSOR",
         "2ND STAGE MP GAS COMPRESSOR"])

    # zone labels (border grid) --------------------------------------------
    for i, c in enumerate(Sheet.COLS):
        x = (i + 0.5) / 12 * sh.width
        sh.add(Element(f"zc{c}", "zone_label", str(c), (x, 4), layer="BORDER", font_mm=3))
        sh.add(Element(f"zct{c}", "zone_label", str(c), (x, sh.height - 4), layer="BORDER", font_mm=3))
    for j, r in enumerate(Sheet.ROWS):
        y = sh.height - (j + 0.5) / 10 * sh.height
        sh.add(Element(f"zr{r}", "zone_label", r, (4, y), layer="BORDER", font_mm=3))
        sh.add(Element(f"zrr{r}", "zone_label", r, (sh.width - 4, y), layer="BORDER", font_mm=3))

    # title block ----------------------------------------------------------
    tb_x, tb_y = sh.width - 200, 8
    fields = {"tf_drawno": f"0D204-PID-{system}-{equip_seq}-001",
              "tf_title": sh.meta["service"], "tf_equip": equip,
              "tf_rev": "REV A", "tf_sheet": f"SHEET {sheet_no} OF {of}",
              "tf_date": "2026-01-15"}
    for i, (eid, txt) in enumerate(fields.items()):
        sh.add(Element(eid, "title_field", txt, (tb_x, tb_y + 6 * i),
                       attrs={"field": eid[3:]}, layer="TITLE"))
    sh.add(Element("rev_a", "rev_row", "A  2026-01-15  ISSUED FOR DESIGN",
                   (tb_x, tb_y + 40), attrs={"rev": "A"}, layer="TITLE"))

    # datasheet ------------------------------------------------------------
    duty = rng.choice([776, 1120, 1835])
    flow = rng.choice([19057, 40210, 62809])
    ds = {"ds_tag": f"TAG NUMBER {equip}",
          "ds_duty": f"DUTY kW {duty}",
          "ds_flow": f"FLOW RATE kg/h {flow}",
          "ds_dp": "DISCHARGE / SUCTION DESIGN PRESS. (MAX) Barg FV / 286 / FV / 286",
          "ds_vendor": "VENDOR MAN ENERGY SOLUTIONS"}
    for i, (eid, txt) in enumerate(ds.items()):
        sh.add(Element(eid, "datasheet_row", txt, (sh.width - 260, 200 - 6 * i),
                       attrs={"value": txt}, layer="TEXT"))

    # notes block, with DELETED placeholders in real positions -------------
    n_notes = rng.randint(14, 20)
    deleted_idx = set(rng.sample(range(3, n_notes), rng.randint(2, 4)))
    notes = rng.sample(NOTE_POOL, n_notes - len(deleted_idx))
    ni = 0
    for i in range(n_notes):
        eid = f"note{i:02d}"
        if i in deleted_idx:
            role, body = "note_deleted", "DELETED."
        else:
            role, body = "note", notes[ni]; ni += 1
        y = 560 - 7 * i
        el = Element(eid, role, f"{i + 1}. {body}", (12, y),
                     attrs={"note_no": i + 1, "body": body}, layer="NOTES")
        sh.add(el)
        sh.note_order.append(eid)

    # DCN note (revision record note, always last — see exemplar note 34/36)
    dcn = Element(f"note{n_notes:02d}", "dcn_note",
                  f"{n_notes + 1}. THIS P&ID CONTAINS DCN-KP-0273-1.",
                  (12, 560 - 7 * n_notes),
                  attrs={"note_no": n_notes + 1, "dcns": ["DCN-KP-0273-1"]},
                  layer="NOTES")
    sh.add(dcn)
    sh.note_order.append(dcn.eid)

    # instruments with per-family sequence base (renumberable) -------------
    loop_base = rng.choice([9010, 9050])
    for k in range(rng.randint(8, 14)):
        func = rng.choice(INSTR_FUNCS)
        x = rng.uniform(120, sh.width - 300)
        y = rng.uniform(120, sh.height - 80)
        _instrument(sh, rng, f"inst{k:02d}", func, loop_base + k, system, x, y,
                    with_sd=(rng.random() < 0.5))

    # line tags ------------------------------------------------------------
    line_base = rng.choice([9000, 9030])
    for k in range(rng.randint(10, 16)):
        lt = LineTag(size=rng.choice(['2"', '3"', '4"', '6"', '8"', '10"']),
                     service=rng.choice(SERVICES), system=rng.choice(["26", "40", "43", "63"]),
                     seq=str(line_base + k), pipe_class=rng.choice(PIPE_CLASSES),
                     insul=rng.choice(["00", "03", "08", "38"]))
        x = rng.uniform(100, sh.width - 280)
        y = rng.uniform(60, sh.height - 60)
        sh.add(Element(f"line{k:02d}", "line_tag", lt.render(), (x, y),
                       attrs=vars(lt), layer="TEXT"))

    # valve tags + nozzles -------------------------------------------------
    for k in range(rng.randint(8, 12)):
        body = rng.choice(["BL", "GT", "GB", "CB", "CH"])
        sh.add(Element(f"valve{k:02d}", "valve_tag",
                       f"{rng.choice(['26','40','43'])}{body}{9000 + rng.randint(0, 400)}",
                       (rng.uniform(100, sh.width - 280), rng.uniform(60, sh.height - 60)),
                       attrs={"body": body}, layer="TEXT", font_mm=2.0))
    noz_base = 4200
    for k in range(rng.randint(6, 10)):
        sh.add(Element(f"noz{k:02d}", "nozzle", f"N{noz_base + k}",
                       (rng.uniform(150, sh.width - 300), rng.uniform(100, sh.height - 100)),
                       attrs={"noz_no": noz_base + k}, layer="TEXT", font_mm=2.0))

    # sparse geometry (process lines + equipment circle) -------------------
    for k in range(6):
        x0 = rng.uniform(80, sh.width - 350); y0 = rng.uniform(80, sh.height - 120)
        sh.add(Element(f"gline{k}", "geom_line", "",
                       (x0, y0), attrs={"dx": rng.uniform(60, 220), "dy": 0.0},
                       layer="GEOM"))
    sh.add(Element("gequip", "geom_circle", "", (sh.width * 0.45, sh.height * 0.5),
                   attrs={"r": 30.0}, layer="GEOM"))
    sh.add(Element("equip_lbl", "equipment_tag", equip,
                   (sh.width * 0.45, sh.height * 0.5 + 36),
                   attrs={"system": system, "type": "KA", "seq": equip_seq}, layer="TEXT",
                   font_mm=3.5))
    return sh
