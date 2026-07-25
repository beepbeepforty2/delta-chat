"""Semantic sheet model: the ground-truth substrate.

The generator edits THIS model, not the rendered artifact. DXF/PDF are
renderings of it. Every element has a stable eid that survives edits, which
is what makes exact correspondence labels (L2) possible.

Design notes (from the two real P&IDs, 26-KA-901 / 26-KA-902):
- Notes block with "DELETED." placeholders is standard practice.
- Tags are composite keys (size-service-system-seq-class-insul), not strings.
- Border zone grid (A..J x 1..12) is the domain-native location primitive.
- Instrument = one semantic element with attributes (bubble + setpoints),
  even though it renders as several text clusters.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

Role = Literal[
    "note", "note_deleted", "title_field", "rev_row", "zone_label",
    "instrument", "line_tag", "valve_tag", "nozzle", "equipment_tag",
    "datasheet_row", "geom_line", "geom_circle", "dcn_note",
]

CHANGE_KINDS = ("add", "remove", "modify", "move")


@dataclass
class LineTag:
    """Composite pipe line identifier, e.g. 4"-PV-26-9048-GC11S-38."""
    size: str          # 4"
    service: str       # PV
    system: str        # 26
    seq: str           # 9048
    pipe_class: str    # GC11S
    insul: str         # 38 or 00

    def render(self) -> str:
        return f'{self.size}-{self.service}-{self.system}-{self.seq}-{self.pipe_class}-{self.insul}'

    @classmethod
    def parse(cls, s: str) -> "LineTag":
        m = re.match(r'([\d/"x ]+)-([A-Z]{2})-(\d{2})-(\d{4})-([A-Z0-9]+)-(\d{2})', s)
        if not m:
            raise ValueError(f"unparseable line tag: {s}")
        return cls(*m.groups())


@dataclass
class Element:
    eid: str
    role: Role
    text: str                          # rendered text content
    anchor: tuple[float, float]        # sheet mm, origin bottom-left
    attrs: dict = field(default_factory=dict)   # parsed fields (loop, system, setpoints, fields of LineTag, note index...)
    refs: list[str] = field(default_factory=list)  # eids this depends on (note-number refs, dim->geom)
    layer: str = "TEXT"
    font_mm: float = 2.5

    def zone(self, sheet: "Sheet") -> str:
        return sheet.zone_of(self.anchor)


@dataclass
class Sheet:
    sheet_no: int
    of: int
    rev: str
    width: float = 841.0    # A1 landscape, mm
    height: float = 594.0
    elements: dict[str, Element] = field(default_factory=dict)
    note_order: list[str] = field(default_factory=list)   # ordered eids of notes block
    meta: dict = field(default_factory=dict)

    ROWS = list("ABCDEFGHIJ")
    COLS = list(range(1, 13))

    def zone_of(self, anchor: tuple[float, float]) -> str:
        x, y = anchor
        col = min(11, max(0, int(x / self.width * 12))) + 1
        row = self.ROWS[min(9, max(0, int((self.height - y) / self.height * 10)))]
        return f"{row}-{col}"

    def add(self, el: Element) -> Element:
        assert el.eid not in self.elements, f"duplicate eid {el.eid}"
        self.elements[el.eid] = el
        return el

    def clone(self) -> "Sheet":
        return copy.deepcopy(self)

    # ---- notes-block helpers (renumbering cascade machinery) -------------
    def renumber_notes(self) -> list[str]:
        """Reassign visible note numbers from note_order; DELETED collapse
        ranges are formed by the renderer, numbering here is per-element.
        Returns eids whose visible number changed (cascade victims)."""
        changed = []
        for i, eid in enumerate(self.note_order, start=1):
            el = self.elements[eid]
            old = el.attrs.get("note_no")
            el.attrs["note_no"] = i
            new_text = re.sub(r"^\d+\.", f"{i}.", el.text)
            if old is not None and old != i:
                changed.append(eid)
            el.text = new_text
        return changed

    def to_json(self) -> str:
        d = {
            "sheet_no": self.sheet_no, "of": self.of, "rev": self.rev,
            "width": self.width, "height": self.height,
            "note_order": self.note_order, "meta": self.meta,
            "elements": {k: asdict(v) for k, v in self.elements.items()},
        }
        return json.dumps(d, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Sheet":
        d = json.loads(s)
        sh = cls(d["sheet_no"], d["of"], d["rev"], d["width"], d["height"])
        sh.note_order = d["note_order"]
        sh.meta = d.get("meta", {})
        for k, v in d["elements"].items():
            v["anchor"] = tuple(v["anchor"])
            sh.elements[k] = Element(**v)
        return sh


@dataclass
class GTDelta:
    """One ground-truth change. L3 label."""
    did: str
    kind: str                      # add | remove | modify | move
    role: str
    eid_a: Optional[str]
    eid_b: Optional[str]
    sheet: int
    zone_a: Optional[str]
    zone_b: Optional[str]
    field_changes: dict            # {"seq": ["9048","9020"], "pipe_class": ["GC11S","FC11S"]}
    description: str
    is_cascade: bool = False
    primary_did: Optional[str] = None
    semantic_null: bool = False    # representation changed, meaning did not

    def to_dict(self):
        return asdict(self)


def diff_sheets(a: Sheet, b: Sheet) -> list[GTDelta]:
    """Trivially-correct reference differ over the MODEL (eids shared).
    Used as the round-trip validator: applying an edit script and diffing
    must reproduce the emitted GTDelta set (up to cascade/primary tags,
    which only the operators know)."""
    out, i = [], 0
    ae, be = a.elements, b.elements
    for eid in sorted(set(ae) - set(be)):
        el = ae[eid]
        out.append(GTDelta(f"ref{i:04d}", "remove", el.role, eid, None, a.sheet_no,
                           el.zone(a), None, {}, f"removed: {el.text[:60]}")); i += 1
    for eid in sorted(set(be) - set(ae)):
        el = be[eid]
        out.append(GTDelta(f"ref{i:04d}", "add", el.role, None, eid, b.sheet_no,
                           None, el.zone(b), {}, f"added: {el.text[:60]}")); i += 1
    for eid in sorted(set(ae) & set(be)):
        ea, eb = ae[eid], be[eid]
        moved = ea.anchor != eb.anchor
        modified = ea.text != eb.text or ea.attrs != eb.attrs or ea.layer != eb.layer
        if modified:
            fc = {k: [ea.attrs.get(k), eb.attrs.get(k)]
                  for k in set(ea.attrs) | set(eb.attrs)
                  if ea.attrs.get(k) != eb.attrs.get(k)}
            out.append(GTDelta(f"ref{i:04d}", "modify", ea.role, eid, eid, a.sheet_no,
                               ea.zone(a), eb.zone(b), fc,
                               f"'{ea.text[:40]}' -> '{eb.text[:40]}'")); i += 1
        elif moved:
            out.append(GTDelta(f"ref{i:04d}", "move", ea.role, eid, eid, a.sheet_no,
                               ea.zone(a), eb.zone(b), {},
                               f"moved: {ea.text[:50]}")); i += 1
    return out
