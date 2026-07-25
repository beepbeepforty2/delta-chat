"""Composite-tag parsers: turn rendered P&ID text into structured fields.

Pure functions, no fitz/OCR dependency, so both the native-PDF adapter
(exact text) and a future scanned-PDF adapter (noisier OCR text) can call
the same parsers. Regexes mirror the shapes produced by
eval/datasets/generator/{model,content,ops}.py — kept in sync by hand since
src/ intentionally does not depend on eval/'s GT-authoring code.
"""
from __future__ import annotations

import re
from typing import Optional

# Line tag: 4"-PV-26-9048-GC11S-38  (size-service-system-seq-pipe_class-insul)
# Mirrors eval/datasets/generator/model.py::LineTag.parse field-for-field.
LINE_TAG_RE = re.compile(r'([\d/"x ]+)-([A-Z]{2})-(\d{2})-(\d{4})-([A-Z0-9]+)-(\d{2})')

# Instrument bubble: "PIT 9055 26" or "PIT 9055 26 SD HH:245 LL:120"
INSTRUMENT_RE = re.compile(
    r'^([A-Z]{2,4})\s+(-?\d{3,5})\s+(\d{2})(?:\s+SD\s+HH:(-?\d+)\s+LL:(-?\d+))?$'
)

# Valve tag: 26BL9123  (system + 2-letter body + 3-4 digit seq, no separators)
VALVE_TAG_RE = re.compile(r'^(\d{2})([A-Z]{2})(\d{3,4})$')

# Nozzle: N4203
NOZZLE_RE = re.compile(r'^N(\d{3,4})$')

# Equipment tag: 26-KA-905  (system-type-seq)
EQUIPMENT_TAG_RE = re.compile(r'^(\d{2})-([A-Z]{2,4})-(\d{3})$')

# Title-block fields
DRAWNO_RE = re.compile(r'^[A-Z0-9]+-PID-\d+-\d+-\d+$')
REV_RE = re.compile(r'^REV\s+([A-Z])$')
SHEET_RE = re.compile(r'^SHEET\s+(\d+)\s+OF\s+(\d+)$')
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
DCN_RE = re.compile(r'DCN-KP-\d{4}-\d')


def parse_line_tag(text: str) -> Optional[dict]:
    m = LINE_TAG_RE.match(text.strip())
    if not m:
        return None
    size, service, system, seq, pipe_class, insul = m.groups()
    return {"size": size, "service": service, "system": system,
            "seq": seq, "pipe_class": pipe_class, "insul": insul}


def parse_instrument(text: str) -> Optional[dict]:
    m = INSTRUMENT_RE.match(text.strip())
    if not m:
        return None
    func, loop, system, hh, ll = m.groups()
    out = {"func": func, "loop": int(loop), "system": system}
    if hh is not None and ll is not None:
        out["setpoints"] = {"HH": int(hh), "LL": int(ll)}
    return out


def parse_valve_tag(text: str) -> Optional[dict]:
    m = VALVE_TAG_RE.match(text.strip())
    if not m:
        return None
    system, body, seq = m.groups()
    return {"system": system, "body": body, "seq": seq}


def parse_nozzle(text: str) -> Optional[dict]:
    m = NOZZLE_RE.match(text.strip())
    if not m:
        return None
    return {"noz_no": int(m.group(1))}


def parse_equipment_tag(text: str) -> Optional[dict]:
    m = EQUIPMENT_TAG_RE.match(text.strip())
    if not m:
        return None
    system, type_, seq = m.groups()
    return {"system": system, "type": type_, "seq": int(seq)}


def parse_title_field(text: str) -> Optional[dict]:
    """Classify+parse a title-block field string. Returns {"field": ..., ...}
    or None if it doesn't match any known title-block shape."""
    t = text.strip()
    if DRAWNO_RE.match(t):
        return {"field": "drawno", "value": t}
    m = REV_RE.match(t)
    if m:
        return {"field": "rev", "value": m.group(1)}
    m = SHEET_RE.match(t)
    if m:
        return {"field": "sheet", "sheet_no": int(m.group(1)), "of": int(m.group(2))}
    if DATE_RE.match(t):
        return {"field": "date", "value": t}
    return None
