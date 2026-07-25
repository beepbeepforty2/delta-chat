"""Canonical representation: the format-agnostic seam.

Layered IR (see README):
  L0 raster page (retained for markup + recall net)
  L1 elements    (primary diff space, citation unit)
  L2 relations   (sparse; containment, text<->symbol association)
  L3 embeddings  (match-cost term only, never a citation target)

Every element retains a pullback to source space (page + bbox + transform)
so markup and citations are always possible.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

ElementType = Literal[
    "text", "note", "note_deleted", "dimension", "instrument", "line_tag",
    "valve_tag", "nozzle", "title_field", "table_cell", "rev_row",
    "datasheet_row", "equipment_tag", "dcn_note", "geometry", "zone_label",
    "unknown",
]

@dataclass
class BBox:
    # normalized [0,1] to sheet, top-left origin, y-down (matches fitz and
    # raster/PNG images natively; adapters must normalize into this frame)
    x0: float; y0: float; x1: float; y1: float

@dataclass
class CanonicalElement:
    id: str
    type: ElementType
    content: str
    bbox: BBox
    sheet: int
    zone: Optional[str]                  # border-grid zone if detectable, e.g. "F-7"
    extraction_confidence: float         # 1.0 native; OCR conf for scans
    attrs: dict = field(default_factory=dict)   # parsed composite-key fields
    relations: list[tuple[str, str]] = field(default_factory=list)  # (kind, other_id)

@dataclass
class CanonicalSheet:
    number: int
    width: float
    height: float
    elements: list[CanonicalElement]

@dataclass
class CanonicalDocument:
    pid: str
    source_format: Literal["pdf_native", "pdf_scanned", "dwg"]
    revision_label: Optional[str]
    sheets: list[CanonicalSheet]
    raster_paths: dict[int, str] = field(default_factory=dict)  # sheet -> png (L0 retained)
