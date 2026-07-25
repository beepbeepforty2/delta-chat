"""Shared helpers for loading eval dataset ground truth into
CanonicalElement/CanonicalSheet shapes, for tests that isolate alignment/
classification logic from pdf_native extraction noise. Not a test file
itself (leading underscore keeps pytest from collecting it)."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "eval" / "datasets"))
from generator.model import Sheet  # noqa: E402

from src.canonical.model import BBox, CanonicalElement, CanonicalSheet

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"
EDITED_PAIRS = [f"edited_{i:03d}" for i in range(6)]

ROLE_TO_TYPE = {
    "note": "note", "note_deleted": "note_deleted", "title_field": "title_field",
    "rev_row": "rev_row", "zone_label": "zone_label", "instrument": "instrument",
    "line_tag": "line_tag", "valve_tag": "valve_tag", "nozzle": "nozzle",
    "equipment_tag": "equipment_tag", "datasheet_row": "datasheet_row",
    "geom_line": "geometry", "geom_circle": "geometry", "dcn_note": "dcn_note",
}


def gt_sheet(pair_dir: pathlib.Path, side: str) -> CanonicalSheet:
    sheet_model = Sheet.from_json((pair_dir / side / "model.json").read_text())
    gt_elements = json.loads((pair_dir / "gt" / f"elements_{side}.json").read_text())
    elements = []
    for e in gt_elements:
        x_mm, y_mm = e["anchor_mm"]
        x_norm = x_mm / sheet_model.width
        y_norm = (sheet_model.height - y_mm) / sheet_model.height
        elements.append(CanonicalElement(
            id=e["eid"], type=ROLE_TO_TYPE[e["role"]], content=e["text"],
            bbox=BBox(x_norm, y_norm, x_norm, y_norm), sheet=1, zone=e["zone"],
            extraction_confidence=1.0, attrs=e["attrs"],
        ))
    return CanonicalSheet(number=1, width=sheet_model.width, height=sheet_model.height, elements=elements)


def gt_correspondence(pair_dir: pathlib.Path) -> dict:
    return json.loads((pair_dir / "gt" / "correspondence.json").read_text())


def gt_deltas(pair_dir: pathlib.Path) -> list:
    return json.loads((pair_dir / "gt" / "deltas.json").read_text())
