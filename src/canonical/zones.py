"""Border-grid zone computation: the domain-native location primitive.

compute_zone operates on normalized, top-left/y-down coordinates (the
BBox convention, see src/canonical/model.py) and replicates
eval/datasets/generator/model.py::Sheet.zone_of's formula exactly: that
function computes `(height - y) / height` from its own bottom-left/y-up
mm coordinates, which *is* top-left/y-down normalized y — no extra flip
needed here.
"""
from __future__ import annotations

import os
import re

ROWS = "ABCDEFGHIJ"
N_COLS = 12
N_ROWS = len(ROWS)

# fraction of sheet width/height treated as the zone-label margin band
ZONE_LABEL_MARGIN = float(os.environ.get("ZONE_LABEL_MARGIN", "0.02"))

_COL_LABEL_RE = re.compile(r"^(?:[1-9]|1[0-2])$")
_ROW_LABEL_RE = re.compile(r"^[A-J]$")


def compute_zone(x_norm: float, y_norm: float) -> str:
    """x_norm, y_norm: normalized [0,1] top-left/y-down coordinates."""
    col = min(N_COLS - 1, max(0, int(x_norm * N_COLS))) + 1
    row = ROWS[min(N_ROWS - 1, max(0, int(y_norm * N_ROWS)))]
    return f"{row}-{col}"


def is_zone_label_shaped(text: str, x_norm: float, y_norm: float) -> bool:
    """A zone-label element is a single row/column token sitting in the
    outer margin band of the sheet (matches make_sheet()'s zone-label
    placement: at ~4mm from each edge of an 841x594mm sheet)."""
    t = text.strip()
    if not (_COL_LABEL_RE.match(t) or _ROW_LABEL_RE.match(t)):
        return False
    in_margin = (
        x_norm <= ZONE_LABEL_MARGIN or x_norm >= 1 - ZONE_LABEL_MARGIN or
        y_norm <= ZONE_LABEL_MARGIN or y_norm >= 1 - ZONE_LABEL_MARGIN
    )
    return in_margin
