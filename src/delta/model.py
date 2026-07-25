"""Delta model: the deterministic engine's output shape.

Mirrors eval/datasets/generator/model.py::GTDelta field-for-field where
possible (did/kind/eid_a/eid_b/sheet/zone_a/zone_b/field_changes/
is_cascade/primary_did) so engine output is directly comparable against
`gt/deltas.json` in tests, without a translation layer.

`semantic_null`/`semantic_null_reason` are set by src/delta/semantic_null.py,
an isolated, opt-in pass (DELTA_SEMANTIC_NULL_LLM=1) run after
classify_matches -- per DESIGN.md decision #3, semantic-equivalence
adjudication ("representation changed, meaning did not") is an LLM-
adjudicated, isolated, non-deterministic zone, so it stays a separate pass
rather than something classify_matches() itself does; the deterministic
engine can still run with zero LLM calls (the flag defaults off).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

from src.canonical.model import BBox

DeltaKind = Literal["add", "remove", "modify", "move", "unclassified_visual_change"]
# graphical: identical tag/annotation text on both sides, but the drawn
# symbol changed (e.g. a valve's glyph swapped) -- the case a text-only
# pipeline is structurally blind to.
# geometry: no text-bearing element found near the region on either side
# -- a pure vector-graphics change (e.g. a line reroute).
# extraction_gap: text found on only one side, or non-matching text on
# both -- this SHOULD have been a symbolic add/remove/modify and wasn't;
# a diagnostic signal about the text pipeline, not just a graphical one.
VisualChangeKind = Literal["graphical", "geometry", "extraction_gap"]
# unclassified_visual_change: emitted by src/delta/raster_join.py, fed by
# src/delta/raster_diff.py's opt-in (DELTA_RASTER_DIFF=1) registered
# raster-diff region proposal -- residue the symbolic (text) pipeline
# could not explain. "Raster localizes, symbolic classifies": the region
# proposal never knows about symbolic deltas, and raster_join.py only
# ever uses them to SUPPRESS an already-explained region, never to help
# itself explain one. Always low confidence, never scored through
# eval/metrics.py's per-kind P/R/F1 (no clean GT counterpart to match
# against by construction) -- reported as its own count in the eval
# scorecard instead.


@dataclass
class Delta:
    did: str
    kind: DeltaKind
    element_type: str
    id_a: Optional[str]
    id_b: Optional[str]
    sheet: int
    zone_a: Optional[str]
    zone_b: Optional[str]
    field_changes: dict = field(default_factory=dict)
    confidence: float = 1.0          # match-cost margin x extraction_confidence
    description: Optional[str] = None  # deterministic placeholder; LLM enriches later
    is_cascade: bool = False
    primary_did: Optional[str] = None
    severity: Optional[str] = None   # set by src/delta/severity.py; None until annotated
    semantic_null: bool = False              # set by src/delta/semantic_null.py; opt-in pass
    semantic_null_reason: Optional[str] = None
    # bbox_a/bbox_b: only ever set by src/delta/raster_join.py, for
    # unclassified_visual_change deltas -- these have no id_a/id_b (no
    # CanonicalElement to look up), unlike every symbolic delta, whose
    # bbox is always resolved via id_a/id_b against the element it
    # matched (see markup/overlay.py::_collect_boxes for that existing
    # pattern, unchanged by this addition).
    bbox_a: Optional[BBox] = None
    bbox_b: Optional[BBox] = None
    visual_change_kind: Optional[VisualChangeKind] = None

    def to_dict(self) -> dict:
        return asdict(self)
