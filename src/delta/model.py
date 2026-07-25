"""Delta model: the deterministic engine's output shape.

Mirrors eval/datasets/generator/model.py::GTDelta field-for-field where
possible (did/kind/eid_a/eid_b/sheet/zone_a/zone_b/field_changes/
is_cascade/primary_did) so engine output is directly comparable against
`gt/deltas.json` in tests, without a translation layer.

`semantic_null`/`semantic_null_reason` are set by src/delta/semantic_null.py,
an isolated, opt-in pass (DELTA_SEMANTIC_NULL_LLM=1) run after
classify_matches -- per CLAUDE.md decision #3, semantic-equivalence
adjudication ("representation changed, meaning did not") is an LLM-
adjudicated, isolated, non-deterministic zone, so it stays a separate pass
rather than something classify_matches() itself does; the deterministic
engine can still run with zero LLM calls (the flag defaults off).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

DeltaKind = Literal["add", "remove", "modify", "move", "unclassified_visual_change"]
# unclassified_visual_change: src/delta/raster_recall.py's opt-in (
# DELTA_RASTER_RECALL=1) confidence-gated fallback -- a registered raster
# diff found visual content that never became a CanonicalElement at all
# (unlike a false positive from garbage OCR text, this is the opposite
# failure: real content extraction missed entirely). Always low
# confidence, never scored through eval/metrics.py's per-kind P/R/F1 (no
# clean GT counterpart to match against by construction) -- reported as
# its own count in the eval scorecard instead.


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

    def to_dict(self) -> dict:
        return asdict(self)
