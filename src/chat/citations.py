"""Fixed-format citations. DESIGN.md decision #7: "Answers must emit
citations in a fixed format; post-validate that cited IDs exist in the
retrieved set; refuse when unsupported."

Format: [source:sheet:zone:id], e.g. [A:1:F-7:el_a1b2c3d4e5f6] or
[delta:1:B-2:delta0007]. Embeds the zone (DESIGN.md decision #5: locations
cite border-grid zones) directly in the citation, not just the id -- a
reader sees *where* without a lookup, and post-validation still only needs
the id field to check existence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.chat.retrieval import Chunk

CITATION_RE = re.compile(r"\[([^:\[\]]+):([^:\[\]]+):([^:\[\]]+):([^:\[\]]+)\]")


def format_citation(chunk: Chunk) -> str:
    return f"[{chunk.source}:{chunk.sheet}:{chunk.zone or '-'}:{chunk.id}]"


@dataclass
class ParsedCitation:
    raw: str
    source: str
    sheet: str
    zone: str
    id: str


def parse_citations(text: str) -> list[ParsedCitation]:
    out = []
    for m in CITATION_RE.finditer(text):
        source, sheet, zone, cid = m.groups()
        out.append(ParsedCitation(raw=m.group(0), source=source, sheet=sheet, zone=zone, id=cid))
    return out


@dataclass
class ValidationResult:
    valid: list[ParsedCitation]
    invalid: list[ParsedCitation]

    @property
    def all_valid(self) -> bool:
        return not self.invalid


def validate_citations(citations: list[ParsedCitation], retrieved: list[Chunk]) -> ValidationResult:
    """Post-validate that every cited id exists in the retrieved set --
    the id is the ground truth; a mismatched source/sheet/zone in the
    citation text doesn't independently invalidate it (the LLM could
    typo the display fields while getting the id right), but a citation
    whose id was never retrieved is always invalid -- that's the actual
    "unsupported" case DESIGN.md's refuse-when-unsupported requirement
    means."""
    valid_ids = {c.id for c in retrieved}
    valid, invalid = [], []
    for cite in citations:
        (valid if cite.id in valid_ids else invalid).append(cite)
    return ValidationResult(valid=valid, invalid=invalid)
