"""Pre-check: are these two documents actually revisions of one document?

CLAUDE.md decision #6: compare title-block drawing number / equipment tag
before diffing; refuse to diff siblings. The eval dataset's `not_a_pair`
control and our own real 26-KA-901 vs 26-KA-902 samples (genuinely
different equipment, same vendor template) are exactly this failure mode.

Prefers the drawing number (title_field, field="drawno") when extractable
-- it's the more precise identity signal, revision-independent by
construction (the generator keeps drawno constant across a revision, only
tf_rev changes). Falls back to equipment_tag content when no drawno was
extracted, which is what fires on the real samples (no separate
drawing-number stamp was locatable there, see data/samples/PROVENANCE.md).
If neither is extractable on either document, proceeds with a warning
rather than refusing -- failing open on a diff is safer than silently
blocking one just because title-block extraction under-performed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from src.canonical.model import CanonicalDocument


@dataclass
class PrecheckResult:
    is_pair: bool
    reason: str
    drawing_no_a: Optional[str]
    drawing_no_b: Optional[str]
    equipment_a: Optional[str]
    equipment_b: Optional[str]


def _find_drawno(doc: CanonicalDocument) -> Optional[str]:
    for sheet in doc.sheets:
        for el in sheet.elements:
            if el.type == "title_field" and el.attrs.get("field") == "drawno":
                return el.attrs.get("value")
    return None


def _find_equipment_tag(doc: CanonicalDocument) -> Optional[str]:
    """Most frequent equipment_tag content, not just the first -- a sheet
    can carry incidental cross-references to other equipment; the primary
    tag is the one that recurs."""
    tags = [el.content for sheet in doc.sheets for el in sheet.elements
            if el.type == "equipment_tag" and el.content]
    if not tags:
        return None
    return Counter(tags).most_common(1)[0][0]


def check_same_document(doc_a: CanonicalDocument, doc_b: CanonicalDocument) -> PrecheckResult:
    drawno_a, drawno_b = _find_drawno(doc_a), _find_drawno(doc_b)
    equip_a, equip_b = _find_equipment_tag(doc_a), _find_equipment_tag(doc_b)

    if drawno_a and drawno_b:
        if drawno_a == drawno_b:
            return PrecheckResult(True, "drawing numbers match", drawno_a, drawno_b, equip_a, equip_b)
        return PrecheckResult(False, f"drawing numbers differ: {drawno_a!r} vs {drawno_b!r}",
                               drawno_a, drawno_b, equip_a, equip_b)

    if equip_a and equip_b:
        if equip_a == equip_b:
            return PrecheckResult(True, "equipment tags match", drawno_a, drawno_b, equip_a, equip_b)
        return PrecheckResult(False, f"equipment tags differ: {equip_a!r} vs {equip_b!r}",
                               drawno_a, drawno_b, equip_a, equip_b)

    return PrecheckResult(True, "no drawing number or equipment tag extracted on one or both "
                                 "documents; proceeding without identity confirmation",
                           drawno_a, drawno_b, equip_a, equip_b)
