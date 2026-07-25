"""Pre-check: are these two documents actually revisions of one document?

DESIGN.md decision #6: compare title-block drawing number / equipment tag
before diffing; refuse to diff siblings. The eval dataset's `not_a_pair`
control and our own real 26-KA-901 vs 26-KA-902 samples (genuinely
different equipment, same vendor template) are exactly this failure mode.

Prefers the drawing number (title_field, field="drawno") when extractable
-- it's the more precise identity signal, revision-independent by
construction (the generator keeps drawno constant across a revision, only
tf_rev changes). Falls back to equipment_tag content when no drawno was
extracted, which is what fires on the real samples (no separate
drawing-number stamp was locatable there, see data/samples/PROVENANCE.md).

If neither title-block signal is extractable on either document, falls
back to a third tier: Jaccard overlap of specific tag identifiers
(line/valve/nozzle/equipment/instrument tag content) across the two
documents. Real revision pairs share the vast majority of their tags
unchanged; sibling/unrelated documents share almost none, even on the same
vendor template -- a cheap, generically-available signal needing no new
extraction. Only when there's no comparable tag content on one side
either (truly zero signal of any kind) does it finally proceed with a
warning rather than refuse -- failing open on a diff is safer than
silently blocking one just because extraction under-performed everywhere.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Literal, Optional

from src.canonical.model import CanonicalDocument

TAG_OVERLAP_MIN = float(os.environ.get("PRECHECK_TAG_OVERLAP_MIN", "0.3"))

_TAG_TYPES = {"line_tag", "valve_tag", "nozzle", "equipment_tag", "instrument"}

# Which signal actually decided the result, strongest to weakest. Callers that
# need to treat a weak acceptance differently (src/cli.py warns on one) must
# branch on this, NOT on substrings of `reason` -- `reason` is a human-readable
# message and rewording it silently broke exactly such a check once already.
IdentityTier = Literal["drawno", "equipment", "tag_overlap", "none"]


@dataclass
class PrecheckResult:
    is_pair: bool
    reason: str
    drawing_no_a: Optional[str]
    drawing_no_b: Optional[str]
    equipment_a: Optional[str]
    equipment_b: Optional[str]
    # Defaulted so the six positional call sites below (and any test that
    # constructs one) keep working unchanged.
    identity_tier: IdentityTier = "none"


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


def _tag_content_overlap(doc_a: CanonicalDocument, doc_b: CanonicalDocument) -> Optional[float]:
    """Jaccard overlap of specific tag-identifier content between the two
    documents. None means one side has no comparable tag content at all --
    truly no signal, not just "low overlap"."""
    tags_a = {el.content for sheet in doc_a.sheets for el in sheet.elements
              if el.type in _TAG_TYPES and el.content}
    tags_b = {el.content for sheet in doc_b.sheets for el in sheet.elements
              if el.type in _TAG_TYPES and el.content}
    if not tags_a or not tags_b:
        return None
    return len(tags_a & tags_b) / len(tags_a | tags_b)


def check_same_document(doc_a: CanonicalDocument, doc_b: CanonicalDocument) -> PrecheckResult:
    drawno_a, drawno_b = _find_drawno(doc_a), _find_drawno(doc_b)
    equip_a, equip_b = _find_equipment_tag(doc_a), _find_equipment_tag(doc_b)

    # An EMPTY extracted value is the absence of an identity signal, not a
    # signal that happens to be equal on both sides: two documents whose title
    # blocks both failed to extract must not match at the strongest tier and
    # skip the weaker ones. `is not None` correctly distinguishes "absent" from
    # "present", but presence alone can't decide a match -- hence the explicit
    # non-empty guard on the equality branch, with both-empty falling through.
    if drawno_a is not None and drawno_b is not None:
        if drawno_a != drawno_b:
            return PrecheckResult(False, f"drawing numbers differ: {drawno_a!r} vs {drawno_b!r}",
                                   drawno_a, drawno_b, equip_a, equip_b, "drawno")
        if drawno_a != "":
            return PrecheckResult(True, "drawing numbers match",
                                   drawno_a, drawno_b, equip_a, equip_b, "drawno")

    if equip_a is not None and equip_b is not None:
        if equip_a != equip_b:
            return PrecheckResult(False, f"equipment tags differ: {equip_a!r} vs {equip_b!r}",
                                   drawno_a, drawno_b, equip_a, equip_b, "equipment")
        if equip_a != "":
            return PrecheckResult(True, "equipment tags match",
                                   drawno_a, drawno_b, equip_a, equip_b, "equipment")

    overlap = _tag_content_overlap(doc_a, doc_b)
    if overlap is not None:
        if overlap >= TAG_OVERLAP_MIN:
            return PrecheckResult(True, f"no title-block identity signal extracted; "
                                         f"tag-content overlap {overlap:.0%} indicates same document",
                                   drawno_a, drawno_b, equip_a, equip_b, "tag_overlap")
        return PrecheckResult(False, f"no title-block identity signal extracted; "
                                      f"tag-content overlap only {overlap:.0%}, likely different documents",
                               drawno_a, drawno_b, equip_a, equip_b, "tag_overlap")

    return PrecheckResult(True, "no drawing number, equipment tag, or comparable tag content "
                                 "extracted on either document; proceeding without identity confirmation",
                           drawno_a, drawno_b, equip_a, equip_b, "none")
