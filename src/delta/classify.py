"""Classification: matched pairs -> typed Delta records.

CLAUDE.md decision #3: classify add/remove/modify/move; confidence =
match-cost margin x extraction_confidence. Decision #4: composite tags are
already parsed into fields by src/canonical/classify.py, so diffing
CanonicalElement.attrs directly gives field-level deltas for free -- no
separate tag-parsing step needed here.

Cascade detection (V1, generic): group same-type "modify" deltas by
(field, constant numeric offset) and, above CASCADE_MIN_FAMILY_SIZE
members, tag all but one as is_cascade -- this generalizes the
generator's own SystematicTagRenumber pattern and also naturally catches
InsertNoteWithCascade's note_no shifts (renumbering the tail of the notes
block after an insertion is itself a constant +1 offset). Known scoped-out
gap: cascade members point to an arbitrary group member as `primary_did`,
not necessarily the true root-cause "add" event (e.g. the actually-
inserted note) -- linking to the real causal event needs cross-kind
reasoning left for a later pass. What this V1 gets right: cascade members
are correctly grouped and not double-counted as independent primary
changes, which is the main practical value for a human reading the report.

Severity ranking (CRITICAL/HIGH/MEDIUM/LOW) is annotated as a final pass
after cascade detection (src/delta/severity.py) -- kept in its own module
since it's a separate rule table over the finished Delta, not part of
matching/classification itself.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from src.canonical.model import CanonicalElement
from src.delta.align import MatchedPair, match_group
from src.delta.model import Delta
from src.delta.severity import annotate_severity

MOVE_THRESHOLD = float(os.environ.get("DELTA_MOVE_THRESHOLD", "0.03"))
CASCADE_MIN_FAMILY_SIZE = int(os.environ.get("DELTA_CASCADE_MIN_FAMILY_SIZE", "3"))
# Derived float attrs (e.g. geometry r_norm, computed from bbox extents)
# pick up sub-point rendering jitter from producer variation (see
# eval/datasets/generator/render.py's "alt" producer) -- exact equality on
# floats treats that noise as a real change. Real numeric field changes in
# this domain (tag renumbers, setpoints) are always >= 1, so this
# tolerance is far below anything that should ever be masked.
NUMERIC_EQ_TOL = float(os.environ.get("DELTA_NUMERIC_EQ_TOL", "1e-4"))

_IGNORED_ATTR_KEYS = {"type_confidence", "classification_rule"}  # adapter bookkeeping, not content


def _center(el: CanonicalElement) -> tuple[float, float]:
    return ((el.bbox.x0 + el.bbox.x1) / 2, (el.bbox.y0 + el.bbox.y1) / 2)


def _dist(p, q) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def _values_differ(va, vb) -> bool:
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
            and not isinstance(va, bool) and not isinstance(vb, bool):
        return abs(va - vb) > NUMERIC_EQ_TOL
    return va != vb


def _field_changes(a: CanonicalElement, b: CanonicalElement) -> dict:
    changes = {}
    if a.type != b.type:
        changes["type"] = [a.type, b.type]
    for k in (set(a.attrs) | set(b.attrs)) - _IGNORED_ATTR_KEYS:
        va, vb = a.attrs.get(k), b.attrs.get(k)
        if _values_differ(va, vb):
            changes[k] = [va, vb]
    if a.content != b.content and not changes:
        # attrs diffing found nothing structured -- fall back to a plain
        # content marker so the change isn't silently dropped
        changes["content"] = [a.content, b.content]
    return changes


def _confidence(pair: MatchedPair) -> float:
    """CLAUDE.md decision #3: confidence = match-cost margin x
    extraction_conf_a x extraction_conf_b -- a product of both sides'
    confidence, not min(). min() and product agree whenever both sides are
    1.0 (every native-native pair, since pdf_native.py hardcodes
    extraction_confidence=1.0), which is why an earlier min()-based version
    passed unnoticed for a long time -- the two formulas only diverge once
    a real (<1.0) OCR confidence is involved, and product is strictly more
    conservative there (0.8*0.8=0.64 vs min=0.8), matching the intent that
    two independently-uncertain extractions should compound, not just take
    the worse of the two."""
    if pair.a and pair.b:
        ext_conf = pair.a.extraction_confidence * pair.b.extraction_confidence
        margin_term = 1.0 if pair.margin is None else max(0.0, min(1.0, pair.margin))
        return round(margin_term * ext_conf, 4)
    el = pair.a or pair.b
    return round(el.extraction_confidence, 4)


def _describe(kind: str, a: Optional[CanonicalElement], b: Optional[CanonicalElement], fc: dict) -> str:
    if kind == "add":
        return f"{b.type} added: {b.content[:60]}"
    if kind == "remove":
        return f"{a.type} removed: {a.content[:60]}"
    if kind == "move":
        return f"'{a.content[:50]}' moved {a.zone} -> {b.zone}"
    if len(fc) == 1:
        k = next(iter(fc))
        old, new = fc[k]
        return f"{b.type} {k} changed: {old} -> {new}"
    return f"'{a.content[:40]}' -> '{b.content[:40]}'"


def classify_matches(matches: list[MatchedPair], sheet: int) -> list[Delta]:
    deltas: list[Delta] = []
    counter = [0]

    def did() -> str:
        counter[0] += 1
        return f"delta{counter[0]:04d}"

    for pair in matches:
        if pair.a and pair.b:
            fc = _field_changes(pair.a, pair.b)
            if fc:
                deltas.append(Delta(did(), "modify", pair.b.type, pair.a.id, pair.b.id, sheet,
                                     pair.a.zone, pair.b.zone, fc, _confidence(pair),
                                     description=_describe("modify", pair.a, pair.b, fc)))
            elif _dist(_center(pair.a), _center(pair.b)) > MOVE_THRESHOLD:
                deltas.append(Delta(did(), "move", pair.b.type, pair.a.id, pair.b.id, sheet,
                                     pair.a.zone, pair.b.zone, {}, _confidence(pair),
                                     description=_describe("move", pair.a, pair.b, {})))
            # else: unchanged -- not emitted as a delta
        elif pair.a and not pair.b:
            deltas.append(Delta(did(), "remove", pair.a.type, pair.a.id, None, sheet,
                                 pair.a.zone, None, {}, _confidence(pair),
                                 description=_describe("remove", pair.a, None, {})))
        elif pair.b and not pair.a:
            deltas.append(Delta(did(), "add", pair.b.type, None, pair.b.id, sheet,
                                 None, pair.b.zone, {}, _confidence(pair),
                                 description=_describe("add", None, pair.b, {})))

    _detect_family_offset_cascades(deltas)
    annotate_severity(deltas)  # after cascades: severity depends on the final is_cascade flag
    return deltas


def _detect_family_offset_cascades(deltas: list[Delta]) -> None:
    groups: dict[tuple[str, str, float], list[Delta]] = defaultdict(list)
    for d in deltas:
        # Only a delta whose *entire* change is one numeric field is
        # eligible: a delta that also carries an independent content
        # change (e.g. a dcn_note whose note_no shifted from the cascade
        # AND whose dcns list gained a genuinely new entry) has meaning
        # beyond the renumber and must stay a standalone primary change,
        # not get silently folded into the cascade tag.
        if d.kind != "modify" or len(d.field_changes) != 1:
            continue
        (k, v), = d.field_changes.items()
        if k == "type":
            continue
        old, new = v
        if isinstance(old, (int, float)) and isinstance(new, (int, float)) \
                and not isinstance(old, bool) and not isinstance(new, bool):
            # match_group so note/note_deleted/dcn_note renumbers (all
            # the same "notes block shifted" phenomenon, see align.py's
            # TYPE_MATCH_GROUPS) cascade together, not split by subtype.
            groups[(match_group(d.element_type), k, new - old)].append(d)

    for (_etype, _field, offset), members in groups.items():
        if offset == 0 or len(members) < CASCADE_MIN_FAMILY_SIZE:
            continue
        members = sorted(members, key=lambda d: d.id_b or d.id_a or "")
        primary, rest = members[0], members[1:]
        for m in rest:
            m.is_cascade = True
            m.primary_did = primary.did
