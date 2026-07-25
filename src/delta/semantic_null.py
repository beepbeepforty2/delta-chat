"""Semantic-null detection: two distinct mechanisms for two distinct
sub-cases in the generator's own ops (eval/datasets/generator/ops.py),
per DESIGN.md decision #3(c) ("optional semantic-equivalence
adjudication ... isolated, cached, documented as the non-deterministic
zone").

Without this, every equivalent reword and every DELETED-range collapse is
indistinguishable from a real change in the engine's own output --
eval/metrics.py's semantic_null_emission_rate measures exactly that gap
(it was 1.0 on the last L0 run: 100% of GT semantic-null entries got
matched by a normal predicted delta).

4a. Rule (no LLM): a note whose content is just a DELETED placeholder --
    "N. DELETED." or a collapsed range "N-M. DELETED." (see
    eval/datasets/generator/ops.py::collapse_deleted_range) -- carries no
    real information regardless of exact numbering. Structural, not a
    judgment call.

4b. LLM adjudication: reword_note_equivalent (ops.py) swaps note text for
    a meaning-equivalent paraphrase. Genuinely needs semantic judgment on
    arbitrary future text -- a rule keyed to the generator's own fixed
    paraphrase dictionary would be overfitting to synthetic data, not a
    real capability. Runs only for modify deltas where classify.py's
    _field_changes() fell through to the generic "content" key (no
    structured attrs diff found anything -- exactly the "words changed,
    unclear if meaning did" case) and 4a's rule didn't already resolve it.
    One isolated call per candidate, cached by (old, new) content pair.
"""
from __future__ import annotations

import os
import re
from collections import OrderedDict
from typing import Callable, Optional

from src.delta.model import Delta

# A "note_deleted"-typed element already matched
# src/canonical/classify.py::DELETED_NOTE_RE to get that type in the first
# place -- see the attrs it assigns there: {"deleted": True, "note_no": N}
# for a single placeholder, {"deleted": True, "range": [lo, hi]} for a
# collapsed one. No need to re-match the text here.


def _rule_deleted_placeholder(delta: Delta) -> Optional[str]:
    """Fires ONLY for a MODIFY that transitions a note_deleted element
    INTO a collapsed-range form ("range" appears in field_changes, i.e.
    b.attrs gained a "range" key it didn't have before -- the precise,
    unambiguous signature of CollapseDeletedRange). Two things this
    deliberately does NOT catch, found the hard way via a live eval
    regression (fn went 1->4 on the very first end-to-end run after this
    rule shipped, traced to exactly this):

    - An ORDINARY +1 renumbering cascade of a note_deleted element (e.g.
      "5. DELETED." -> "6. DELETED." because an earlier note was
      inserted) has field_changes = {"note_no": [5, 6]} -- no "range" key
      at all, so it's real, not null. A prior version of this rule
      matched on "field_changes subset of note_deleted's own bookkeeping
      fields," which wrongly caught this case too: a note_deleted swept
      into an ordinary cascade is a real, GT-expected delta (it's exactly
      what is_cascade=True + cascade_recall exist to track), not a
      structural no-op.
    - is_cascade=True deltas are excluded outright as a second,
      independent guard -- _detect_family_offset_cascades (classify.py)
      already runs before this pass, so is_cascade is final by the time
      this rule sees it.

    Also deliberately does NOT fire for add/remove of a note_deleted
    element -- ambiguous (could be an unmatched half of the same collapse,
    or DeleteNoteKeepPlaceholder's real content newly vanishing with only
    the resulting placeholder surviving the match) and a false null there
    is worse than a missed one."""
    if delta.kind != "modify" or delta.element_type != "note_deleted" or delta.is_cascade:
        return None
    if "type" in delta.field_changes:
        return None  # transitioned INTO/OUT OF note_deleted -- a real change
    if "range" not in delta.field_changes:
        return None  # not a collapse -- e.g. an ordinary single renumber
    return "note_deleted placeholder collapsed into a range -- no real content on either side"


JUDGE_SYSTEM_PROMPT = """You are adjudicating whether a change between two \
revisions of a P&ID (piping & instrumentation diagram) note is a real \
engineering change or just a different way of writing the same fact.

Respond with EXACTLY one line: "VERDICT: NULL" if the two strings describe \
the same engineering fact (a reword, a formatting change, no substantive \
difference), or "VERDICT: REAL" if the meaning actually changed. Follow \
with a one-sentence reason on the next line."""

_VERDICT_RE = re.compile(r"VERDICT:\s*(NULL|REAL)", re.IGNORECASE)

# Process-global LLM verdict cache, keyed by (old, new) content pair -- per
# decision #3(c), "isolated, cached ... within and across eval runs on the
# same process" to avoid repeat calls for identical pairs. Bounded by an
# OrderedDict with FIFO eviction so a long-lived process (batch job, future
# server refactor) cannot grow this without limit on user content. The bound
# is generous relative to the realistic working set (a single eval run touches
# O(100s) of distinct pairs) so eviction never fires in normal use.
_CACHE_MAX = 4096
_cache: "OrderedDict[tuple[str, str], tuple[bool, str]]" = OrderedDict()


def _default_call_llm(system: str, user: str) -> str:
    from src.chat.llm import get_client, get_model
    client = get_client()
    resp = client.messages.create(model=get_model(), max_tokens=150, system=system,
                                   messages=[{"role": "user", "content": user}])
    return next((b.text for b in resp.content if b.type == "text"), "")


def adjudicate_semantic_null(old_content: str, new_content: str,
                              call_llm: Optional[Callable[[str, str], str]] = None) -> tuple[bool, str]:
    """Returns (is_null, reason). Cached by (old, new) content pair --
    "isolated, cached" per decision #3(c), avoids repeat calls for
    identical pairs within and across eval runs on the same process."""
    key = (old_content, new_content)
    if key in _cache:
        return _cache[key]

    llm_call = call_llm or _default_call_llm
    user_message = f'Old: "{old_content}"\nNew: "{new_content}"'
    raw = llm_call(JUDGE_SYSTEM_PROMPT, user_message)
    match = _VERDICT_RE.search(raw)
    if not match:
        result = (False, f"unparseable adjudication response: {raw[:200]!r}")
    else:
        reason = raw[match.end():].strip().lstrip(":").strip() or raw
        result = (match.group(1).upper() == "NULL", reason)
    _cache[key] = result
    # FIFO eviction: OrderedDict preserves insertion order, so popitem(last=
    # False) removes the oldest entry once over capacity. Keeps the cache
    # bounded without disturbing the hot path (eviction only fires past the
    # generous _CACHE_MAX, which normal use never reaches).
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return result


def clear_cache() -> None:
    _cache.clear()


def annotate_semantic_null(deltas: list[Delta], call_llm: Optional[Callable[[str, str], str]] = None) -> None:
    """Mutates in place. 4a (rule) always runs; 4b (LLM) only runs when
    DELTA_SEMANTIC_NULL_LLM=1 -- the deterministic engine must be runnable
    with zero LLM calls, and this flag is how."""
    llm_enabled = os.environ.get("DELTA_SEMANTIC_NULL_LLM") == "1"

    for d in deltas:
        reason = _rule_deleted_placeholder(d)
        if reason:
            d.semantic_null, d.semantic_null_reason = True, reason
            continue

        if not llm_enabled or d.kind != "modify" or set(d.field_changes) != {"content"}:
            continue
        old, new = d.field_changes["content"]
        if not isinstance(old, str) or not isinstance(new, str):
            continue
        is_null, reason = adjudicate_semantic_null(old, new, call_llm=call_llm)
        d.semantic_null, d.semantic_null_reason = is_null, reason
