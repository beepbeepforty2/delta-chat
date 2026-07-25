import pathlib

import pytest

from src.delta.model import Delta
from src.delta import semantic_null as sn
from src.delta.semantic_null import adjudicate_semantic_null, annotate_semantic_null, clear_cache

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


def _delta(**kw):
    base = dict(did="d1", kind="modify", element_type="note", id_a="a", id_b="b",
                sheet=1, zone_a="A-1", zone_b="A-1", field_changes={}, is_cascade=False)
    base.update(kw)
    return Delta(**base)


# --- 4a: rule (no LLM) ------------------------------------------------------

def test_rule_fires_for_note_deleted_range_collapse(monkeypatch):
    monkeypatch.delenv("DELTA_SEMANTIC_NULL_LLM", raising=False)
    d = _delta(kind="modify", element_type="note_deleted",
               field_changes={"note_no": [5, None], "range": [None, [5, 6]]})
    calls = []
    annotate_semantic_null([d], call_llm=lambda s, u: calls.append(1) or "VERDICT: NULL\nx")
    assert d.semantic_null is True
    assert "no real content" in d.semantic_null_reason
    assert calls == []  # rule resolved it -- LLM never invoked


def test_rule_does_not_fire_when_type_changed():
    """note -> note_deleted (DeleteNoteKeepPlaceholder) is a REAL change --
    real content vanished, even though the result is a placeholder."""
    d = _delta(kind="modify", element_type="note_deleted",
               field_changes={"type": ["note", "note_deleted"]})
    annotate_semantic_null([d])
    assert d.semantic_null is False


def test_rule_does_not_fire_for_ordinary_renumber_cascade():
    """Regression: an ordinary +1 renumbering cascade of a note_deleted
    element (e.g. "5. DELETED." -> "6. DELETED." because an earlier note
    was inserted) has field_changes={"note_no": [5, 6]} -- no "range" key
    -- and is a REAL, GT-expected delta (what is_cascade/cascade_recall
    exist to track), not a structural no-op. A prior version of this rule
    matched on "field_changes subset of note_deleted's own bookkeeping
    fields," which wrongly caught this too: found via a live eval
    regression (fn went 1->4 on the first end-to-end run after this rule
    shipped) before being caught by a unit test."""
    d = _delta(kind="modify", element_type="note_deleted", is_cascade=True,
               field_changes={"note_no": [5, 6]})
    annotate_semantic_null([d])
    assert d.semantic_null is False


def test_rule_does_not_fire_for_single_renumber_not_yet_a_range():
    """Same underlying case as above but not (yet) grouped into a cascade
    by classify.py -- the "range" key is what actually distinguishes a
    real renumber from a genuine collapse-into-range, not is_cascade
    alone (defense in depth, not the only guard)."""
    d = _delta(kind="modify", element_type="note_deleted", is_cascade=False,
               field_changes={"note_no": [5, 6]})
    annotate_semantic_null([d])
    assert d.semantic_null is False


def test_rule_does_not_fire_for_add_remove_of_note_deleted():
    """Deliberately conservative -- ambiguous whether this is an unmatched
    half of a collapse or a real newly-vanished note; see module docstring."""
    d = _delta(kind="remove", element_type="note_deleted", id_b=None, field_changes={})
    annotate_semantic_null([d])
    assert d.semantic_null is False


def test_rule_does_not_fire_for_unrelated_note_content_change():
    d = _delta(kind="modify", element_type="note", field_changes={"content": ["old text", "new text"]})
    annotate_semantic_null([d])
    assert d.semantic_null is False


# --- 4b: LLM adjudication ----------------------------------------------------

def test_llm_adjudication_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DELTA_SEMANTIC_NULL_LLM", raising=False)
    d = _delta(kind="modify", element_type="note", field_changes={"content": ["old", "new"]})
    calls = []
    annotate_semantic_null([d], call_llm=lambda s, u: calls.append(1) or "VERDICT: NULL\nx")
    assert calls == []
    assert d.semantic_null is False


def test_llm_adjudication_runs_when_flag_set(monkeypatch):
    monkeypatch.setenv("DELTA_SEMANTIC_NULL_LLM", "1")
    clear_cache()
    d = _delta(kind="modify", element_type="note",
               field_changes={"content": ["MAX BACK-PRESSURE 0.005 BARG.", "MAX BACK PRESSURE 0.005 BARG."]})
    annotate_semantic_null([d], call_llm=lambda s, u: "VERDICT: NULL\nsame value, formatting only")
    assert d.semantic_null is True
    assert "formatting" in d.semantic_null_reason


def test_llm_adjudication_real_change_not_flagged(monkeypatch):
    monkeypatch.setenv("DELTA_SEMANTIC_NULL_LLM", "1")
    clear_cache()
    d = _delta(kind="modify", element_type="note",
               field_changes={"content": ["MAX BACK-PRESSURE 0.005 BARG.", "MAX BACK-PRESSURE 0.05 BARG."]})
    annotate_semantic_null([d], call_llm=lambda s, u: "VERDICT: REAL\nvalue actually changed")
    assert d.semantic_null is False


def test_llm_adjudication_only_targets_pure_content_fallback_modifies(monkeypatch):
    monkeypatch.setenv("DELTA_SEMANTIC_NULL_LLM", "1")
    clear_cache()
    calls = []
    d = _delta(kind="modify", element_type="instrument",
               field_changes={"setpoints": [{"HH": 1}, {"HH": 2}]})  # structured, not "content" fallback
    annotate_semantic_null([d], call_llm=lambda s, u: calls.append(1) or "VERDICT: NULL\nx")
    assert calls == []
    assert d.semantic_null is False


def test_adjudicate_semantic_null_caches_by_content_pair():
    clear_cache()
    calls = []

    def fake_llm(system, user):
        calls.append(1)
        return "VERDICT: NULL\nsame"

    adjudicate_semantic_null("old", "new", call_llm=fake_llm)
    adjudicate_semantic_null("old", "new", call_llm=fake_llm)
    assert len(calls) == 1  # second call hit the cache


def test_adjudicate_semantic_null_unparseable_response_is_not_null():
    clear_cache()
    is_null, reason = adjudicate_semantic_null("x", "y", call_llm=lambda s, u: "I'm not sure.")
    assert is_null is False
    assert "unparseable" in reason


# --- live checks against the real dataset -----------------------------------

def test_real_null_reword_pair_gets_semantic_null_flagged(monkeypatch):
    """null_reword_902 has both a real CollapseDeletedRange (rule) and a
    RewordNoteEquivalent (needs the LLM flag) case per the generator."""
    pair_dir = PAIRS_DIR / "null_reword_902"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")
    from src.cli import _resolve_with_pid, compute_deltas
    from src.observability.tracer import Tracer

    monkeypatch.setenv("DELTA_SEMANTIC_NULL_LLM", "1")
    clear_cache()
    doc_a = _resolve_with_pid("A", str(pair_dir / "a" / "L0.pdf"))
    doc_b = _resolve_with_pid("B", str(pair_dir / "b" / "L0.pdf"))
    tracer = Tracer()

    def fake_llm(system, user):
        # A real reword pair should describe the same underlying fact --
        # simulate a reasonable judge without a live call.
        return "VERDICT: NULL\nsame engineering fact, different wording"

    deltas = compute_deltas(doc_a, doc_b, tracer, semantic_null_call_llm=fake_llm)
    tracer.finish()
    assert any(d.semantic_null for d in deltas)


def test_cache_is_bounded_and_evicts_oldest_beyond_cap(monkeypatch):
    """Regression: the module-global _cache used to be an unbounded dict --
    fine for a single CLI run, but an unbounded leak keyed on user content
    for any long-lived process (batch job, future server refactor). The cache
    must now cap at _CACHE_MAX and evict the oldest entry (FIFO) past it.

    We temporarily lower _CACHE_MAX so the test runs in reasonable time; the
    eviction logic itself is what's under test, not the specific bound."""
    clear_cache()
    monkeypatch.setattr(sn, "_CACHE_MAX", 4)
    monkeypatch.setenv("DELTA_SEMANTIC_NULL_LLM", "1")

    def fake_llm(system, user):
        return "VERDICT: REAL\ndistinct content"

    # Fill past the cap; each key is a distinct (old, new) pair.
    for i in range(10):
        adjudicate_semantic_null(f"old-{i}", f"new-{i}", call_llm=fake_llm)

    cache = sn._cache
    assert len(cache) == 4, f"cache should be capped at _CACHE_MAX=4, got {len(cache)}"
    # FIFO eviction: the surviving keys must be the most-recently inserted.
    assert ("old-6", "new-6") in cache
    assert ("old-9", "new-9") in cache
    assert ("old-0", "new-0") not in cache  # evicted long ago
    clear_cache()


def test_cache_clear_still_empties_after_bounding(monkeypatch):
    """clear_cache() (the test seam) must still empty the cache fully even
    after the bounding refactor -- several other tests rely on this."""
    clear_cache()
    monkeypatch.setenv("DELTA_SEMANTIC_NULL_LLM", "1")
    adjudicate_semantic_null("a", "b", call_llm=lambda s, u: "VERDICT: REAL\nx")
    assert len(sn._cache) == 1
    clear_cache()
    assert len(sn._cache) == 0
