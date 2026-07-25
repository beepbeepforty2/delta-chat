from src.chat.chat import ChatAnswer
from src.chat.citations import parse_citations
from src.chat.retrieval import Chunk

from eval.chat_eval import (
    ChatEvalItem,
    citation_content_supported,
    judge_correctness,
    score_correctness_aggregate,
    score_groundedness,
    score_groundedness_aggregate,
    score_refusal_accuracy,
    _sentence_for_citation,
)


def test_sentence_for_citation_isolates_the_right_clause():
    # Padding pushes the two claims further apart than the lookback window
    # (200 chars), so each citation's window excludes the other claim.
    padding = "x" * 220
    text = f"The pipe class stayed the same [A:1:F-7:el_1]. {padding} The setpoint changed to 214 [B:1:F-7:el_2]."
    cites = parse_citations(text)
    s1 = _sentence_for_citation(text, cites[0].raw)
    s2 = _sentence_for_citation(text, cites[1].raw)
    assert "pipe class" in s1 and "setpoint" not in s1
    assert "setpoint changed to 214" in s2 and "pipe class" not in s2


def test_sentence_for_citation_quoted_period_does_not_truncate_content():
    """A cited chunk whose own content ends in a period (e.g. a quoted
    note '5. DELETED.') must not have that period mistaken for the
    answer's own sentence boundary -- regression for a real bug found via
    live testing where this silently discarded the quoted content."""
    text = 'Note 5 is marked "DELETED." in both revisions [A:1:B-1:el_1].'
    cite = parse_citations(text)[0]
    sentence = _sentence_for_citation(text, cite.raw)
    assert "DELETED" in sentence


def test_citation_content_supported_true_for_real_overlap():
    chunk = Chunk(id="el_1", source="A", sheet=1, zone="F-7", content="pipe class GC11S on line 4\"-PV-26")
    text = "The pipe class is GC11S [A:1:F-7:el_1]."
    cite = parse_citations(text)[0]
    assert citation_content_supported(text, cite, chunk)


def test_citation_content_supported_false_for_unrelated_chunk():
    chunk = Chunk(id="el_1", source="A", sheet=1, zone="F-7", content="revision table row B added")
    text = "The HH setpoint is 214 [A:1:F-7:el_1]."
    cite = parse_citations(text)[0]
    assert not citation_content_supported(text, cite, chunk)


def test_score_groundedness_none_when_refused():
    result = ChatAnswer("q", "", refused=True, reason="no citations", citations=[], retrieved_ids=[])
    assert score_groundedness(result, {}) is None


def test_score_groundedness_all_supported():
    chunk = Chunk(id="el_1", source="A", sheet=1, zone="F-7", content="pipe class GC11S")
    text = "Pipe class is GC11S [A:1:F-7:el_1]."
    cites = parse_citations(text)
    result = ChatAnswer("q", text, refused=False, reason=None, citations=cites, retrieved_ids=["el_1"])
    g = score_groundedness(result, {"el_1": chunk})
    assert g["all_supported"] is True
    assert g["n_citations"] == 1


def test_score_refusal_accuracy():
    def item(expected, refused):
        r = ChatAnswer("q", "", refused=refused, reason=None, citations=[], retrieved_ids=[])
        return ChatEvalItem("p", "q", expected, "a", r, None)

    items = [item("refuse", True), item("answer", False), item("answer", True)]
    score = score_refusal_accuracy(items)
    assert score == {"n": 3, "n_correct": 2, "accuracy": round(2 / 3, 4)}


def test_score_groundedness_aggregate_mixes_refused_and_answered():
    def item(groundedness):
        r = ChatAnswer("q", "", refused=groundedness is None, reason=None, citations=[], retrieved_ids=[])
        return ChatEvalItem("p", "q", "answer", "a", r, groundedness)

    items = [
        item(None),
        item({"n_citations": 2, "n_supported": 2, "all_supported": True}),
        item({"n_citations": 2, "n_supported": 1, "all_supported": False}),
    ]
    agg = score_groundedness_aggregate(items)
    assert agg["n_answered_with_citations"] == 2
    assert agg["fraction_fully_supported"] == 0.5
    assert agg["citation_support_rate"] == round(3 / 4, 4)


def test_judge_correctness_parses_verdict():
    def fake_llm(system, user):
        return "VERDICT: CORRECT\nBoth state the flow rate changed to 78511."

    correct, reason = judge_correctness("q", "answer", "flow changed to 78511", False,
                                         "flow rate is now 78511 [delta:1:F-7:d1]", call_llm=fake_llm)
    assert correct is True
    assert "flow" in reason.lower() or "78511" in reason


def test_judge_correctness_unparseable_returns_none():
    def fake_llm(system, user):
        return "I'm not sure."

    correct, reason = judge_correctness("q", "answer", "x", False, "y", call_llm=fake_llm)
    assert correct is None


def test_score_correctness_aggregate_excludes_unparseable():
    def item(correct):
        r = ChatAnswer("q", "", refused=False, reason=None, citations=[], retrieved_ids=[])
        it = ChatEvalItem("p", "q", "answer", "a", r, None)
        it.correct = correct
        return it

    items = [item(True), item(True), item(False), item(None)]
    agg = score_correctness_aggregate(items)
    assert agg == {"n_judged": 3, "n_unparseable": 1, "accuracy": round(2 / 3, 4)}
