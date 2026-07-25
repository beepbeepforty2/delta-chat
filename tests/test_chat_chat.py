from src.chat.chat import LLMResult, answer
from src.observability.tracer import Tracer
from src.chat.retrieval import BM25Index, Chunk

CHUNKS = [
    Chunk(id="el_1", source="A", sheet=1, zone="F-7", content="pipe class GC11S on line 4\"-PV-26-9048"),
    Chunk(id="delta_1", source="delta", sheet=1, zone="F-7", content="pipe class changed: GC11S -> FC11S"),
]


def _index():
    return BM25Index(CHUNKS)


def test_answer_with_valid_citation_not_refused():
    def fake_llm(system, user):
        return "The pipe class changed to FC11S [delta:1:F-7:delta_1]."

    result = answer("what changed with the pipe class?", _index(), call_llm=fake_llm)
    assert not result.refused
    assert len(result.citations) == 1
    assert result.citations[0].id == "delta_1"


def test_answer_with_explicit_model_refusal():
    def fake_llm(system, user):
        return "REFUSED: the retrieved context does not mention sheet 7."

    result = answer("what changed on sheet 7?", _index(), call_llm=fake_llm)
    assert result.refused
    assert "sheet 7" in result.reason


def test_answer_with_no_citations_is_forced_refusal():
    """Even if the model doesn't say REFUSED, an uncited answer must not
    pass through -- DESIGN.md decision #7: refuse when unsupported."""
    def fake_llm(system, user):
        return "The pipe class changed to FC11S."  # plausible, but no citation

    result = answer("what changed?", _index(), call_llm=fake_llm)
    assert result.refused
    assert "no citations" in result.reason


def test_answer_with_hallucinated_citation_is_forced_refusal():
    def fake_llm(system, user):
        return "Something changed [A:1:F-7:el_totally_made_up]."

    result = answer("what changed?", _index(), call_llm=fake_llm)
    assert result.refused
    assert "el_totally_made_up" in result.reason


def test_answer_no_retrieval_hits_refuses_without_calling_llm():
    calls = []

    def fake_llm(system, user):
        calls.append(1)
        return "should never be called"

    result = answer("completely unrelated gibberish query zzzqqq", _index(), call_llm=fake_llm)
    assert result.refused
    assert calls == []  # short-circuited before any LLM call


def test_answer_retrieved_ids_populated():
    def fake_llm(system, user):
        return "The pipe class changed to FC11S [delta:1:F-7:delta_1]."

    result = answer("pipe class", _index(), call_llm=fake_llm)
    assert set(result.retrieved_ids) >= {"delta_1"}


def test_estimate_cost_uses_configured_rates(monkeypatch):
    from src.chat.chat import _estimate_cost
    monkeypatch.setenv("LLM_COST_PER_1K_INPUT", "0.003")
    monkeypatch.setenv("LLM_COST_PER_1K_OUTPUT", "0.015")
    assert _estimate_cost(100, 20) == round(100 / 1000 * 0.003 + 20 / 1000 * 0.015, 6)


def test_estimate_cost_none_when_unconfigured(monkeypatch):
    from src.chat.chat import _estimate_cost
    monkeypatch.delenv("LLM_COST_PER_1K_INPUT", raising=False)
    monkeypatch.delenv("LLM_COST_PER_1K_OUTPUT", raising=False)
    assert _estimate_cost(100, 20) is None


def test_llm_span_captures_tokens_and_cost(tmp_path):
    """Regression: a first pass only recorded model/prompt/response on the
    LLM span, silently missing tokens/cost -- caught by inspecting a real
    trace, not by a test, which is exactly why this one exists now."""
    def fake_llm(system, user):
        return LLMResult(text="ok [delta:1:F-7:delta_1].", tokens_in=100, tokens_out=20, cost_usd=0.0018)

    tracer = Tracer(trace_dir=str(tmp_path))
    with tracer.span("request"):
        answer("pipe class", _index(), tracer=tracer, call_llm=fake_llm)
    tracer.finish()

    llm_span = next(c for c in tracer._roots[0].children if c.name == "chat_answer")
    assert llm_span.attrs["tokens_in"] == 100
    assert llm_span.attrs["tokens_out"] == 20
    assert llm_span.attrs["cost_usd"] == 0.0018
