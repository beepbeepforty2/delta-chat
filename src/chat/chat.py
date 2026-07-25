"""Grounded chat: retrieval -> LLM -> citation post-validation ->
refuse-when-unsupported. CLAUDE.md decision #7.

The LLM's only role here (decision #3): write the answer prose with
citations. Retrieval, citation parsing, and validation are all
deterministic -- only the prose is generated, and even that is checked
before being trusted: an answer with no citations, or with a citation
whose id was never retrieved, is deterministically overridden into a
refusal rather than returned as-is. The model's own "REFUSED: ..." framing
is respected too, but isn't the only path to a refusal.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Optional

from src.chat.citations import ParsedCitation, format_citation, parse_citations, validate_citations
from src.chat.llm import MODEL, get_client
from src.chat.retrieval import BM25Index, Chunk
from src.observability.tracer import Tracer

TOP_K = int(os.environ.get("CHAT_TOP_K", "8"))
MAX_TOKENS = int(os.environ.get("CHAT_MAX_TOKENS", "1024"))
REFUSAL_PREFIX = "REFUSED:"

def _estimate_cost(tokens_in: int, tokens_out: int) -> Optional[float]:
    """Cost is provider-specific and not hardcoded (config over hardcoding):
    unset by default, so cost_usd stays None for a provider we don't know
    the price table for (e.g. a third-party Anthropic-compatible proxy)
    rather than silently reporting a wrong number under real Anthropic
    pricing. Read fresh each call, not cached at import time, so tests can
    monkeypatch the env without needing a module reload."""
    cost_in = os.environ.get("LLM_COST_PER_1K_INPUT")
    cost_out = os.environ.get("LLM_COST_PER_1K_OUTPUT")
    if cost_in is None or cost_out is None:
        return None
    return round(tokens_in / 1000 * float(cost_in) + tokens_out / 1000 * float(cost_out), 6)

SYSTEM_PROMPT = """You are answering questions about two revisions of a P&ID \
(piping & instrumentation diagram) engineering document, called PID A and \
PID B, and a structured delta report describing what changed between them.

You will be given a set of retrieved context chunks, each tagged with a \
citation key in the exact format [source:sheet:zone:id], where source is \
"A", "B", or "delta".

Rules:
1. Answer ONLY using the provided chunks. Do not use outside knowledge.
2. Every factual claim in your answer must end with a citation copied \
VERBATIM from the chunk it came from -- exact characters, e.g. \
[A:1:F-7:el_a1b2c3d4e5f6]. Never invent, alter, or abbreviate a citation.
3. If the provided chunks do not contain enough information to answer, \
respond with EXACTLY: "REFUSED: " followed by a one-sentence reason. Do \
not guess or answer partially from outside knowledge."""


def _render_context(chunks: list[Chunk]) -> str:
    lines = []
    for c in chunks:
        tag = format_citation(c)
        kind = c.element_type or c.kind or "?"
        lines.append(f"{tag} ({c.source}, {kind}): {c.content}")
    return "\n".join(lines)


@dataclass
class LLMResult:
    """Richer return type for call_llm, carrying the telemetry CLAUDE.md
    decision #8 requires LLM spans to capture (prompt/response/model/
    tokens/cost). A plain str is also accepted from call_llm (auto-wrapped
    below) so test fakes can stay simple -- only the real default call
    needs to report real usage."""
    text: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None


def _default_call_llm(system: str, user: str) -> LLMResult:
    client = get_client()
    resp = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "")
    tokens_in = resp.usage.input_tokens
    tokens_out = resp.usage.output_tokens
    return LLMResult(text=text, tokens_in=tokens_in, tokens_out=tokens_out,
                      cost_usd=_estimate_cost(tokens_in, tokens_out))


@dataclass
class ChatAnswer:
    question: str
    text: str
    refused: bool
    reason: Optional[str]
    citations: list[ParsedCitation]
    retrieved_ids: list[str]


def answer(
    question: str,
    index: BM25Index,
    tracer: Optional[Tracer] = None,
    call_llm: Optional[Callable[[str, str], "str | LLMResult"]] = None,
) -> ChatAnswer:
    """call_llm(system, user) -> str | LLMResult is injectable so tests
    don't need a live API call; defaults to the real LLM via
    src/chat/llm.py. A plain str is fine for fakes; LLMResult additionally
    carries token/cost telemetry for the LLM span."""
    with (tracer.span("retrieval", question=question) if tracer else nullcontext(None)) as s:
        retrieved = index.search(question, top_k=TOP_K)
        chunks = [c for c, _score in retrieved]
        if s:
            s.set("n_hits", len(chunks))

    if not chunks:
        return ChatAnswer(question, "", refused=True,
                           reason="no relevant context retrieved for this question",
                           citations=[], retrieved_ids=[])

    context = _render_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    llm_call = call_llm or _default_call_llm

    with (tracer.llm_span("chat_answer", model=MODEL) if tracer else nullcontext(None)) as s:
        raw = llm_call(SYSTEM_PROMPT, user_message)
        result_obj = raw if isinstance(raw, LLMResult) else LLMResult(text=raw)
        response_text = result_obj.text
        if s:
            s.set("prompt", user_message)
            s.set("response", response_text)
            if result_obj.tokens_in is not None:
                s.set("tokens_in", result_obj.tokens_in)
            if result_obj.tokens_out is not None:
                s.set("tokens_out", result_obj.tokens_out)
            if result_obj.cost_usd is not None:
                s.set("cost_usd", result_obj.cost_usd)

    stripped = response_text.strip()
    if stripped.startswith(REFUSAL_PREFIX):
        return ChatAnswer(question, stripped, refused=True,
                           reason=stripped[len(REFUSAL_PREFIX):].strip(),
                           citations=[], retrieved_ids=[c.id for c in chunks])

    parsed = parse_citations(stripped)
    if not parsed:
        return ChatAnswer(question, stripped, refused=True,
                           reason="answer contained no citations",
                           citations=[], retrieved_ids=[c.id for c in chunks])

    result = validate_citations(parsed, chunks)
    if not result.all_valid:
        bad = ", ".join(c.raw for c in result.invalid)
        return ChatAnswer(question, stripped, refused=True,
                           reason=f"answer cited unsupported id(s): {bad}",
                           citations=result.valid, retrieved_ids=[c.id for c in chunks])

    return ChatAnswer(question, stripped, refused=False, reason=None,
                       citations=result.valid, retrieved_ids=[c.id for c in chunks])
