"""Chat evaluation: CLAUDE.md's eval requirements name three metrics --
"correctness (LLM-judge, judged against 5 hand-checked answers to validate
the judge -- report agreement) + deterministic groundedness (cited element
ids exist and contain the claimed content) + refusal accuracy on
refuse-expected probes."

Uses each pair's qa.jsonl (templated Q&A with a reference answer, an
expected_behavior of "answer"/"refuse", and reference citation targets --
see eval/datasets/generator/generate.py:make_qa). Reference citation ids
are the *dataset generator's* internal ids (e.g. "d0002", "inst00"), not
our engine's content-hash ids, so they can't be used to check our own
citations directly -- groundedness instead checks our OWN answer's
citations against our OWN retrieved chunks (id-existence is already
enforced live by src/chat/chat.py's forced-refusal gate; the NEW check
here is content-containment, which chat.py deliberately does not gate on
live -- see its module docstring -- so it lives here as a scoring-only
check instead of risking over-refusal on legitimate paraphrase).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from rapidfuzz import fuzz

from src.chat.chat import ChatAnswer, answer as chat_answer
from src.chat.citations import ParsedCitation
from src.chat.retrieval import BM25Index, Chunk, build_chunks

CONTENT_SUPPORT_THRESHOLD = 45  # rapidfuzz.fuzz.partial_ratio floor; calibrated
# below against real transcripts, same methodology as eval/metrics.py's
# MIN_DESC_SIM -- a sentence built from a short chunk plus the model's own
# phrasing scores lower than two same-source descriptions would, so this
# floor is intentionally looser than MIN_DESC_SIM=60.


def load_qa(pair_dir: Path) -> list[dict]:
    path = pair_dir / "qa.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


_SENTENCE_WINDOW = 200  # chars of lookback before a citation


def _sentence_for_citation(text: str, citation_raw: str) -> str:
    """The text immediately preceding a citation, used as the unit that
    must lexically overlap the cited chunk's content -- checking the whole
    answer against the whole chunk would pass even when a citation is
    attached to the wrong claim within a multi-sentence answer.

    A fixed lookback window, not "back to the previous '.'": quoted source
    content routinely ends in its own period (e.g. citing a note whose
    text is `"5. DELETED."`), which a naive sentence-boundary split
    mistakes for the end of the *answer's* sentence -- discarding exactly
    the quoted content that should overlap the chunk, right before the
    citation that cites it."""
    idx = text.find(citation_raw)
    if idx == -1:
        return text
    start = max(0, idx - _SENTENCE_WINDOW)
    return text[start:idx + len(citation_raw)].strip()


def citation_content_supported(answer_text: str, citation: ParsedCitation, chunk: Chunk,
                                threshold: int = CONTENT_SUPPORT_THRESHOLD) -> bool:
    sentence = _sentence_for_citation(answer_text, citation.raw)
    return fuzz.partial_ratio(sentence.lower(), chunk.content.lower()) >= threshold


def score_groundedness(result: ChatAnswer, chunks_by_id: dict[str, Chunk]) -> Optional[dict]:
    """None when the answer was refused (nothing to ground); otherwise the
    fraction of its citations whose cited chunk content actually supports
    the sentence it's attached to."""
    if result.refused or not result.citations:
        return None
    supported = [citation_content_supported(result.text, c, chunks_by_id[c.id])
                 for c in result.citations if c.id in chunks_by_id]
    return {
        "n_citations": len(result.citations),
        "n_supported": sum(supported),
        "all_supported": all(supported) if supported else True,
    }


JUDGE_SYSTEM_PROMPT = """You are grading whether a system's answer to a \
question about a P&ID engineering document revision is substantively \
correct, compared against a known-correct reference answer.

Rules:
- If the reference says the system should refuse (expected_behavior=refuse), \
the system's answer is CORRECT only if it also refused or otherwise \
indicated it could not answer -- a confident direct answer is INCORRECT.
- If the reference expects an answer, the system's answer is CORRECT if it \
conveys the same substantive facts as the reference (the same entities, \
numbers, and yes/no conclusion), even if worded completely differently. It \
is INCORRECT if it states a different value, omits the key fact, or \
refuses when an answer was expected.
- Ignore citation tags like [A:1:F-7:el_abc123] entirely when judging -- \
judge only the substantive content of the answer.

Respond with EXACTLY one line in this format: "VERDICT: CORRECT" or \
"VERDICT: INCORRECT", followed on the next line by a one-sentence reason."""


def _judge_user_message(question: str, expected_behavior: str, expected_answer: str,
                         actual_refused: bool, actual_text: str) -> str:
    actual = f"[REFUSED] {actual_text}" if actual_refused else actual_text
    return (f"Question: {question}\n"
            f"Expected behavior: {expected_behavior}\n"
            f"Reference answer: {expected_answer}\n"
            f"System's actual answer: {actual}")


_VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)", re.IGNORECASE)

JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "200"))


def _default_judge_call_llm(system: str, user: str) -> str:
    """Deliberately NOT src.chat.chat._default_call_llm: the judge is a
    structurally separate seam (src/chat/llm.py::get_judge_client/
    JUDGE_MODEL) from the chat backend being judged, even though -- with
    no second credential configured in this environment -- they resolve to
    the same backend today (see src/chat/llm.py::judge_is_same_backend).
    Keeping the call path separate means a real second credential is a
    one-line .env change, not a code change."""
    from src.chat.llm import get_judge_client, get_judge_model

    client = get_judge_client()
    resp = client.messages.create(
        model=get_judge_model(), max_tokens=JUDGE_MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")


def judge_correctness(question: str, expected_behavior: str, expected_answer: str,
                       actual_refused: bool, actual_text: str,
                       call_llm: Optional[Callable[[str, str], str]] = None) -> tuple[Optional[bool], str]:
    """Returns (is_correct, reason). is_correct is None if the judge's
    response couldn't be parsed (treated as a judge failure, not scored
    either way, by the caller)."""
    llm_call = call_llm or _default_judge_call_llm
    user_message = _judge_user_message(question, expected_behavior, expected_answer, actual_refused, actual_text)
    raw = llm_call(JUDGE_SYSTEM_PROMPT, user_message)
    match = _VERDICT_RE.search(raw)
    if not match:
        return None, f"unparseable judge response: {raw[:200]!r}"
    reason = raw[match.end():].strip().lstrip(":").strip() or raw
    return match.group(1).upper() == "CORRECT", reason


@dataclass
class ChatEvalItem:
    pair_id: str
    question: str
    expected_behavior: str
    expected_answer: str
    result: ChatAnswer
    groundedness: Optional[dict]
    correct: Optional[bool] = None
    judge_reason: Optional[str] = None


def run_chat_for_pair(doc_a, doc_b, deltas, qa_items: list[dict], pair_id: str,
                       call_llm: Optional[Callable] = None) -> list[ChatEvalItem]:
    chunks = build_chunks(doc_a, doc_b, deltas)
    chunks_by_id = {c.id: c for c in chunks}
    index = BM25Index(chunks)
    items = []
    for qa in qa_items:
        result = chat_answer(qa["q"], index, call_llm=call_llm)
        items.append(ChatEvalItem(
            pair_id=pair_id, question=qa["q"], expected_behavior=qa["expected_behavior"],
            expected_answer=qa["a"], result=result,
            groundedness=score_groundedness(result, chunks_by_id),
        ))
    return items


def score_refusal_accuracy(items: list[ChatEvalItem]) -> dict:
    correct = sum(1 for it in items if (it.expected_behavior == "refuse") == it.result.refused)
    return {"n": len(items), "n_correct": correct,
            "accuracy": round(correct / len(items), 4) if items else None}


def score_groundedness_aggregate(items: list[ChatEvalItem]) -> dict:
    graded = [it.groundedness for it in items if it.groundedness is not None]
    if not graded:
        return {"n_answered_with_citations": 0, "fraction_fully_supported": None,
                "citation_support_rate": None}
    fully = sum(1 for g in graded if g["all_supported"])
    total_cites = sum(g["n_citations"] for g in graded)
    supported_cites = sum(g["n_supported"] for g in graded)
    return {
        "n_answered_with_citations": len(graded),
        "fraction_fully_supported": round(fully / len(graded), 4),
        "citation_support_rate": round(supported_cites / total_cites, 4) if total_cites else None,
    }


def run_judge_on_items(items: list[ChatEvalItem], call_llm: Optional[Callable[[str, str], str]] = None) -> None:
    """Mutates items in place, setting .correct / .judge_reason."""
    for it in items:
        correct, reason = judge_correctness(
            it.question, it.expected_behavior, it.expected_answer,
            it.result.refused, it.result.text, call_llm=call_llm,
        )
        it.correct, it.judge_reason = correct, reason


def score_correctness_aggregate(items: list[ChatEvalItem]) -> dict:
    judged = [it for it in items if it.correct is not None]
    correct = sum(1 for it in judged if it.correct)
    return {"n_judged": len(judged), "n_unparseable": len(items) - len(judged),
            "accuracy": round(correct / len(judged), 4) if judged else None}


@dataclass(frozen=True)
class HandCheckedCase:
    """A real (question, expected, actual) transcript from a live run,
    manually reviewed once and frozen here as ground truth for judge
    validation -- CLAUDE.md: "correctness (LLM-judge, judged against 5
    hand-checked answers to validate the judge -- report agreement)".
    Captured from real `python -m eval.chat_eval` transcripts during
    development, not invented -- see README's chat findings section for
    the full narrative behind each one."""
    pair_id: str
    question: str
    expected_behavior: str
    expected_answer: str
    actual_refused: bool
    actual_text: str
    hand_verdict: bool
    hand_reasoning: str


HAND_CHECKED_SAMPLE: list[HandCheckedCase] = [
    HandCheckedCase(
        pair_id="edited_000",
        question="What is the HH setpoint of FIT-9050 in the revised document?",
        expected_behavior="answer", expected_answer="150",
        actual_refused=False,
        actual_text="The HH setpoint of FIT-9050 is 150. [B:1:C-6:el_b5dabb69207b]",
        hand_verdict=True,
        hand_reasoning="Exact value match (150), correctly grounded in the instrument element.",
    ),
    HandCheckedCase(
        pair_id="edited_002",
        question="Did anything change in zone A-1?",
        expected_behavior="answer",
        expected_answer="Yes: note inserted at position 5: TIE-IN POINT TO BE VERIFIED AT SITE.",
        actual_refused=False,
        actual_text='Yes. In zone A-1, a note was added: "5. TIE-IN POINT TO BE VERIFIED AT SITE." [delta:1:A-1:delta0018]',
        hand_verdict=True,
        hand_reasoning="Same fact (note 5, TIE-IN POINT text, zone A-1), correctly grounded in the delta.",
    ),
    HandCheckedCase(
        pair_id="edited_002",
        question="What changed on sheet 7?",
        expected_behavior="refuse",
        expected_answer="There is no sheet 7; the document has 1 sheet.",
        actual_refused=False,  # did NOT trigger our own refused flag -- see reasoning
        actual_text="The provided context does not contain information about sheet 7. All cited "
                     "chunks refer to sheet 1 [delta:1:B-1:delta0002][A:1:J-10:el_0478a2186923]"
                     "[B:1:D-7:el_1c8d461f2d5c].",
        hand_verdict=True,
        hand_reasoning=(
            "Substantively conveys the same conclusion as the reference (no information exists "
            "for sheet 7) even though the model phrased it without the literal 'REFUSED:' prefix, "
            "so chat.py's own refused flag reads False here -- a real, documented detection gap "
            "(see README): the model can express a refusal in substance without the exact marker "
            "our forced-refusal gate looks for. Judged on content, this answer is correct; refusal "
            "*accuracy* (a separate metric, scored on the refused flag) correctly counts it as a miss."
        ),
    ),
    HandCheckedCase(
        pair_id="edited_000",
        question="Did any alarm or trip setpoints change?",
        expected_behavior="answer",
        expected_answer="No setpoint changes between these revisions.",
        actual_refused=True,
        actual_text="REFUSED: The provided context does not contain information about alarm or "
                     "trip setpoints.",
        hand_verdict=False,
        hand_reasoning=(
            "The reference wants an informative negative ('no setpoint changes'); the system "
            "instead refuses outright. This is a structural limitation, not a bug: a retrieval-"
            "grounded, citation-required chat system has no chunk to cite for the absence of a "
            "change, so it cannot assert a negative without violating its own citation "
            "requirement. Principled, but it still fails to deliver the expected answer, so it's "
            "marked incorrect rather than given a pass for being 'reasonable'."
        ),
    ),
    HandCheckedCase(
        pair_id="null_ident_900",
        question="What changed on sheet 1?",
        expected_behavior="answer", expected_answer="0 primary change(s): ",
        actual_refused=False,
        actual_text=(
            "On sheet 1, two notes were deleted:\n\n- Note 8 was deleted [B:1:B-1:el_56bd2b5cf6ae].\n\n"
            "The remaining changes shown in the context appear as deleted notes present in PID A "
            "but with no corresponding entry in PID B, including notes 4 [A:1:A-1:el_53e7cf17be79], "
            "11 [A:1:B-1:el_65d2e1d6f4b3], and 13 [A:1:B-1:el_3924676b82c4]. Note 4 also appears as "
            "deleted in PID B [B:1:A-1:el_53e7cf17be79]."
        ),
        hand_verdict=False,
        hand_reasoning=(
            "null_ident_900 is an identical clone -- there are zero real deltas, and the reference "
            "answer is '0 primary change(s)'. The system instead describes elements whose PRINTED "
            "TEXT happens to be the word 'DELETED.' (a pre-existing note-placeholder convention in "
            "this drawing, present unchanged in both revisions -- see CLAUDE.md's DELETED-"
            "placeholder-collapse operator) as though they were newly-deleted changes. Every "
            "citation resolves to a real element, so this passes id-existence and would even "
            "plausibly pass a loose content-overlap check, but the substance is a hallucinated "
            "characterization of static content as change -- exactly the kind of null-pair false "
            "positive CLAUDE.md's eval requirements call out as needing its own reporting column."
        ),
    ),
]


def validate_judge(call_llm: Optional[Callable[[str, str], str]] = None) -> dict:
    """Runs the judge against HAND_CHECKED_SAMPLE and reports agreement
    with the hand verdicts -- the actual, measured number CLAUDE.md's eval
    requirements ask for, not an assumed one."""
    results = []
    for case in HAND_CHECKED_SAMPLE:
        judge_verdict, judge_reason = judge_correctness(
            case.question, case.expected_behavior, case.expected_answer,
            case.actual_refused, case.actual_text, call_llm=call_llm,
        )
        results.append({
            "pair_id": case.pair_id, "question": case.question,
            "hand_verdict": case.hand_verdict, "judge_verdict": judge_verdict,
            "agree": judge_verdict == case.hand_verdict, "judge_reason": judge_reason,
        })
    n_agree = sum(1 for r in results if r["agree"])
    return {"cases": results, "n": len(results), "n_agree": n_agree,
            "agreement_rate": round(n_agree / len(results), 4) if results else None}
