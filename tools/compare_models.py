"""Quick comparative usage check: same PID pair, same credential
(LLM_BASE_URL/AUTH_TOKEN/API_KEY from env -- see src/chat/llm.py), different
model names, side by side. Deliberately NOT a backend-comparison
abstraction: model names are just strings passed on the command line, no
per-model credential config -- matches the project's established
preference for plain env-var-driven model switching over dedicated
multi-backend tooling (see memory: feedback_prefer_env_var_llm_switching).

Reuses the SAME call_llm injection point every other LLM call site in this
project already has (src/chat/chat.py::answer, eval/baselines/
llm_direct.py::run_llm_direct) -- one shared metered-call closure per
model, no env mutation, no module reloading.

Usage:
    python -m tools.compare_models --a revA.pdf --b revB.pdf \
        --models glm-5.2,glm-4.5-air [--question "what changed?"]
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

from eval.baselines.llm_direct import run_llm_direct
from src.chat.chat import answer as chat_answer
from src.chat.llm import get_client
from src.chat.retrieval import BM25Index, build_chunks
from src.cli import _resolve_with_pid, compute_deltas
from src.observability.tracer import Tracer

DEFAULT_QUESTION = "What changed on sheet 1?"


def _make_metered_call(model: str, sink: dict):
    """One closure, one shape (system, user, *extra) -> str, that fits
    both call sites' call_llm signatures (chat.answer wants (system, user);
    run_llm_direct wants (system, user, temperature)) -- extra positional
    args are just ignored. Usage/latency land in `sink` as a side channel
    since run_llm_direct's call_llm interface only returns text, with
    nowhere else to carry token counts back out."""
    def _call(system: str, user: str, *_extra) -> str:
        client = get_client()
        t0 = time.time()
        resp = client.messages.create(
            model=model, max_tokens=2048, system=system,
            messages=[{"role": "user", "content": user}],
        )
        sink["latency_s"] = round(time.time() - t0, 3)
        sink["tokens_in"] = resp.usage.input_tokens
        sink["tokens_out"] = resp.usage.output_tokens
        return next((b.text for b in resp.content if b.type == "text"), "")
    return _call


def compare(path_a: str, path_b: str, models: list[str], question: str = DEFAULT_QUESTION) -> list[dict]:
    doc_a = _resolve_with_pid("A", path_a)
    doc_b = _resolve_with_pid("B", path_b)
    tracer = Tracer()
    deltas = compute_deltas(doc_a, doc_b, tracer)
    tracer.finish()
    chunks = build_chunks(doc_a, doc_b, deltas)
    index = BM25Index(chunks)

    rows = []
    for model in models:
        direct_sink: dict = {}
        direct_deltas = run_llm_direct(doc_a, doc_b, call_llm=_make_metered_call(model, direct_sink))
        rows.append({
            "model": model, "task": "llm_direct",
            "tokens_in": direct_sink.get("tokens_in"), "tokens_out": direct_sink.get("tokens_out"),
            "latency_s": direct_sink.get("latency_s"), "result": f"{len(direct_deltas)} deltas",
        })

        chat_sink: dict = {}
        result = chat_answer(question, index, call_llm=_make_metered_call(model, chat_sink))
        rows.append({
            "model": model, "task": "chat",
            "tokens_in": chat_sink.get("tokens_in"), "tokens_out": chat_sink.get("tokens_out"),
            "latency_s": chat_sink.get("latency_s"),
            "result": "REFUSED" if result.refused else result.text[:70].replace("\n", " "),
        })
    return rows


def print_comparison(rows: list[dict]) -> None:
    header = f"{'model':16s} {'task':11s} {'tokens_in':>9s} {'tokens_out':>10s} {'latency_s':>9s}  result"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:16s} {r['task']:11s} {str(r['tokens_in']):>9s} "
              f"{str(r['tokens_out']):>10s} {str(r['latency_s']):>9s}  {r['result']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--models", required=True, help="comma-separated model names, same credential for all")
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = compare(args.a, args.b, models, question=args.question)
    print_comparison(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
