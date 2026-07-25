"""Cost/latency/determinism comparison across configured LLM backends.

DESIGN.md decision #10's "swappable provider" claim is proven by ONE
alternate provider today (GLM via z.ai's Anthropic-compatible API) -- this
script is what would prove it with 2+ genuinely different backends side by
side (tokens, cost, latency, output variance on the same fixed prompts).
No second credential is configured in this environment (confirmed: only
`LLM_*` is set, no `BACKEND_2_*`/`BACKEND_3_*`), so this is scaffolded and
runnable, not run -- it reports "N/1 backends configured" and exits
cleanly rather than fabricating a comparison. Configure a real
`BACKEND_2_*` (see `.env.example`) to actually run it.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import anthropic

FIXED_PROMPTS = [
    "In one sentence, what is a P&ID?",
    "In one sentence, what does HH mean on an instrument setpoint?",
]


@dataclass
class BackendConfig:
    name: str
    base_url: Optional[str]
    auth_token: Optional[str]
    api_key: Optional[str]
    model: str


def _backend_from_env(prefix: str) -> Optional[BackendConfig]:
    model = os.environ.get(f"{prefix}_MODEL")
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    auth_token = os.environ.get(f"{prefix}_AUTH_TOKEN")
    api_key = os.environ.get(f"{prefix}_API_KEY")
    if not any((model, base_url, auth_token, api_key)):
        return None
    name = os.environ.get(f"{prefix}_NAME", prefix.lower())
    return BackendConfig(name, base_url, auth_token, api_key, model or "claude-sonnet-5")


def configured_backends() -> list[BackendConfig]:
    """The primary LLM_* backend always counts as backend #1 (chat's own
    connection); BACKEND_2_*/BACKEND_3_* are optional additions."""
    primary = BackendConfig(
        "primary", os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_AUTH_TOKEN"),
        os.environ.get("LLM_API_KEY"), os.environ.get("LLM_MODEL", "claude-sonnet-5"),
    )
    backends = [primary]
    for i in (2, 3):
        b = _backend_from_env(f"BACKEND_{i}")
        if b:
            backends.append(b)
    return backends


def _default_call(backend: BackendConfig, prompt: str) -> tuple[str, int, int]:
    kwargs: dict = {}
    if backend.base_url:
        kwargs["base_url"] = backend.base_url
    if backend.auth_token:
        kwargs["auth_token"] = backend.auth_token
    elif backend.api_key:
        kwargs["api_key"] = backend.api_key
    client = anthropic.Anthropic(**kwargs)
    resp = client.messages.create(model=backend.model, max_tokens=150,
                                   messages=[{"role": "user", "content": prompt}])
    text = next((c.text for c in resp.content if c.type == "text"), "")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def run_backend(backend: BackendConfig,
                 call_fn: Optional[Callable[[BackendConfig, str], tuple[str, int, int]]] = None) -> dict:
    """call_fn is injectable (same DI pattern as src/chat/chat.py::answer
    and eval/baselines/llm_direct.py::run_llm_direct) so this is testable
    without a live API call."""
    call = call_fn or _default_call
    runs = []
    for prompt in FIXED_PROMPTS:
        t0 = time.time()
        text, tokens_in, tokens_out = call(backend, prompt)
        latency = time.time() - t0
        runs.append({"prompt": prompt, "latency_s": round(latency, 3),
                      "tokens_in": tokens_in, "tokens_out": tokens_out,
                      "response_preview": text[:120]})
    return {"backend": backend.name, "model": backend.model, "runs": runs}


def main() -> int:
    backends = configured_backends()
    if len(backends) < 2:
        print(f"{len(backends)}/1 backends configured -- comparison needs at least 2. "
              "Configure BACKEND_2_* (and optionally BACKEND_3_*) in .env, see .env.example.")
        return 0

    print(f"=== backend comparison: {len(backends)} backends configured ===\n")
    for backend in backends:
        try:
            result = run_backend(backend)
        except Exception as e:
            print(f"{backend.name} ({backend.model}): ERROR -- {e}")
            continue
        total_latency = sum(r["latency_s"] for r in result["runs"])
        total_tokens = sum(r["tokens_in"] + r["tokens_out"] for r in result["runs"])
        print(f"{backend.name} ({backend.model}): total_latency={total_latency:.2f}s total_tokens={total_tokens}")
        for r in result["runs"]:
            print(f"  [{r['latency_s']}s, in={r['tokens_in']} out={r['tokens_out']}] {r['response_preview']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
