"""LLM interface: one seam, swappable provider. CLAUDE.md decision #10:
provider/model from env, keys never committed, prompt caching for the
two-document context (added once chat's retrieval path exists).

Two credential shapes are supported because they are genuinely different on
the wire, not just two names for the same thing: `api_key` sends the
credential via the `x-api-key` header (the real Anthropic API's convention);
`auth_token` sends it via `Authorization: Bearer` (what some Anthropic-
compatible proxies -- e.g. GLM/z.ai -- expect instead). `auth_token` is
tried first because that's the credential shape the working z.ai config
this project's LLM_* vars were copied from actually used.

Judge/chat backend decoupling: `eval/chat_eval.py`'s LLM-judge defaults to
`get_judge_client()`/`JUDGE_MODEL`, a structurally separate seam from
`get_client()`/`MODEL` -- so a second credential is a one-line `.env`
change (`JUDGE_LLM_*`), not a code change. No second credential is
configured in this environment, so `get_judge_client()` currently falls
back to the exact same config as `get_client()` -- the judge and the
system it's judging share a backend today. `judge_is_same_backend()` lets
callers report that honestly (see README's chat-eval findings section)
rather than silently presenting the judge as independent.
"""
from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()  # no-op if .env is absent or vars are already set in the environment

# Model names are read fresh on each call (get_model / get_judge_model below),
# NOT cached at import time, so tests can monkeypatch LLM_MODEL / JUDGE_MODEL
# in os.environ without a module reload -- the same principle _estimate_cost
# in chat.py applies to cost rates. Reading them once into module constants
# here would freeze the value at first import and silently ignore later env
# changes, which is exactly the inconsistency this refactor removes.
DEFAULT_MODEL = "claude-sonnet-5"


def get_model() -> str:
    """Chat/backend model name, read fresh each call (env: LLM_MODEL)."""
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)


def get_judge_model() -> str:
    """Judge model name, read fresh each call. Defaults to the chat model
    when JUDGE_MODEL is unset, preserving the documented same-backend default
    that judge_is_same_backend() relies on."""
    return os.environ.get("JUDGE_MODEL", get_model())


def _client_from_env(base_url: str | None, auth_token: str | None, api_key: str | None) -> anthropic.Anthropic:
    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    if auth_token:
        kwargs["auth_token"] = auth_token
    elif api_key:
        kwargs["api_key"] = api_key
    else:
        raise RuntimeError(
            "no LLM credentials found -- set LLM_AUTH_TOKEN or LLM_API_KEY in .env "
            "(see .env.example)"
        )
    return anthropic.Anthropic(**kwargs)


def get_client() -> anthropic.Anthropic:
    return _client_from_env(
        os.environ.get("LLM_BASE_URL") or None,
        os.environ.get("LLM_AUTH_TOKEN"),
        os.environ.get("LLM_API_KEY"),
    )


def judge_is_same_backend() -> bool:
    """True when no JUDGE_LLM_* override is configured -- the judge is
    reusing the chat backend's own credentials/model, a self-judging risk
    worth surfacing explicitly rather than leaving implicit."""
    return not any(os.environ.get(k) for k in ("JUDGE_LLM_BASE_URL", "JUDGE_LLM_AUTH_TOKEN", "JUDGE_LLM_API_KEY")) \
        and os.environ.get("JUDGE_MODEL") is None


def get_judge_client() -> anthropic.Anthropic:
    return _client_from_env(
        os.environ.get("JUDGE_LLM_BASE_URL") or os.environ.get("LLM_BASE_URL") or None,
        os.environ.get("JUDGE_LLM_AUTH_TOKEN") or os.environ.get("LLM_AUTH_TOKEN"),
        os.environ.get("JUDGE_LLM_API_KEY") or os.environ.get("LLM_API_KEY"),
    )
