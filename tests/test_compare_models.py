import pathlib
from dataclasses import dataclass, field

import pytest

from tools.compare_models import _make_metered_call, compare, print_comparison

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


@dataclass
class _Block:
    type: str
    text: str


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeResponse:
    content: list = field(default_factory=list)
    usage: _Usage = None


class _FakeMessages:
    def __init__(self, text: str, tokens_in: int, tokens_out: int):
        self._text, self._tokens_in, self._tokens_out = text, tokens_in, tokens_out
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(content=[_Block(type="text", text=self._text)],
                              usage=_Usage(input_tokens=self._tokens_in, output_tokens=self._tokens_out))


class _FakeClient:
    def __init__(self, text, tokens_in, tokens_out):
        self.messages = _FakeMessages(text, tokens_in, tokens_out)


def test_metered_call_captures_usage_and_latency_in_sink(monkeypatch):
    fake_client = _FakeClient("hello", tokens_in=42, tokens_out=7)
    monkeypatch.setattr("tools.compare_models.get_client", lambda: fake_client)

    sink = {}
    call = _make_metered_call("test-model", sink)
    result = call("system prompt", "user message")

    assert result == "hello"
    assert sink["tokens_in"] == 42
    assert sink["tokens_out"] == 7
    assert isinstance(sink["latency_s"], float)
    assert fake_client.messages.calls[0]["model"] == "test-model"


def test_metered_call_ignores_extra_positional_args(monkeypatch):
    """run_llm_direct's call_llm passes a third (temperature) arg; chat's
    call_llm passes only two -- one shared function must accept both."""
    fake_client = _FakeClient("x", 1, 1)
    monkeypatch.setattr("tools.compare_models.get_client", lambda: fake_client)
    call = _make_metered_call("m", {})
    assert call("s", "u") == "x"
    assert call("s", "u", 0.0) == "x"


def test_print_comparison_renders_all_rows(capsys):
    rows = [
        {"model": "glm-5.2", "task": "llm_direct", "tokens_in": 100, "tokens_out": 50,
         "latency_s": 1.2, "result": "5 deltas"},
        {"model": "glm-4.5-air", "task": "chat", "tokens_in": 80, "tokens_out": 20,
         "latency_s": 0.8, "result": "REFUSED"},
    ]
    print_comparison(rows)
    out = capsys.readouterr().out
    assert "glm-5.2" in out and "glm-4.5-air" in out
    assert "5 deltas" in out and "REFUSED" in out


def test_compare_real_pair_with_fake_client(monkeypatch):
    """End-to-end wiring against a real ingested pair, without a live
    call -- confirms compare() actually drives run_llm_direct + chat.answer
    correctly, not just that the metered-call helper works in isolation."""
    pair_dir = PAIRS_DIR / "edited_003"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")

    fake_client = _FakeClient(
        '[{"kind": "modify", "element_type": "instrument", "sheet": 1, '
        '"zone_a": "C-3", "zone_b": "C-3", "description": "setpoint changed"}]',
        tokens_in=500, tokens_out=40,
    )
    monkeypatch.setattr("tools.compare_models.get_client", lambda: fake_client)

    rows = compare(str(pair_dir / "a" / "L0.pdf"), str(pair_dir / "b" / "L0.pdf"),
                    models=["fake-model-1", "fake-model-2"])

    assert len(rows) == 4  # 2 models x (llm_direct + chat)
    models_seen = {r["model"] for r in rows}
    assert models_seen == {"fake-model-1", "fake-model-2"}
    tasks_seen = {r["task"] for r in rows}
    assert tasks_seen == {"llm_direct", "chat"}
    for r in rows:
        assert r["tokens_in"] == 500
        assert r["tokens_out"] == 40
