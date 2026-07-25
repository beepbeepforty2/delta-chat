import io
from contextlib import redirect_stdout

from eval.baselines.backend_compare import (
    BackendConfig,
    configured_backends,
    main,
    run_backend,
)


def _clear_backend_env(monkeypatch):
    for var in ("LLM_MODEL", "LLM_BASE_URL", "LLM_AUTH_TOKEN", "LLM_API_KEY",
                "BACKEND_2_NAME", "BACKEND_2_MODEL", "BACKEND_2_BASE_URL",
                "BACKEND_2_AUTH_TOKEN", "BACKEND_2_API_KEY",
                "BACKEND_3_NAME", "BACKEND_3_MODEL", "BACKEND_3_BASE_URL",
                "BACKEND_3_AUTH_TOKEN", "BACKEND_3_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_configured_backends_always_includes_primary(monkeypatch):
    _clear_backend_env(monkeypatch)
    backends = configured_backends()
    assert len(backends) == 1
    assert backends[0].name == "primary"


def test_configured_backends_picks_up_backend_2(monkeypatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("BACKEND_2_MODEL", "some-other-model")
    monkeypatch.setenv("BACKEND_2_NAME", "gpt-thing")
    backends = configured_backends()
    assert len(backends) == 2
    assert backends[1].name == "gpt-thing"
    assert backends[1].model == "some-other-model"


def test_run_backend_uses_injected_call_fn():
    backend = BackendConfig("test", None, None, None, "test-model")
    calls = []

    def fake_call(b, prompt):
        calls.append(prompt)
        return f"answer to: {prompt}", 10, 5

    result = run_backend(backend, call_fn=fake_call)
    assert result["backend"] == "test"
    assert len(result["runs"]) == 2
    assert len(calls) == 2
    for r in result["runs"]:
        assert r["tokens_in"] == 10
        assert r["tokens_out"] == 5


def test_main_reports_insufficient_backends_and_exits_cleanly(monkeypatch):
    _clear_backend_env(monkeypatch)
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main()
    assert code == 0
    assert "1/1 backends configured" in buf.getvalue()
