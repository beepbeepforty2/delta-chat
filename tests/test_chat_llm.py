from src.chat.llm import (
    DEFAULT_MODEL,
    get_client,
    get_judge_client,
    get_judge_model,
    get_model,
    judge_is_same_backend,
)


def _clear(monkeypatch, *names):
    for n in names:
        monkeypatch.delenv(n, raising=False)


def test_get_client_uses_llm_env_vars(monkeypatch):
    _clear(monkeypatch, "LLM_BASE_URL", "LLM_AUTH_TOKEN", "LLM_API_KEY")
    monkeypatch.setenv("LLM_BASE_URL", "https://a.example.com")
    monkeypatch.setenv("LLM_AUTH_TOKEN", "token-a")
    client = get_client()
    assert client.base_url == "https://a.example.com"
    assert client.auth_token == "token-a"


def test_judge_client_falls_back_to_llm_config_when_judge_unset(monkeypatch):
    _clear(monkeypatch, "JUDGE_LLM_BASE_URL", "JUDGE_LLM_AUTH_TOKEN", "JUDGE_LLM_API_KEY", "JUDGE_MODEL")
    monkeypatch.setenv("LLM_BASE_URL", "https://a.example.com")
    monkeypatch.setenv("LLM_AUTH_TOKEN", "token-a")
    judge_client = get_judge_client()
    assert judge_client.base_url == "https://a.example.com"
    assert judge_client.auth_token == "token-a"
    assert judge_is_same_backend() is True


def test_judge_client_uses_judge_override_when_set(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://a.example.com")
    monkeypatch.setenv("LLM_AUTH_TOKEN", "token-a")
    monkeypatch.setenv("JUDGE_LLM_BASE_URL", "https://b.example.com")
    monkeypatch.setenv("JUDGE_LLM_AUTH_TOKEN", "token-b")
    judge_client = get_judge_client()
    assert judge_client.base_url == "https://b.example.com"
    assert judge_client.auth_token == "token-b"
    assert judge_is_same_backend() is False


def test_judge_is_same_backend_false_when_only_judge_model_set(monkeypatch):
    _clear(monkeypatch, "JUDGE_LLM_BASE_URL", "JUDGE_LLM_AUTH_TOKEN", "JUDGE_LLM_API_KEY")
    monkeypatch.setenv("JUDGE_MODEL", "some-other-model")
    assert judge_is_same_backend() is False


def test_get_model_reads_env_lazily_without_module_reload(monkeypatch):
    """Regression: MODEL/JUDGE_MODEL used to be module constants read once at
    import time, so setting LLM_MODEL in os.environ AFTER import had no effect
    -- contradicting the project's own stated principle (see chat.py's
    _estimate_cost docstring). get_model() must read fresh each call."""
    _clear(monkeypatch, "LLM_MODEL")
    assert get_model() == DEFAULT_MODEL  # unset -> default

    monkeypatch.setenv("LLM_MODEL", "claude-test-model-12345")
    # No reload; the value must be picked up live.
    assert get_model() == "claude-test-model-12345"

    _clear(monkeypatch, "LLM_MODEL")
    assert get_model() == DEFAULT_MODEL


def test_get_judge_model_defaults_to_chat_model_when_unset(monkeypatch):
    """When JUDGE_MODEL is unset, the judge model tracks the chat model live
    (so overriding LLM_MODEL also moves the judge unless JUDGE_MODEL pins it)."""
    _clear(monkeypatch, "LLM_MODEL", "JUDGE_MODEL")
    monkeypatch.setenv("LLM_MODEL", "chat-model-x")
    assert get_judge_model() == "chat-model-x"

    monkeypatch.setenv("JUDGE_MODEL", "pinned-judge")
    assert get_judge_model() == "pinned-judge"  # pinned, ignores chat model
