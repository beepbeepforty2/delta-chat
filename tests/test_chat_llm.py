from src.chat.llm import get_client, get_judge_client, judge_is_same_backend


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
