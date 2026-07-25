from src.observability.print_trace import render

SAMPLE_TRACE = {
    "correlation_id": "abc123",
    "spans": [
        {
            "name": "request", "duration_ms": 120.5, "status": "ok",
            "attrs": {"pid_a": "a.pdf", "pid_b": "b.pdf"},
            "error_type": None, "error_message": None,
            "children": [
                {
                    "name": "ingest_a", "duration_ms": 40.2, "status": "ok",
                    "attrs": {"n_elements": 123}, "error_type": None, "error_message": None,
                    "children": [],
                },
                {
                    "name": "chat_answer", "duration_ms": 800.0, "status": "ok",
                    "attrs": {"model": "claude-sonnet-5", "tokens_in": 512,
                              "tokens_out": 24, "cost_usd": 0.0031},
                    "error_type": None, "error_message": None, "children": [],
                },
                {
                    "name": "risky", "duration_ms": 5.0, "status": "error",
                    "attrs": {}, "error_type": "ValueError", "error_message": "boom",
                    "children": [],
                },
            ],
        },
    ],
}


def test_render_includes_correlation_id_and_span_names():
    out = render(SAMPLE_TRACE)
    assert "abc123" in out
    assert "request [120.5ms] OK" in out
    assert "ingest_a" in out
    assert "n_elements=123" in out


def test_render_marks_llm_span():
    out = render(SAMPLE_TRACE)
    assert "model=claude-sonnet-5 tokens_in=512 tokens_out=24 cost_usd=0.0031" in out


def test_render_marks_error_span():
    out = render(SAMPLE_TRACE)
    assert "risky [5.0ms] ERROR" in out
    assert "ValueError: boom" in out
