import json
import pathlib
import time

import pytest

from src.observability.tracer import Tracer


def test_span_records_timing_and_nesting(tmp_path):
    tracer = Tracer(trace_dir=str(tmp_path))
    with tracer.span("request", pid_a="a", pid_b="b") as root:
        assert root.name == "request"
        with tracer.span("ingest_a") as s1:
            time.sleep(0.001)
            s1.set("n_elements", 42)
        with tracer.span("ingest_b") as s2:
            s2.set("n_elements", 40)
    tracer.finish()

    assert len(tracer._roots) == 1
    root_span = tracer._roots[0]
    assert root_span.name == "request"
    assert root_span.attrs == {"pid_a": "a", "pid_b": "b"}
    assert len(root_span.children) == 2
    assert [c.name for c in root_span.children] == ["ingest_a", "ingest_b"]
    assert root_span.children[0].attrs["n_elements"] == 42
    for span in [root_span] + root_span.children:
        assert span.end_ts >= span.start_ts
        assert span.duration_ms >= 0
        assert span.status == "ok"


def test_correlation_id_consistent_across_spans(tmp_path):
    tracer = Tracer(trace_dir=str(tmp_path))
    with tracer.span("a"):
        with tracer.span("b"):
            pass
    tracer.finish()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(events) == 2
    assert all(e["correlation_id"] == tracer.correlation_id for e in events)


def test_exception_inside_span_is_captured_and_reraised(tmp_path):
    tracer = Tracer(trace_dir=str(tmp_path))
    with pytest.raises(ValueError, match="boom"):
        with tracer.span("request"):
            with tracer.span("risky"):
                raise ValueError("boom")
    trace_path = tracer.finish()  # must not raise, must still write the trace

    data = json.loads(pathlib.Path(trace_path).read_text())
    root = data["spans"][0]
    assert root["status"] == "error"  # failure propagates up to the parent span
    risky = root["children"][0]
    assert risky["status"] == "error"
    assert risky["error_type"] == "ValueError"
    assert risky["error_message"] == "boom"


def test_finish_writes_trace_json_and_events_jsonl(tmp_path):
    tracer = Tracer(trace_dir=str(tmp_path))
    with tracer.span("request"):
        pass
    trace_path = tracer.finish()

    trace_file = pathlib.Path(trace_path)
    assert trace_file.exists()
    data = json.loads(trace_file.read_text())
    assert data["correlation_id"] == tracer.correlation_id
    assert len(data["spans"]) == 1

    events_file = tmp_path / "events.jsonl"
    assert events_file.exists()
    lines = events_file.read_text().splitlines()
    assert len(lines) == 1
    json.loads(lines[0])  # valid JSON


def test_finish_appends_across_multiple_tracers(tmp_path):
    t1 = Tracer(trace_dir=str(tmp_path))
    with t1.span("request"):
        pass
    t1.finish()

    t2 = Tracer(trace_dir=str(tmp_path))
    with t2.span("request"):
        pass
    t2.finish()

    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    ids = {json.loads(l)["correlation_id"] for l in lines}
    assert ids == {t1.correlation_id, t2.correlation_id}


def test_llm_span_captures_telemetry(tmp_path):
    tracer = Tracer(trace_dir=str(tmp_path))
    with tracer.span("request"):
        with tracer.llm_span("chat_answer") as s:
            s.set("model", "claude-sonnet-5")
            s.set("prompt", "What changed?")
            s.set("response", "The pipe class changed.")
            s.set("tokens_in", 512)
            s.set("tokens_out", 24)
            s.set("cost_usd", 0.0031)
    tracer.finish()

    llm = tracer._roots[0].children[0]
    assert llm.name == "chat_answer"
    assert llm.attrs["model"] == "claude-sonnet-5"
    assert llm.attrs["tokens_in"] == 512
    assert llm.attrs["cost_usd"] == 0.0031
    assert llm.duration_ms >= 0


def test_default_trace_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "custom_traces"))
    import importlib
    from src.observability import tracer as tracer_mod
    importlib.reload(tracer_mod)
    try:
        t = tracer_mod.Tracer()
        assert t.trace_dir == tmp_path / "custom_traces"
    finally:
        monkeypatch.delenv("TRACE_DIR", raising=False)
        importlib.reload(tracer_mod)
