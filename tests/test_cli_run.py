"""End-to-end CLI integration test: `run` produces both a delta report and
a trace, mirroring the tmp_path pattern already used in
tests/test_pdf_native_real_samples.py for env-var-configured output dirs."""
import json
import pathlib

import pytest

from src.cli import cmd_run
from src.observability import tracer as tracer_mod

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


class _Args:
    def __init__(self, a, b, out, html=False):
        self.a, self.b, self.out, self.html = a, b, out, html


def test_run_produces_report_and_trace(tmp_path, monkeypatch):
    pair_dir = PAIRS_DIR / "edited_002"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")

    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    monkeypatch.setattr(tracer_mod, "TRACE_DIR", str(trace_dir))

    out_dir = tmp_path / "reports"
    args = _Args(str(pair_dir / "a" / "L0.pdf"), str(pair_dir / "b" / "L0.pdf"), str(out_dir))
    rc = cmd_run(args)
    assert rc == 0

    assert (out_dir / "delta_report.json").exists()
    assert (out_dir / "delta_report.md").exists()

    trace_files = list(trace_dir.glob("*.json"))
    assert len(trace_files) == 1
    trace = json.loads(trace_files[0].read_text())
    root = trace["spans"][0]
    assert root["name"] == "request"
    child_names = {c["name"] for c in root["children"]}
    assert {"ingest", "precheck", "register", "align", "classify", "report"} <= child_names
    assert root["status"] == "ok"

    events_file = trace_dir / "events.jsonl"
    assert events_file.exists()
    lines = events_file.read_text().splitlines()
    assert len(lines) > 5
    assert all(json.loads(l)["correlation_id"] == trace["correlation_id"] for l in lines)


def test_run_with_html_flag_also_writes_report_html(tmp_path, monkeypatch):
    """--html is opt-in (default False, covered by test_run_produces_report_
    and_trace above never touching report.html) -- this is the flag's own
    positive path, through the real cmd_run, not just html_report.py in
    isolation."""
    pair_dir = PAIRS_DIR / "edited_002"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")

    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    monkeypatch.setattr(tracer_mod, "TRACE_DIR", str(trace_dir))

    out_dir = tmp_path / "reports"
    args = _Args(str(pair_dir / "a" / "L0.pdf"), str(pair_dir / "b" / "L0.pdf"), str(out_dir), html=True)
    rc = cmd_run(args)
    assert rc == 0

    assert (out_dir / "delta_report.json").exists()
    assert (out_dir / "report.html").exists()
    assert "__DATA_JSON__" not in (out_dir / "report.html").read_text()


def test_run_failure_still_writes_trace(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    monkeypatch.setattr(tracer_mod, "TRACE_DIR", str(trace_dir))

    args = _Args("does/not/exist_a.pdf", "does/not/exist_b.pdf", str(tmp_path / "reports"))
    with pytest.raises(ValueError):
        cmd_run(args)

    trace_files = list(trace_dir.glob("*.json"))
    assert len(trace_files) == 1
    trace = json.loads(trace_files[0].read_text())
    root = trace["spans"][0]
    assert root["status"] == "error"
