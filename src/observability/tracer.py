"""Homegrown observability: context-manager spans, correlation id per
request, per-span timings, JSONL structured logs, one JSON trace file per
request. CLAUDE.md decision #8: zero infra, fully understood, vs OTel --
see README for the justification.

Usage:
    tracer = Tracer()
    try:
        with tracer.span("request", pid_a=a, pid_b=b):
            with tracer.span("ingest_a") as s:
                doc_a = ingest(a)
                s.set("n_elements", n)
            ...
    finally:
        tracer.finish()   # always writes the trace + log, success or failure
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

TRACE_DIR = os.environ.get("TRACE_DIR", "traces")

# Attrs whose presence marks a span as an LLM call, for print_trace.py's
# special-case rendering. Not a separate span type -- llm_span() is a thin
# semantic wrapper over span() so LLM calls get the exact same timing and
# failure-capture behavior as everything else, no separate code path.
LLM_SPAN_MARKER_ATTRS = ("tokens_in", "tokens_out", "cost_usd")


@dataclass
class Span:
    name: str
    start_ts: float
    end_ts: Optional[float] = None
    status: str = "ok"
    attrs: dict = field(default_factory=dict)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    children: list["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_ts is None:
            return None
        return round((self.end_ts - self.start_ts) * 1000, 3)

    def set(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def to_dict(self) -> dict:
        return {
            "name": self.name, "start_ts": self.start_ts, "end_ts": self.end_ts,
            "duration_ms": self.duration_ms, "status": self.status,
            "attrs": self.attrs, "error_type": self.error_type,
            "error_message": self.error_message,
            "children": [c.to_dict() for c in self.children],
        }


class Tracer:
    """One per request. Not thread-safe (this is a CLI, not a server) --
    the parent-stack is process-local, ordering-based nesting."""

    def __init__(self, correlation_id: Optional[str] = None, trace_dir: Optional[str] = None):
        self.correlation_id = correlation_id or uuid.uuid4().hex[:12]
        self.trace_dir = Path(trace_dir or TRACE_DIR)
        self._stack: list[Span] = []
        self._roots: list[Span] = []
        self._events: list[dict] = []

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[Span]:
        s = Span(name=name, start_ts=time.time(), attrs=dict(attrs))
        (self._stack[-1].children if self._stack else self._roots).append(s)
        self._stack.append(s)
        try:
            yield s
        except Exception as e:
            # Failure is recorded on the span, never swallowed: the
            # exception still propagates after this.
            s.status = "error"
            s.error_type = type(e).__name__
            s.error_message = str(e)
            raise
        finally:
            s.end_ts = time.time()
            self._stack.pop()
            self._log_event(s)

    @contextmanager
    def llm_span(self, name: str = "llm_call", **attrs) -> Iterator[Span]:
        """Semantic alias for span() -- set model/prompt/response/tokens_in/
        tokens_out/cost_usd on the yielded span once the call completes.
        Same timing and failure-capture behavior as any other span."""
        with self.span(name, **attrs) as s:
            yield s

    def _log_event(self, s: Span) -> None:
        self._events.append({
            "correlation_id": self.correlation_id, "span": s.name,
            "start_ts": s.start_ts, "end_ts": s.end_ts, "duration_ms": s.duration_ms,
            "status": s.status, "attrs": s.attrs,
            "error_type": s.error_type, "error_message": s.error_message,
        })

    def finish(self) -> str:
        """Writes traces/{correlation_id}.json (the full nested tree for
        this request) and appends to traces/events.jsonl (flat structured
        log, correlation id on every line). Always safe to call, even
        after a failure -- this is what makes failures 'captured, not
        swallowed' rather than just 'not crashing the tracer'."""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.trace_dir / f"{self.correlation_id}.json"
        trace_path.write_text(json.dumps({
            "correlation_id": self.correlation_id,
            "spans": [r.to_dict() for r in self._roots],
        }, indent=2))
        events_path = self.trace_dir / "events.jsonl"
        with open(events_path, "a") as f:
            for e in self._events:
                f.write(json.dumps(e) + "\n")
        return str(trace_path)
