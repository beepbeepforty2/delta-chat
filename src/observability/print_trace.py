"""Pretty-print a trace file: span tree with durations, LLM telemetry,
custom attrs (e.g. delta counts) -- the "inspectable metrics" requirement,
homegrown rather than a dashboard. Wired to `make trace ID=<correlation_id>`.

Usage: python -m src.observability.print_trace <correlation_id-or-path>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.observability.tracer import LLM_SPAN_MARKER_ATTRS, TRACE_DIR


def _resolve_path(id_or_path: str) -> Path:
    p = Path(id_or_path)
    if p.exists():
        return p
    candidate = Path(TRACE_DIR) / f"{id_or_path}.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"no trace found for {id_or_path!r} (checked {p} and {candidate})")


def _is_llm_span(span: dict) -> bool:
    return any(k in span["attrs"] for k in LLM_SPAN_MARKER_ATTRS)


def _format_span(span: dict, depth: int) -> list[str]:
    indent = "  " * depth
    status_mark = "OK" if span["status"] == "ok" else "ERROR"
    dur = f"{span['duration_ms']:.1f}ms" if span["duration_ms"] is not None else "?"
    lines = [f"{indent}{span['name']} [{dur}] {status_mark}"]
    if span["status"] == "error":
        lines.append(f"{indent}  ! {span['error_type']}: {span['error_message']}")
    if _is_llm_span(span):
        a = span["attrs"]
        lines.append(f"{indent}  model={a.get('model')} tokens_in={a.get('tokens_in')} "
                      f"tokens_out={a.get('tokens_out')} cost_usd={a.get('cost_usd')}")
    elif span["attrs"]:
        lines.append(f"{indent}  " + ", ".join(f"{k}={v}" for k, v in span["attrs"].items()))
    for child in span["children"]:
        lines.extend(_format_span(child, depth + 1))
    return lines


def render(trace: dict) -> str:
    lines = [f"trace {trace['correlation_id']}"]
    for root in trace["spans"]:
        lines.extend(_format_span(root, 1))
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m src.observability.print_trace <correlation_id-or-path>", file=sys.stderr)
        return 1
    path = _resolve_path(sys.argv[1])
    trace = json.loads(path.read_text())
    print(render(trace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
