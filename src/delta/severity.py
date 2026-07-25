"""Severity ranking (HIGH -> LOW): a rule-based, deterministic classifier
over Delta.kind/element_type/field_changes -- deliberately NOT an LLM
judgment call. Severity here is about *what kind of field changed on what
kind of element*, which is exactly the kind of structured, rule-derivable
fact DESIGN.md decision #3 keeps in the deterministic engine rather than
handing to the LLM (whose only roles are description prose, chat answers,
and semantic-equivalence adjudication).

No eval ground truth exists for this (gt/deltas.json has no severity
field -- the dataset generator predates this capability), so it isn't
scored through eval/metrics.py's P/R/F1 machinery; it's spot-checked
against known real deltas in tests/test_delta_severity.py instead. Kept as
its own module rather than folded into classify.py so the rule table reads
as one thing.

Rules are ordered by priority (first match wins), narrowest
safety-critical case first:
  - a cascade member (renumbering side-effect, no independent engineering
    content -- see classify.py's cascade detection) is always LOW
    regardless of what element type it's on
  - a MOVE (position only, content unchanged) is always LOW
  - a trip/alarm SETPOINT change on an instrument is CRITICAL -- directly
    safety-instrumented (decision #4's core P&ID concern: HH/LL on a
    bubble like PDIT-9017)
  - a PIPE_CLASS change on a line tag is HIGH -- a pressure/material
    rating change, mechanical integrity, not administrative
  - an equipment-tag identity change, or a process/safety element
    (instrument/line/valve/nozzle/equipment tag) appearing or
    disappearing, is HIGH
  - any other field-level engineering change on a process element or a
    datasheet spec value (duty, flow rate, design pressure) is MEDIUM
  - administrative/documentation elements (notes, DCN references,
    revision table rows, title fields) default to LOW
"""
from __future__ import annotations

from typing import Literal

from src.delta.model import Delta

Severity = Literal["critical", "high", "medium", "low"]

SEVERITY_ORDER: dict[str, int] = {"critical": 3, "high": 2, "medium": 1, "low": 0}

_PROCESS_ELEMENT_TYPES = {"instrument", "line_tag", "valve_tag", "nozzle", "equipment_tag"}


def classify_severity(delta: Delta) -> Severity:
    if delta.is_cascade:
        return "low"
    if delta.kind == "move":
        return "low"

    if delta.kind == "modify":
        if delta.element_type == "instrument" and "setpoints" in delta.field_changes:
            return "critical"
        if delta.element_type == "line_tag" and "pipe_class" in delta.field_changes:
            return "high"
        if delta.element_type == "equipment_tag":
            return "high"
        if delta.element_type in _PROCESS_ELEMENT_TYPES or delta.element_type == "datasheet_row":
            return "medium"
        return "low"

    if delta.kind in ("add", "remove"):
        if delta.element_type in _PROCESS_ELEMENT_TYPES:
            return "high"
        if delta.element_type == "datasheet_row":
            return "medium"
        return "low"

    return "low"


def annotate_severity(deltas: list[Delta]) -> None:
    """Mutates in place. Must run after cascade detection -- severity
    depends on the final is_cascade flag, which classify.py only settles
    once every delta in the batch has been produced."""
    for d in deltas:
        d.severity = classify_severity(d)
