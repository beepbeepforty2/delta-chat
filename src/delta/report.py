"""Delta report emission: JSON (machine-readable) + Markdown (human-readable).

`description` fields are the deterministic placeholders from classify.py;
CLAUDE.md decision #3 reserves human-readable description authoring as one
of the LLM's roles, layered on top of this report later, not required for
the deltas themselves to be correct or reportable now.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.delta.model import Delta
from src.delta.severity import SEVERITY_ORDER

KIND_ORDER = ("add", "remove", "modify", "move", "unclassified_visual_change")
_KIND_LABELS = {"unclassified_visual_change": "Unclassified Visual Changes (raster recall net)"}


def render_markdown(deltas: list[Delta], pid_a: str, pid_b: str) -> str:
    primary = [d for d in deltas if not d.is_cascade]
    cascade = [d for d in deltas if d.is_cascade]

    lines = [f"# Delta Report: {pid_a} -> {pid_b}", ""]
    lines.append(f"{len(primary)} primary change(s), {len(cascade)} cascade change(s).")

    severity_counts = {sev: sum(1 for d in primary if d.severity == sev) for sev in SEVERITY_ORDER}
    if any(severity_counts.values()):
        lines.append("Severity: " + ", ".join(
            f"{sev}={n}" for sev, n in severity_counts.items() if n
        ))
    lines.append("")

    by_kind: dict[str, list[Delta]] = {}
    for d in primary:
        by_kind.setdefault(d.kind, []).append(d)

    for kind in KIND_ORDER:
        group = by_kind.get(kind, [])
        if not group:
            continue
        # Most severe first within a kind -- a human reviewer scanning the
        # report sees safety-critical changes (setpoints, pipe class)
        # before administrative ones (notes, DCN references) of the same kind.
        group = sorted(group, key=lambda d: -SEVERITY_ORDER.get(d.severity, 0))
        label = _KIND_LABELS.get(kind, kind.capitalize())
        lines.append(f"## {label} ({len(group)})")
        for d in group:
            loc = d.zone_b or d.zone_a or "?"
            sev_tag = f"[{d.severity.upper()}] " if d.severity else ""
            lines.append(f"- {sev_tag}**Sheet {d.sheet}, zone {loc}** (confidence {d.confidence:.2f}): {d.description}")
            children = [c for c in cascade if c.primary_did == d.did]
            if children:
                lines.append(f"  - +{len(children)} related cascade change(s)")
        lines.append("")

    if not primary:
        lines.append("No changes detected.")

    return "\n".join(lines)


def write_report(deltas: list[Delta], pid_a: str, pid_b: str, out_dir: str) -> tuple[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "delta_report.json"
    json_path.write_text(json.dumps(
        {"pid_a": pid_a, "pid_b": pid_b, "deltas": [d.to_dict() for d in deltas]},
        indent=2,
    ))

    md_path = out / "delta_report.md"
    md_path.write_text(render_markdown(deltas, pid_a, pid_b))

    return str(json_path), str(md_path)
