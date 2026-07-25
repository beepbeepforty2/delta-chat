import json

from src.delta.model import Delta
from src.delta.report import render_markdown, write_report

SAMPLE_DELTAS = [
    Delta("d1", "add", "note", None, "note27", 1, None, "A-3", {}, 1.0, "note added: FOO"),
    Delta("d2", "modify", "instrument", "inst00", "inst00", 1, "B-2", "B-2",
          {"loop": [9055, 9016]}, 0.9, "instrument loop changed: 9055 -> 9016"),
    Delta("d3", "modify", "instrument", "inst01", "inst01", 1, "B-3", "B-3",
          {"loop": [9056, 9017]}, 0.9, "instrument loop changed: 9056 -> 9017",
          is_cascade=True, primary_did="d2"),
]


def test_render_markdown_has_sections():
    md = render_markdown(SAMPLE_DELTAS, "pid_a", "pid_b")
    assert "# Delta Report: pid_a -> pid_b" in md
    assert "2 primary change(s), 1 cascade change(s)." in md
    assert "## Add (1)" in md
    assert "## Modify (1)" in md
    assert "+1 related cascade change(s)" in md


def test_render_markdown_no_changes():
    md = render_markdown([], "pid_a", "pid_b")
    assert "No changes detected." in md


def test_write_report(tmp_path):
    json_path, md_path = write_report(SAMPLE_DELTAS, "pid_a", "pid_b", str(tmp_path))
    data = json.loads(open(json_path).read())
    assert data["pid_a"] == "pid_a"
    assert len(data["deltas"]) == 3
    assert data["deltas"][1]["field_changes"] == {"loop": [9055, 9016]}
    assert "# Delta Report" in open(md_path).read()


def test_render_markdown_shows_severity_summary_and_orders_critical_first():
    d_low = Delta("d1", "modify", "note", "n1", "n1", 1, "A-1", "A-1",
                   {"body": ["old", "new"]}, 1.0, "note revised", severity="low")
    d_critical = Delta("d2", "modify", "instrument", "i1", "i1", 1, "B-2", "B-2",
                        {"setpoints": [{"HH": 1}, {"HH": 2}]}, 1.0,
                        "setpoint changed: 1 -> 2", severity="critical")
    md = render_markdown([d_low, d_critical], "pid_a", "pid_b")
    assert "critical=1" in md and "low=1" in md
    assert "[CRITICAL]" in md and "[LOW]" in md
    # within the "Modify" section, the critical item must be listed first
    modify_section = md.split("## Modify")[1]
    assert modify_section.index("[CRITICAL]") < modify_section.index("[LOW]")


def test_render_markdown_shows_visual_change_kind_hint_for_unclassified():
    from src.canonical.model import BBox
    d = Delta("raster0001", "unclassified_visual_change", "unclassified_visual_change",
              None, None, 1, "C-4", "C-4", {"tags": ["26GT9143"]}, 0.3,
              "graphical change near 26GT9143; not characterized by text engine",
              bbox_a=BBox(0.1, 0.1, 0.2, 0.2), bbox_b=BBox(0.1, 0.1, 0.2, 0.2),
              visual_change_kind="graphical")
    md = render_markdown([d], "pid_a", "pid_b")
    assert "## Unclassified Visual Changes (raster recall net) (1)" in md
    assert "[graphical]" in md
    assert "zone C-4" in md


def test_write_report_includes_severity_in_json(tmp_path):
    d = Delta("d1", "modify", "instrument", "i1", "i1", 1, "B-2", "B-2",
              {"setpoints": [{"HH": 1}, {"HH": 2}]}, 1.0, "setpoint changed",
              severity="critical")
    json_path, _ = write_report([d], "pid_a", "pid_b", str(tmp_path))
    data = json.loads(open(json_path).read())
    assert data["deltas"][0]["severity"] == "critical"
