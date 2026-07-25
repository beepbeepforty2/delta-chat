import json
import pathlib
import re

import pytest
from PIL import Image

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalSheet
from src.delta.model import Delta
from src.markup.html_report import render_html_report

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


def _el(id_, content, x0, y0, x1, y1, sheet=1, zone="A-1"):
    return CanonicalElement(id=id_, type="note", content=content, bbox=BBox(x0, y0, x1, y1),
                             sheet=sheet, zone=zone, extraction_confidence=1.0)


def _doc(pid, elements, raster_paths):
    return CanonicalDocument(pid=pid, source_format="pdf_native", revision_label=None,
                              sheets=[CanonicalSheet(number=1, width=1.0, height=1.0, elements=elements)],
                              raster_paths=raster_paths)


def _blank_png(path, size=(400, 300)):
    Image.new("RGB", size, (255, 255, 255)).save(path)
    return str(path)


def _extract_embedded(content: str, name: str) -> dict:
    m = re.search(rf"const {name} = (\{{.*?\}});\n", content, re.DOTALL)
    assert m, f"could not find embedded {name} in report"
    return json.loads(m.group(1))


def test_render_html_report_writes_a_single_file_with_no_leftover_placeholders(tmp_path):
    raster_a = _blank_png(tmp_path / "a1.png")
    raster_b = _blank_png(tmp_path / "b1.png")
    el_a = _el("a1", "old note", 0.1, 0.1, 0.3, 0.15)
    el_b = _el("b1", "new note", 0.1, 0.1, 0.3, 0.15)
    doc_a = _doc("A", [el_a], {1: raster_a})
    doc_b = _doc("B", [el_b], {1: raster_b})
    deltas = [Delta("d1", "modify", "note", "a1", "b1", 1, "A-1", "A-1", {"content": ["old", "new"]},
                     description="note text changed", severity="low")]

    out_path = tmp_path / "report.html"
    result = render_html_report(doc_a, doc_b, deltas, "pid_a.pdf", "pid_b.pdf", str(out_path))

    assert result == str(out_path)
    content = out_path.read_text()
    assert "__DATA_JSON__" not in content
    assert "__KIND_COLOR_JSON__" not in content
    assert "__KIND_LABEL_JSON__" not in content
    assert "__TITLE__" not in content
    assert "pid_a.pdf" in content and "pid_b.pdf" in content


def test_render_html_report_embeds_bbox_and_severity_for_a_modify_delta(tmp_path):
    raster_a = _blank_png(tmp_path / "a1.png")
    raster_b = _blank_png(tmp_path / "b1.png")
    el_a = _el("a1", "old note", 0.1, 0.1, 0.3, 0.15)
    el_b = _el("b1", "new note", 0.1, 0.1, 0.3, 0.15)
    doc_a = _doc("A", [el_a], {1: raster_a})
    doc_b = _doc("B", [el_b], {1: raster_b})
    deltas = [Delta("d1", "modify", "note", "a1", "b1", 1, "A-1", "B-2", {"content": ["old", "new"]},
                     description="note text changed", severity="medium", confidence=0.87)]

    out_path = tmp_path / "report.html"
    render_html_report(doc_a, doc_b, deltas, "A", "B", str(out_path))
    data = _extract_embedded(out_path.read_text(), "DATA")

    assert len(data["deltas"]) == 1
    rec = data["deltas"][0]
    assert rec["did"] == "d1"
    assert rec["kind"] == "modify"
    assert rec["severity"] == "medium"
    assert rec["confidence"] == 0.87
    assert rec["zone"] == "B-2"  # zone_b preferred, matching report.py's own convention
    assert rec["box_a"] == [0.1, 0.1, 0.3, 0.15]
    assert rec["box_b"] == [0.1, 0.1, 0.3, 0.15]
    assert data["summary"]["n_primary"] == 1
    assert data["summary"]["severity_counts"]["medium"] == 1


def test_render_html_report_add_delta_has_no_box_a(tmp_path):
    raster_a = _blank_png(tmp_path / "a1.png")
    raster_b = _blank_png(tmp_path / "b1.png")
    el_b = _el("b1", "new note", 0.6, 0.6, 0.8, 0.65)
    doc_a = _doc("A", [], {1: raster_a})
    doc_b = _doc("B", [el_b], {1: raster_b})
    deltas = [Delta("d1", "add", "note", None, "b1", 1, None, "A-1", {}, severity="low")]

    out_path = tmp_path / "report.html"
    render_html_report(doc_a, doc_b, deltas, "A", "B", str(out_path))
    data = _extract_embedded(out_path.read_text(), "DATA")

    rec = data["deltas"][0]
    assert rec["box_a"] is None
    assert rec["box_b"] is not None


def test_render_html_report_unclassified_visual_change_has_no_box_either_side(tmp_path):
    """A Delta with no id_a/id_b AND no bbox_a/bbox_b set (the shape this
    kind had before raster_join.py started carrying a real bbox) -- the
    report must still list it (with a "no exact location" note
    client-side), not silently drop it the way it would if it only
    iterated element-resolved boxes. See the sibling test below for the
    now-common case where bbox_a/bbox_b ARE set."""
    raster_a = _blank_png(tmp_path / "a1.png")
    raster_b = _blank_png(tmp_path / "b1.png")
    doc_a = _doc("A", [], {1: raster_a})
    doc_b = _doc("B", [], {1: raster_b})
    deltas = [Delta("d1", "unclassified_visual_change", "unclassified_visual_change", None, None,
                     1, "C-4", "C-4", {}, confidence=0.2,
                     description="unclassified visual change detected at zone C-4", severity="low")]

    out_path = tmp_path / "report.html"
    render_html_report(doc_a, doc_b, deltas, "A", "B", str(out_path))
    data = _extract_embedded(out_path.read_text(), "DATA")

    rec = data["deltas"][0]
    assert rec["kind"] == "unclassified_visual_change"
    assert rec["box_a"] is None and rec["box_b"] is None
    assert rec["zone"] == "C-4"


def test_render_html_report_unclassified_visual_change_uses_its_own_bbox(tmp_path):
    """The now-common case: raster_join.py sets bbox_a/bbox_b directly on
    the Delta (no CanonicalElement to resolve, but a real region was
    found) -- the report should use that as the box location, not fall
    back to "no exact location" just because there's no id_a/id_b."""
    raster_a = _blank_png(tmp_path / "a1.png")
    raster_b = _blank_png(tmp_path / "b1.png")
    doc_a = _doc("A", [], {1: raster_a})
    doc_b = _doc("B", [], {1: raster_b})
    deltas = [Delta("raster0001", "unclassified_visual_change", "unclassified_visual_change", None, None,
                     1, "C-4", "C-4", {"tags": ["26GT9143"]}, confidence=0.3,
                     description="graphical change near 26GT9143; not characterized by text engine",
                     bbox_a=BBox(0.4, 0.4, 0.6, 0.6), bbox_b=BBox(0.4, 0.4, 0.6, 0.6),
                     visual_change_kind="graphical")]

    out_path = tmp_path / "report.html"
    render_html_report(doc_a, doc_b, deltas, "A", "B", str(out_path))
    data = _extract_embedded(out_path.read_text(), "DATA")

    rec = data["deltas"][0]
    assert rec["box_a"] == [0.4, 0.4, 0.6, 0.6]
    assert rec["box_b"] == [0.4, 0.4, 0.6, 0.6]
    assert rec["visual_change_kind"] == "graphical"


def test_render_html_report_cascade_and_semantic_null_flags_survive_to_the_record(tmp_path):
    raster_a = _blank_png(tmp_path / "a1.png")
    raster_b = _blank_png(tmp_path / "b1.png")
    el_a = _el("a1", "5. DELETED.", 0.1, 0.1, 0.3, 0.15)
    el_b = _el("b1", "5-6. DELETED.", 0.1, 0.1, 0.3, 0.15)
    doc_a = _doc("A", [el_a], {1: raster_a})
    doc_b = _doc("B", [el_b], {1: raster_b})
    deltas = [
        Delta("d1", "modify", "note", "a1", "b1", 1, "A-1", "A-1", {"range": ["5", "5-6"]},
              severity="low", semantic_null=True, semantic_null_reason="DELETED-placeholder collapse"),
        Delta("d2", "modify", "note", "a1", "b1", 1, "A-1", "A-1", {},
              severity="low", is_cascade=True, primary_did="d1"),
    ]

    out_path = tmp_path / "report.html"
    render_html_report(doc_a, doc_b, deltas, "A", "B", str(out_path))
    data = _extract_embedded(out_path.read_text(), "DATA")

    by_did = {r["did"]: r for r in data["deltas"]}
    assert by_did["d1"]["semantic_null"] is True
    assert by_did["d1"]["semantic_null_reason"] == "DELETED-placeholder collapse"
    assert by_did["d2"]["is_cascade"] is True
    assert by_did["d2"]["primary_did"] == "d1"
    assert data["summary"]["n_cascade"] == 1
    # semantic_null delta is still a primary, non-cascade delta -- counted
    # as such (it's the eval harness's job to treat it as a soft FP, not
    # this report's -- this report only needs to surface the flag).
    assert data["summary"]["n_primary"] == 1


def test_render_html_report_multi_sheet_document(tmp_path):
    raster_a1 = _blank_png(tmp_path / "a1.png")
    raster_a2 = _blank_png(tmp_path / "a2.png")
    raster_b1 = _blank_png(tmp_path / "b1.png")
    doc_a = CanonicalDocument(
        pid="A", source_format="pdf_native", revision_label=None,
        sheets=[
            CanonicalSheet(number=1, width=1.0, height=1.0, elements=[]),
            CanonicalSheet(number=2, width=1.0, height=1.0, elements=[]),
        ],
        raster_paths={1: raster_a1, 2: raster_a2},
    )
    doc_b = _doc("B", [], {1: raster_b1})

    out_path = tmp_path / "report.html"
    render_html_report(doc_a, doc_b, [], "A", "B", str(out_path))
    data = _extract_embedded(out_path.read_text(), "DATA")

    sheet_nos = {s["number"] for s in data["sheets"]}
    assert sheet_nos == {1, 2}
    sheet2 = next(s for s in data["sheets"] if s["number"] == 2)
    assert sheet2["img_a"] is not None
    assert sheet2["img_b"] is None  # doc_b has no sheet 2 -- client shows "no page 2 in B"


def test_render_html_report_real_pair_end_to_end(tmp_path):
    """Live check against a real generated pair and the real engine, same
    pattern as test_markup_overlay.py's own end-to-end test -- confirms
    real Delta objects, real raster paths, and real bbox values all
    survive through to the embedded report data."""
    from src.cli import _resolve_with_pid, compute_deltas
    from src.observability.tracer import Tracer

    pair_dir = PAIRS_DIR / "edited_003"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")
    doc_a = _resolve_with_pid("A", str(pair_dir / "a" / "L0.pdf"))
    doc_b = _resolve_with_pid("B", str(pair_dir / "b" / "L0.pdf"))
    tracer = Tracer()
    deltas = compute_deltas(doc_a, doc_b, tracer)
    tracer.finish()
    assert deltas  # sanity: this pair does have real changes

    out_path = tmp_path / "report.html"
    render_html_report(doc_a, doc_b, deltas, str(pair_dir / "a" / "L0.pdf"),
                        str(pair_dir / "b" / "L0.pdf"), str(out_path))
    assert out_path.exists()
    content = out_path.read_text()
    data = _extract_embedded(content, "DATA")
    assert len(data["deltas"]) == len(deltas)
    assert data["sheets"]  # at least one sheet with an inlined raster
    # every non-cascade delta should carry a resolvable severity string
    assert all(r["severity"] in ("critical", "high", "medium", "low") for r in data["deltas"])
