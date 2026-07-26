"""The payload is shared by the offline report and the web API. These
tests exist to make that sharing load-bearing rather than incidental --
the first one would fail if html_report.py ever grew its own private
serializer again, which is exactly the drift the refactor prevents."""
import json
import pathlib
import re

import pytest
from PIL import Image

from src.canonical.model import (
    BBox,
    CanonicalDocument,
    CanonicalElement,
    CanonicalSheet,
)
from src.delta.model import Delta
from src.markup.html_report import render_html_report
from src.markup.payload import KIND_LABELS, build_payload, kind_color_map

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


def _el(id_, content, x0, y0, x1, y1, sheet=1, zone="A-1"):
    return CanonicalElement(id=id_, type="note", content=content, bbox=BBox(x0, y0, x1, y1),
                             sheet=sheet, zone=zone, extraction_confidence=1.0)


def _doc(pid, elements, raster_paths, n_sheets=1):
    sheets = [CanonicalSheet(number=n, width=1.0, height=1.0,
                              elements=[e for e in elements if e.sheet == n])
              for n in range(1, n_sheets + 1)]
    return CanonicalDocument(pid=pid, source_format="pdf_native", revision_label=None,
                              sheets=sheets, raster_paths=raster_paths)


def _blank_png(path, size=(400, 300)):
    Image.new("RGB", size, (255, 255, 255)).save(path)
    return str(path)


def _extract_embedded(content: str, name: str) -> dict:
    m = re.search(rf"const {name} = (\{{.*?\}});\n", content, re.DOTALL)
    assert m, f"could not find embedded {name} in report"
    return json.loads(m.group(1))


@pytest.fixture
def pair(tmp_path):
    raster_a = _blank_png(tmp_path / "a1.png")
    raster_b = _blank_png(tmp_path / "b1.png")
    doc_a = _doc("A", [_el("a1", "old note", 0.1, 0.1, 0.3, 0.15)], {1: raster_a})
    doc_b = _doc("B", [_el("b1", "new note", 0.1, 0.1, 0.3, 0.15)], {1: raster_b})
    deltas = [Delta("d1", "modify", "note", "a1", "b1", 1, "A-1", "A-1",
                     {"content": ["old", "new"]},
                     description="note text changed", severity="low")]
    return doc_a, doc_b, deltas


def test_report_html_embeds_exactly_the_shared_payload(tmp_path, pair):
    """The welding test. report.html's __DATA_JSON__ IS build_payload's
    output -- not a lookalike -- so the web API and the downloadable file
    cannot drift as Delta gains fields."""
    doc_a, doc_b, deltas = pair
    out_path = tmp_path / "report.html"
    render_html_report(doc_a, doc_b, deltas, "pid_a.pdf", "pid_b.pdf", str(out_path))

    embedded = _extract_embedded(out_path.read_text(), "DATA")
    expected = build_payload(doc_a, doc_b, deltas, "pid_a.pdf", "pid_b.pdf", inline_images=True)

    assert embedded == expected


def test_report_html_embeds_the_shared_kind_vocabulary(tmp_path, pair):
    doc_a, doc_b, deltas = pair
    out_path = tmp_path / "report.html"
    render_html_report(doc_a, doc_b, deltas, "A", "B", str(out_path))
    content = out_path.read_text()

    assert _extract_embedded(content, "KIND_COLOR") == kind_color_map()
    assert _extract_embedded(content, "KIND_LABEL") == KIND_LABELS


def test_inline_images_false_drops_base64_but_keeps_everything_else(pair):
    """The web client fetches PDFs and renders with pdf.js, so inlining
    megabytes it will never draw is pure first-paint cost. Only `sheets`
    may differ between the two modes."""
    doc_a, doc_b, deltas = pair
    web = build_payload(doc_a, doc_b, deltas, "A", "B", inline_images=False)
    offline = build_payload(doc_a, doc_b, deltas, "A", "B", inline_images=True)

    assert web["deltas"] == offline["deltas"]
    assert web["summary"] == offline["summary"]
    assert web["pid_a"] == offline["pid_a"] and web["pid_b"] == offline["pid_b"]

    assert web["sheets"] == [{"number": 1, "has_a": True, "has_b": True}]
    assert "img_a" not in web["sheets"][0]
    assert offline["sheets"][0]["img_a"] is not None


def test_sheets_are_the_union_so_a_one_sided_sheet_still_gets_a_tab(tmp_path):
    """A sheet added or removed between revisions exists on one side only.
    Hiding it would hide the change itself."""
    raster_a1 = _blank_png(tmp_path / "a1.png")
    raster_a2 = _blank_png(tmp_path / "a2.png")
    raster_b1 = _blank_png(tmp_path / "b1.png")
    doc_a = _doc("A", [], {1: raster_a1, 2: raster_a2}, n_sheets=2)
    doc_b = _doc("B", [], {1: raster_b1})

    payload = build_payload(doc_a, doc_b, [], "A", "B", inline_images=False)

    assert payload["sheets"] == [
        {"number": 1, "has_a": True, "has_b": True},
        {"number": 2, "has_a": True, "has_b": False},
    ]


def test_summary_counts_primary_only(tmp_path):
    """Cascade deltas are renumbering side-effects of a primary change.
    Counting them as independent findings inflates every total a reviewer
    sees -- report.py's markdown headline makes the same distinction."""
    raster = _blank_png(tmp_path / "a1.png")
    doc_a = _doc("A", [], {1: raster})
    doc_b = _doc("B", [], {1: raster})
    deltas = [
        Delta("d1", "modify", "instrument", None, None, 1, "A-1", "A-1", severity="critical"),
        Delta("d2", "modify", "note", None, None, 1, "A-1", "A-1", severity="low",
              is_cascade=True, primary_did="d1"),
        Delta("d3", "modify", "note", None, None, 1, "A-1", "A-1", severity="low",
              semantic_null=True),
    ]

    summary = build_payload(doc_a, doc_b, deltas, "A", "B", inline_images=False)["summary"]

    assert summary["n_primary"] == 2
    assert summary["n_cascade"] == 1
    assert summary["severity_counts"]["critical"] == 1
    assert summary["n_semantic_null"] == 1
    # ...but the cascade delta still has a record, so the UI can reveal it.
    assert len(build_payload(doc_a, doc_b, deltas, "A", "B", inline_images=False)["deltas"]) == 3


def test_unclassified_visual_change_falls_back_to_its_own_bbox(tmp_path):
    """raster_join.py finds pixels, not elements, so these deltas have no
    id_a/id_b and must resolve through Delta.bbox_a/bbox_b instead."""
    raster = _blank_png(tmp_path / "a1.png")
    doc_a = _doc("A", [], {1: raster})
    doc_b = _doc("B", [], {1: raster})
    deltas = [Delta("d1", "unclassified_visual_change", "geometry", None, None, 1, "C-4", "C-4",
                     bbox_a=BBox(0.2, 0.3, 0.4, 0.5), bbox_b=BBox(0.2, 0.3, 0.4, 0.5),
                     visual_change_kind="graphical", severity="low")]

    record = build_payload(doc_a, doc_b, deltas, "A", "B", inline_images=False)["deltas"][0]

    assert record["box_a"] == [0.2, 0.3, 0.4, 0.5]
    assert record["box_b"] == [0.2, 0.3, 0.4, 0.5]
    assert record["visual_change_kind"] == "graphical"


def test_delta_with_no_resolvable_location_gets_null_boxes_not_dropped(tmp_path):
    raster = _blank_png(tmp_path / "a1.png")
    doc_a = _doc("A", [], {1: raster})
    doc_b = _doc("B", [], {1: raster})
    deltas = [Delta("d1", "modify", "note", "missing_a", "missing_b", 1, "A-1", "A-1",
                     severity="low")]

    records = build_payload(doc_a, doc_b, deltas, "A", "B", inline_images=False)["deltas"]

    assert len(records) == 1
    assert records[0]["box_a"] is None and records[0]["box_b"] is None


@pytest.mark.skipif(not (PAIRS_DIR / "edited_001").exists(),
                    reason="run `make dataset` to generate the seeded eval pairs")
def test_payload_on_a_real_pair_end_to_end(tmp_path):
    """Same invariant, but through the real ingest + compute_deltas path
    rather than hand-built fixtures."""
    from src.cli import _resolve_with_pid, compute_deltas
    from src.observability.tracer import Tracer

    pair_dir = PAIRS_DIR / "edited_001"
    doc_a = _resolve_with_pid("A", str(pair_dir / "a" / "L0.pdf"))
    doc_b = _resolve_with_pid("B", str(pair_dir / "b" / "L0.pdf"))
    tracer = Tracer(trace_dir=str(tmp_path / "traces"))
    deltas = compute_deltas(doc_a, doc_b, tracer)

    payload = build_payload(doc_a, doc_b, deltas, "a.pdf", "b.pdf", inline_images=False)

    assert len(payload["deltas"]) == len(deltas)
    assert payload["summary"]["n_primary"] == sum(1 for d in deltas if not d.is_cascade)
    # Every record must be JSON-serializable -- it goes over the wire as-is.
    json.dumps(payload)
    for rec in payload["deltas"]:
        for box in (rec["box_a"], rec["box_b"]):
            if box is not None:
                assert len(box) == 4
                assert all(0.0 <= v <= 1.0 for v in box), f"{rec['did']} box not normalized: {box}"
