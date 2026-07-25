"""End-to-end CLI integration test for ``markup``: mirrors tests/test_cli_run.py
and tests/test_cli_chat.py. Exercises both --format branches (pdf = real
annotation objects, png = raster preview) through the real cmd_markup and the
shared pipeline, asserting the output files exist and the trace shape is
correct. Closes the review-noted gap that only cmd_run was covered end-to-end."""
import json
import pathlib

import pytest

from src.cli import cmd_markup
from src.observability import tracer as tracer_mod

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


class _Args:
    def __init__(self, a, b, out, fmt="pdf"):
        self.a, self.b, self.out, self.format = a, b, out, fmt


@pytest.mark.skipif(not (PAIRS_DIR / "edited_002").exists(),
                    reason="run `make dataset` first")
def test_markup_pdf_writes_annotated_pdfs_for_both_revisions(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    monkeypatch.setattr(tracer_mod, "TRACE_DIR", str(trace_dir))

    pair_dir = PAIRS_DIR / "edited_002"
    out_dir = tmp_path / "markup"
    args = _Args(str(pair_dir / "a" / "L0.pdf"), str(pair_dir / "b" / "L0.pdf"),
                 str(out_dir), fmt="pdf")
    rc = cmd_markup(args)
    assert rc == 0

    # render_pdf_markup returns/writes one annotated PDF per revision; assert
    # at least one file landed under out_dir for each side.
    pdfs = sorted(pathlib.Path(out_dir).glob("*.pdf"))
    assert len(pdfs) >= 2, f"expected >=2 annotated PDFs, got {pdfs}"
    # Each must be a real, non-empty PDF (magic bytes %PDF).
    for p in pdfs:
        assert p.read_bytes()[:4] == b"%PDF", f"{p.name} is not a valid PDF"

    trace_files = list(trace_dir.glob("*.json"))
    assert len(trace_files) == 1
    trace = json.loads(trace_files[0].read_text())
    root = trace["spans"][0]
    assert root["name"] == "request"
    assert root["attrs"].get("mode") == "markup"
    child_names = {c["name"] for c in root["children"]}
    assert {"ingest", "precheck", "markup"} <= child_names
    assert root["status"] == "ok"


@pytest.mark.skipif(not (PAIRS_DIR / "edited_002").exists(),
                    reason="run `make dataset` first")
def test_markup_png_writes_raster_previews_for_both_revisions(tmp_path, monkeypatch):
    """The --format png branch: render_markup writes one PNG per sheet per
    revision (raster preview, not real annotations). Distinct code path from
    the pdf branch above -- both must be exercised."""
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("TRACE_DIR", str(trace_dir))
    monkeypatch.setattr(tracer_mod, "TRACE_DIR", str(trace_dir))

    pair_dir = PAIRS_DIR / "edited_002"
    out_dir = tmp_path / "markup_png"
    args = _Args(str(pair_dir / "a" / "L0.pdf"), str(pair_dir / "b" / "L0.pdf"),
                 str(out_dir), fmt="png")
    rc = cmd_markup(args)
    assert rc == 0

    pngs = sorted(pathlib.Path(out_dir).glob("*.png"))
    assert len(pngs) >= 2, f"expected >=2 PNG previews, got {pngs}"
    for p in pngs:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{p.name} is not a valid PNG"


def test_markup_refuses_not_a_pair(tmp_path, monkeypatch, capsys):
    pair_dir = PAIRS_DIR / "not_a_pair_903"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")

    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setattr(tracer_mod, "TRACE_DIR", str(tmp_path / "traces"))

    args = _Args(str(pair_dir / "a" / "L0.pdf"), str(pair_dir / "b" / "L0.pdf"),
                 str(tmp_path / "out"), fmt="pdf")
    rc = cmd_markup(args)
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err
    # And no markup output should have been produced for a refused pair.
    assert not list(pathlib.Path(tmp_path / "out").glob("*.pdf"))
