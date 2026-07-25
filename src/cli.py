"""CLI entrypoint: python -m src.cli <subcommand>. See Makefile for usage."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.canonical.model import CanonicalDocument
from src.chat.chat import answer as chat_answer
from src.chat.retrieval import BM25Index, build_chunks
from src.delta.align import match_elements, match_sheets
from src.delta.classify import classify_matches
from src.delta.model import Delta
from src.delta.precheck import check_same_document
from src.delta.raster_recall import detect_unclassified_visual_changes
from src.delta.register import register
from src.delta.report import write_report
from src.delta.semantic_null import annotate_semantic_null
from src.delta.severity import annotate_severity
from src.ingest.base import FormatAdapter
from src.markup.html_report import render_html_report
from src.markup.overlay import render_markup
from src.markup.pdf_annotate import render_pdf_markup
from src.ingest.dwg import DWGAdapter
from src.ingest.pdf_native import PdfNativeAdapter
from src.ingest.pdf_scanned import PdfScannedAdapter
from src.observability.tracer import Tracer

# Order matters: PdfScannedAdapter.detect() is the deliberate complement of
# PdfNativeAdapter.detect() (sparse-vs-rich text layer, same threshold), so
# native must be tried first for a document that could pass either check to
# resolve to the more accurate (deterministic, extraction_confidence=1.0)
# adapter.
ADAPTERS: list[FormatAdapter] = [PdfNativeAdapter(), PdfScannedAdapter(), DWGAdapter()]


def _resolve_with_pid(pid: str, path: str) -> CanonicalDocument:
    for adapter in ADAPTERS:
        if adapter.detect(path):
            return adapter.ingest(pid, path)
    raise ValueError(f"no adapter claims {path}")


def compute_deltas(doc_a: CanonicalDocument, doc_b: CanonicalDocument, tracer: Tracer,
                    semantic_null_call_llm=None) -> list[Delta]:
    with tracer.span("register") as s:
        transform = register(doc_a, doc_b)
        s.set("scale", transform.scale)

    deltas: list[Delta] = []
    with tracer.span("align") as s:
        sheet_pairs = match_sheets(doc_a, doc_b)
        all_matches = []
        for sheet_a, sheet_b in sheet_pairs:
            with tracer.span("align_sheet", sheet=(sheet_a or sheet_b).number) as s_sheet:
                matches = match_elements(sheet_a, sheet_b, transform)
                s_sheet.set("n_matches", len(matches))
                all_matches.append(((sheet_a or sheet_b).number, matches))
        s.set("n_sheets", len(sheet_pairs))

    with tracer.span("classify") as s:
        for sheet_no, matches in all_matches:
            deltas.extend(classify_matches(matches, sheet=sheet_no))
        primary = sum(1 for d in deltas if not d.is_cascade)
        cascade = sum(1 for d in deltas if d.is_cascade)
        by_kind: dict[str, int] = {}
        for d in deltas:
            if not d.is_cascade:
                by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
        s.set("primary_count", primary)
        s.set("cascade_count", cascade)
        s.set("by_kind", by_kind)

    with tracer.span("semantic_null") as s:
        # The rule half (4a, no LLM) always runs; the LLM half (4b) only
        # when DELTA_SEMANTIC_NULL_LLM=1 -- see src/delta/semantic_null.py.
        # This is the one place a live LLM call can sit in an otherwise
        # fully deterministic path, so it's opt-in and explicit.
        annotate_semantic_null(deltas, call_llm=semantic_null_call_llm)
        s.set("n_flagged", sum(1 for d in deltas if d.semantic_null))
        s.set("llm_enabled", os.environ.get("DELTA_SEMANTIC_NULL_LLM") == "1")

    with tracer.span("raster_recall") as s:
        # Opt-in (DELTA_RASTER_RECALL=1), no-op otherwise -- see
        # src/delta/raster_recall.py. Confidence-gated: never fires on a
        # native-native pair (extraction_confidence is always 1.0 there).
        recall_deltas = detect_unclassified_visual_changes(doc_a, doc_b, deltas, transform)
        deltas.extend(recall_deltas)
        if recall_deltas:
            # These deltas didn't exist during the classify span's
            # annotate_severity() call above -- re-run it (idempotent,
            # pure function of kind/element_type/field_changes) so every
            # delta in the report has a severity, not just the ones that
            # existed before this pass.
            annotate_severity(deltas)
        s.set("n_detected", len(recall_deltas))
        s.set("enabled", os.environ.get("DELTA_RASTER_RECALL") == "1")

    return deltas


def cmd_run(args: argparse.Namespace) -> int:
    tracer = Tracer()
    try:
        with tracer.span("request", pid_a=args.a, pid_b=args.b):
            with tracer.span("ingest"):
                with tracer.span("ingest_a", path=args.a) as s:
                    doc_a = _resolve_with_pid("A", args.a)
                    s.set("n_elements", sum(len(sh.elements) for sh in doc_a.sheets))
                with tracer.span("ingest_b", path=args.b) as s:
                    doc_b = _resolve_with_pid("B", args.b)
                    s.set("n_elements", sum(len(sh.elements) for sh in doc_b.sheets))

            with tracer.span("precheck") as s:
                precheck = check_same_document(doc_a, doc_b)
                s.set("is_pair", precheck.is_pair)
                s.set("reason", precheck.reason)

            if not precheck.is_pair:
                print(f"REFUSED: {precheck.reason}", file=sys.stderr)
                print(f"  A: drawno={precheck.drawing_no_a!r} equipment={precheck.equipment_a!r}", file=sys.stderr)
                print(f"  B: drawno={precheck.drawing_no_b!r} equipment={precheck.equipment_b!r}", file=sys.stderr)
                return 1
            if "no drawing number" in precheck.reason:
                print(f"WARNING: {precheck.reason}", file=sys.stderr)

            deltas = compute_deltas(doc_a, doc_b, tracer)

            with tracer.span("report") as s:
                json_path, md_path = write_report(deltas, args.a, args.b, args.out)
                s.set("json_path", json_path)
                s.set("md_path", md_path)

            if args.html:
                # Opt-in end-user report: a self-contained HTML file with
                # annotated pages + a filterable delta list, distinct from
                # `markup`'s output (real PDF annotations / PNG preview) --
                # this is the "show me in a webpage" view, off by default
                # so a normal run stays report.json/.md only.
                with tracer.span("html_report") as s:
                    html_path = render_html_report(doc_a, doc_b, deltas, args.a, args.b,
                                                    str(Path(args.out) / "report.html"))
                    s.set("html_path", html_path)
                    print(f"wrote {html_path}")

            primary = sum(1 for d in deltas if not d.is_cascade)
            cascade = sum(1 for d in deltas if d.is_cascade)
            by_kind: dict[str, int] = {}
            for d in deltas:
                if not d.is_cascade:
                    by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
            print(f"{primary} primary change(s), {cascade} cascade change(s): {by_kind}")
            print(f"wrote {json_path}")
            print(f"wrote {md_path}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
    finally:
        trace_path = tracer.finish()
        print(f"trace: {tracer.correlation_id} ({trace_path})", file=sys.stderr)


def _ask(question: str, index: BM25Index, tracer: Tracer) -> None:
    result = chat_answer(question, index, tracer=tracer)
    print(f"Q: {question}")
    if result.refused:
        print(f"REFUSED: {result.reason}")
    else:
        print(f"A: {result.text}")
    print()


def cmd_chat(args: argparse.Namespace) -> int:
    tracer = Tracer()
    try:
        with tracer.span("request", pid_a=args.a, pid_b=args.b, mode="chat"):
            with tracer.span("ingest"):
                with tracer.span("ingest_a", path=args.a) as s:
                    doc_a = _resolve_with_pid("A", args.a)
                    s.set("n_elements", sum(len(sh.elements) for sh in doc_a.sheets))
                with tracer.span("ingest_b", path=args.b) as s:
                    doc_b = _resolve_with_pid("B", args.b)
                    s.set("n_elements", sum(len(sh.elements) for sh in doc_b.sheets))

            with tracer.span("precheck") as s:
                precheck = check_same_document(doc_a, doc_b)
                s.set("is_pair", precheck.is_pair)
                s.set("reason", precheck.reason)

            if not precheck.is_pair:
                print(f"REFUSED: {precheck.reason}", file=sys.stderr)
                return 1

            deltas = compute_deltas(doc_a, doc_b, tracer)

            with tracer.span("build_index") as s:
                chunks = build_chunks(doc_a, doc_b, deltas)
                index = BM25Index(chunks)
                s.set("n_chunks", len(chunks))

            print(f"Indexed {len(chunks)} chunks (PID A, PID B, {len(deltas)} deltas). "
                  f"Ask a question ('quit' to exit).", file=sys.stderr)

            if args.question:
                for q in args.question:
                    _ask(q, index, tracer)
            else:
                for line in sys.stdin:
                    q = line.strip()
                    if not q or q.lower() in ("quit", "exit"):
                        break
                    _ask(q, index, tracer)
        return 0
    finally:
        trace_path = tracer.finish()
        print(f"trace: {tracer.correlation_id} ({trace_path})", file=sys.stderr)


def cmd_markup(args: argparse.Namespace) -> int:
    tracer = Tracer()
    try:
        with tracer.span("request", pid_a=args.a, pid_b=args.b, mode="markup"):
            with tracer.span("ingest"):
                with tracer.span("ingest_a", path=args.a) as s:
                    doc_a = _resolve_with_pid("A", args.a)
                    s.set("n_elements", sum(len(sh.elements) for sh in doc_a.sheets))
                with tracer.span("ingest_b", path=args.b) as s:
                    doc_b = _resolve_with_pid("B", args.b)
                    s.set("n_elements", sum(len(sh.elements) for sh in doc_b.sheets))

            with tracer.span("precheck") as s:
                precheck = check_same_document(doc_a, doc_b)
                s.set("is_pair", precheck.is_pair)
                s.set("reason", precheck.reason)

            if not precheck.is_pair:
                print(f"REFUSED: {precheck.reason}", file=sys.stderr)
                return 1

            deltas = compute_deltas(doc_a, doc_b, tracer)

            with tracer.span("markup", format=args.format) as s:
                if args.format == "png":
                    # Raster preview: fast to inspect (Read a PNG directly,
                    # as used throughout development) but not real PDF
                    # annotations -- see src/markup/overlay.py's docstring.
                    paths_a, paths_b = render_markup(doc_a, doc_b, deltas, args.out)
                    s.set("n_sheets_a", len(paths_a))
                    s.set("n_sheets_b", len(paths_b))
                    for _sheet_no, path in sorted(paths_a.items()):
                        print(f"wrote {path}")
                    for _sheet_no, path in sorted(paths_b.items()):
                        print(f"wrote {path}")
                else:
                    # Default: real PDF annotation objects (page.add_rect_annot)
                    # on the original PDFs -- appear in Acrobat's/Bluebeam's
                    # markup list, toggleable, never rasterize source content.
                    out_a, out_b = render_pdf_markup(doc_a, doc_b, deltas, args.a, args.b, args.out)
                    s.set("out_a", out_a)
                    s.set("out_b", out_b)
                    print(f"wrote {out_a}")
                    print(f"wrote {out_b}")
        return 0
    finally:
        trace_path = tracer.finish()
        print(f"trace: {tracer.correlation_id} ({trace_path})", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m src.cli")
    sub = ap.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="ingest a pair and produce a delta report")
    run_p.add_argument("--a", required=True)
    run_p.add_argument("--b", required=True)
    run_p.add_argument("--out", default="reports/")
    run_p.add_argument("--html", action="store_true",
                        help="also write an interactive report.html (annotated pages + "
                             "filterable delta list) alongside the json/md report")
    run_p.set_defaults(func=cmd_run)

    chat_p = sub.add_parser("chat", help="grounded chat over PID A + PID B + delta report")
    chat_p.add_argument("--a", required=True)
    chat_p.add_argument("--b", required=True)
    chat_p.add_argument("--question", action="append",
                         help="ask one question and exit (repeatable); omit for interactive stdin")
    chat_p.set_defaults(func=cmd_chat)

    markup_p = sub.add_parser("markup", help="render delta annotations onto each revision (bonus)")
    markup_p.add_argument("--a", required=True)
    markup_p.add_argument("--b", required=True)
    markup_p.add_argument("--out", default="reports/markup")
    markup_p.add_argument("--format", choices=["pdf", "png"], default="pdf",
                           help="pdf: real PDF annotation objects (default, Acrobat/Bluebeam-compatible); "
                                "png: raster preview, quick to inspect but not a real annotation")
    markup_p.set_defaults(func=cmd_markup)

    args = ap.parse_args()
    try:
        return args.func(args)
    except Exception as e:
        if args.command != "run":  # cmd_run already printed its own ERROR line
            print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
