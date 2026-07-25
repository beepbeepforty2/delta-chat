#!/usr/bin/env python3
"""Build a held-out revision pair from a REAL P&ID PDF.

The base document stays real (real fonts, drafting, density, symbols); only the
edit is ours. That gives real-world extraction difficulty WITH exact ground
truth -- which a found revision pair could not (found pairs have real edits but
no labels).

Usage
-----
  # 1. see what's on the page and pick targets
  python make_real_pair.py inspect base.pdf --page 0 > elements.txt
  python make_real_pair.py inspect base.pdf --page 0 --grep "HH|PSV|barg"

  # 2. write an edit spec (see EXAMPLE_SPEC below), then
  python make_real_pair.py edit base.pdf spec.json --out pairs/real_001 \
         --page 0 --pair-id real_001

  # 3. optional scanned variant of the same pair (same GT)
  python make_real_pair.py degrade pairs/real_001 --level 3

Emits the same layout the generator does, so run_eval.py needs no new plumbing:
  pairs/real_001/a/L0.pdf  b/L0.pdf
  pairs/real_001/gt/elements_a.json  elements_b.json
  pairs/real_001/gt/correspondence.json  deltas.json  provenance.json
  pairs/real_001/qa.jsonl

Edit mechanics: PyMuPDF redaction removes the original glyphs cleanly and
leaves the rest of the drawing byte-identical, then text is reinserted at the
same baseline. That is why this produces a *real* input pair rather than a
re-render (a re-render would change every glyph's rasterization and make the
pair trivially different everywhere).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import fitz  # PyMuPDF

EXAMPLE_SPEC = {
    "provenance": {
        "source_url": "https://hmis.hanford.gov/files.cfm/HNF-64103_-_Rev_00.pdf",
        "source_name": "DOE Hanford HNF-64103",
        "license": "US Government work / public domain",
        "retrieved": "2026-07-25",
        "page_used": 12,
        "notes": "base document real; revision B edits authored by us",
    },
    "edits": [
        {"op": "replace_text", "find": "HH: 160", "with": "HH: 145",
         "role": "instrument", "field": "trip_hh", "severity": "HIGH",
         "desc": "high-high trip setpoint lowered 160 -> 145"},
        {"op": "replace_text", "find": "GC11S", "with": "FC11S", "nth": 0,
         "role": "line_tag", "field": "pipe_class", "severity": "HIGH",
         "desc": "pipe class GC11S -> FC11S (material spec change)"},
        {"op": "delete_text", "find": "26BL9077",
         "role": "valve_tag", "severity": "MEDIUM",
         "desc": "valve 26BL9077 removed"},
        {"op": "add_text", "text": "27. TIE-IN POINT TO BE VERIFIED AT SITE.",
         "at": [40, 470], "size": 6.0,
         "role": "note", "severity": "LOW", "desc": "note 27 added"},
        {"op": "move_text", "find": "N4212", "to": [300, 210],
         "role": "nozzle", "severity": "LOW",
         "desc": "nozzle label relocated (content unchanged)"},
        {"op": "swap_symbol", "rect": [250, 300, 268, 318], "glyph": "globe",
         "role": "geometry", "severity": "HIGH",
         "desc": "valve symbol gate -> globe at constant tag "
                 "(graphical change; symbolic engine cannot see this)"},
    ],
}


# --------------------------------------------------------------------------- #
# element extraction (mirrors what the ingest adapter sees)
# --------------------------------------------------------------------------- #
@dataclass
class El:
    eid: str
    text: str
    bbox: tuple[float, float, float, float]
    page: int
    size: float = 0.0

    def to_gt(self, page_rect) -> dict:
        w, h = page_rect.width, page_rect.height
        x0, y0, x1, y1 = self.bbox
        return {
            "eid": self.eid, "text": self.text, "page": self.page,
            "bbox_pt": list(self.bbox),
            "bbox_norm": [x0 / w, y0 / h, x1 / w, y1 / h],
            "zone": zone_of(self.bbox, page_rect),
        }


def zone_of(bbox, page_rect, rows="ABCDEFGHIJ", ncols=12) -> str:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    col = min(ncols - 1, max(0, int(cx / page_rect.width * ncols))) + 1
    row = rows[min(len(rows) - 1, max(0, int(cy / page_rect.height * len(rows))))]
    return f"{row}-{col}"


def stable_eid(prefix: str, text: str, bbox) -> str:
    """Content+position derived, so citations survive re-extraction."""
    key = f"{text}|{round(bbox[0], 1)},{round(bbox[1], 1)}"
    return f"{prefix}:{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def extract(doc: fitz.Document, page_no: int, prefix: str) -> list[El]:
    page = doc[page_no]
    out = []
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for line in blk["lines"]:
            txt = "".join(s["text"] for s in line["spans"]).strip()
            if not txt:
                continue
            bbox = tuple(line["bbox"])
            size = max((s["size"] for s in line["spans"]), default=0.0)
            out.append(El(stable_eid(prefix, txt, bbox), txt, bbox, page_no, size))
    return out


# --------------------------------------------------------------------------- #
# edit application
# --------------------------------------------------------------------------- #
@dataclass
class GTDelta:
    did: str
    kind: str
    role: str
    eid_a: str | None
    eid_b: str | None
    page: int
    zone_a: str | None
    zone_b: str | None
    field_changes: dict = field(default_factory=dict)
    description: str = ""
    severity: str = "MEDIUM"
    detectable_by: str = "symbolic"   # symbolic | raster_only
    semantic_null: bool = False
    # eval/run_eval.py::_gt_row_found reads row["sheet"] -- emitting only
    # "page" raises KeyError there. Sheets are 1-indexed in the canonical
    # model (src/canonical/model.py) while `page` here is 0-indexed, so this
    # is a genuine conversion, not an alias.
    sheet: int = 1


def _find_rects(page, needle: str, nth: int | None):
    hits = page.search_for(needle)
    if not hits:
        return []
    return [hits[nth]] if nth is not None and nth < len(hits) else hits[:1]


def _insert(page, rect, text, size, fontname="helv"):
    # baseline sits near rect.y1; nudge to sit on the original baseline
    page.insert_text((rect.x0, rect.y1 - size * 0.18), text,
                     fontsize=size, fontname=fontname, color=(0, 0, 0))


def _glyph(page, rect, kind, orient="h"):
    """Draw an ISA-5.1-shaped valve body in `rect`, replacing what was there.

    Two things this deliberately does NOT do, both learned from overlaying the
    output on the real drawing:

    - **No bounding border.** The original drew a black `draw_rect` outline
      around every swapped glyph. On a real P&ID that box is not a symbol --
      it reads as an annotation artifact, and it makes the raster diff trivial
      for the wrong reason (a big rectangle appearing, rather than the valve
      body actually changing shape).
    - **Filled triangles, not an X.** A gate valve is a bowtie: two solid
      triangles meeting at the centre. Two crossing lines look similar at
      thumbnail size but leave the interior white, so the pixel delta against
      a real filled bowtie is dominated by the fill, not by the symbol change.

    `orient` follows the pipe: "h" for a bowtie whose triangles meet
    left-right, "v" for one that meets top-bottom.
    """
    r = fitz.Rect(rect)
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    page.draw_rect(r, color=None, fill=(1, 1, 1))          # clear the original
    black = (0, 0, 0)

    if orient == "v":
        tri_a = [(r.x0, r.y0), (r.x1, r.y0), (cx, cy)]     # top triangle
        tri_b = [(r.x0, r.y1), (r.x1, r.y1), (cx, cy)]     # bottom triangle
    else:
        tri_a = [(r.x0, r.y0), (r.x0, r.y1), (cx, cy)]     # left triangle
        tri_b = [(r.x1, r.y0), (r.x1, r.y1), (cx, cy)]     # right triangle

    for tri in (tri_a, tri_b):
        page.draw_polyline(tri + [tri[0]], color=black, fill=black, width=0.3)

    if kind == "globe":
        # globe = gate body + the disc at the seat, per ISA-5.1
        page.draw_circle((cx, cy), min(r.width, r.height) * 0.30,
                         color=black, fill=(1, 1, 1), width=0.5)
    elif kind == "check":
        page.draw_line((r.x0, r.y1), (r.x1, r.y1), color=black, width=0.5)


def apply_edits(src: Path, spec: dict, page_no: int, out_b: Path) -> list[GTDelta]:
    doc = fitz.open(src)
    page = doc[page_no]
    rect = page.rect
    deltas: list[GTDelta] = []
    pending_inserts = []          # (rect, text, size) applied after redactions
    n = 0

    for e in spec["edits"]:
        op = e["op"]
        n += 1
        did = f"r{n:04d}"

        if op in ("replace_text", "delete_text", "move_text"):
            rects = _find_rects(page, e["find"], e.get("nth"))
            if not rects:
                # Hard failure, not a skip. Silently dropping the edit produces
                # a pair whose ground truth claims edits the document does not
                # contain -- the single worst outcome for a gold-standard
                # holdout, because every later measurement against it is then
                # quietly wrong. Common cause: the base page has no extractable
                # text layer at all (a scanned/raster drawing), where NO text
                # op can ever match.
                raise SystemExit(
                    f"edit target not found on page {page_no}: {e['find']!r}\n"
                    f"  If this base is a raster/scanned drawing it has no text "
                    f"to search -- use swap_symbol/erase_region instead, or pick "
                    f"a vector PDF. Run `inspect` to see what text exists.")
            r = rects[0]
            size = e.get("size") or max(6.0, r.height * 0.82)
            page.add_redact_annot(r)                       # remove original glyphs
            if op == "replace_text":
                new_full = e.get("with_full") or e["find"].replace(
                    e["find"], e["with"]) if "with_full" not in e else e["with_full"]
                pending_inserts.append((r, e.get("with_full", e["with"]), size))
                deltas.append(GTDelta(
                    did, "modify", e.get("role", "text"),
                    stable_eid("A", e["find"], tuple(r)),
                    stable_eid("B", e.get("with_full", e["with"]), tuple(r)),
                    page_no, zone_of(tuple(r), rect), zone_of(tuple(r), rect),
                    {e.get("field", "text"): [e["find"], e["with"]]},
                    e.get("desc", f"{e['find']} -> {e['with']}"),
                    e.get("severity", "MEDIUM")))
            elif op == "delete_text":
                deltas.append(GTDelta(
                    did, "remove", e.get("role", "text"),
                    stable_eid("A", e["find"], tuple(r)), None,
                    page_no, zone_of(tuple(r), rect), None, {},
                    e.get("desc", f"removed {e['find']}"),
                    e.get("severity", "MEDIUM")))
            else:  # move_text
                tx, ty = e["to"]
                nr = fitz.Rect(tx, ty, tx + r.width, ty + r.height)
                pending_inserts.append((nr, e["find"], size))
                deltas.append(GTDelta(
                    did, "move", e.get("role", "text"),
                    stable_eid("A", e["find"], tuple(r)),
                    stable_eid("B", e["find"], tuple(nr)),
                    page_no, zone_of(tuple(r), rect), zone_of(tuple(nr), rect), {},
                    e.get("desc", f"moved {e['find']}"), e.get("severity", "LOW")))

        elif op == "add_text":
            x, y = e["at"]
            size = e.get("size", 6.0)
            nr = fitz.Rect(x, y, x + len(e["text"]) * size * 0.5, y + size)
            pending_inserts.append((nr, e["text"], size))
            deltas.append(GTDelta(
                did, "add", e.get("role", "text"), None,
                stable_eid("B", e["text"], tuple(nr)),
                page_no, None, zone_of(tuple(nr), rect), {},
                e.get("desc", f"added {e['text'][:40]}"), e.get("severity", "LOW")))

        elif op == "swap_symbol":
            r = fitz.Rect(*e["rect"])
            _glyph(page, r, e.get("glyph", "globe"), e.get("orient", "h"))
            # Shaped to match the generator's own ChangeValveSymbol GT rows,
            # NOT this tool's original kind="graphical"/role="geometry" form.
            # eval/run_eval.py::_is_raster_only_gt_row only recognizes a
            # raster-only row as role=="valve_tag" carrying a "symbol_type"
            # field change (or role=="geom_line" with dx/dy), and score_pair
            # only knows the kinds add/remove/modify/move. Emitting the
            # original shape parsed fine and then scored as nothing at all --
            # silently excluded from the raster-recall measurement this pair
            # exists to produce. Reusing the generator's vocabulary is what
            # lets the holdout be scored by the identical code path as the
            # seeded set, which is the whole point of a holdout.
            deltas.append(GTDelta(
                did, "modify", e.get("role", "valve_tag"), None, None,
                page_no, zone_of(tuple(r), rect), zone_of(tuple(r), rect),
                {"symbol_type": [e.get("from_glyph", "gate"), e.get("glyph", "globe")]},
                e.get("desc", "symbol changed"), e.get("severity", "HIGH"),
                detectable_by="raster_only"))

        elif op == "erase_region":
            r = fitz.Rect(*e["rect"])
            page.add_redact_annot(r)
            deltas.append(GTDelta(
                did, "graphical", e.get("role", "geometry"), None, None,
                page_no, zone_of(tuple(r), rect), None, {},
                e.get("desc", "geometry erased"), e.get("severity", "HIGH"),
                detectable_by="raster_only"))
        else:
            raise ValueError(f"unknown op {op}")

    page.apply_redactions()
    for r, txt, size in pending_inserts:
        _insert(page, r, txt, size)
    out_b.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_b, garbage=3, deflate=True)
    doc.close()
    return deltas


# --------------------------------------------------------------------------- #
# emit
# --------------------------------------------------------------------------- #
# Our op names -> the generator's operator names, which eval/run_eval.py's
# _RASTER_ONLY_OPS matches against when selecting pairs for the raster-recall
# measurement. Anything unmapped passes through unchanged.
_OP_TO_GENERATOR_NAME = {
    "swap_symbol": "ChangeValveSymbol",
    "erase_region": "RerouteLine",
}


def upsert_manifest(dataset_dir: Path, row: dict) -> None:
    """Append-or-replace one row in the dataset's manifest.jsonl, keyed on
    pair_id. run_eval.py discovers every pair through this file, so a pair
    that isn't listed here simply does not exist as far as the scorecard is
    concerned -- the original tool emitted no manifest at all, which is why
    its output looked complete on disk but scored nothing."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "manifest.jsonl"
    rows = []
    if path.exists():
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    rows = [r for r in rows if r.get("pair_id") != row["pair_id"]] + [row]
    rows.sort(key=lambda r: r["pair_id"])
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def emit_null(base_pair: Path, out: Path, pair_id: str, provenance: dict) -> None:
    """Producer-variation null control: rev A re-saved through a different
    PyMuPDF write path, so content is identical and only the byte-level
    encoding differs. Ground truth is EMPTY -- every delta is a false
    positive, and the raster-region count on this pair is the calibration
    number for the morphological cleanup (see docs/findings.md).

    Deliberately reuses the manifest kind "null_prod", so run_eval.py's
    existing eval_null_pairs path scores it with no new code."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "a").mkdir(exist_ok=True); (out / "b").mkdir(exist_ok=True)
    (out / "gt").mkdir(exist_ok=True)

    src = fitz.open(base_pair)
    src.save(out / "a" / "L0.pdf", garbage=4, deflate=True)
    # different producer settings on the same content: no garbage collection,
    # no deflate, and a clean/pretty rewrite -> different bytes, same drawing
    src.save(out / "b" / "L0.pdf", garbage=0, deflate=False, clean=True, pretty=True)
    src.close()

    (out / "gt" / "deltas.json").write_text("[]")
    (out / "gt" / "correspondence.json").write_text(json.dumps({
        "method": "null control -- content identical by construction",
        "matched": [], "ambiguous_a": [], "ambiguous_b": [], "not_a_pair": False,
    }, indent=1))
    prov = dict(provenance)
    prov.update({"pair_id": pair_id, "held_out": True, "kind": "null_prod",
                 "notes": "rev A re-saved through a different producer path; "
                          "ground truth is empty by construction"})
    (out / "gt" / "provenance.json").write_text(json.dumps(prov, indent=1))
    (out / "qa.jsonl").write_text(json.dumps({
        "q": "What changed between these two revisions?",
        "expected_behavior": "answer",
        "a": "Nothing changed; the two files are the same drawing re-saved.",
        "citations": [],
    }) + "\n")

    upsert_manifest(out.parent.parent, {
        "pair_id": pair_id, "kind": "null_prod", "ops": [],
        "n_deltas": 0, "n_primary": 0, "n_cascade": 0, "n_semantic_null": 0,
        "held_out": True, "source": "real base document, re-saved (no edits)",
    })
    print(f"\n{pair_id}: null control (GT empty)  -> {out}")


def emit(base: Path, spec_path: Path, out: Path, page_no: int, pair_id: str):
    spec = json.loads(spec_path.read_text())
    out = out.resolve()
    (out / "a").mkdir(parents=True, exist_ok=True)
    (out / "b").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)

    # rev A = the real page, extracted unchanged
    src = fitz.open(base)
    a_doc = fitz.open()
    a_doc.insert_pdf(src, from_page=page_no, to_page=page_no)
    a_doc.save(out / "a" / "L0.pdf", garbage=3, deflate=True)
    a_doc.close(); src.close()

    deltas = apply_edits(out / "a" / "L0.pdf", spec, 0, out / "b" / "L0.pdf")

    da, db = fitz.open(out / "a" / "L0.pdf"), fitz.open(out / "b" / "L0.pdf")
    els_a, els_b = extract(da, 0, "A"), extract(db, 0, "B")
    ra, rb = da[0].rect, db[0].rect

    (out / "gt" / "elements_a.json").write_text(
        json.dumps([e.to_gt(ra) for e in els_a], indent=1))
    (out / "gt" / "elements_b.json").write_text(
        json.dumps([e.to_gt(rb) for e in els_b], indent=1))

    # correspondence by exact text where unambiguous; the rest is the residue
    ta = {e.text: e for e in els_a if sum(x.text == e.text for x in els_a) == 1}
    tb = {e.text: e for e in els_b if sum(x.text == e.text for x in els_b) == 1}
    shared = sorted(set(ta) & set(tb))
    (out / "gt" / "correspondence.json").write_text(json.dumps({
        "method": "unique-exact-text (unambiguous subset only)",
        "matched": [[ta[t].eid, tb[t].eid] for t in shared],
        "ambiguous_a": sorted({e.text for e in els_a} - set(ta)),
        "ambiguous_b": sorted({e.text for e in els_b} - set(tb)),
        "not_a_pair": False,
    }, indent=1))
    for d in deltas:
        d.sheet = d.page + 1        # canonical sheets are 1-indexed; page is 0-indexed
    (out / "gt" / "deltas.json").write_text(
        json.dumps([asdict(d) for d in deltas], indent=1))

    prov = dict(spec.get("provenance", {}))
    prov.update({"pair_id": pair_id, "base_sha256":
                 hashlib.sha256(Path(base).read_bytes()).hexdigest()[:16],
                 "held_out": True, "edits_applied": len(deltas),
                 "elements_a": len(els_a), "elements_b": len(els_b)})
    (out / "gt" / "provenance.json").write_text(json.dumps(prov, indent=1))

    qa = [{"q": "What changed between these two revisions?",
           "expected_behavior": "answer",
           "a": "; ".join(d.description for d in deltas
                          if d.detectable_by == "symbolic"),
           "citations": [d.did for d in deltas if d.detectable_by == "symbolic"]}]
    high = [d for d in deltas if d.severity == "HIGH"]
    if high:
        qa.append({"q": "Are there any safety-relevant changes?",
                   "expected_behavior": "answer",
                   "a": "; ".join(d.description for d in high),
                   "citations": [d.did for d in high]})
    qa.append({"q": f"What changed on page 9 of this drawing?",
               "expected_behavior": "refuse",
               "a": "This pair has 1 page.", "citations": []})
    (out / "qa.jsonl").write_text("\n".join(json.dumps(q) for q in qa))

    upsert_manifest(out.parent.parent, {
        "pair_id": pair_id,
        "kind": "edited",
        # eval/run_eval.py::eval_raster_recall_pairs selects pairs by these
        # exact op names (_RASTER_ONLY_OPS). A pair whose ops don't match is
        # silently skipped from the raster measurement rather than reported
        # as zero -- so the names have to be the generator's, not ours.
        "ops": sorted({_OP_TO_GENERATOR_NAME.get(e["op"], e["op"])
                       for e in spec["edits"]}),
        "n_deltas": len(deltas),
        "n_primary": len(deltas),
        "n_cascade": 0,
        "n_semantic_null": 0,
        "held_out": True,
        "source": "real base document; edits authored here",
    })

    print(f"\n{pair_id}: {len(deltas)} edits  "
          f"(symbolic {sum(d.detectable_by=='symbolic' for d in deltas)}, "
          f"raster_only {sum(d.detectable_by=='raster_only' for d in deltas)})")
    print(f"  elements: A={len(els_a)} B={len(els_b)}   -> {out}")
    for d in deltas:
        print(f"  {d.did} {d.kind:9s} {d.severity:8s} {d.detectable_by:11s} "
              f"{d.description[:60]}")
    da.close(); db.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inspect", help="list text elements to pick edit targets")
    i.add_argument("pdf"); i.add_argument("--page", type=int, default=0)
    i.add_argument("--grep", default=None, help="regex filter on text")

    e = sub.add_parser("edit", help="apply edit spec -> rev B + ground truth")
    e.add_argument("pdf"); e.add_argument("spec")
    e.add_argument("--out", required=True); e.add_argument("--page", type=int, default=0)
    e.add_argument("--pair-id", default="real_001")

    c = sub.add_parser("crop", help="crop a page to a rect (drop surrounding chrome)")
    c.add_argument("pdf"); c.add_argument("--page", type=int, default=0)
    c.add_argument("--rect", required=True, help="x0,y0,x1,y1 in points")
    c.add_argument("--out", required=True)

    nl = sub.add_parser("null", help="producer-variation null control (GT empty)")
    nl.add_argument("pdf", help="the rev-A PDF to re-save")
    nl.add_argument("--out", required=True); nl.add_argument("--pair-id", default="null_001")
    nl.add_argument("--provenance", default=None, help="json file with a provenance block")

    sub.add_parser("example-spec", help="print a starter spec to stdout")

    a = ap.parse_args()
    if a.cmd == "example-spec":
        print(json.dumps(EXAMPLE_SPEC, indent=2)); return
    if a.cmd == "inspect":
        doc = fitz.open(a.pdf)
        pg = doc[a.page]
        els = extract(doc, a.page, "A")
        rx = re.compile(a.grep, re.I) if a.grep else None
        print(f"# {a.pdf} page {a.page}  {pg.rect.width:.0f}x{pg.rect.height:.0f}pt"
              f"  {len(els)} text elements")
        for el in els:
            if rx and not rx.search(el.text):
                continue
            x0, y0, x1, y1 = (round(v, 1) for v in el.bbox)
            print(f"{zone_of(el.bbox, pg.rect):5s} [{x0},{y0},{x1},{y1}] "
                  f"sz{el.size:.1f}  {el.text}")
        return
    if a.cmd == "crop":
        x0, y0, x1, y1 = (float(v) for v in a.rect.split(","))
        src = fitz.open(a.pdf)
        out_doc = fitz.open()
        out_doc.insert_pdf(src, from_page=a.page, to_page=a.page)
        # set_cropbox BEFORE anything else -- the ordering data/samples/
        # real_pair/PROVENANCE.md documents as load-bearing.
        out_doc[0].set_cropbox(fitz.Rect(x0, y0, x1, y1))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out_doc.save(a.out, garbage=4, deflate=True)
        out_doc.close(); src.close()
        chk = fitz.open(a.out)
        print(f"cropped -> {a.out}  rect={chk[0].rect}  "
              f"words_remaining={len(chk[0].get_text('text').split())}")
        chk.close()
        return
    if a.cmd == "null":
        prov = json.loads(Path(a.provenance).read_text()).get("provenance", {}) \
            if a.provenance else {}
        emit_null(Path(a.pdf), Path(a.out), a.pair_id, prov)
        return
    emit(Path(a.pdf), Path(a.spec), Path(a.out), a.page, a.pair_id)


if __name__ == "__main__":
    main()
