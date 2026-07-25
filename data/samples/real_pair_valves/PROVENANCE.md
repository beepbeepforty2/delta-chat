# Real, hand-edited revision pair — geometry / valve changes

A second real pair, same process as `data/samples/real_pair/` (crop real
vendor P&ID content, hand-edit with PyMuPDF), but this time deliberately
targeting **diagram-level geometry** — moving, adding, and removing an
actual valve symbol's vector paths — rather than text/notes, to see how the
delta engine and markup tooling handle a real component-diagram change.

## Crop

Same source, `data/samples/Lift Gas compressor-P&ID.pdf` (26-KA-901),
columns 1-3 full height — but `fitz.Rect(0, 0, 255, 822)`, 5pt wider than
`real_pair/`'s crop, because that width was clipping the last digit of a
valve tag (`40GT9311`) this pair specifically needed intact.

## Edits made to produce revision B

1. **Line tag pipe-class change** (same edit type as `real_pair/`, kept as
   a known-good control case): `2"-WC-40-9014-AC21-00` → `...-AC31-00`.
   Still correctly `modify`/`line_tag`, `HIGH` severity.
2. **MOVE a real valve**: the gate-valve "bowtie" icon at tag `40GT9313`
   (4 filled triangles meeting at a center point — this drawing's actual
   symbol for that valve type) plus its tag and spec label, slid +25pt
   along its own horizontal pipe run.
3. **REMOVE a real valve**: the icon, tag (`40GT9311`), and spec label at
   the next pipe run down, deleted entirely — the pipe itself continues
   uninterrupted, as a real valve removal would look.
4. **ADD a new valve**: the same 4-triangle icon redrawn from scratch in
   genuinely blank space, with a new tag (`40GT9399`) — not present in A.

## What this took, concretely (answering "how hard is this to fabricate")

Text edits (pair 1) are close to mechanical: find the span, redact its
exact bbox, `insert_text` the replacement at the same origin. Geometry
edits were a different order of effort, for reasons specific to how a
real PDF stores a diagram, not this project's tooling:

- **There is no "valve" object to select.** A PDF content stream has no
  concept of a named component — `page.get_drawings()` returns bare path
  primitives (lines, filled polygons) with no grouping. Finding "the 4
  triangles that make up this valve" meant clustering small filled-path
  rects by size and proximity and cross-checking against nearby tag text
  by hand (see the exploratory scan in this session — not something you
  can point a library function at). A different drafting template could
  draw the "same" conceptual valve with a completely different primitive
  count (2 triangles, an outline-only bowtie, a circle+lines), so this
  clustering heuristic is specific to this drawing, not reusable as-is.
- **Redacting a shape doesn't reliably delete it.** `add_redact_annot(...,
  fill=(1,1,1))` with `graphics=PDF_REDACT_LINE_ART_NONE` visually covers
  the old triangles with a white patch — but the underlying path objects
  are untouched. A geometry-aware ingest (`get_drawings()`, not rendered
  pixels — exactly what this project's own adapter does) still sees them,
  unchanged, so the "removed" valve produced **no delta at all** on first
  attempt. Fixed with `PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED`, verified
  safe at this tiny (~6×12pt) scale first (checked that it doesn't also
  consume the long pipe line passing through the same area — this
  drawing stores each short stroke as its own path object, not one
  monolithic line, so it doesn't).
- **Real annotation density breaks assumptions text edits don't.** The
  valve's tag, its spec label, and an unrelated neighboring valve's own
  label (`CSO`) are stacked closely enough that their glyph bboxes
  overlap by under a point. A single redaction rect wide enough to cover
  the tag+spec pair clipped into `CSO` (`CSO` → `CS`) or left fragments of
  the spec label behind (`3/4"GTAC00R` → `3`, `R`) three times before a
  two-rect split (one bounded by the neighbor's x-extent, one by its
  y-extent) worked cleanly. None of this shows up on a quick visual
  render — it only surfaces by re-extracting text and checking exactly
  what survived.
- **A real bug this pair caught, not the first one**: `markup/
  pdf_annotate.py` crashed (`rect is infinite or empty`) the first time a
  delta touched a perfectly horizontal or vertical `geometry` element
  (zero-width/height bbox by construction, per `pdf_native.py`'s own
  documented design — a real element, not an extraction bug). Text-only
  deltas never have a degenerate bbox, so `real_pair/` never exercised
  this path. `overlay.py`'s PNG path already pads for exactly this reason
  (`PAD_PX`); `pdf_annotate.py` never did. Fixed with an equivalent
  point-space pad (`PAD_PT`), clamped to the page rect; regression test
  added (`test_render_pdf_markup_handles_degenerate_zero_area_bbox`).

## What the delta engine actually produced

`20 primary changes: {move: 6, modify: 10, add: 3, remove: 1}` — plus the
one clean `HIGH` pipe-class modify. Two genuine, worth-reporting
limitations, both **specific to geometry, not present in the text-only
pair**:

- **Anonymous geometry cross-matches.** A `geometry` element carries no
  content (`classify_geometry` returns `{"geom_kind": ...}`, nothing
  else) — with three near-identical valve icons changing in the same
  small neighborhood at once (moved, added, removed), the bipartite
  matcher's position/shape cost function paired some of the *removed*
  valve's triangles against the *moved* or *added* valve's triangles
  (`moved E-11 -> E-12`, `geom_kind changed: line -> rect`) rather than
  resolving them as clean adds/removes. The *valve tag text* had the same
  problem for a different reason: `40GT9313 -> 9399` and `9311 -> 9313`
  both got matched as `modify`/`valve_tag` — real tag-numbering
  conventions are repetitive enough (shared prefix, sequential digits)
  that content-similarity matching treats an unrelated new tag as a
  plausible edit of an old one. Text edits with more distinctive content
  (a full sentence, a distinct pipe spec) don't have this problem; short,
  templated, visually-similar geometry and tags do.
- **Geometry is never elevated in severity.** `severity.py`'s rule table
  has no case for `element_type == "geometry"` — every geometry delta
  above is `LOW`, identical to administrative noise, regardless of
  whether it's a removed safety valve or an anti-aliasing artifact. This
  is an honest, real architectural gap: the system currently cannot tell
  "a valve disappeared from the diagram" from "a stray pixel changed,"
  because it has no notion of "valve" as a concept at all above the
  individual-triangle level.

## Regenerating

Same as `real_pair/`: a one-off hand-built script, not a `tools/`
utility — these two files are the artifact.
