# Engineering findings

A detailed, chronological account of real bugs found while building
delta-chat, and — more importantly — *how* each was actually caught.
Almost none of these were caught by a unit test in isolation; they showed
up by running the real pipeline against real data (the generated eval
dataset, the two real vendor P&IDs in `data/samples/`, or a live LLM
call) and looking at the actual output. That pattern repeats enough times
below that it's worth naming up front: a synthetic unit test can pass
while the real pipeline is wrong, and the only reliable way to catch that
is to actually run it.

See [README.md](../README.md) for the project overview, architecture, and
quick start; this document is the detail behind the "what building X
caught" claims made there.

---

## Native-PDF adapter, validated against real vendor P&IDs

The real 26-KA-901/902 PDFs run ~800 text elements + ~5000 geometry paths
per sheet, versus the generator's synthetic ~80-120 — validating
`pdf_native.py` against them (rather than only the synthetic dataset)
surfaced a real gap: instrument bubbles in the real drawings split
system/function/loop across three stacked text baselines (e.g. `26` /
`PI` / `9055` on separate lines), not the generator's single-line `FUNC
LOOP SYS` format `parse_instrument` expects. Same-baseline clustering
correctly keeps them separate — they are genuinely distinct text runs —
so this isn't a clustering bug, it's a composition-format gap. Fixing it
needs 2D proximity-based grouping instead of same-baseline-only
clustering; tracked as an `xfail` in `tests/test_pdf_native_real_samples.py`
rather than silently passing.

Everything else transferred cleanly: all 44 zone labels found on both
real sheets, and line tags / valve tags / nozzles / DELETED-placeholder
notes all parse correctly via the same Tier-1 regexes the generator's
format was modeled from.

## Delta engine, validated against ground truth

Three real bugs surfaced while validating the alignment/classification
logic against `eval/datasets/v0`'s ground truth, not by inspection:

- **Type-bucketed matching initially couldn't recover
  `DeleteNoteKeepPlaceholder`** (a note's classified type changes from
  `note` to `note_deleted` as *part of* the edit) — bucketing strictly by
  exact type meant that correspondence was structurally unrecoverable.
  Fixed with a small type-compatibility-group concept
  (`align.py::TYPE_MATCH_GROUPS`); the general lesson is that type itself
  can be the subject of a delta, not just a partition key for matching.
- **Cascade detection needed a "single-field-only" guard.** A `dcn_note`
  whose `note_no` shifts from a renumbering cascade *and* whose `dcns`
  list gains a genuinely new entry must stay a standalone primary change
  — it has meaning beyond the renumber. Grouping only fires on deltas
  whose *entire* change is one numeric field.
- **Producer-jitter float noise was flagged as a real change** on the
  `null_prod` null pair (same content, two PDF producers): a derived
  geometry attribute (`r_norm`) differed at the 1e-8 scale from
  sub-point rendering jitter, and exact float equality treated that as a
  delta. `null_prod` must emit zero deltas by construction — caught by
  actually running the null pair, not by the isolated GT-element tests
  (which don't extract geometry from a real render). Fixed with a numeric
  tolerance (`DELTA_NUMERIC_EQ_TOL`, default `1e-4`, far below any real
  integer field change in this domain).

Known scoped-out gap, documented in `classify.py`: cascade members link to
an arbitrary group member as `primary_did`, not necessarily the true
root-cause `add` event (e.g. the note that was actually inserted).
Cascade *grouping* is correct (members aren't double-counted as
independent primary changes); cascade *attribution* to the literal
triggering event needs cross-kind reasoning left for a later pass.

Alignment logic validated exactly (0 missing / 0 extra) against
`gt/correspondence.json` across all 6 edited pairs, including the
DELETED-placeholder ambiguity (multiple identical `"N. DELETED."` strings
per sheet, distinguished only by position). Classification validated
against `gt/deltas.json` with a 1-item slack per pair (extraction/move-
threshold edge cases) rather than exact equality.

## Scanned-PDF adapter, validated against real degraded pages

A spike against a real degraded page (the generator's rasterized
degradation ladder) surfaced the same *class* of clustering bug
`pdf_native.py` hit above, but a worse variant of it. Tesseract's own
line/block grouping merged an entire row of widely-spaced zone-grid
digits (~540px apart) into one polluted string, so word-level OCR output
plus manual gap-based re-clustering was the right approach going in
(confirmed against real gap measurements: ~7-9px between words in one
line vs. ~540px between unrelated zone digits, a ~60x separation any
reasonable threshold clears).

The bug that slipped through the first pass: OCR word boxes jitter a
pixel or two vertically even on one physical line (unlike fitz's exact
vector coordinates, where every word from one `drawString` call shares an
identical origin y). Sorting by `(round(y0, 4), x0)` as a single key put
jittered words from the *same* line into different sort buckets, which
silently scrambled left-to-right order and corrupted the gap merge for
whichever words happened to jitter across a rounding boundary. Caught by
running the real CLI end-to-end (native PDF vs. a scanned revision, not
just the adapter's own unit tests): a genuine note, "OIL CHANGE BY USING
TEMPORARY ARRANGEMENT WITH HOSES.", fragmented into four spurious
add/remove deltas against its native-PDF counterpart. Fixed with a proper
two-pass cluster: band words into rows by y-proximity first
(order-independent), *then* sweep each band left-to-right by x — the
same gap merge as before, just no longer dependent on a fragile sort key.
Reduced a cross-format smoke-test run from 112 primary deltas down to 82,
with the real semantic changes (an inserted note, a revision bump) now
showing up as single clean adds instead of noise.

What's left, honestly: OCR misreads on small/degraded text still produce
some low-confidence noise deltas (e.g. zone-grid digits read as
punctuation), and `extraction_confidence` correctly reflects that — this
is expected OCR behavior, not silently hidden, and exactly the signal the
eval harness's per-format-level scoring (L0 vs L2/L3) is meant to
quantify. No CV-based geometry extraction (line/circle detection from the
raster) is attempted — a real computer-vision problem, a bigger lift than
the OCR text path for the same time budget, and a deliberate, documented
cut rather than a silent one.

## Registration: from an honest no-op to a real, verified similarity transform

`src/delta/register.py` always returned an identity transform until this
was built — a real, extensible seam, but no actual anchor-based
estimation behind it. Replaced with: elements from the drawing's fixed
scaffolding (title-block fields, the equipment tag, border-grid zone
labels — never touched by any edit operator) whose content is identical
between A and B *and* unique within the sheet become anchor
correspondences; two or more anchors give a full similarity transform
(uniform scale + rotation + translation) via the closed-form
Umeyama/Kabsch least-squares solution; fewer than two falls back to
identity rather than force-fitting an underdetermined estimate.

Verified two ways: `tests/test_delta_register.py` constructs a *known*
non-trivial transform (scale 1.1, rotation 0.05 rad, translation),
derives synthetic anchor pairs from it, and checks `register()` recovers
it to 1e-6 — including on a held-out point never used as an anchor, the
actual property `match_elements`' spatial cost term depends on. Against
real native pairs it correctly converges on near-identity (as it should —
no skew to correct).

Honest finding, not glossed over: it does *not* show any effect against
this dataset's own scanned pairs, even at L3 (nominal 0.545° skew).
Digging in rather than assuming the code was wrong: skew genuinely does
perturb OCR-extracted anchor positions relative to that same document's
own L0 render (confirmed directly — e.g. one anchor moves from
`(0.7721, 0.9445)` at L0 to `(0.7746, 0.9406)` at L3). But
`render_transforms.json` applies the *same* `skew_deg` value to both
revision A and revision B at a given level — the degradation is
deterministic per level, not independently randomized per document — so
the two documents drift together and there is no *relative* misalignment
between them left for registration to correct; the anchor pairs come out
matching to floating-point precision either way. This is a property of
the dataset's synthetic degradation ladder, not evidence against the
registration math, which the synthetic-ground-truth tests verify
independently and directly. Real independently-scanned document pairs —
two separate physical scans of the same drawing revision — are the case
this seam exists for.

## Chat, validated against a real GLM connection

Two real findings from actually running questions through the live
model, not just unit tests with a fake LLM call:

- **The LLM span was missing two of the five telemetry fields it was
  supposed to capture.** The first pass only recorded `model`/`prompt`/
  `response` on the span — caught by inspecting a real trace after the
  first live chat session, not by any test (the unit tests all used a
  fake `call_llm` returning a bare string, which never exercised the real
  response's `usage` object at all). Fixed by introducing `LLMResult`
  (text + tokens + cost, cost `None` unless `LLM_COST_PER_1K_*` is
  configured — no hardcoded pricing for a third-party provider) and
  adding a regression test that asserts the span actually carries
  `tokens_in`/`tokens_out`/`cost_usd`, not just that the call succeeds.
- **A refusal-expected probe got an answer instead of a refusal — and it
  was arguably the better response.** Asked "What changed on sheet 7?"
  (the document has one sheet), the model didn't say `REFUSED:` — it
  explained there's no sheet 7, correctly identified that the "7"s it
  found in context were zone labels rather than a sheet number, and cited
  all of it validly. That passes the citation post-validation (real
  citations, all ids exist in the retrieved set), so it wasn't overridden
  into a refusal. Whether that's "correct" is a real, unresolved judgment
  call — an honest, cited explanation of why the premise is false
  arguably beats a bare refusal, but it diverges from strict refuse/
  answer grading. This shows up directly in the eval numbers as a gap
  between refusal accuracy (scored on the literal flag) and LLM-judge
  correctness (scored on content) — see "Eval scorecard" below.

## Retrieval, validated against the eval harness

Running real `qa.jsonl` questions through the live chat path (not a fake
`call_llm`) surfaced three real `BM25Index` bugs in one sitting — none
visible from hand-built fixture chunks, all visible within the first few
live transcripts:

- **Hyphenated tags didn't match their own drawing.** A question about
  `FIT-9050` shares zero tokens with the printed form `FIT 9050 26 SD
  HH:150 LL:120` (space-separated on the page, hyphenated the way an
  engineer would actually ask about it) — retrieval's plain tokenizer
  never split on it. Confirmed as a real miss before fixing it: the model
  refused, citing "the provided context does not include the revised
  document's version of FIT-9050." Fixed by making `_tokenize`
  additionally emit hyphen-split sub-tokens (`"fit-9050"` also indexes as
  `"fit"` + `"9050"`), additive so the whole-token match is never lost.
- **Zone was unsearchable.** `Chunk.zone` is metadata, never part of
  `.content`, so "did anything change in zone B-1?" had no tokens to
  match against at all — a hard zero-result miss, not a ranking problem.
  Fixed by folding `zone` tokens into the indexed surface (never into
  `.content`, so citations still show only real drawing text). That
  promptly caused a second, more interesting bug: a `zone_label` element
  (the single literal digit/letter rendered at the sheet border to make
  the zone grid itself, e.g. content `"B"`) is an extremely short
  "document," and BM25's own length normalization makes a 1-token match
  against a 1-token document score enormously higher than the same match
  against a real multi-word note — so the zone letter's own rendering
  artifact dominated every zone query. Fixed by excluding `zone_label`
  from `build_chunks` entirely.
- **Short static content still out-ranked the actual delta.** Even after
  the zone fix, the real "note added" delta for a live zone query ranked
  19th, behind a run of unrelated one-line static notes sharing the same
  zone tokens, for the same length-normalization reason. Fixed with a
  modest source-weighting boost (`DELTA_SOURCE_BOOST = 1.5`) for
  `source == "delta"` chunks, tuned against the live case until it
  consistently surfaced the right chunk rather than guessed at.

A fourth bug was in the eval harness's own groundedness check, not
retrieval: the first-pass sentence-extraction heuristic walked back to
the nearest `.` to isolate the clause a citation is attached to, and a
cited chunk whose own content ends in a period (e.g. quoting a note whose
text is `"5. DELETED."`) tripped that boundary and discarded exactly the
quoted content that should have overlapped the chunk — silently reporting
real, correct citations as unsupported. Fixed by switching to a fixed
200-char lookback window instead of naive sentence splitting.

**Retrieval alias expansion** (added in a later pass): BM25 is purely
lexical (deliberate — no vector DB), so a natural question sharing zero
literal tokens with the drawing's own notation retrieves nothing: "trip"
shares no token with a setpoint chunk's actual printed content
(`"...SD HH:150 LL:120"`), "spec" shares none with the field name
`pipe_class`. `config/domain.yaml` is a curated first-pass alias table
(query phrase → literal drawing tokens), expanded into the query's token
set additively — never replaces the base tokenization. Live-verified: the
query "did the trip setpoint change?" retrieved zero results before this
change and now correctly surfaces the actual instrument chunks on the
first try, including the one whose setpoint actually changed in that
pair.

What's still honestly unresolved, not silently tuned around: a broad
"what changed on sheet 1?" query has almost no distinctive vocabulary to
rank against (delta descriptions are templated, not prose), so which
specific deltas BM25 surfaces for a *summary* question is somewhat
arbitrary — targeted questions (a zone, a tag, a specific field) retrieve
reliably; broad ones don't consistently surface every primary change. A
production system would likely route "what changed on sheet N" to a
deterministic tool call over the structured delta list rather than
through retrieval at all, since the full answer already exists losslessly
upstream of the chat layer. Also unresolved: a model can express a
refusal in substance without the literal `REFUSED:` prefix the
forced-refusal gate looks for, so the `refused` flag reads `False` even
though the content is a correct refusal — this is the same gap noted
above under "Chat."

## Judge/chat backend decoupling

The LLM-judge originally called the chat backend's own default-call
function directly — the literal same function, model, and client
answering the questions it was grading. Self-judging bias was real and
undocumented next to the reported correctness number. Fixed
structurally, not just documented around: `src/chat/llm.py` has
`get_judge_client()`/`JUDGE_MODEL`, a separate seam that falls back to
the chat backend's own config when `JUDGE_LLM_*`/`JUDGE_MODEL` are unset
(true today — no second credential is configured) and switches with a
one-line `.env` change the moment one exists. `judge_is_same_backend()`
reports this live: the scorecard prints an explicit
`*** SAME BACKEND AS CHAT -- self-judging risk, treat as an upper bound ***`
next to the correctness number, and flags the 5-hand-checked validation
sample size explicitly too — both caveats surfaced where the number is
reported, not left to go stale in a doc nobody re-reads.

The three-backend cost/latency/determinism comparison table itself
(`eval/baselines/backend_compare.py`) is scaffolded, not run — no second
real credential is available in this environment, and fabricating a
comparison without one would be a lie by construction, so it reports its
actual state honestly instead: `1/1 backends configured -- comparison
needs at least 2`, verified via a real invocation, and exits 0.

`tools/compare_models.py` is a lighter, since-added companion for the
common case of one credential with several model names (e.g. two GLM
variants) — same pair, same credential, model names swapped, reusing the
existing `call_llm` injection points on `run_llm_direct`/`chat.answer`
rather than needing per-model credential config. Live-run comparison
between `glm-5.2` and `glm-4.5-air` on the same real pair: both answered
a setpoint-change question correctly, but `glm-4.5-air` used roughly 4x
fewer output tokens for the same substance — a genuine, measured
data point, not a guess.

## Markup: from raster PNG to real PDF annotations

The first version (`src/markup/overlay.py`) drew colored, legended
delta boxes onto each revision's retained raster (PNG) — correct on both
A and B sides, but a PNG is a flat image: no toggle, no structured
metadata, never appears in a PDF viewer's own markup/comments list, and
permanently rasterizes content that should stay vector. Live-verified
against a real pair (`edited_003`): revision A showed the *old* setpoint,
note text, and DCN reference correctly amber; revision B showed the *new*
values plus a green box around the newly-added revision row, nothing
marked where nothing changed. One design note: the legend occupies a
fixed top-left panel that a small synthetic test image can trivially
collide with (caught early via two tests that were unknowingly sampling
the legend instead of the delta box, fixed to sample elsewhere) — a real
full-resolution raster didn't visibly collide with drawing content in
practice, but it's a fixed position, not content-aware, so that remains a
real, undefended edge case on some other drawing's layout.

`src/markup/pdf_annotate.py` replaced this as the default: real native
PDF "Square" annotation objects (`page.add_rect_annot()`), directly on
the *original* PDF pages, not the raster cache — appearing in Acrobat's/
Bluebeam's own markup/comments list, toggleable, and never rasterizing
the underlying vector drawing. `overlay.py`'s raster version is kept as a
`--format png` opt-in (a PNG can be visually inspected directly in a way
a PDF can't, which is how the live checks throughout this project were
actually done).

One naming gap worth flagging: an earlier spec called for
`page.add_square_annot()`; that method doesn't exist in the installed
PyMuPDF (1.28.0) — confirmed via direct introspection before writing any
code around it. The real method is `add_rect_annot()`, producing the
same annotation *type* ("Square" is the PDF spec's own name for it, not a
PyMuPDF quirk).

Denormalizing bbox coordinates against `page.rect.width`/`height` (not
raster pixel size) works for both native and scanned inputs without any
format-specific branching: bbox is normalized [0,1] as a *fraction* of
the page, and that fraction is basis-independent as long as the OCR'd
raster is a full-page, non-cropped render.

Live-verified two ways: `tests/test_markup_pdf_annotate.py` builds a real
PDF, runs annotation, reopens the *output* file and asserts on real
`page.annots()` objects (content, color, type) — a round trip, not "a
file was written." And a live `make markup` run against `edited_003`,
inspected both by listing the real annotation objects (content strings
matching each delta's own description exactly) and by rendering the
annotated PDF to an image and looking at it.

### A third markup path: an interactive report for the end user, not the reviewer

Both paths above assume the audience already has a PDF markup tool open.
`src/markup/html_report.py` (opt-in, `run --html`) targets a different
audience: the person who asked for the diff, wants to see it, and
shouldn't need Acrobat or a terminal to do so. It's a single
self-contained HTML file — same two-pane-plus-sidebar shape as
`tools/visual_diff.py`'s debug viewer, reused deliberately for the layout
only; the data underneath is the real engine's own `Delta` objects, not
that tool's independent naive matcher. Reusing a debug tool's *UI shape*
while refusing to reuse its *matching logic* is the same principle
DESIGN.md decision #2 applies to comparison generally: a good layout
isn't the thing that was supposed to stay independent, the matcher was.

Two implementation choices worth recording:

- **Boxes are positioned client-side as raw bbox percentages**, not
  denormalized to pixels server-side the way `overlay.py`'s PIL path
  does. `CanonicalElement.bbox` is already normalized [0,1] against the
  same raster both adapters retain, so `left: x0*100%` is correct with no
  pixel-size lookup at all — simpler than the PNG path, which only
  denormalizes because PIL draws server-side and needs real pixel
  coordinates to do it.
- **`_collect_boxes()` (`overlay.py`) was changed to yield whole `Delta`
  objects** instead of a hand-picked `(kind, is_cascade, description)`
  tuple, once a third consumer needed fields the first two never did
  (severity, confidence, semantic_null, zone). `pdf_annotate.py`'s
  unpacking loop had to be updated in the same change — a real,
  momentarily-broken intermediate state (a `ValueError` on tuple
  unpacking) caught by re-running `tests/test_markup_pdf_annotate.py`
  before moving on, not by inspection alone.

`unclassified_visual_change` deltas (from the opt-in raster recall net)
have no `id_a`/`id_b` by construction — no element for `_collect_boxes` to
resolve a box against. Rather than silently drop them from the report the
way they're silently absent from the two PDF-based markup paths (neither
of which has anywhere to show a delta with no box at all), the sidebar
lists them anyway, tagged "no exact location — zone only," since the
entire point of that pass is surfacing what extraction missed — hiding it
a second time here would defeat it. Live-verified against `edited_003`
(`tests/test_markup_html_report.py::test_render_html_report_real_pair_end_to_end`)
and by opening the generated `report.html` directly.

## Severity ranking

No eval ground truth exists for this (the dataset generator predates the
capability), so it isn't scored through the P/R/F1 machinery — it's a
rule table (`src/delta/severity.py`) validated against known real deltas
instead: the test suite runs the actual engine (not synthetic `Delta`
objects) against a real trip-setpoint change (PDIT-9017 HH 235→240) and a
real pipe-class change (AC21S→GC11S) and asserts CRITICAL / HIGH
respectively. Live-checked via `make run`: the setpoint change is the
only CRITICAL entry and sorts first within "Modify," while the note edit,
DCN update, and title-field REV bump all correctly land as LOW and sort
after it. `is_cascade` overrides every other rule — a cascade member is
always LOW no matter what field or element type it landed on, since a
renumbering side-effect has no independent engineering content.

## Semantic-null detection, and a real bug the live eval caught that unit tests didn't

`Delta.semantic_null` didn't exist for most of this project; every
equivalent reword and every DELETED-range collapse was indistinguishable
from a real change in the engine's own output. `semantic_null_emission_rate`
had already measured the cost of that gap honestly (1.0 before this work
— 100% of GT null entries got matched by a normal, unflagged delta)
without anything actually closing it.

Two mechanisms for two genuinely different sub-cases:

- **Rule, no LLM** (`src/delta/semantic_null.py::_rule_deleted_placeholder`):
  fires only when a `note_deleted` element transitions *into* a collapsed
  range (`"range"` appears in `field_changes` — the precise, unambiguous
  signature a DELETED-range collapse leaves, since only the collapsed
  form ever carries a `range` attr). Deliberately excludes `is_cascade`
  deltas and add/remove of a `note_deleted` element (ambiguous — could be
  an unmatched half of the same collapse, or real content newly vanishing
  behind a placeholder; a false null is worse than a missed one).
- **LLM adjudication** (opt-in via `DELTA_SEMANTIC_NULL_LLM=1`, default
  off): one isolated, cached call per candidate — `modify` deltas where
  structured field diffing found nothing (exactly the "words changed,
  unclear if meaning did" case) and the rule didn't already resolve it.
  Cached by `(old, new)` content pair.

**A real precision bug, caught by the live eval, not a unit test.** The
first version of the rule matched on "the entire field_changes is a
subset of `note_deleted`'s own bookkeeping fields" — which also matched
an *ordinary* +1 renumbering cascade of a `note_deleted` element (e.g.
"5. DELETED." → "6. DELETED." because an earlier note was inserted
elsewhere). That's a real, GT-expected delta — exactly what
`is_cascade`/cascade recall exist to track — not a structural no-op.
Every unit test for the rule passed; the bug only showed up running the
*actual* eval pipeline end to end: overall recall on the native-pair
level dropped from 0.98 to 0.93 (false negatives 1→4), traced via a new
`semantic_null_detection` P/R column showing 3 real deltas wrongly
flagged null. Fixed by tightening the rule to the precise `"range"`
transition signature plus an `is_cascade` guard, re-verified against the
same live run: recall back to 0.98, semantic-null-detection precision
back to 1.0. Regression tests for both the original bug and the fix
exist. This is the same lesson the ingest adapters taught earlier: a
synthetic unit test can pass while the real pipeline is wrong.

Live-verified with a real GLM call: a DELETED-collapse case correctly
flagged by the rule alone (no LLM call made); a reword case
("ATMOSPHERIC VENT." → "VENT TO ATMOSPHERE.") correctly flagged by the
live LLM adjudication with a sensible reason; and a genuinely ambiguous
unmatched remove correctly left unflagged — the deliberate conservatism
working as designed.

One correction to how this gets measured: the *existing*
`semantic_null_emission_rate` metric does **not** move when the engine's
own flag improves — it measures whether a normal, unflagged prediction
happens to description-match a null GT entry, unrelated to the flag by
construction. The metric that actually moves is the *new*
`semantic_null_detection` P/R column: on one null-reword pair, recall
goes 0.33 (rule alone) → 0.67 (rule + live LLM), precision 1.0 in both
cases.

## Confidence: a real formula bug, and a calibration check that immediately found something

`_confidence()` used `min(ext_conf_a, ext_conf_b)` where the design
specifies a product (`margin × ext_conf_a × ext_conf_b`). It went
unnoticed because the two formulas agree whenever both sides are 1.0 —
every native-native pair, since the native adapter hardcodes
`extraction_confidence=1.0` — so months of native-pair-heavy manual
testing could never have caught it. Fixed to the literal product;
single-sided add/remove deltas were never affected.

Nothing had ever validated whether confidence actually predicted
correctness. Added a confidence-calibration check: buckets matched vs.
false-positive predicted deltas by confidence band and reports precision
per band, wired into the scorecard. First real run immediately surfaced
something worth knowing rather than assuming: on the scanned (OCR'd)
level, precision is **not** monotonically increasing with confidence
band — the lowest band (0.0-0.5) actually scores higher precision (0.35)
than the middle bands (0.0 and 0.04), with the top band only partially
recovering (0.12). A well-calibrated engine's confidence should track
correctness; this one's doesn't, at least not on OCR'd input. Genuine,
previously invisible finding, now on record.

## Raster recall net

**This section describes a full rewrite.** An earlier, simpler version
(`src/delta/raster_recall.py`: raw pixel-diff, PIL affine warp, fixed
0.2 confidence, confidence/density-gated trigger) shipped first and is
now retired entirely, replaced by `src/delta/raster_diff.py` (region
*proposal*) + `src/delta/raster_join.py` (*adjudication*), built around
one explicit design principle: **raster localizes, symbolic classifies**
— the raw diff mask is never emitted as deltas directly; only the
residue the symbolic pipeline couldn't explain becomes a
`unclassified_visual_change` delta.

**Why the trigger gate was dropped, not carried over.** The retired
module only ran when mean extraction confidence was low or geometry
density was high — a fair heuristic for "OCR missed content entirely,"
but wrong for this rewrite's actual purpose: a valve symbol swapping
type at an unchanged tag, or a rerouted line, is exactly as likely on a
perfectly clean **native** pair (extraction confidence always 1.0) as on
a scanned one. Keeping that gate would have silently suppressed the
stage on the exact cases — `ChangeValveSymbol`, `RerouteLine`, both new
generator operators built specifically to exercise this path — it exists
to catch. The new master switch (`DELTA_RASTER_DIFF`) is unconditional.

**Mechanics (`raster_diff.py`):** warps B's raster into A's frame via
`cv2.warpAffine`, using a 2x3 matrix derived from `register.py`'s own
`Transform` (validated the same way the registration transform itself
was — a point-algebra check against `Transform.apply()`'s own
predictions, plus an image-level marker test, not trusted from the
algebra alone); diffs via SSIM (`skimage.metrics.structural_similarity`,
`1 - ssim_map` as the difference magnitude) rather than raw `|A-B|`,
specifically because a raw pixel diff lights up the 1px anti-alias
fringe on every unchanged glyph and line, while SSIM compares local
structure and tolerates sub-pixel misalignment; morphological open then
dilate cleans up the thresholded mask; connected components, filtered by
min/max area, become `ChangeRegion`s.

**A real bug the very first synthetic test caught, before any real PDF
was touched:** `cv2.normalize`'s global min-max contrast stretch,
applied independently per image, turns a near-blank page into all-zero
pixels (nothing to stretch a constant image *to*), while the same call
on the other, content-bearing image correctly spans the full range —
comparing them then reports a spurious whole-page "difference" instead
of the one real injected change. Fixed by dropping the global stretch
entirely (SSIM's own luminance/contrast terms already normalize locally,
per-window, which is the right place for this); Otsu binarization (an
optional, well-defined per-image threshold) was kept.

**A real bug the align.py bucketing already had, only exposed once real
content raised geometry density:** every geometry shape — line, circle,
rect — shares the one coarse `CanonicalElement.type` value `"geometry"`;
the actual shape lives only in `attrs["geom_kind"]`. `align.py`'s
type-bucketing (deliberately hard-partitions cross-type matches as
"never semantically correct," e.g. a `line_tag` can never become a
`valve_tag`) never extended that same guarantee to geometry sub-kinds,
because with ~7 geometry elements per synthetic sheet it never mattered.
Once valve-symbol glyphs (2-3 real vector shapes per valve, dozens per
sheet) pushed density up, the Hungarian matcher genuinely started
pairing a line against a circle on a **producer-variation null pair** —
content similarity is always 1.0 for empty-content geometry, so shape
carried zero cost signal without a fix. `align.py` now sub-buckets
geometry by `geom_kind` too (`_bucket_key`), restoring the same
"cross-shape was never correct" guarantee every other type already had.

**The interaction this design explicitly accepts, not hides:** because
"explained" is a bare bbox/centroid overlap check with no kind
requirement, a valve's own drawn glyph (2-3 raw vector shapes,
independently visible to `pdf_native.py`'s geometry extraction, exactly
like a real vendor drawing's valve icon — see
`data/samples/real_pair_valves/PROVENANCE.md`, which measured the same
thing on real content) can produce a coincidental symbolic geometry
delta (e.g. a "circle removed" when a globe valve becomes a gate valve)
that suppresses the raster net's own contribution at that exact spot.
Confirmed on the generator's own `ChangeValveSymbol` GT case: the
symbolic pipeline correctly cannot match the change by *kind* (it sees
`remove`, not `modify` — the GT's own kind), but that same `remove`
delta's bbox still counts as "explained," so the raster net doesn't get
a chance to flag it as `unclassified_visual_change` either. On this
dataset's one seed, this drove the measured recall lift on
`ChangeValveSymbol`/`RerouteLine` GT rows to **0.0** — a real, honestly
disappointing number on this specific case, not a hidden one.

**The most important honest number, from the null-pair calibration
check itself:** a true self-identical pair (`null_ident_900`, and
separately the real `26-KA-901` vendor PDF compared against itself)
produces **0** raster regions — clean. But `null_prod_901` (the
generator's actual producer-variation null pair — same content, rendered
with a different font/producer, no jitter-free re-export) produced
**71** `unclassified_visual_change` deltas in a live `make eval` run.
Root cause, confirmed directly: `null_prod`'s ground truth is (correctly)
empty — identical content means zero symbolic deltas — so there is
*nothing* for `raster_join.py`'s explained-check to suppress against,
and every real pixel-level difference a full font substitution produces
(Helvetica → Courier, across the whole page) survives straight through
as residue. This does **not** violate the null-pair false-positive
contract (`score_pair` excludes `unclassified_visual_change` from that
count by construction, same as the retired module — `hard_false_positives`
stays 0), but it is a real, honest limitation worth stating plainly:
**this stage is not robust to producer/font-rendering variation the way
the symbolic layer is**, because it has no semantic notion of "same
content, different glyph" — only pixels. The symbolic layer solves this
exact case correctly (fuzzy content matching ignores font entirely);
the raster layer, by design, cannot.

### An ensemble fix, and why it only partly closes the gap above

The obvious next question, asked directly: the symbolic layer *does*
know, independently, that the text at a given spot is unchanged (same
extracted content both sides) — shouldn't that be usable to suppress a
raster hit there, the same way a symbolic *change* already suppresses
one? Yes, and `raster_join.py` now does exactly this
(`_is_text_confirmed_unchanged`) — but it cannot be "matching text nearby
→ suppress" as a blanket rule, because `ChangeValveSymbol` (the generator
operator built specifically to exercise this stage) produces the
identical signal: a tag's text is unchanged *by design* while the valve's
drawn symbol next to it really changed. From "is there matching text
nearby?" alone, a font-substitution false positive and a genuine
valve-symbol change look the same. The real discriminator is *where* the
diff sits relative to the matched text, not *whether* it exists nearby:
a font change sits directly on the glyphs themselves; a valve-symbol
change sits in the padding margin around the tag (`render.py`'s
`_draw_valve_symbol_pdf`/`_draw_valve_symbol_dxf` draw the glyph ~8mm
offset from the tag's own anchor, deliberately non-overlapping). So the
fix suppresses only when a region is *directly, substantially* covered
(`cfg.text_confirm_overlap_frac`, default 0.6) by an element with
identical content on both sides — not merely nearby.

A live check immediately surfaced a second, narrower bug in the fix's
first version: it required *each side independently* to clear the
overlap threshold, but a monospace font renders the same string at a
different width than a proportional one, so the identical element
legitimately covered the region at, e.g., 0.56 on one side and 0.62 on
the other — an artifact of the font substitution itself, not evidence
the match was wrong. Requiring both sides to independently clear the
same fixed bar rejected exactly the case the fix exists for. Changed to
threshold the *pair's average* overlap instead (`(frac_a + frac_b) / 2
>= cfg.text_confirm_overlap_frac`) — a small, targeted correction, with
a dedicated regression test (`test_asymmetric_font_width_still_suppresses_via_pair_average`)
reproducing the exact near-miss shape.

**The honest result: 71 → 61, not 71 → 0.** Investigated directly rather
than assumed: the ensemble fix closes 10 of the 71 regions — real, and
verified with no regression on `ChangeValveSymbol`/`RerouteLine` recall
(that metric only ever adds suppression, never removes it; confirmed
directly it stayed at 0.0, not worse). The dominant remainder is a
qualitatively different problem this fix was never designed to solve. A
full-page font substitution doesn't produce one diff region per changed
glyph — `raster_diff.py`'s own morphological dilation (by design, to
merge a cluster of changed strokes into one region instead of fifty
specks) merges dozens of individual per-glyph font differences into a
handful of **giant, multi-element regions** — one, inspected directly,
spanned roughly 62% of the entire page, covering the whole numbered-notes
column and every zone label in one connected blob. No single matched
text element can ever satisfy a "does one element's own bbox account for
this region" check at that scale; the fix above only ever asks that
question, which is the right one for a region the size of a single note
or tag, and structurally the wrong one for a region the size of half the
drawing. Correctly closing this remainder would need a different check
entirely — does the *union* of many matched, content-identical elements
collectively tile the region, not "does one element cover it" — a
genuinely bigger, separate design (identifying every element overlapping
a large region, confirming each is itself unchanged, and accumulating
coverage across all of them until the region is accounted for or isn't).
Left as a stated, honest limit rather than forced through in this pass.

## Architecture review: confirmed-clean items

Two claims checked and found already correct, not fixed because nothing
was wrong:

- **Markup already wrote both revisions.** A review raised the concern
  that only one side might be annotated (leaving removals invisible);
  checking the actual code showed both A and B sides were already being
  written correctly.
- **Delta descriptions are pure templates.** A review asked to confirm no
  LLM call sits in the delta-computation path. The description-generation
  function is 100% f-string templating; a repo-wide search for LLM/API
  client references in the delta engine turns up zero hits outside
  docstrings describing intentionally-deferred future enrichment.

## A second external review: 2 real bugs fixed, 1 already tracked, 1 already mitigated

A review from a different LLM flagged 4 issues. Each was checked against
the actual code before any fix, not taken at face value.

**Real, novel bug: add/remove confidence gap.** `align.py`'s Hungarian
matcher rejects a candidate pair whenever its cost exceeds
`MAX_MATCH_COST` — but the rejected cost itself was being discarded, so
both elements became single-sided `MatchedPair`s reported with confidence
= `extraction_confidence` alone, which is hardcoded to `1.0` for every
native-PDF element. A genuinely ambiguous near-miss (a plausible match
that got rejected) and a truly unambiguous add/remove (no candidate
existed at all) were indistinguishable in the reported confidence.
Verified concretely, not just in theory: before this fix, every
false-positive `remove` delta in the L0 eval dataset — including all 5
`remove` deltas the engine produces on `not_a_pair_903` (the
deliberately-mismatched refusal-control pair) — would have reported
confidence `1.0`. Fixed by propagating the rejected candidate's cost onto
the pair as `near_miss_cost` and scaling confidence down the closer that
cost sits to `MAX_MATCH_COST` (see `align.py`'s `_match_bucket` and
`classify.py`'s `_confidence`). Live re-run after the fix: every one of
those same false-positive removes now reports confidence `0.0` instead of
`1.0`.

**Real, deliberate design gap: `precheck.py`'s fail-open fallback.** When
neither drawing number nor equipment tag is extractable on either
document, the pre-fix code proceeded with a bare warning and zero
secondary heuristic. Note this path wasn't exercised by the existing
real-sample sibling test (`test_real_sibling_drawings_are_refused`) at
all — the real 26-KA-901/902 samples both have extractable
`equipment_tag`, so the existing tier 2 already catches them; the gap
only bites in the narrower, previously-untested case. Added a third tier:
Jaccard overlap of specific tag identifiers (line/valve/nozzle/equipment/
instrument tag content) between the two documents — real revision pairs
share the vast majority of their tags unchanged; unrelated documents share
almost none, even on the same vendor template. Only when there's truly no
comparable tag content on one side either does it still fail open.

**Already tracked: `pdf_native.py`'s instrument-bubble format gap.** Real
vendor instrument bubbles stack system/function/loop text across three
separate baselines (e.g. `"26"` / `"PI"` / `"9055"` on distinct lines),
unlike the synthetic generator's single-line format. This was already
known and tracked via `@pytest.mark.xfail` — the reviewer independently
spotted a real, already-documented gap, not new information. Fixed with a
second pass (`_stack_instrument_bubbles`) gated on real circle geometry
(never a free-floating vertical-stacking heuristic, to avoid false merges
on a dense real sheet): finds circles that "own" exactly one func-shaped,
one system-shaped, and one loop-shaped orphan token, reassembles them in
`parse_instrument`'s expected order, and replaces the three orphans with
one `instrument` element. The padding radius around a circle's own bbox
needed real tuning, not a guess — inspecting the real samples directly
found the system/unit label conventionally sits *outside* the bubble to
one side, at roughly half the circle's own width away, while func/loop
sit inside it. Tuned empirically against both real samples: detected
count plateaus at a padding ratio of 0.6-1.0 and grows only marginally
even at 2x that, confirming the fix isn't bridging to unrelated nearby
bubbles. The `xfail` marker has been removed — the real-sample test now
passes for real, not as an aspiration.

**Already mitigated, not re-fixed: `retrieval.py`'s lexical-only BM25.**
`config/domain.yaml`'s alias-expansion table already exists specifically
to soften this; a full fix would mean embeddings-based retrieval (the
declared-but-unimplemented L3 layer noted elsewhere in this document).
The reviewer's broader point stands as a real, already-acknowledged
limitation, not something this pass re-addressed.
