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

A confidence-gated fallback for content extraction misses *entirely* —
deliberately not the same problem as the scanned-format precision
collapse (false positives from garbage OCR *text*, i.e. wrong classified
deltas). This targets the opposite failure: real visual content with no
extracted element at all, so the deterministic match/classify pipeline
has nothing to work with in the first place.

**Trigger:** per sheet, mean extraction confidence below a threshold, or
geometry-element count above a density threshold (the "dense geometry"
case — real vendor P&IDs run ~5000 geometry paths/sheet vs. this
project's synthetic dataset's ~80-120). Both thresholds are structurally
incapable of firing on a native-native pair — a built-in safety property,
not a special case that has to be remembered.

**Mechanics:** warps revision B's raster into revision A's frame using
the registration transform (a real similarity transform via PIL's affine
transform), takes absolute grayscale pixel difference, thresholds, finds
connected regions, drops any region overlapping an already-classified
delta's bbox, and emits the rest as a low-confidence
"unclassified visual change" with a generic templated description. No
LLM call anywhere in this path; opt-in via an env flag, default off.

The affine-transform matrix used for the warp is the *inverse* of the
registration transform — a derivation, not a guess, validated the same
way the registration transform itself was: a known non-trivial
transform, a marker at a known position, warped, and its actual output
position checked against the transform's own prediction, not trusted on
inspection of the algebra alone.

A second real bug caught before it shipped, not after: the new
`unclassified_visual_change` kind never matches any ground-truth kind by
construction, so leaving it in the normal prediction pool would count
every single raster-recall hit as an automatic false positive against a
P/R/F1 bar it was never meant to be judged by. Excluded from that pool
the same way a semantic-null-flagged delta is, reported as a plain count
instead.

The report's kind-ordering list needed the new kind added too, or these
deltas would render into the raw JSON but silently vanish from the
human-readable markdown report — caught before shipping by actually
reading the rendered report, not assumed correct from the code.

Live-verified against a real degraded pair: triggers correctly, found 15
candidate regions after dedup against already-classified deltas on that
sheet. Honestly, there's no ground truth for "genuinely missed visual
content" in this dataset to check precision against, so it isn't
possible to claim these 15 are all real misses rather than partly
compression/noise artifacts from that level's own degradation — a real,
open limitation of this feature as built, stated plainly.

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
