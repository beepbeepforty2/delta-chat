# delta-chat — Document Delta & Grounded Chat

Given two PIDs (two revisions of a document: native PDF, scanned PDF, or DWG),
compute a structured delta, emit a human+machine readable delta report, and
chat over both revisions and the report with citations.

**License:** shared publicly for hiring-assessment evaluation only — see
[LICENSE](LICENSE). No rights are granted to use this code beyond
evaluation without prior written permission.

## Status

Repo scaffold, canonical-representation seam, a **seeded, validated dataset
generator** producing labeled revision pairs with three-layer ground truth,
and the **native-PDF adapter** (`src/ingest/pdf_native.py`): fitz-based
extraction, custom span clustering (survives both PDF producer variants the
generator emits), heuristic element-type classification
(`src/canonical/classify.py`), composite-tag parsing
(`src/canonical/tags.py`), and border-grid zone computation
(`src/canonical/zones.py`) that matches the generator's own `Sheet.zone_of`
exactly. Also done: the **deterministic delta engine**
(`src/delta/{precheck,register,align,classify,report}.py`) — a not-a-pair
pre-check, an identity registration seam (real, extensible, but a no-op
for native-native pairs), type-bucketed bipartite element matching (scipy
Hungarian) with match-cost-margin confidence, and add/remove/modify/move
classification with generic per-family constant-offset cascade detection
— plus `src/cli.py`'s `run` subcommand (`make run A=... B=...`). Alignment
logic is validated exactly against the eval dataset's `gt/correspondence.json`
across all edited pairs; classification against `gt/deltas.json`. Also
done: **homegrown observability** (`src/observability/tracer.py`) —
context-manager spans, correlation id, per-span timings, JSONL structured
logs, one JSON trace file per request, failures captured (never swallowed),
threaded through `src/cli.py`'s `run` command end to end
(`make trace ID=<correlation_id>` to inspect one). Ready for chat's LLM
spans without changes once step 5 lands. Also done: the **scanned-PDF
adapter** (`src/ingest/pdf_scanned.py`) — pytesseract OCR, band-then-sweep
word clustering (see "What building this against real degraded pages
caught" below), real per-element `extraction_confidence` from OCR
confidence (not the native adapter's always-1.0), reusing
`src/canonical/{classify,zones,tags}.py` unchanged from the native
adapter -- direct validation that keeping those format-agnostic in step 2
was the right call. `PdfNativeAdapter`/`PdfScannedAdapter` detect() are
exact complements (same text-layer-richness threshold), so `≥2 formats
working end-to-end` (native PDF + scanned) is met; DWG stays a real stub.
Also done: the **eval scorecard** (`eval/{metrics,run_eval}.py`,
`make eval`) — delta P/R/F1 (overall, per change type, per format level),
primary/cascade recall split, a semantic-null column (quantifies the "no
LLM adjudication yet" gap honestly instead of hiding it), null-pair false-
positive counts, not-a-pair refusal check, results written to
`eval/results/{timestamp}.json` and diffed against the previous run. First
real run: **L0 (native) scores P=1.00 R=0.98 F1=0.99; L2 (scanned) drops to
P=0.21 R=0.88 F1=0.34** — OCR noise costs precision badly (182 false
positives from garbage text) while recall holds up reasonably (0.88),
exactly the honest per-format-level signal the eval requirements ask for.
`llm_direct` baseline arm now built (`eval/baselines/llm_direct.py`) and
wired into the scorecard — see "Eval: chat metrics and the llm_direct
baseline" below for the measured numbers. Also done: **grounded chat** (`src/chat/{retrieval,citations,
chat,llm}.py`, `make chat A=... B=...`) — homegrown BM25 retrieval over PID
A + PID B + delta-report chunks (no vector DB, same "zero infra, fully
understood" reasoning as the tracer's own homegrown choice), a fixed
citation format `[source:sheet:zone:id]` embedding border-grid zones
directly, and deterministic post-validation: an answer with no citations,
or a citation whose id was never retrieved, is programmatically overridden
into a refusal rather than trusted as given — the model's own judgment
isn't the only gate. `LLMResult` on the LLM span carries prompt/response/
model/tokens/cost per CLAUDE.md decision #8 (cost stays `None` for
providers without a configured price table — not hardcoded to Anthropic's
own pricing). Live-tested end to end against GLM 5.2 via z.ai's Anthropic-
compatible API (`base_url` + `auth_token` override on the same `anthropic`
SDK client — no separate provider code path). All lettered deliverables
(A-D) and both cross-cutting requirements (observability, eval) are now in
place, including the previously-pending eval closeout: chat correctness
(LLM-judge)/groundedness/refusal-accuracy metrics and the `llm_direct`
baseline are both live in `make eval`'s scorecard now that a real LLM
connection exists (see "Eval: chat metrics and the llm_direct baseline"
below). Registration (`src/delta/register.py`) also moved from an
always-identity placeholder to a real similarity-transform estimate from
high-confidence anchor correspondences (title-block fields, equipment tag,
zone labels), verified against synthetic ground truth with exact recovery
— see that section below for why it can't be empirically demonstrated
against this dataset's own scanned pairs despite being correctly built.
Two further additions close out the remaining bonus/architecture items:
**severity ranking** (`src/delta/severity.py`) — a deterministic
CRITICAL/HIGH/MEDIUM/LOW rule table over `Delta.kind`/`element_type`/
`field_changes` (trip/alarm setpoint changes are CRITICAL, pipe-class and
process-element identity changes are HIGH, cascades and moves are always
LOW regardless of what they land on), surfaced in the markdown report with
a per-report severity summary and critical-first ordering within each kind
— and the **markup overlay bonus** (`src/markup/overlay.py`,
`make markup A=... B=...`), which draws colored, legended delta
annotations directly onto each revision's retained L0 raster (the same
raster both adapters already produce, so one code path covers native and
scanned inputs identically). Both are live-tested against real generated
pairs, not just unit tests — see the findings sections below.

**Architecture review response (this pass).** A review against the
original spec flagged 7 deviations; two turned out to already be
non-issues on inspection (markup already wrote both A and B sides; delta
descriptions were already pure templating, zero LLM calls in the delta
path), five were real and are now closed: the confidence formula now
matches decision #3's literal product (`margin × ext_conf_a × ext_conf_b`,
was `min(ext_conf_a, ext_conf_b)`) plus a new confidence-calibration table
in the scorecard; the LLM-judge is now a structurally separate backend
seam from chat's own (`JUDGE_MODEL`/`get_judge_client()`), with the
self-judging caveat surfaced directly in the scorecard output, not just
docs, until a second credential exists; markup now writes real PDF
annotation objects (`page.add_rect_annot`) by default, appearing in
Acrobat's/Bluebeam's own markup list, with the PNG version demoted to a
`--format png` preview opt-in; semantic-null detection is built (a rule
for the DELETED-placeholder-collapse case, opt-in LLM adjudication for
reword-equivalence); and BM25 retrieval gained a curated domain-alias
table (`config/domain.yaml`) so natural phrasing ("trip", "spec") resolves
to the literal tokens actually printed on the drawing ("HH"/"LL"/"SD",
`pipe_class`). A sixth, larger item — a confidence-gated raster recall net
for content extraction misses entirely (as opposed to the L2 scorecard's
existing false-positive problem, a different failure mode) — is also
built and opt-in. See the dedicated findings sections below for each,
including a real precision regression the semantic-null rule introduced
and how it was caught (a live full-eval run, not a unit test) and fixed.

## Quick start

```bash
make install
make dataset          # generates eval/datasets/v0 (seeded, reproducible)
```

The scanned-PDF adapter needs the `tesseract` binary on `PATH` (a system
dependency, not pip-installable): `brew install tesseract` on macOS,
`apt install tesseract-ocr` on Debian/Ubuntu. `pytesseract` (already a
`pyproject.toml` dependency) is just a thin wrapper around it.

### Docker

```bash
docker build -t delta-chat .
docker run --rm delta-chat
```

The build bakes in `make dataset` and runs the full test suite
(`make test`) as a build step — a failed test fails the build, so a
successfully built image is itself evidence the containerized environment
is correct, not just the dev environment it was built on. Fully hermetic:
no credential is needed to build (every chat-related test injects a fake
`call_llm`, never a live call).

**Caveat, stated plainly:** this `Dockerfile` was written carefully against
this project's actual dependencies but has not been verified with a real
`docker build` — Docker isn't available in the sandbox this was developed
in. If the build fails on your machine, paste the error and it'll get
fixed; the design (base image, system deps, build steps) is documented
above precisely so any failure is easy to diagnose.

With no arguments, the container runs the deterministic-only scorecard
(delta P/R/F1, confidence calibration, semantic-null detection, null-pair
and not-a-pair checks) — no credential, no volume mount needed. Three
other things you'll likely want:

```bash
# full scorecard including chat correctness/groundedness/refusal +
# llm_direct baseline -- needs a real LLM credential, passed at run time,
# never baked into the image
docker run --rm --env-file .env delta-chat \
  python -m eval.run_eval --dataset eval/datasets/v0

# run against your own two PDFs -- mount a host directory in, write
# reports back out to it
docker run --rm -v "$(pwd)/mypdfs:/data" delta-chat \
  python -m src.cli run --a /data/revA.pdf --b /data/revB.pdf --out /data/reports

# chat over your own pair (needs a credential too)
docker run --rm -v "$(pwd)/mypdfs:/data" --env-file .env delta-chat \
  python -m src.cli chat --a /data/revA.pdf --b /data/revB.pdf --question "what changed?"
```

`ENTRYPOINT` is intentionally left unset (only `CMD` is set) — any command
after the image name fully replaces the default, so `make test`,
`make markup A=/data/a.pdf B=/data/b.pdf`, or a bare shell (`bash`) all
work the same way they would outside the container.

## Design decisions so far

**Canonical representation is a layered IR, not a feature space.**
L0 retained raster → L1 typed elements with bbox + zone + extraction
confidence (the diff space and the citation unit) → L2 sparse relations →
L3 embeddings used only inside the match cost. Rationale: grounding requires
discrete addressable entities; markup requires an invertible pullback to
source coordinates. Comparison happens in canonical space, not in extracted
text streams — post-hoc comparison of independently-noisy extractions is the
raw-text-diff failure mode (OCR-diff precision ≈36% in the published
document-change-detection literature).

**Composite tags are parsed, not treated as strings.** Line numbers
(`4"-PV-26-9048-GC11S-38`), instrument loops (`PIT 9055 26`), and equipment
tags decompose into fields. This turns "text changed" into "pipe class
GC11S→FC11S (material spec change)" and gives the matcher a structured
similarity metric.

**Border-grid zones (A–J × 1–12) are the location primitive** — the
domain-native way engineers cite regions, robust to scale/skew because they
are re-derived from detected border labels.

**Delta detection is deterministic; the LLM writes descriptions and answers
chat.** Detection must be reproducible and regression-testable; a bipartite
matcher's cost margin is a real confidence signal, a generated float is not.

**Observability is homegrown, not OpenTelemetry.** `src/observability/
tracer.py` is ~150 lines: context-manager spans (nesting via a process-
local stack), a correlation id per request, per-span timings, one JSON
trace file per request plus an append-only JSONL event log, and LLM spans
that capture model/prompt/response/tokens/cost the same way any other span
captures its attrs (no separate code path). Chosen over OTel because this
project needs zero infrastructure to run `make run`/`make eval` in a fresh
clone, and the entire trace format is fully understood end to end by
whoever's reading this — not a wire protocol and a collector to stand up
for a project this size. `make trace ID=<correlation_id>` pretty-prints one
(`src/observability/print_trace.py`) as the "inspectable metrics"
requirement without a dashboard. A failure inside any span is recorded
(`status=error`, exception type/message) *and* still propagates — the
tracer never swallows an exception, it just makes sure the trace gets
written before the caller sees it. The design already anticipates
multi-stage request spans and LLM spans, so chat (step 5) reuses the exact
same `Tracer` with no changes.

**Eval dataset is generated, with recorded — not invented — edit
operators.** Every operator is grounded in an edit observed between the two
real P&IDs shipped with the assignment (26-KA-901 vs 26-KA-902):
non-monotonic note-renumbering cascades, `DELETED.` placeholder collapse
(semantic null), per-family systematic tag renumbering (instruments −39,
nozzles −1000 in the real pair), pipe-class changes, setpoint changes,
equivalent rewording. Ground truth is exact by construction and labeled at
three layers: element inventory (L1), correspondence map (L2), typed deltas
with cascade/primary/semantic-null tags (L3). Null pairs (identity,
producer-variation, reword-only) and a not-a-pair sibling drawing measure
the false-positive behavior that determines real-world trust.

**Generator is self-validating**: a round-trip model differ must recover
every emitted delta; a render-fidelity check asserts no text is dropped; a
canary baseline (exact string set-difference) must score mid-range —
too high means the set is trivial, too low means it is broken.

## Repo layout

```
src/
  ingest/        FormatAdapter seam; pdf_native (done), pdf_scanned, dwg (real stub)
  canonical/     layered IR (model.py); zones.py, tags.py, classify.py
  delta/         precheck -> register -> align (bipartite matcher) -> classify
                 -> severity -> semantic_null -> raster_recall -> report
  cli.py         `run`/`chat`/`markup` subcommands
  chat/          retrieval (BM25 + domain aliases) over PID A + PID B +
                 delta report; llm.py (chat + judge backend seams); cited answers
  markup/        overlay.py (raster PNG preview) + pdf_annotate.py (real PDF
                 annotations, default)
  observability/ homegrown tracer: spans, correlation ids, LLM telemetry
config/
  domain.yaml    BM25 query-alias table (chat retrieval)
eval/
  datasets/generator/   seeded pair generator (model, ops, render, generate)
  baselines/            llm_direct.py, backend_compare.py
  calibration.py        confidence-band precision check
  run_eval.py           scorecard: delta P/R/F1, calibration, semantic-null
                         detection, chat correctness/groundedness/refusal
data/samples/           provenance-documented pairs (see PROVENANCE.md)
```

## Plan

1. ~~Dataset generator with layered GT~~ (this commit)
2. ~~Native-PDF adapter → canonical; zone detection; tag parsing~~ (this commit)
3. ~~Alignment (register → bipartite match → classify) + delta report~~ (this commit)
4. ~~Scanned-PDF adapter (OCR) — degradation ladder already generated~~ (this commit)
5. ~~Chat with citation post-validation; refuse-on-unsupported~~ (this commit)
6. ~~Tracer threaded through; eval runner + scorecard incl. llm_direct baseline~~ (this commit)
7. ~~Markup overlay (bonus)~~ (this commit)

## What was deliberately cut (so far)

- DWG parsing: real stub behind the real seam (`src/ingest/dwg.py` documents
  the ODA/LibreDWG→DXF→ezdxf path). The generator's DXF leg proves entity
  compatibility.
- Web UI: CLI chat.
- Multi-hundred-sheet scaling: single-sheet pairs; the sheet-matching stage
  and per-sheet-pair delta design leave the parallelization seam in place.

## Known limitation, found validating against the real samples

The real 26-KA-901/902 PDFs run ~800 text elements + ~5000 geometry paths
per sheet, versus the generator's ~80-120 — validating `pdf_native.py`
against them (rather than only the synthetic dataset) surfaced a real gap:
instrument bubbles in the real drawings split system/function/loop across
three stacked text baselines (e.g. `26` / `PI` / `9055` on separate lines),
not the generator's single-line `FUNC LOOP SYS` format `parse_instrument`
expects. Same-baseline clustering correctly keeps them separate — they are
genuinely distinct text runs — so this isn't a clustering bug, it's a
composition-format gap. Fixing it needs 2D proximity-based grouping instead
of same-baseline-only clustering; tracked as an `xfail` in
`tests/test_pdf_native_real_samples.py` rather than silently passing.
Everything else transferred cleanly: all 44 zone labels found on both real
sheets, and line tags / valve tags / nozzles / DELETED-placeholder notes
all parse correctly via the same Tier-1 regexes the generator's format was
modeled from.

## Delta engine: what building it against real GT actually caught

Three real bugs surfaced while validating the alignment/classification
logic against `eval/datasets/v0`'s ground truth, not by inspection:

- **Type-bucketed matching initially couldn't recover `DeleteNoteKeepPlaceholder`**
  (a note's classified type changes from `note` to `note_deleted` as *part
  of* the edit) -- bucketing strictly by exact type meant that
  correspondence was structurally unrecoverable. Fixed with a small
  type-compatibility-group concept (`align.py::TYPE_MATCH_GROUPS`); the
  general lesson (kept in the code comment) is that type itself can be the
  subject of a delta, not just a partition key for matching.
- **Cascade detection needed a "single-field-only" guard.** A `dcn_note`
  whose `note_no` shifts from a renumbering cascade *and* whose `dcns`
  list gains a genuinely new entry must stay a standalone primary change
  -- it has meaning beyond the renumber. Grouping only fires on deltas
  whose *entire* change is one numeric field.
- **Producer-jitter float noise was flagged as a real change** on the
  `null_prod` null pair (same content, two PDF producers): a derived
  geometry attribute (`r_norm`) differed at the 1e-8 scale from sub-point
  rendering jitter, and exact float equality treated that as a delta.
  `null_prod` must emit zero deltas by construction (any emission is a
  false positive) -- caught by actually running the null pair, not by the
  isolated GT-element tests (which don't extract geometry from a real
  render). Fixed with a numeric tolerance (`DELTA_NUMERIC_EQ_TOL`, default
  `1e-4`, far below any real integer field change in this domain).

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

## Scanned-PDF adapter: what building it against real degraded pages caught

A spike against a real degraded page (`eval/datasets/v0/pairs/*/L1-L3.pdf`,
the generator's rasterized degradation ladder) surfaced the same *class* of
clustering bug `pdf_native.py` hit in step 2, but a worse variant of it.
Tesseract's own line/block grouping merged an entire row of widely-spaced
zone-grid digits (~540px apart) into one polluted string, so word-level
OCR output plus manual gap-based re-clustering was the right approach
going in (confirmed against real gap measurements: ~7-9px between words in
one line vs. ~540px between unrelated zone digits, a ~60x separation any
reasonable threshold clears).

The bug that slipped through the first pass: OCR word boxes jitter a pixel
or two vertically even on one physical line (unlike fitz's exact vector
coordinates, where every word from one `drawString` call shares an
identical origin y). Sorting by `(round(y0, 4), x0)` as a single key put
jittered words from the *same* line into different sort buckets, which
silently scrambled left-to-right order and corrupted the gap merge for
whichever words happened to jitter across a rounding boundary. Caught by
running the real CLI end-to-end (native PDF vs. a scanned revision, not
just the adapter's own unit tests): a genuine note, "OIL CHANGE BY USING
TEMPORARY ARRANGEMENT WITH HOSES.", fragmented into four spurious
add/remove deltas against its native-PDF counterpart. Fixed with a proper
two-pass cluster: band words into rows by y-proximity first (order-
independent), *then* sweep each band left-to-right by x — the same gap
merge as before, just no longer dependent on a fragile sort key. Reduced a
cross-format smoke-test run from 112 primary deltas down to 82, with the
real semantic changes (an inserted note, a revision bump) now showing up
as single clean adds instead of noise.

What's left, honestly: OCR misreads on small/degraded text still produce
some low-confidence noise deltas (e.g. zone-grid digits read as
punctuation), and `extraction_confidence` correctly reflects that — this
is expected OCR behavior, not silently hidden, and exactly the signal the
eval harness's per-format-level scoring (L0 vs L1-L3) is meant to
quantify once it exists. No CV-based geometry extraction (line/circle
detection from the raster) is attempted — a real computer-vision problem,
a bigger lift than the OCR text path for the same time budget, and a
deliberate, documented cut rather than a silent one.

## Chat: what building it against a real GLM connection caught

Two real findings from actually running questions through the live model,
not just unit tests with a fake LLM call:

- **The LLM span was missing two of the five fields CLAUDE.md decision #8
  requires.** The first pass only recorded `model`/`prompt`/`response` on
  the span — caught by inspecting a real trace after the first live chat
  session, not by any test (the unit tests all used a fake `call_llm`
  returning a bare string, which never exercised the real response's
  `usage` object at all). Fixed by introducing `LLMResult` (text + tokens +
  cost, cost `None` unless `LLM_COST_PER_1K_*` is configured — no
  hardcoded Anthropic pricing for a third-party provider) and adding a
  regression test that asserts the span actually carries `tokens_in`/
  `tokens_out`/`cost_usd`, not just that the call succeeds.
- **A refusal-expected probe got an answer instead of a refusal — and it
  was arguably the better response.** Asked "What changed on sheet 7?"
  (the document has one sheet; this exact probe shape is in the eval
  dataset's `qa.jsonl` with `expected_behavior: refuse`), the model didn't
  say `REFUSED:` — it explained there's no sheet 7, correctly identified
  that the "7"s it found in context were zone labels rather than a sheet
  number, and cited all of it validly. That passes this codebase's own
  post-validation (real citations, all ids exist in the retrieved set), so
  it wasn't overridden into a refusal. Whether that's "correct" is a real,
  unresolved judgment call — an honest, cited explanation of why the
  premise is false arguably beats a bare refusal, but it diverges from the
  dataset's strict refuse/answer grading, which will show up as a
  refusal-accuracy miss once the eval harness's chat metrics are built.
  Documented here rather than silently tuned away.

Live-tested via `make chat A=... B=...` against a generated pair over
GLM 5.2 (z.ai's Anthropic-compatible API): correctly grounded, correctly
cited answers to in-scope questions; correct refusal on an out-of-domain
question ("what is the capital of France?"); the sheet-7 case above.

## Retrieval: what building the eval harness against it caught

Building `eval/chat_eval.py` (running real `qa.jsonl` questions through the
live chat path, not fake `call_llm`) surfaced three real `BM25Index` bugs
in one sitting — none visible from `test_chat_retrieval.py`'s hand-built
fixture chunks, all visible within the first few live transcripts:

- **Hyphenated tags didn't match their own drawing.** A question about
  `FIT-9050` shares zero tokens with the printed form `FIT 9050 26 SD
  HH:150 LL:120` (space-separated on the page, hyphenated the way an
  engineer would actually ask about it) — the exact composite-tag
  inconsistency CLAUDE.md decision #4 exists to parse, but retrieval's
  plain tokenizer never split on it. Confirmed as a real miss before
  fixing it: the model refused, citing "the provided context does not
  include the revised document's version of FIT-9050." Fixed by making
  `_tokenize` additionally emit hyphen-split sub-tokens (`"fit-9050"` also
  indexes as `"fit"` + `"9050"`), additive so the whole-token match is
  never lost.
- **Zone was unsearchable.** `Chunk.zone` is metadata, never part of
  `.content`, so "did anything change in zone B-1?" had no tokens to match
  against at all — a hard zero-result miss, not a ranking problem. Fixed
  by folding `zone` tokens into the indexed surface (never into
  `.content`, so citations still show only real drawing text). That
  promptly caused a second, more interesting bug: a `zone_label` element
  (the single literal digit/letter rendered at the sheet border to make
  the zone grid itself, e.g. content `"B"`) is an extremely short
  "document," and BM25's own length normalization makes a 1-token match
  against a 1-token document score enormously higher than the same match
  against a real multi-word note — so the zone letter's own rendering
  artifact dominated every zone query. Fixed by excluding `zone_label`
  from `build_chunks` entirely (it's redundant with the `.zone` field
  every other chunk already carries; a citation to the zone letter itself
  is meaningless).
- **Short static content still out-ranked the actual delta.** Even after
  the zone fix, the real "note added" delta for a live zone query ranked
  19th, behind a run of unrelated one-line static notes sharing the same
  zone tokens, for the same length-normalization reason. Fixed with a
  modest source-weighting boost (`DELTA_SOURCE_BOOST = 1.5`) for
  `source == "delta"` chunks — a defensible field weighting (delta chunks
  are what "what changed" questions are actually asking for, and there
  are always far fewer of them than raw A/B element chunks), tuned against
  the live case until it consistently surfaced the right chunk rather than
  guessed at.

A fourth bug was in the eval harness's own groundedness check, not
retrieval: `eval/chat_eval.py`'s first-pass sentence-extraction heuristic
walked back to the nearest `.` to isolate the clause a citation is
attached to, and a cited chunk whose own content ends in a period (e.g.
quoting a note whose text is `"5. DELETED."`) tripped that boundary and
discarded exactly the quoted content that should have overlapped the
chunk — silently reporting real, correct citations as unsupported. Fixed
by switching to a fixed 200-char lookback window instead of naive sentence
splitting.

What's still honestly unresolved, not silently tuned around: a broad
"what changed on sheet 1?" query has almost no distinctive vocabulary to
rank against (this codebase's delta descriptions are templated, not
prose), so which specific deltas BM25 surfaces for a *summary* question is
somewhat arbitrary — targeted questions (a zone, a tag, a specific field)
retrieve reliably; broad ones don't consistently surface every primary
change. A production system would likely route "what changed on sheet N"
to a deterministic tool call over the structured delta list rather than
through retrieval at all, since the full answer already exists
losslessly upstream of the chat layer. Also unresolved: a model can
express a refusal in substance ("the provided context does not contain
information about sheet 7") without the literal `REFUSED:` prefix our
forced-refusal gate looks for, so `chat.py`'s own `refused` flag reads
`False` even though the content is a correct refusal — this shows up as a
real, measured gap between refusal-accuracy (scored on the flag) and
LLM-judge correctness (scored on content) in the eval numbers below.

## Registration: from an honest no-op to a real, verified similarity transform

`src/delta/register.py` always returned `Transform()` (identity) until
this pass — a real, extensible seam, but with no actual anchor-based
estimation behind it. Replaced with the design CLAUDE.md decision #3
specifies: elements from the drawing's fixed scaffolding (title-block
fields, the equipment tag, border-grid zone labels — never touched by any
of the generator's edit operators) whose content is identical between A
and B *and* unique within the sheet become anchor correspondences; two or
more anchors give a full similarity transform (uniform scale + rotation +
translation) via the closed-form Umeyama/Kabsch least-squares solution;
fewer than two falls back to identity rather than force-fitting an
underdetermined estimate.

Verified two ways: `tests/test_delta_register.py` constructs a *known*
non-trivial transform (scale 1.1, rotation 0.05 rad, translation), derives
synthetic anchor pairs from it, and checks `register()` recovers it to
1e-6 — including on a held-out point never used as an anchor, the actual
property `match_elements`' spatial cost term depends on. Against real
native pairs it correctly converges on near-identity (as it should — no
skew to correct).

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
the dataset's synthetic degradation ladder (a fixed recipe per level, not
per-document jitter), not evidence against the registration math, which
the synthetic-ground-truth tests verify independently and directly. Real
independently-scanned document pairs — e.g. two separate physical scans of
the same drawing revision, which is what `data/samples/` would need to
truly exercise this — are the case this seam exists for.

## Eval: chat metrics and the llm_direct baseline

First full run against GLM 5.2, all 43 questions across `eval/datasets/v0`
(6 edited + 3 null pairs' `qa.jsonl`; `not_a_pair`'s single refuse-expected
probe is scored without a live call, exactly like `cmd_chat` short-circuits
at precheck):

```
refusal accuracy:        0.79  (34/43)
groundedness:             fraction_fully_supported=0.81  citation_support_rate=0.87  (n=26 answered-with-citations)
correctness (LLM-judge):  0.60  (43/43 judged, 0 unparseable)
judge validation:         5/5 agreement vs hand-checked answers
```

The gap between refusal accuracy (0.79) and LLM-judge correctness (0.60) is
real, not noise, and traces directly to the two structural findings above:
refusal accuracy alone flatters the system (a "sheet 7" question phrased as
a substantive explanation instead of a literal `REFUSED:` scores as a
refusal-accuracy *miss* even when the judge — reading content, not the
flag — correctly calls it right), while correctness is dragged down by
questions with no chunk to cite for an absence ("did any setpoints
change?" when none did) and by the broad-summary retrieval weakness (a
"what changed on sheet 1?" answer that's fully grounded in real citations
but incomplete relative to the reference's fuller list still reads as
substantively wrong to the judge). Groundedness (0.81-0.87) is the
strongest number and the most direct measurement of decision #7 doing its
job: when the system does answer, what it cites usually says what the
answer claims it says. The judge itself is trustworthy for this report —
5/5 agreement against hand-labeled ground truth, including on the hardest
case in the sample (a null-pair hallucination where every citation
resolves to a real element yet the substance is wrong, see
`HAND_CHECKED_SAMPLE` in `eval/chat_eval.py`).

`llm_direct` baseline (3 runs/pair @ temperature=0, scored through the
identical `score_pair` path the real engine uses):

```
edited_000   F1 mean=0.9667  stdev=0.0577   n_deltas mean=10     stdev=0.0
edited_001   F1 mean=1.0000  stdev=0.0000   n_deltas mean=5      stdev=0.0
edited_002   F1 mean=0.8070  stdev=0.3143   n_deltas mean=16.33  stdev=8.9629
edited_003   F1 mean=1.0000  stdev=0.0000   n_deltas mean=5      stdev=0.0
edited_004   F1 mean=1.0000  stdev=0.0000   n_deltas mean=5      stdev=0.0
edited_005   F1 mean=0.9091  stdev=0.0000   n_deltas mean=11     stdev=0.0
aggregate F1 mean=0.9471   within-pair F1 stdev (mean)=0.062
```

This is the actual, measured version of the determinism argument CLAUDE.md
asks for, not an assumed one — and it's more nuanced than "the LLM is bad
at this." On raw F1 the baseline is close to the deterministic engine's own
L0 score (0.9471 vs 0.9910) — a capable model can zero-shot a reasonable
delta list when handed the same zone-annotated extracted text the real
engine works from (see the llm_direct docstring for why raw PDF bytes
aren't used: a live spike showed this provider's vision path silently
hallucinates instead of reading the page). What raw F1 hides is
*reproducibility*: `edited_002` swung from F1=1.00 to F1=0.50 to F1≈0.93 in
three back-to-back "temperature=0" runs, with the emitted delta count
ranging 8-24 for the exact same input — real, measured non-determinism
this project's actual submission would have to explain away every time a
reviewer re-ran it. The deterministic engine, by construction, cannot do
that: same input, same output, every time, which is the entire argument
CLAUDE.md decision #3 is making, now backed by a number instead of an
assertion.

## Severity ranking

No eval ground truth exists for this (the dataset generator predates the
capability, so `gt/deltas.json` has no severity field), so it isn't scored
through `eval/metrics.py`'s P/R/F1 machinery — it's a rule table
(`src/delta/severity.py`) validated against known real deltas instead:
`tests/test_delta_severity.py` runs the actual engine (not synthetic
`Delta` objects) against `edited_003` (a real trip-setpoint change,
PDIT-9017 HH 235→240) and `edited_004` (a real pipe-class change,
AC21S→GC11S) and asserts CRITICAL / HIGH respectively. Live-checked via
`make run` on `edited_003`: the setpoint change is the only CRITICAL entry
and sorts first within "Modify," while the note edit, DCN update, and
title-field REV bump all correctly land as LOW and sort after it —
exactly the "surface the safety-relevant change first" behavior the rule
table is for. `is_cascade` overrides every other rule (a cascade member is
always LOW no matter what field or element type it landed on) — a
renumbering side-effect has no independent engineering content, matching
`report.py`'s own "primary changes first" priority.

## Markup overlay (bonus)

**Update:** this section describes the original raster (PNG) version,
`src/markup/overlay.py`. It's still there and still useful (a PDF can't be
`Read` and visually inspected in a terminal the way a PNG can, which is
exactly how the live checks below were done) but is now the
`--format png` opt-in, not the default — see "Markup: from raster preview
to real PDF annotations" further down for why and what replaced it as the
default.

Draws on each revision's retained L0 raster
(`CanonicalDocument.raster_paths`, populated by both adapters already, per
`model.py`'s own docstring: "retained for markup + recall net") rather
than the original PDF's vector page — the same code path works
identically for native and scanned inputs, since `CanonicalElement.bbox`
is normalized against that same raster for both. One real subtlety caught
before it became a test bug rather than after: `CanonicalSheet.width`/
`height` hold the *pre-rasterization* page size (points for native, pixels
for scanned) — not the raster's own resolution, which can differ once
`RASTER_DPI` is applied — so denormalizing bbox coordinates has to use the
opened raster image's actual pixel size, not those fields.

Colored by kind (green add / red remove / amber modify / blue move,
unfilled), padded outward 4px so a box doesn't clip the glyphs it's
highlighting, with a fixed legend and cascade members drawn thinner and
unfilled so primary changes read as visually dominant. Live-verified
against a real pair (`edited_003`, `make markup`), not just synthetic
fixtures — visually inspected both output PNGs directly: revision A shows
the *old* setpoint (HH:235), pipe-class-style note text, single DCN
reference, and "REV A," all correctly amber; revision B shows the *new*
values and an additional green box around the newly-added revision-table
row, with nothing marked on either side for content that didn't change.
One design note worth flagging, not silently avoided: the legend occupies
a fixed top-left panel, which a synthetic small test image can trivially
collide with (caught early — two tests initially sampled pixels the
legend itself was covering, not the delta box underneath, and were fixed
to sample elsewhere) — on a real full-resolution P&ID raster this is a
small corner of the page and did not visibly collide with drawing content
in the live check above, but it's a fixed position, not a content-aware
one, so a title block or dense drawing content occupying that exact corner
remains a real, undefended edge case.

## Confidence: a real formula bug, and a calibration check that immediately found something

`_confidence()` (`src/delta/classify.py`) used `min(ext_conf_a, ext_conf_b)`
where CLAUDE.md decision #3 specifies a product
(`margin × ext_conf_a × ext_conf_b`). It went unnoticed because the two
formulas agree whenever both sides are 1.0 — every native-native pair,
since `pdf_native.py` hardcodes `extraction_confidence=1.0` — so months of
native-pair-heavy manual testing could never have caught it. Fixed to the
literal product; single-sided add/remove deltas were never affected (they
only ever had one side's confidence to begin with).

"Confidence appears nowhere as a stage and there's no reliability check"
was also real — nothing validated whether confidence actually predicted
correctness. Added `eval/calibration.py`: buckets matched vs.
false-positive predicted deltas by confidence band and reports precision
per band, wired into the scorecard. First real run immediately surfaced
something worth knowing rather than assuming: on the L2 (scanned) level,
precision is **not** monotonically increasing with confidence band —
`0.0-0.5` scores 0.35 precision, `0.5-0.75` scores 0.0, `0.75-0.9` scores
0.04, and `0.9-1.0` only partially recovers to 0.12. A well-calibrated
engine's confidence should track correctness; this one's doesn't, at
least not on OCR'd input. That's a genuine, previously invisible finding
now on record instead of an assumed-but-unverified formula — "a confidence
you never validated is decoration," borne out exactly as flagged.

## Judge/chat backend decoupling, and the three-backend table

`eval/chat_eval.py`'s LLM-judge called `src.chat.chat._default_call_llm`
directly — the literal same function, model, and client answering the
questions it was grading. Self-judging bias was real and undocumented next
to the reported 0.60 correctness number. Fixed structurally, not just
documented around: `src/chat/llm.py` now has `get_judge_client()`/
`JUDGE_MODEL`, a separate seam that falls back to the chat backend's own
config when `JUDGE_LLM_*`/`JUDGE_MODEL` are unset (true today — no second
credential is configured in this environment) and switches with a one-line
`.env` change the moment one exists. `judge_is_same_backend()` reports
this live: the scorecard now prints
`*** SAME BACKEND AS CHAT -- self-judging risk, treat as an upper bound ***`
directly next to the correctness number, and flags the 5-hand-checked
validation sample size explicitly too (`n=5 is small enough that a perfect
score is not strong evidence on its own`) — both caveats surfaced where
the number is reported, not left to go stale in a doc nobody re-reads.

The three-backend cost/latency/determinism comparison table itself is
scaffolded, not run — `eval/baselines/backend_compare.py`, config via
`BACKEND_2_*`/`BACKEND_3_*` in `.env.example`. No second real credential
is available in this environment (confirmed: only `LLM_*` is set), and
fabricating a comparison without one would be a lie by construction, so it
reports its actual state honestly instead: `1/1 backends configured --
comparison needs at least 2`, verified via a real invocation
(`python -m eval.baselines.backend_compare`), and exits 0. The moment a
second credential exists, it runs for real with zero code changes.

## Markup: from raster preview to real PDF annotations

`overlay.py`'s PNG output already wrote both A and B sides correctly (a
concern raised in review didn't hold once checked against the actual
code), but a PNG is a flat image: no toggle, no structured metadata, never
appears in Acrobat's or Bluebeam's own markup/comments list, and
permanently rasterizes content that should stay vector. `src/markup/
pdf_annotate.py` uses `page.add_rect_annot()` — real native PDF "Square"
annotation objects, directly on the *original* PDF pages, not the raster
cache — and is now `make markup`'s default (`--format png` keeps the old
behavior as an opt-in preview).

One naming gap worth flagging: the original spec asked for
`page.add_square_annot()`; that method doesn't exist in the installed
PyMuPDF (1.28.0) — confirmed via direct introspection before writing any
code around it. The real method is `add_rect_annot()`, producing the same
annotation *type* ("Square" is the PDF spec's own name for it, not a
PyMuPDF quirk) — a naming mismatch in the spec, not a missing capability.

Denormalizing bbox coordinates against `page.rect.width`/`height` (not
raster pixel size) works for both native and scanned inputs without any
format-specific branching, for the same reason `overlay.py`'s raster
version does: bbox is normalized [0,1] as a *fraction* of the page, and
that fraction is basis-independent as long as the OCR'd raster is a
full-page, non-cropped render (`get_pixmap()` already is).

Live-verified two ways, not just unit tests: `tests/test_markup_pdf_
annotate.py` builds a real PDF, runs annotation, reopens the *output* file
and asserts on real `page.annots()` objects (content, color, type) — a
round trip, not "a file was written." And a live `make markup` run against
`edited_003`, inspected both by listing the real annotation objects
(`fitz`, content strings matching each delta's own description exactly)
and by rendering the annotated PDF to an image and looking at it: amber
boxes on the exact setpoint/note/DCN/REV locations that changed, a green
box around the newly-added revision row, nothing marked where nothing
changed, and the underlying vector drawing fully intact underneath (no
legend baked in this time — a real PDF viewer's own comments panel *is*
the legend).

## Semantic-null detection: a rule, an LLM pass, and a bug the live eval caught that a unit test didn't

`Delta.semantic_null` didn't exist; every equivalent reword and every
DELETED-range collapse was indistinguishable from a real change in this
engine's own output. `eval/metrics.py`'s `semantic_null_emission_rate` had
already measured the cost of that gap honestly (1.0 on the last run before
this work — 100% of GT null entries got matched by a normal, unflagged
delta) without anything actually closing it.

Two mechanisms for two genuinely different sub-cases, per CLAUDE.md
decision #3(c) ("isolated, cached, documented as the non-deterministic
zone"):

- **Rule, no LLM** (`src/delta/semantic_null.py::_rule_deleted_placeholder`):
  fires only when a `note_deleted` element transitions *into* a collapsed
  range (`"range"` appears in `field_changes` — the precise, unambiguous
  signature `CollapseDeletedRange` leaves, since only the collapsed form
  ever carries a `range` attr). Deliberately excludes `is_cascade=True`
  deltas and add/remove of a `note_deleted` element (ambiguous — could be
  an unmatched half of the same collapse, or real content newly vanishing
  via `DeleteNoteKeepPlaceholder`; a false null is worse than a missed
  one).
- **LLM adjudication** (`adjudicate_semantic_null`, opt-in via
  `DELTA_SEMANTIC_NULL_LLM=1`, default off): one isolated, cached call per
  candidate — `modify` deltas where `_field_changes()` fell through to the
  generic `"content"` key (structured attrs diffing found nothing, exactly
  the "words changed, unclear if meaning did" case) and the rule didn't
  already resolve it. Cached by `(old, new)` content pair so a repeated
  pair costs one call, not one per occurrence.

**A real precision bug, caught by the live eval, not a unit test.** The
first version of the rule matched on "the entire field_changes is a
subset of `note_deleted`'s own bookkeeping fields" — which also matched an
*ordinary* +1 renumbering cascade of a `note_deleted` element (e.g. "5.
DELETED." → "6. DELETED." because an earlier note was inserted elsewhere).
That's a real, GT-expected delta — exactly what `is_cascade`/
`cascade_recall` exist to track — not a structural no-op. Every unit test
for the rule passed; the bug only showed up running the *actual* eval
pipeline end to end: L0's overall recall dropped from 0.98 to 0.93 (fn
1→4), traced via the new `semantic_null_detection` P/R column
(`tp=0 fp=3 fn=1` — 3 real deltas wrongly flagged null). Fixed by tightening
the rule to the precise `"range"` transition signature plus an
`is_cascade` guard, re-verified against the same live run: L0 back to
`P=1.00 R=0.98 F1=0.99`, `semantic_null_detection` precision back to 1.0.
Regression tests for both the original bug and the fix are in
`tests/test_delta_semantic_null.py`. This is the same lesson `pdf_native.py`
and `pdf_scanned.py` each taught earlier in this project: a synthetic unit
test can pass while the real pipeline is wrong, and the only thing that
reliably catches that is actually running it.

Live-verified with a real GLM call (`DELTA_SEMANTIC_NULL_LLM=1 make run`
against `null_reword_902`): the DELETED-collapse case
(`'10. DELETED.' -> '9-10. DELETED.'`) correctly flagged by the rule alone
(no LLM call made); the reword case
(`'6. ATMOSPHERIC VENT.' -> '6. VENT TO ATMOSPHERE.'`) correctly flagged
by the live LLM adjudication with a sensible reason; and a genuinely
ambiguous unmatched remove (`'9. DELETED.'`) correctly left unflagged —
the deliberate conservatism working exactly as designed, visible in real
output.

One correction to how this gets measured: the *existing*
`semantic_null_emission_rate` metric does **not** move when the engine's
own flag improves — it measures whether a *normal, unflagged* prediction
happens to description-match a null GT entry, which is unrelated to our
own flag by construction. The metric that actually moves is the *new*
`semantic_null_detection` P/R column: on `null_reword_902`, recall goes
0.33 (rule alone, catches 1 of 3 null GT entries) → 0.67 (rule + live LLM,
catches 2 of 3), precision 1.0 in both cases. Both columns are kept
deliberately — the first measures the cost of doing nothing, the second
measures whether the actual detector is any good — but conflating them
would have been a wrong claim in this same write-up if not checked
against the real numbers first.

## Retrieval alias expansion

BM25 is purely lexical (deliberate — no vector DB), so a natural question
sharing zero literal tokens with the drawing's own notation retrieves
nothing: "trip" shares no token with a setpoint chunk's actual printed
content (`"...SD HH:150 LL:120"`), "spec" shares none with the field name
`pipe_class`. `config/domain.yaml` is a curated first-pass alias table
(query phrase → literal drawing tokens), loaded by
`src/chat/retrieval.py` and expanded into the query's token set additively
— never replaces the base tokenization, so it can only help an
already-working query, never hurt one. A missing config file is not an
error (empty table, base lexical matching only) — retrieval never
*requires* this file to function.

Live-verified against a real pair (`edited_003`): the query "did the trip
setpoint change?" retrieved zero results before this change (no shared
tokens at all) and now correctly surfaces the actual instrument chunks
(`PDIT 9017 26 SD HH:235 LL:110`, among others) on the first try, including
the one whose setpoint actually changed in this pair.

## Raster recall net

A confidence-gated fallback for content extraction misses *entirely* —
deliberately not the same problem as the L2 scorecard's existing
precision collapse (182 false positives from garbage OCR *text*, i.e.
wrong classified deltas). This targets the opposite failure: real visual
content with no `CanonicalElement` at all, so the deterministic match/
classify pipeline has nothing to work with in the first place. Both are
real scanned-input problems; this only addresses the second.

**Trigger** (`should_run_raster_recall`): per sheet, mean
`extraction_confidence` below 0.75, or geometry-element count above 2000
(the "dense geometry" case — real vendor P&IDs run ~5000 geometry
paths/sheet vs. this project's synthetic dataset's ~80-120, per the
earlier real-sample validation). Both thresholds are structurally
incapable of firing on a native-native pair: `extraction_confidence` is
hardcoded `1.0` there and geometry counts stay in the low hundreds — a
built-in safety property, not a special case that has to be remembered.

**Mechanics**: warps B's raster into A's frame using `register.py`'s
`Transform` (a real similarity transform: scale + rotation + translation,
via PIL's `Image.transform(AFFINE, ...)`), takes absolute grayscale pixel
difference, thresholds, finds connected regions (`scipy.ndimage.label`),
drops any region overlapping an already-classified delta's bbox (already
explained), and emits the rest as `kind="unclassified_visual_change"` with
a fixed, deliberately low confidence (`0.2` — not a computed number
pretending to be precise) and a generic templated description. No LLM
call anywhere in this path — it stays inside the deterministic-by-default
engine, opt-in via `DELTA_RASTER_RECALL=1` (default off, so it never
changes an existing native-pair scorecard number unless explicitly
enabled).

The PIL affine-transform matrix is the *inverse* of `register.py`'s own
`Transform.apply` — a derivation, not a guess, and one non-trivial enough
that it was validated the same way `register.py`'s own transform was: a
known non-trivial `Transform` (scale=1.1, rotation=0.05, translation), a
marker at a known position, warped, and its actual output position
checked against `Transform.apply`'s prediction — not trusted on
inspection of the algebra alone (`tests/test_delta_raster_recall.py::
test_warp_places_marker_at_transform_predicted_position`).

A second real bug caught before it shipped, not after: the new
`unclassified_visual_change` kind never matches any GT kind by
construction (GT only has add/remove/modify/move), so leaving it in the
normal prediction pool would count every single raster-recall hit as an
automatic false positive against a P/R/F1 bar it was never meant to be
judged by. `eval/metrics.py::score_pair` excludes it from the "real
change" pool the same way a semantic-null-flagged delta is excluded, and
reports a plain count (`n_unclassified_visual_change`) instead — visible
in the scorecard, never silently tanking precision the moment the flag is
turned on.

`report.py`'s `KIND_ORDER` needed the new kind added too, or these deltas
would render into the raw JSON but silently vanish from the human-readable
markdown report (`by_kind` grouping happens for every kind, but the
section-printing loop only iterated the original four) — caught before
shipping by actually reading the rendered report, not assumed correct
from the code.

Live-verified against a real degraded pair (`edited_000` at L3,
`DELTA_RASTER_RECALL=1`): triggers correctly (L3's OCR confidence sits
below threshold), found 15 candidate regions after dedup against the 16
already-classified deltas on that sheet. Honestly, there's no ground
truth for "genuinely missed visual content" in this dataset to check
precision against, so it isn't possible to claim these 15 are all real
misses rather than partly JPEG-compression/noise artifacts from L3's own
degradation ladder (`noise_sigma=5.29`, `jpeg_q=70`) — that's a real,
open limitation of this feature as built, stated plainly rather than
implied away by a clean-looking count.

## Item 7, confirmed: delta descriptions are pure templates

The review asked to confirm no LLM call sits in the delta path.
`_describe()` (`classify.py`) is 100% f-string templating
(`f"{b.type} added: {b.content[:60]}"` and similar); a grep of all of
`src/delta/` for `llm`/`anthropic`/`LLM` turns up zero hits outside
docstrings describing future, not-yet-built enrichment. No code change —
this one was already correct, verification only.
