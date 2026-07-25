# delta-chat — Document Delta & Grounded Chat

Given two PIDs (two revisions of a piping & instrumentation diagram:
native PDF, scanned PDF, or DWG), compute a structured delta, emit a
human+machine-readable delta report with real PDF markup annotations, and
chat over both revisions and the delta with citations.

**Video walkthrough:** https://youtu.be/C2SbISJMrWI

**License:** shared publicly for hiring-assessment evaluation only — see
[LICENSE](LICENSE). No rights are granted to use this code beyond
evaluation without prior written permission.

## A note of thanks

I built this on a simple principle: **a human as the guide, AI as the tool
and the coder.** Every decision that mattered — what to build, what to cut,
which trade-off to accept, when a result looked too good to trust — was
mine; the code and much of the careful reasoning behind it came from models
working under that direction. My thanks to **Claude Code**, the harness this
was written in; to **Claude Opus** and **Claude Sonnet** for the design
conversations and the implementation; to **GLM 5.2** and its harness, which
powers the grounded-chat feature you'll see running in the traces and the
scorecard, not just the building of it; and to **Gemini** and **ChatGPT**
for planning and review passes that caught real issues from outside the main
thread of work. Having them check each other's output turned out to be one
of the most useful things here — several of the bugs recorded in
[`docs/findings.md`](docs/findings.md) were found exactly that way, then
verified against real data rather than taken on trust. The judgement, the
direction, and the responsibility for what this repo claims are mine.

## What this is

Ingest → canonical layered representation → deterministic delta engine →
report + markup → grounded chat, with homegrown observability and a full
eval harness threaded through all of it:

1. **Ingest** — a native-PDF adapter (vector extraction) and a
   scanned-PDF adapter (OCR) both produce the same canonical
   representation; a DWG adapter is a documented, real stub.
2. **Canonical IR** — a layered representation (retained raster, typed
   elements with bounding boxes and computed zones, sparse relations,
   embeddings) so every downstream stage compares structured data, never
   raw OCR text streams against each other (the classic 36%-precision
   failure mode that approach has in the published literature).
3. **Delta engine** — fully deterministic: precheck (is this actually a
   revision pair?) → registration (similarity transform from
   high-confidence anchors) → bipartite element matching (Hungarian
   algorithm) → classification (add/remove/modify/move, on parsed
   structured fields, not raw text) → cascade grouping → severity ranking
   → semantic-null detection (rule + optional isolated LLM adjudication)
   → an opt-in registered raster-diff layer that proposes candidate
   change regions with pure CV (SSIM structural diff, morphology,
   connected components) and reports only the residue the symbolic
   pipeline couldn't explain, as low-confidence `unclassified_visual_change`
   deltas — catching valve/symbol changes at a constant tag, line
   reroutes, and other purely graphical edits the text-only pipeline is
   structurally blind to. The LLM never sits in this path unless
   explicitly opted into (raster recall is pure CV, no LLM either way).
4. **Report + markup** — a JSON + Markdown delta report, and real PDF
   annotation objects stamped onto each revision (visible in Acrobat's/
   Bluebeam's own markup list, not a flattened image).
5. **Chat** — homegrown BM25 retrieval over both revisions plus the delta
   report, with citation post-validation: an uncited claim, or a citation
   to an id that was never retrieved, gets the answer overridden into a
   refusal before it's returned.

**If you're reviewing this, start with [Reviewer's guide: one real pair,
stage by stage](#reviewers-guide-one-real-pair-stage-by-stage) below** —
it walks the same pipeline as concrete data from a real traced run,
showing the actual structure at each stage and how it changes. It's a
faster way in than the abstract list above.

See [`docs/architecture.svg`](docs/architecture.svg) /
[`docs/architecture.html`](docs/architecture.html) for a diagram of the
full pipeline, and [`docs/findings.md`](docs/findings.md) for a detailed,
honest account of real bugs found while building each piece — almost all
caught by actually running the pipeline against real data, not by
inspection or unit tests alone.

## Reviewer's guide: one real pair, stage by stage

The list above is the design. This is what actually happened to one real
pair — `eval/datasets/v0/pairs/edited_000` at L0 (native), 1338ms end to
end, taken from a real observability trace and re-run to capture the
payloads the trace only counts. Every number below is reproducible
(`make dataset` is seeded; traces land in `traces/{correlation_id}.json`,
gitignored but written on every run).

**The shape of the whole thing:** two independent lists (133, 134
elements) → one paired list (134) → typed deltas (8) → annotated in place
(8) → plus an independent pixel list (11) → filtered and re-typed into the
same shape (4) → **12 deltas**. Every narrowing is a deliberate
subtraction, and each stage's evidence survives into the next rather than
being recomputed.

### 0 — IR (`CanonicalDocument`)

```
doc_a: pid=A  fmt=pdf_native  rev=A  sheets=1  rasters=[1]
n_elements  A=133  B=134
by type: zone_label 44, geometry 20, line_tag 15, note 13, valve_tag 9,
         instrument 9, nozzle 7, datasheet_row 5, title_field 5, ...
```

One element, in full:

```
id      = el_77e42cf8453a
type    = instrument
content = 'PIT 9056 26 SD HH:245 LL:110'
bbox    = BBox(0.5709, 0.2187, 0.6128, 0.2245)   # normalized, y-down
sheet=1  zone=C-7  extraction_confidence=1.0
attrs   = {'func':'PIT','loop':9056,'system':'26',
           'setpoints':{'HH':245,'LL':110},
           'type_confidence':1.0,'classification_rule':'regex:instrument'}
```

Two flat lists. Nothing relates A to B yet. Note `attrs` is already
*parsed* — downstream stages diff structured fields, never raw text.

### 1 — precheck → `PrecheckResult` *(0.06ms)*

```
is_pair      = True
reason       = 'drawing numbers match'
drawing_no_a = '0D204-PID-26-902-001'   drawing_no_b = '0D204-PID-26-902-001'
equipment_a  = '26-KA-902'              equipment_b  = '26-KA-902'
```

Tier 1 hit. A gate, not a transform — nothing changes shape.

### 2 — register → `Transform` *(0.2ms)*

```
Transform(scale=1.0000000000000002, rotation=1.8e-17, tx=-5.5e-16, ty=-4.4e-16)
```

Same producer here, so an identity to float noise. One object, carried
forward and applied lazily — the element lists are never rewritten.

### 3 — align → `list[MatchedPair]` *(2.3ms)*

`match_sheets` → `[(1, 1)]`, then per sheet **133 + 134 elements → 134 pairs**:

```
both=133   only_a=0   only_b=1
```

A matched pair, carrying its own evidence:

```
MatchedPair(
  a.id='el_18a1d3f05d47'  '16. THIS P&ID CONTAINS DCN-KP-0273-1.'
  b.id='el_ee4f2d530b81'  '16. THIS P&ID CONTAINS DCN-KP-0273-1/1002-'
  cost=0.0539  margin=0.3393  near_miss_cost=None)
```

The one single-sided pair:

```
MatchedPair(a=None, b='el_b1e48be41cfc', type=rev_row,
            'B 2026-06-30 REVISED AS PER DCN',
            near_miss_cost=0.2493)
```

Two lists became **one list of pairs**. `cost` / `margin` /
`near_miss_cost` are the matcher's evidence — everything downstream reads
them instead of re-deriving them.

### 4 — classify → `list[Delta]` *(0.15ms)*

**134 pairs → 8 deltas.** The 126 unchanged pairs emit *nothing* — that
collapse is the point of the stage.

```
by_kind = {modify: 2, move: 5, add: 1}      by_severity = {low: 8}

[delta0001] modify  dcn_note     conf=0.3393  C-1→C-1
   fields={'content': ['16. …DCN-KP-0273-1.', '16. …DCN-KP-0273-1/1002-1.']}
[delta0005] add     rev_row      conf=0.0     None→J-10
   fields={}                          ← near_miss_cost=0.2493 ⇒ conf 0.0, not 1.0
[delta0006] modify  title_field  conf=0.3434  J-10→J-10
   fields={'value': ['A', 'B']}       ← the revision bump
[delta0007] move    valve_tag    conf=0.3132  B-7→C-6   '43GT9067'
[delta0008] move    valve_tag    conf=0.3711  C-5→C-5   '26CB9038'
```

Pairs became **typed, field-level records**. `delta0001` is a field diff
(`content`), not a string comparison. `delta0005` is where the
`near_miss_cost` guard shows up in real data: a plausible candidate was
rejected, so it reports `0.0` — before that fix it read `1.0`.

### 5 — semantic_null *(0.009ms)*

```
n_flagged = 0   (llm_enabled=False — rule half only)
```

**Annotates in place.** Same 8 objects, `.semantic_null` set. No
reshaping. This is the one stage that can call an LLM, and only when
`DELTA_SEMANTIC_NULL_LLM=1`.

### 6 — raster_diff → `list[ChangeRegion]` *(947ms — 71% of total)*

Pixels only. Has never heard of stages 0–5.

```
n_regions = 11
  area_px=2628  mag=0.631  bbox=(0.5005, 0.1847, 0.5112, 0.2001)
  area_px=1612  mag=0.552  bbox=(0.5140, 0.1904, 0.5279, 0.1973)
  …
  area_px=6976  mag=0.621  bbox=(0.7608, 0.9056, 0.8160, 0.9131)   ← title block
  area_px=556   mag=0.322  bbox=(0.7679, 0.9510, 0.7725, 0.9581)
```

A **parallel, independent list** — no ids, no types, just *where* and
*how much*. This is the "raster localizes" half.

### 7 — raster_join → residue `list[Delta]` *(0.23ms)*

```
11 regions in → 4 residue deltas out   (7 suppressed)
```

The 7 dropped are regions stage 4 already explained (the valve moves, the
rev bump), or that identical text on both sides confirms unchanged.
Survivors are re-typed into the *same* `Delta` shape everything else uses:

```
[raster0001] unclassified_visual_change  kind=extraction_gap  conf=0.4137  B-7
             fields={'candidate_element_ids': ['el_e6542fe5bac1']}
[raster0004] unclassified_visual_change  kind=graphical       conf=0.3254  C-5
             fields={'tags': ['40GT9248']}
```

Confidences are all ≤ 0.45 by cap — this stage locates, it never
outranks a symbolic finding. That's the "symbolic classifies" half.

### Final

```
8 symbolic + 4 raster residue = 12 deltas → report / markup / chat
```

## Quick start

```bash
make install
make dataset          # generates eval/datasets/v0 (seeded, reproducible)
make run A=path/to/revA.pdf B=path/to/revB.pdf     # delta report
make html-report A=path/to/revA.pdf B=path/to/revB.pdf  # + an interactive report.html
make markup A=path/to/revA.pdf B=path/to/revB.pdf  # annotated PDFs
make chat A=path/to/revA.pdf B=path/to/revB.pdf    # grounded Q&A (needs an LLM credential, see .env.example)
make eval             # full scorecard against the seeded dataset
```

`make html-report` (`src.cli run --html`) writes `reports/report.html` alongside
the usual json/md — a single self-contained file (rasters inlined, no
external assets) with both revisions' pages side by side, delta boxes
colored by kind, and a filterable/searchable sidebar (kind, severity,
cascade toggle, free-text search) that stays in sync with the page view.
It's the end-user-facing view of the same real `Delta` objects the json/md
report renders — opt-in and off by default so a plain `make run` doesn't
pay for it.

Format (native vs. scanned) is auto-detected per file — no flag needed.
If the two files aren't actually revisions of the same drawing (different
drawing number/equipment tag), `run`/`chat`/`markup` refuse rather than
emit a bogus diff.

Dependencies and the virtualenv are managed by
[uv](https://docs.astral.sh/uv/) (`make install` runs `uv sync --extra dev`;
`uv.lock` pins exact versions for a reproducible install) — install uv
itself first if you don't have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`
(macOS/Linux) or see uv's docs for other platforms. Every Makefile target
runs through `uv run`, so there's never a venv to manually activate.

The scanned-PDF adapter needs the `tesseract` binary on `PATH` (a system
dependency, not pip-installable): `brew install tesseract` on macOS,
`apt install tesseract-ocr` on Debian/Ubuntu.

### Docker — zero-setup run

Nothing to install but Docker itself. No Python, no `uv`, no `tesseract`,
no credential, no dataset build, and **no volume mounts**:

```bash
docker compose run --rm demo    # real vendor revision pair -> delta report
docker compose run --rm eval    # deterministic scorecard
```

`demo` reproduces section 1 of [`DEMO.md`](DEMO.md) and prints the delta
report to stdout. Both build the image on first use.

| Service | What it does | Credential |
|---|---|---|
| `demo` | Delta report on the real vendor pair in `data/samples/` | — |
| `eval` | Deterministic scorecard (delta P/R/F1, calibration, null-pair controls) | — |
| `chat` | Interactive grounded chat over the sample pair | `.env` |
| `eval-full` | Full scorecard incl. chat metrics + `llm_direct` baseline | `.env` |
| `mine` | Your own PDFs — drop `revA.pdf`/`revB.pdf` in `./mypdfs` | — |
| `shell` | A shell in the image | — |

The sample pairs are baked into the image, which is what lets `demo` and
`eval` run with no mount and no arguments. The default services
deliberately **don't** bind-mount: the image runs as a non-root user
(uid 1000), and a host directory owned by a different uid — routine on
Linux — would make a mounted output path unwritable and turn a one-command
demo into a permissions bug. To keep the generated files:

```bash
docker compose run --rm -v "$PWD/reports:/app/reports" demo
```

Plain `docker` works too, if you'd rather not use compose:

```bash
docker build -t delta-chat .
docker run --rm delta-chat                        # deterministic scorecard
docker run --rm --env-file .env delta-chat \
  uv run python -m eval.run_eval --dataset eval/datasets/v0   # full scorecard
```

The build bakes in `make dataset` and runs the full test suite as a build
step — a failed test fails the build, so a built image is itself evidence
the containerized environment works. Fully hermetic: no credential needed
to build (every chat-related test injects a fake LLM call, never a live
one), and `.dockerignore` keeps `.env` out of the build context so a
credential can never be baked into a layer.

`ENTRYPOINT` is left unset — any command after the image name replaces the
default. Commands needing the project's dependencies must go through
`uv run` (`uv run python ...`, `make test`, a bare `bash` are all fine —
`make`'s targets already call `uv run` internally); a bare `python -m ...`
would hit the base image's system Python, which has none of the project's
dependencies, since those live in the `uv`-managed `.venv`.

#### Verified, and what verifying it caught

The image builds and the services run — `docker build` passes with **339
tests green inside the container**, and `docker compose run --rm demo` and
`... eval` were both executed end to end. Building it for real immediately
found two bugs that no amount of reading would have:

1. **`opencv-python` doesn't import in a slim image.** Every cv2-importing
   test failed collection with `ImportError: libGL.so.1: cannot open shared
   object file`. The default OpenCV wheel links a GUI stack this project
   never calls into (no `imshow`, no `waitKey` anywhere — only `warpAffine`,
   morphology, connected components). Fixed by switching the dependency to
   `opencv-python-headless`, which is the correct wheel for a non-GUI
   consumer, rather than installing ~100MB of X11/mesa to satisfy a symbol.
2. **A YAML folded scalar silently split one command into four.** The
   `demo` service used `command: >` with indented continuation lines; `>`
   only folds newlines between lines at the *same* indentation, so the
   deeper-indented arguments kept their newlines and the shell ran
   `--a ...` as its own command (`sh: 2: --a: not found`). Now an explicit
   single-line exec-form list.

**A determinism result worth recording**, from comparing the container run
against the host run (macOS/arm64 vs linux/amd64):

- **L0 (native, deterministic) is identical** — `P=0.82 R=0.86 F1=0.84
  (tp=37 fp=8 fn=6)` on both, and `docker compose run --rm demo` produces a
  byte-identical `delta_report.md` to `make run` on the host.
- **L2 (scanned) is not** — `fp=217` on the host vs `fp=226` in the
  container. The OCR path depends on the platform's `tesseract` build, so
  the reproducibility guarantee holds precisely where the design claims it
  (the symbolic engine) and not in the OCR front end. Worth knowing before
  anyone treats an L2 number as a fixed baseline.

## Eval scorecard (current)

Deterministic delta engine, against the seeded synthetic dataset:

| Level | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| L0 (native) | 0.82 | 0.86 | 0.84 | includes GT rows from `ChangeValveSymbol`/`RerouteLine` the symbolic engine is *designed* to miss (see Raster recall net, below) — those show up as real `modify` false negatives here by construction |
| L2 (scanned) | 0.13 | 0.77 | 0.23 | OCR noise costs precision badly; recall holds up |

Null pairs (identical / re-rendered / reworded-only content): **0 false
positives** on any of the three. `not_a_pair` (sibling drawing, not a
revision): correctly refused, not diffed.

### Raster recall net (`DELTA_RASTER_DIFF=1`, opt-in)

**Raster localizes, symbolic classifies.** A registered raster-diff
layer (`src/delta/raster_diff.py`) proposes candidate change regions
using pure CV (SSIM structural diff, morphological cleanup, connected
components — no LLM anywhere in this stage) in `A`'s frame; a join step
(`src/delta/raster_join.py`) subtracts out anything the symbolic pipeline
already explained and emits only the residue, as low-confidence
`unclassified_visual_change` deltas. The raw diff mask is never emitted
as deltas directly. This is what catches a valve symbol changing type at
an unchanged tag, a rerouted line, or any other purely graphical edit
the text-only pipeline is structurally blind to — and, just as
importantly, what it explicitly does **not** do: classify *what*
changed. It locates; it never assigns more than a low, capped confidence
or a coarse `graphical`/`geometry`/`extraction_gap` hint.

Measured, honestly, not claimed:

| Check | Result |
|---|---|
| Registration/tolerance calibration: true self-identical pair (`null_ident_900`, and the real 26-KA-901 vendor PDF vs. itself) | **0** regions |
| The generator's actual producer-variation null pair (`null_prod_901` — same content, different font/producer) | **61** residue deltas, down from 71 — see below |
| Recall lift on this dataset's `ChangeValveSymbol`/`RerouteLine` GT rows, raster on vs. off | **0.0** (recall stayed 0.0 both ways on this seed) |

`raster_join.py` includes an ensemble check for exactly the
`null_prod_901` case: if the symbolic layer independently confirms the
text under a region is **unchanged** (identical extracted content on
both sides, directly under the region — not just nearby, which would
also eat the `ChangeValveSymbol` case this stage exists to catch), the
region is suppressed the same way a symbolic *change* already suppresses
one. This is a real, tested fix (it catches, and was built specifically
around, a confirmed near-miss: the same identical-content note covering
a region at 0.5 on one side and 0.9 on the other, because a monospace
font renders the same string at a different width than a proportional
one — requiring each side to independently clear a fixed threshold
rejected this for no good reason; checking the *pair's average* coverage
doesn't) — but it only closed **10 of 71** regions on this dataset.
Investigated directly, not assumed: the dominant remainder is a
different, harder problem than the one this check solves. A full-page
font substitution doesn't produce one diff region per changed glyph —
`raster_diff.py`'s own morphological dilation (by design, to merge a
cluster of changed strokes into one region instead of fifty specks)
merges dozens of individual per-glyph font differences into a handful of
**giant, multi-element regions**, one of which covered 62% of the entire
page in a live check. No single matched text element can ever "explain"
a region that large — the fix above only ever asks "does one element's
own bbox account for this region," which is the right question for a
region the size of one note or tag, and the wrong one for a region the
size of half the drawing. Closing that gap for real needs a different
check (does the *union* of many matched, content-identical elements
collectively tile the region — not "one element covers it") — a
genuinely bigger, separate piece of work, left as a stated, honest limit
rather than forced through here.

The 0.0 recall lift is a separate honest finding, unaffected by the
above: on this dataset's one seed, a valve's own drawn glyph is also
independently visible to the symbolic geometry pipeline (a circle
appearing/disappearing when a globe valve becomes a gate valve), and
that coincidental symbolic delta suppresses the raster net's own
contribution at the same spot — confirmed with no regression after the
ensemble fix above (it only ever adds suppression, never removes it,
verified directly). Full detail on all of this, including the two real
bugs this rewrite caught along the way (a contrast-normalization bug
that fabricated a whole-page false diff on a blank page, and a
cross-shape matching bug in `align.py` exposed only once real content
raised geometry density), is in [`docs/findings.md`](docs/findings.md).

### Held-out real-P&ID set (`make eval-holdout`)

A **gold-standard outsider test**, reported separately and never pooled with
the seeded numbers above. The base is a real EPA P&ID (EPA-600/8-80-028,
Figure 15 — cryogenic oxygen generation, public domain); only the edits are
ours. No test asserts anything about detection on it — `tests/test_holdout_integrity.py`
checks only that the fixture is well-formed and still *scoreable*, because a
holdout that gates development stops being held out.

| Check | Seeded set (`v0`) | **Held-out (real)** |
|---|---|---|
| Raster recall lift (raster on vs off) | 0.0 | **1.0** |
| Residue precision | n/a (no hits) | **1.0** |
| Null pair — hard false positives | 0 | **0** |
| Null pair — raster regions emitted | **61** | **0** |

**The headline finding is the reversal.** On the seeded set the raster recall
net has always measured a `0.0` lift — the number that made it look like dead
weight. On real content it catches **both** valve-symbol swaps that the
symbolic engine structurally cannot see, at `1.0` residue precision. The
synthetic set was *understating* the feature, for a reason now understood: its
valve glyphs happen to be independently visible to the symbolic geometry
pipeline, which suppresses the raster hit at the same spot.

The null control reverses too, in the opposite direction: the synthetic
`null_prod` pair emits **61** residue regions (a full font substitution changes
every glyph), while a real producer re-save emits **0**. The synthetic pair was
*overstating* the false-positive rate.

Both numbers moved the moment real data was used — which is exactly what a
holdout is for, and why the seeded figures should not be read as the system's
true behavior on real drawings.

**Honest limitation:** this base is a raster scan with no OCR-recoverable text
at any DPI, so it exercises only the graphical path. On the symbolic side it
produces ~28 false positives from OCR noise, and a symbolic holdout still needs
a text-bearing vector P&ID — see "What I'd do next".

### External review: 3 fixes

A second-opinion review flagged 4 issues; each was checked against the
actual code, not taken at face value. Full detail in `docs/findings.md`.

- **Add/remove confidence gap (real bug, fixed).** A rejected near-miss
  candidate and a truly unambiguous add/remove were indistinguishable in
  the reported confidence (both `1.0` on native PDFs). Verified
  concretely: every false-positive `remove` delta in the L0 eval dataset
  now reports confidence `0.0` instead of `1.0` after the fix (propagate
  the rejected candidate's cost as `near_miss_cost`, scale confidence down
  by how close it sits to `MAX_MATCH_COST`).
- **`precheck.py`'s fail-open fallback (real design gap, fixed).** Added a
  third tier — tag-content Jaccard overlap — before conceding blind when
  neither drawing number nor equipment tag is extractable on either
  document.
- **Instrument-bubble format gap (already tracked, fixed).** Real vendor
  bubbles stack func/loop/system text across separate baselines; a
  position-gated second pass (`_stack_instrument_bubbles`, keyed off real
  circle geometry) now recovers them. The real-sample `xfail` is gone —
  it passes for real now.
- **Retrieval's lexical-only BM25 (already mitigated, not re-fixed)** —
  `config/domain.yaml`'s alias table already softens this; a full fix
  means embeddings (the declared-but-unimplemented L3 layer).

Chat, all 43 questions across the dataset's `qa.jsonl`:

| Metric | Value |
|---|---|
| Refusal accuracy | 0.98 (42/43) |
| Groundedness (citations fully supported) | 0.75 |
| Correctness (LLM-judge)* | 0.72 |
| Judge validated against 5 hand-checked answers | 5/5 agreement |

\* the judge shares a backend with the chat model being judged until a
second LLM credential is configured — see `docs/findings.md` for the
structural fix in place for when one is (`JUDGE_MODEL`/`get_judge_client`),
and the scorecard itself prints this caveat live next to the number.

`llm_direct` baseline (both PDFs handed directly to the LLM, no
deterministic engine, 3 runs at temperature 0, scored through the
identical metrics path): aggregate F1 mean 0.93, but with real measured
non-determinism — one pair swung from F1=1.00 to F1=0.50 to F1≈0.93
across three "temperature=0" runs on the exact same input. That's the
measured version of the determinism argument for building a deterministic
engine at all, not an assumed one.

Run `make eval` for the full, current scorecard, including a confidence-
calibration table and semantic-null-detection precision/recall.

## Design decisions

- **Canonical representation is a layered IR, not a feature space** — a
  retained raster, typed elements (bbox + zone + extraction confidence,
  the diff space and citation unit), sparse relations, and embeddings used
  only inside the match cost. Comparison happens in this canonical space,
  never by diffing two independently-extracted text streams.
- **Composite tags are parsed into fields, not treated as strings.** Line
  tags, instrument loops, and equipment tags decompose into structured
  fields, so a change reads as "pipe class GC11S → FC11S" rather than an
  opaque "text changed."
- **Border-grid zones (A–J × 1–12) are the location primitive** — the
  domain-native way engineers cite regions, re-derived from detected
  border labels so it's robust to scale/skew.
- **Delta detection is fully deterministic; the LLM only writes
  descriptions, answers chat, and (optionally) adjudicates semantic
  equivalence.** A bipartite matcher's cost margin is a real, reproducible
  confidence signal; a generated float is not.
- **Observability is homegrown, not OpenTelemetry** — context-manager
  spans, a correlation id per request, per-span timings, one JSON trace
  file per request. Chosen for zero infrastructure to run this project in
  a fresh clone, and a trace format fully understood end to end rather
  than a wire protocol and collector to stand up.
- **The eval dataset is generated, with edit operators recorded from a
  real edit** (not invented) between the two real P&IDs in
  `data/samples/`: renumbering cascades, DELETED-placeholder collapse,
  systematic per-family tag renumbering, pipe-class/setpoint changes,
  equivalent rewording. Ground truth is exact by construction, labeled at
  three layers (element inventory, correspondence map, typed deltas), and
  the generator is self-validating (a round-trip differ must recover
  every emitted delta).

## What's built

All 7 planned steps plus the bonus markup deliverable:

1. Dataset generator with layered ground truth
2. Native-PDF adapter → canonical representation; zone detection; tag parsing
3. Alignment (register → bipartite match → classify) + delta report
4. Scanned-PDF adapter (OCR)
5. Chat with citation post-validation; refuse-on-unsupported
6. Tracer threaded through; eval runner + scorecard incl. `llm_direct` baseline
7. Markup overlay (real PDF annotations, bonus)

Plus, from a later architecture review: a corrected confidence formula
with a calibration check, judge/chat backend decoupling, semantic-null
detection (rule + opt-in LLM), a BM25 domain-alias table, and an opt-in
registered raster-diff recall net (see "Raster recall net" above) for
purely graphical edits — valve symbol changes, line reroutes — the
text-only pipeline is structurally blind to. Full detail on all of the
above, including what each one's live testing actually caught, is in
[`docs/findings.md`](docs/findings.md).

Also: an opt-in interactive HTML report (`--html`, see Quick start) — the
end-user-facing counterpart to `report.py`'s json/md, reusing
`tools/visual_diff.py`'s two-pane-plus-sidebar layout but powered by the
real delta engine's output rather than that tool's own independent naive
matcher.

## Deliberately not built

- **DWG parsing** — a real stub behind a real seam (`src/ingest/dwg.py`
  documents the ODA/LibreDWG→DXF→ezdxf path); the generator's DXF leg
  proves entity compatibility without needing a full parser.
- **A served web app** — the CLI can emit a webpage (`--html`, a static,
  self-contained file, see Quick start) but there's no server, no
  multi-user state, no upload flow; every run is still `python -m src.cli
  run --a ... --b ...` on the command line.
- **Multi-hundred-sheet scaling** — this project targets single-sheet
  pairs; the sheet-matching stage and per-sheet delta design leave the
  parallelization seam in place, but it isn't exercised.
- **L2 relations / L3 embeddings** — declared in the canonical model's
  own layering but never populated or implemented; a real, acknowledged
  gap, not hidden.
- **A three-backend LLM cost/latency comparison table** — scaffolded
  (`eval/baselines/backend_compare.py`), but only one credential is
  configured in this environment, so it reports its own "not enough
  backends configured" state honestly rather than fabricating a
  comparison. `tools/compare_models.py` covers the lighter, more common
  case of comparing model names under one shared credential.

## What I'd do next with more time

Ordered by measured pain, not by interest. Each item names the number in the
scorecard it exists to move and the seam it would attach to — the layered IR
was designed with these in mind, which is why most are additive rather than
rewrites.

### 1. L3 embeddings: semantic similarity, in the two places it actually pays

**The measured pain.** `DEMO.md` exchange 2d is the cleanest example: asked
which notes were *"deleted or reworded"*, retrieval missed two real changes
whose delta descriptions say *"content changed"* and *"note removed"*. BM25
cannot bridge that — it matches tokens, and the question and the answer share
none of the right ones. `config/domain.yaml`'s alias table is a curated
patch over the same hole, and it only scales as far as someone maintains it.
Chat correctness sits at **0.70** and groundedness at **0.71**; retrieval
recall is the ceiling on both.

**Where it attaches.** L3 is already declared in `src/canonical/model.py`'s
layering with a deliberate constraint worth keeping: *embeddings are a
match-cost term, never a citation target.* Two consumers:

- **Retrieval** (`src/chat/retrieval.py`) — hybrid BM25 + dense, fused with
  reciprocal-rank fusion rather than replacing the lexical leg. Exact tag
  strings (`26-KA-901`, `AC21`) are precisely where lexical matching wins and
  embeddings blur; a P&ID is full of near-identical identifiers where a
  dense-only retriever would be actively worse. Hybrid, not swap.
- **`align.py`'s cost function** — a third term alongside text and spatial,
  gated to low weight. It would catch the reworded-note case (`"OIL CHANGE BY
  USING TEMPORARY ARRANGEMENT WITH HOSES."` → `"OIL CHANGE USING TEMPORARY
  HOSE ARRANGEMENT."`) that rapidfuzz scores as a weak match today, and feed
  `semantic_null.py` a rule-based signal where it currently needs an LLM call.

**The cost to be honest about.** Determinism. The delta engine is currently
reproducible end to end (verified: byte-identical reports across macOS and
Linux). A local, pinned, CPU embedding model keeps that; a hosted embedding
API does not, and would put a network call in the one path that is currently
guaranteed offline. I would pin a small local model and version the weights
before I would take an API's quality.

### 2. L2 relations: make the drawing a graph, not a bag of elements

**The measured pain.** `remove` precision is **0.00** on L0 — every removal
the engine reports on the edited pairs is a false positive. The root cause is
structural: elements are matched independently, so nothing knows that a valve
tag *belongs to* the valve symbol beside it, or that a note is *referenced by*
a DCN callout. When one member of a group is matched badly, nothing pulls its
siblings along.

**Where it attaches.** `CanonicalElement.relations: list[tuple[str, str]]`
already exists at `src/canonical/model.py:39` — declared, never written to.
Populating it is the whole feature:

- **Containment** — text inside an instrument bubble. The geometry to do this
  already runs: `_stack_instrument_bubbles` in `src/ingest/pdf_native.py`
  computes exactly this circle-contains-token relation and then throws the
  structure away after composing the string. Emitting a relation instead of
  discarding it is close to free.
- **Text↔symbol association** — a valve tag and its glyph, currently only
  implicitly linked by `raster_join.py`'s `tag_proximity_norm` padding.
- **Reference edges** — `dcn_note` → the notes it cites; these are already
  parsed into `attrs["dcns"]` and simply not linked.

Then matching becomes **graph-aware**: score a candidate pairing partly on
whether its neighbours also matched. That is the standard fix for exactly the
failure mode here — a locally-plausible wrong match that is obviously wrong
one hop out. It would also let cascade detection link to the true root cause
rather than an arbitrary group member, which `src/delta/classify.py`'s own
docstring already flags as a known gap.

### 3. Visual geometry awareness: make the raster stage understand shape

**The measured pain.** Raster recall lift is **0.0** — the opt-in CV stage
contributed nothing on this dataset — and the producer-variation null pair
still emits **61** residue regions. Two distinct, diagnosed causes:

- **Regions are blobs, not shapes.** `raster_diff.py` proposes connected
  components with an area and a mean magnitude. A region knows *where* it is
  and *how much* changed, never *what* it is. The `unclassified_visual_change`
  label is honest precisely because the stage genuinely cannot say more.
- **Suppression is per-element, so it fails on large regions.** The
  text-confirmed-unchanged check asks "does *one* element's bbox account for
  this region?" — the right question for a tag-sized region, the wrong one for
  a region covering 62% of the page after morphological dilation merges
  hundreds of per-glyph font differences.

**What I would build, in order:**

1. **Union-coverage suppression** — accumulate coverage across *all* matched,
   content-identical elements overlapping a region instead of testing one at a
   time. This is the direct fix for the 61 residue regions and needs no new
   dependency. Highest value per unit of work in this whole list.
2. **Vector-native geometry diffing.** For native PDFs the vector paths are
   right there in `page.get_drawings()` — currently flattened to a bbox and a
   coarse `geom_kind`. Diffing actual path geometry (endpoints, segment counts,
   topology) would catch a rerouted line as a *route* change rather than as
   "some pixels differ", and would fix a real weakness in `align.py`: geometry
   elements have empty content, so `_text_sim` is always `1.0` and shape
   contributes *zero* cost signal beyond the `geom_kind` bucketing added after
   the matcher was caught pairing a line against a circle.
3. **Symbol recognition.** Template matching first (P&ID symbol sets are
   small, closed, and standardized — ISA-5.1), a small trained classifier only
   if that plateaus. This is what turns `unclassified_visual_change` into
   `"gate valve → globe valve"`, which is the answer a reviewer actually wants.
4. **Connectivity extraction** — trace line-work into a topological graph of
   what connects to what. This is the genuinely hard one and the most
   valuable: it makes *"is this line still routed to the same vessel?"* a
   question the system can answer at all. It also feeds straight back into
   item 2 as another relation type.

### 4. Fixing what the scorecard already says is broken

Not new capability — just the numbers the eval is currently shouting about:

- **L2 (scanned) precision 0.13**, 217 false positives. OCR noise is treated
  as content. Needs OCR-confidence-weighted matching and a noise model, not
  more matching cleverness.
- **Confidence is inverted on L2** — the `0.9-1.0` band scores precision
  `0.00` while `0.0-0.5` scores `0.146`. A confidence signal that
  anti-correlates with correctness is worse than none, because people act on
  it. I would fix this before adding any new feature.
- **`semantic_null` rule recall 0.14** — the rule half catches one case in
  seven; the LLM half is off by default.

### Deliberately *not* on this list

- **An LLM in the core delta path.** The `llm_direct` baseline already scores
  higher F1 (0.90 vs 0.84) on this dataset, and I could close the gap fastest
  by calling a model. I would not: it costs determinism, bounding boxes,
  per-stage traces, and a confidence signal — everything that makes the output
  reviewable. The right place for an LLM here is adjudicating a *specific*
  ambiguity (`semantic_null.py`), which is where the one optional call already
  sits.
- **A bigger synthetic dataset.** More seeded pairs would make the numbers
  smoother without making them more true. Real vendor pairs are what this
  needs, and they're the scarce input.

## Repo layout

```
src/
  ingest/        FormatAdapter seam: pdf_native, pdf_scanned, dwg (real stub)
  canonical/     layered IR (model.py); zones.py, tags.py, classify.py
  delta/         precheck -> register -> align (bipartite matcher) -> classify
                 -> severity -> semantic_null -> raster_diff -> raster_join -> report
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
  datasets/holdout/     held-out real EPA P&ID pair -- committed, never tuned
                         against, scored separately (`make eval-holdout`)
  baselines/            llm_direct.py, backend_compare.py
  calibration.py        confidence-band precision check
  run_eval.py           scorecard: delta P/R/F1, calibration, semantic-null
                         detection, chat correctness/groundedness/refusal
data/samples/           provenance-documented real vendor P&IDs (see PROVENANCE.md)
docs/
  architecture.{svg,html,txt,mmd}   pipeline diagram, four formats
  findings.md                       detailed engineering findings (see above)
  DEMO_SCRIPT.md                    shot list for the video walkthrough
tools/
  compare_models.py     same pair, same credential, different model names
  visual_diff.py        human-in-the-loop debug viewer
  holdout/              builder for the held-out set (make_real_pair.py) and
                         its source/licensing notes
tests/                  339 tests; `make test`
```

Root: [`DESIGN.md`](DESIGN.md) (decisions of record, cited from source
docstrings), [`DEMO.md`](DEMO.md) (walkthrough with real captured output),
[`docs/findings.md`](docs/findings.md) (what broke and why).
