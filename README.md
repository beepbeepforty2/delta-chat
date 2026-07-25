# delta-chat — Document Delta & Grounded Chat

Given two PIDs (two revisions of a piping & instrumentation diagram:
native PDF, scanned PDF, or DWG), compute a structured delta, emit a
human+machine-readable delta report with real PDF markup annotations, and
chat over both revisions and the delta with citations.

**License:** shared publicly for hiring-assessment evaluation only — see
[LICENSE](LICENSE). No rights are granted to use this code beyond
evaluation without prior written permission.

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

See [`docs/architecture.svg`](docs/architecture.svg) /
[`docs/architecture.html`](docs/architecture.html) for a diagram of the
full pipeline, and [`docs/findings.md`](docs/findings.md) for a detailed,
honest account of real bugs found while building each piece — almost all
caught by actually running the pipeline against real data, not by
inspection or unit tests alone.

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

### Docker

```bash
docker build -t delta-chat .
docker run --rm delta-chat        # deterministic-only scorecard, no credential needed
```

The build bakes in `make dataset` and runs the full test suite as a build
step — a failed test fails the build. Fully hermetic: no credential
needed to build (every chat-related test injects a fake LLM call, never a
live one).

```bash
# full scorecard incl. chat + llm_direct baseline -- needs a credential,
# passed at run time, never baked into the image
docker run --rm --env-file .env delta-chat \
  uv run python -m eval.run_eval --dataset eval/datasets/v0

# run against your own two PDFs
docker run --rm -v "$(pwd)/mypdfs:/data" delta-chat \
  uv run python -m src.cli run --a /data/revA.pdf --b /data/revB.pdf --out /data/reports
```

`ENTRYPOINT` is left unset — any command after the image name replaces
the default. Commands that need the project's dependencies must go
through `uv run` (`uv run python ...`, `make test`, a bare `bash` are all
fine — `make`'s own targets already call `uv run` internally); a bare
`python -m ...` without `uv run` would hit the base image's system
Python, which has none of the project's dependencies installed, since
those live in the `uv`-managed `.venv`. (Caveat: this `Dockerfile` was
written carefully against the project's real dependencies but hasn't been
verified with an actual `docker build` — Docker wasn't available in the
environment this was developed in.)

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
  baselines/            llm_direct.py, backend_compare.py
  calibration.py        confidence-band precision check
  run_eval.py           scorecard: delta P/R/F1, calibration, semantic-null
                         detection, chat correctness/groundedness/refusal
data/samples/           provenance-documented real vendor P&IDs (see PROVENANCE.md)
docs/
  architecture.{svg,html,txt,mmd}   pipeline diagram, four formats
  findings.md                       detailed engineering findings (see above)
tools/
  compare_models.py     same pair, same credential, different model names
  visual_diff.py        human-in-the-loop debug viewer
```
