# DESIGN.md — design decisions of record

The numbered decisions below are cited directly from source docstrings
("DESIGN.md decision #3") as the rationale for choices that would otherwise
look arbitrary. Read it alongside [`docs/findings.md`](docs/findings.md),
which records what happened when these decisions met real data.

## What this is

Given two PIDs (document revisions — native PDF / scanned PDF / DWG), ingest
to a canonical representation, compute a structured delta, emit a delta
report, and serve grounded chat with citations. Observability and a runnable
eval harness are first-class requirements, not extras. Depth over breadth.

## Design decisions

1. **Canonical IR is layered and addressable** (`src/canonical/model.py`):
   L0 retained raster, L1 typed elements (diff space + citation unit,
   normalized bbox + zone + extraction_confidence), L2 sparse relations,
   L3 embeddings used ONLY as a match-cost term. Every element keeps a
   pullback to source coordinates (markup + citations need invertibility).
2. **Comparison happens in canonical space**, never by diffing two
   independently-extracted text streams (that is the 36%-precision
   OCR-diff failure mode from the literature).
3. **Delta detection is deterministic.** Pipeline: page-level registration
   (similarity transform from high-confidence anchors) → sheet matching →
   per-sheet bipartite element matching (scipy Hungarian; cost =
   w_text·(1−sim) + w_spatial·dist, weights in .env) → classify
   add/remove/modify/move. Type is handled by *bucketing* candidates before
   the Hungarian runs rather than by a type-mismatch cost term: cross-type
   matches are never semantically correct here (a line_tag never "becomes" a
   valve_tag), so a hard partition is a faithful simplification and keeps
   each matrix small at real density (`src/delta/align.py`).
   Confidence = match-cost margin (best vs second-best) ×
   extraction_confidence_a × extraction_confidence_b (both sides multiplied,
   not the min of the two; they agree only when both sides are 1.0, i.e.
   every native-native pair, which is why a min()-based version passed
   unnoticed for a while). A single-sided add/remove additionally carries a
   `near_miss_cost` guard, so a rejected-but-plausible match reports low
   confidence instead of a misleading 1.0.
   The LLM's only roles:
   (a) writing human-readable change descriptions, (b) chat answers,
   (c) optional semantic-equivalence adjudication — each isolated,
   cached, documented as the non-deterministic zone.
4. **Composite tags are parsed into fields** (line tags
   `4"-PV-26-9048-GC11S-38` → size/service/system/seq/pipe_class/insul;
   instrument bubbles → func/loop/system + setpoints). Field-level deltas,
   not string deltas. Detect per-family constant offsets (e.g. all
   instrument loops −39) and emit one grouped delta + member cascades.
5. **Locations cite border-grid zones** ("Sheet 1, zone F-7") derived from
   detected zone labels; bbox retained underneath.
6. **Pre-check: are these actually revisions of one document?** Compare
   title-block equipment tag / drawing number first; refuse to diff
   siblings (dataset contains a `not_a_pair` control for this).
7. **Chat**: retrieval over PID A + PID B + delta-report JSON entries;
   every chunk carries {source, sheet, zone, element_id}. Answers must
   emit citations in a fixed format; post-validate that cited IDs exist
   in the retrieved set; refuse when unsupported. `answer()` is stateless
   and takes a prebuilt index, so a caller owns session lifetime — that is
   what let the web UI reuse it unchanged (decision 11).
8. **Observability is homegrown** (`src/observability/`): context-manager
   spans, correlation id per request, per-span timings, LLM spans capture
   prompt/response/model/tokens/cost, JSONL structured logs, one JSON
   trace file per request in `traces/`. Chosen over OTel for zero infra and
   full transparency (justified in the README). Failures recorded as failed
   spans, never swallowed.
9. **Formats**: native PDF + scanned PDF end-to-end; DWG is a deliberate
   stub behind a real seam (`src/ingest/dwg.py` documents the
   ODA/LibreDWG→DXF→ezdxf path), cut in favour of depth on the OCR path.
10. **LLM behind one interface** (`src/chat/llm.py`), provider/model from
    env, keys never committed, prompt caching for the two-document context.
11. **The web UI is a presentation layer, not a second engine**
    (`src/web/`). It calls the same `compute_deltas` the CLI and the eval
    scorecard call, and renders `markup/payload.py::build_payload` — the
    same function that produces the JSON embedded in the downloadable
    `report.html`, so the browser view and the offline file cannot drift
    (pinned by `tests/test_markup_payload.py`). It does *not* reuse
    `cli._run_pipeline`, which prints refusals to stderr and returns an int
    exit code; a browser needs the structured `PrecheckResult` to explain a
    refusal and offer an override. Frontend is dependency-free vanilla JS
    served as static files — no Node, no build step — with pdf.js vendored
    for vector-crisp zoom. Boxes stay normalized `[0,1]` all the way to the
    browser and are laid out as CSS percentages, so no second coordinate
    system exists alongside `overlay.py` and `pdf_annotate.py`. Single-user
    and localhost-bound by design: in-process job store, no auth.

## Dataset

`eval/datasets/generator/` emits seeded pairs with three-layer GT:
- L1 `gt/elements_{a,b}.json` (inventory: measures extraction in isolation)
- L2 `gt/correspondence.json` (eid map: measures alignment given gold L1)
- L3 `gt/deltas.json` (typed deltas; `is_cascade`/`primary_did`/
  `semantic_null` tags; measures classification)
Pairs: 6 edited + null_ident + null_prod + null_reword + not_a_pair.
Degradation ladder L0 (vector) → L1/L2/L3 rasters with recorded transforms
(`gt/render_transforms.json`) — apply them to L1 bboxes when scoring
against degraded inputs. `qa.jsonl` has templated Q&A with citation
targets and refuse-expected probes. Regenerate with `make dataset`.
Operators are harvested from the real assignment pair (26-KA-901/902):
note-insertion renumbering cascades, DELETED-placeholder collapse,
per-family tag renumbering, pipe-class changes, setpoint changes,
equivalent rewording. `data/samples/` (see `PROVENANCE.md` there): the two
real vendor P&IDs — format exemplars and now also used to validate/
recalibrate `src/ingest/pdf_native.py` against real element density
(~800 text + ~5000 geometry elements per sheet vs. the generator's ~80-120)
and real layout (datasheet block bottom-left, not bottom-right as the
generator assumes — `src/canonical/classify.py`'s Tier-2 region rects carry
both as candidates). Redistribution rights confirmed. Real-file gap found and
since closed: instrument bubbles split system/function/loop across three
stacked baselines in the real drawings, unlike the generator's single-line
format. It was carried as an `xfail` for a while rather than passed over
silently, and is now handled by `_stack_instrument_bubbles`
(`src/ingest/pdf_native.py`), a second pass gated on real circle geometry —
`tests/test_pdf_native_real_samples.py` asserts it for real.

A **held-out** set (`eval/datasets/holdout/`, `make eval-holdout`) sits
alongside the seeded one: a real EPA P&ID that nothing is tuned against,
scored by the identical code path and reported separately. Its numbers are
the realism check on everything above — see `docs/findings.md`.

## Eval harness requirements

- `make eval` prints a scorecard; results also written to
  `eval/results/{timestamp}.json` and diffed vs previous run.
- Delta metrics: P/R/F1 matched by (location IoU or zone + type), reported
  overall, per change type, per format level (L0 vs L3), and separately
  for primary vs cascade vs semantic-null populations. Correct handling of
  semantic-nulls: flagging them as changes is a (soft) false positive;
  report as its own column.
- Null pairs: any emitted delta is a false positive; report FP count per
  null kind. not_a_pair: correct behavior is refusal.
- Chat: correctness (LLM-judge, judged against 5 hand-checked answers to
  validate the judge — report agreement) + deterministic groundedness
  (cited element ids exist and contain the claimed content) + refusal
  accuracy on refuse-expected probes.
- Include baseline arm `eval/baselines/llm_direct.py`: both PDFs to the
  LLM with a delta-schema prompt, parsed into the same Delta type, scored
  through the identical metrics path; run 3× at temperature 0 and report
  output variance (the determinism argument, measured).

## Engineering constraints

- Python 3.11+, stdlib + deps declared in `pyproject.toml`; no LangChain or
  LlamaIndex — the retrieval and agent loops here are small enough to own
  outright, and owning them is what makes them instrumentable.
- Config over hardcoding: model, thresholds and paths come from env/`.env`
  (every tunable constant in the delta engine has an ablation path — see
  `.env.example`).
- Tests concentrate where correctness is load-bearing: the delta engine, tag
  parsing, citation validation, cascade grouping, and resource hygiene.
- `make dataset && make eval` stays runnable; the seeded dataset is
  regenerated from seed 42, never committed.
- Trade-offs, deliberate cuts and the candid failure table live in the
  README and `docs/findings.md`.
