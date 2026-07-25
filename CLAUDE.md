# CLAUDE.md — project context for Claude Code

## What this is

Take-home: given two PIDs (document revisions — native PDF / scanned PDF /
DWG), ingest to a canonical representation, compute a structured delta,
emit a delta report, and serve grounded chat with citations. Observability
and a runnable eval harness are required. Depth over breadth.

## Established design decisions — do not silently revisit

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
   w_text·(1−sim) + w_spatial·dist + w_type·mismatch, weights in .env) →
   classify add/remove/modify/move. Confidence = match-cost margin
   (best vs second-best) × extraction_confidence_a × extraction_confidence_b
   (both sides' extraction confidence, multiplied — not the min of the
   two; they agree only when both sides are 1.0, i.e. every native-native
   pair, which is why a min()-based version passed unnoticed for a while).
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
   in the retrieved set; refuse when unsupported. CLI is sufficient.
8. **Observability is homegrown** (`src/observability/`): context-manager
   spans, correlation id per request, per-span timings, LLM spans capture
   prompt/response/model/tokens/cost, JSONL structured logs, one JSON
   trace file per request in `traces/`. Justify vs OTel in README
   (zero infra, fully understood). Failures recorded as failed spans,
   never swallowed.
9. **Formats**: native PDF + scanned PDF end-to-end; DWG stays a REAL stub
   (`src/ingest/dwg.py` documents ODA/LibreDWG→DXF→ezdxf path). Do not
   implement DWG parsing without being asked.
10. **LLM behind one interface** (`src/chat/llm.py`), provider/model from
    env, keys never committed, prompt caching for the two-document context.

## Steps

1. [x] Dataset generator with layered GT (first commit)
2. [x] Native-PDF adapter -> canonical; zone detection; tag parsing
3. [x] Alignment (register -> bipartite match -> classify) + delta report
4. [x] Scanned-PDF adapter (OCR) -- degradation ladder already generated
5. [x] Chat with citation post-validation; refuse-on-unsupported
6. [x] Tracer threaded through; eval runner + scorecard incl. llm_direct baseline
   -- chat correctness (LLM-judge, validated 5/5 vs hand-checked)/
   groundedness/refusal-accuracy metrics and the llm_direct baseline
   (3x @ temperature=0, measured output variance) are both wired into
   `make eval`'s scorecard now that a live LLM connection exists.
7. [x] Markup overlay (bonus) -- `src/markup/overlay.py`, `make markup A=... B=...`

Update checkboxes as steps land; keep README's "Plan" section as the
narrative version of the same list (don't let the two drift — CLAUDE.md is
the checklist, README explains the "so far" rationale).

## Dataset (already built — use it, don't rebuild it)

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
both as candidates). Redistribution rights confirmed. Known real-file gap:
instrument bubbles split system/function/loop across three stacked
baselines in the real drawings, unlike the generator's single-line format —
`tests/test_pdf_native_real_samples.py` documents this as an expected
failure (`xfail`), not silently passing.

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

## Working agreements

- Python 3.11+, stdlib + deps in pyproject only; no LangChain/LlamaIndex.
- Config over hardcoding: model, thresholds, paths from env/.env.
- Tests where they matter: delta engine, tag parser, citation validator,
  cascade grouping. `make test` must stay green.
- Every commit leaves `make dataset && make eval` runnable.
- README records every trade-off and every deliberate cut, including the
  candid failure table once eval runs exist.
