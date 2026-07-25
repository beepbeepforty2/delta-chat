# DEMO — one delta, one grounded chat exchange, one scorecard

Every command, output, and number below is **verbatim from a real run** on
this repo — nothing is illustrative or reconstructed. Reproduce any of it
with the commands shown.

The pair used throughout is `data/samples/real_pair/` — a **real** revision
pair, not synthetic generator output: a crop of genuine vendor P&ID content
(MAN Energy Solutions drawing 26-KA-901), hand-edited with real PDF editing
tools. Full provenance and the exact list of edits made:
[`data/samples/real_pair/PROVENANCE.md`](data/samples/real_pair/PROVENANCE.md).

**Fastest way to reproduce section 1 yourself**, with nothing installed but
Docker — no Python, no `uv`, no `tesseract`, no credential, no mounts:

```bash
docker compose run --rm demo
```

Or natively:

```bash
make install
```

---

## 1 — The delta

```bash
make run A=data/samples/real_pair/a/L0.pdf B=data/samples/real_pair/b/L0.pdf
```

```
trace: a80f8f30019c (traces/a80f8f30019c.json)
11 primary change(s), 0 cascade change(s): {'modify': 3, 'remove': 2, 'add': 6}
wrote reports/delta_report.json
wrote reports/delta_report.md
```

(The correlation id is fresh per run; the report content is not — running
this twice produces a byte-identical `delta_report.md`, verified with
`diff`. The delta engine is deterministic end to end, no LLM in this path.)

### `delta_report.md` (verbatim)

```markdown
# Delta Report: data/samples/real_pair/a/L0.pdf -> data/samples/real_pair/b/L0.pdf

11 primary change(s), 0 cascade change(s).
Severity: high=1, low=10

## Add (6)
- [LOW] **Sheet 1, zone J-2** (confidence 0.00): note_deleted added: 13-14. DELETED.
- [LOW] **Sheet 1, zone F-10** (confidence 1.00): geometry added:
- [LOW] **Sheet 1, zone I-6** (confidence 1.00): geometry added:
- [LOW] **Sheet 1, zone J-6** (confidence 1.00): geometry added:
- [LOW] **Sheet 1, zone J-2** (confidence 1.00): geometry added:
- [LOW] **Sheet 1, zone J-1** (confidence 0.24): unknown added: J

## Remove (2)
- [LOW] **Sheet 1, zone J-2** (confidence 0.00): note removed: 13. UPSTREAM STRAIGHT RUN MIN. 10xD. DOWNSTREAM STRAIGHT RUN
- [LOW] **Sheet 1, zone J-1** (confidence 0.00): text removed: J 14. VENT ROUTED TO SAFE LOCATION.

## Modify (3)
- [HIGH] **Sheet 1, zone F-8** (confidence 0.02): line_tag pipe_class changed: AC21 -> AC31
- [LOW] **Sheet 1, zone I-2** (confidence 0.26): note content changed: 5. OIL CHANGE BY USING TEMPORARY ARRANGEMENT WITH HOSES. -> 5. OIL CHANGE USING TEMPORARY HOSE ARRANGEMENT.
- [LOW] **Sheet 1, zone J-2** (confidence 0.01): note_deleted note_no changed: 15 -> 14
```

**What to notice.** Exactly one change is ranked `HIGH`, and it's the right
one: `pipe_class AC21 -> AC31` is a mechanical rating change on a line tag.
Everything else — a note reword, a `DELETED.` placeholder collapse, some
geometry — is correctly `LOW`. The engine is not counting changes, it is
ranking them.

Note also the second `modify`: `5. OIL CHANGE BY USING TEMPORARY ARRANGEMENT
WITH HOSES.` → `5. OIL CHANGE USING TEMPORARY HOSE ARRANGEMENT.` The text
differs, the meaning does not. This is the case the semantic-null stage
exists for (rule-based by default; `DELTA_SEMANTIC_NULL_LLM=1` adds LLM
adjudication).

### The same delta, machine-readable

From `reports/delta_report.json` — the `HIGH` entry in full:

```json
{
  "did": "delta0009",
  "kind": "modify",
  "element_type": "line_tag",
  "id_a": "el_e1664bc7666e",
  "id_b": "el_1279e6096b8e",
  "sheet": 1,
  "zone_a": "F-8",
  "zone_b": "F-8",
  "field_changes": { "pipe_class": ["AC21", "AC31"] },
  "confidence": 0.0214,
  "description": "line_tag pipe_class changed: AC21 -> AC31",
  "is_cascade": false,
  "primary_did": null,
  "severity": "high",
  "semantic_null": false,
  "semantic_null_reason": null,
  "bbox_a": null,
  "bbox_b": null,
  "visual_change_kind": null
}
```

`field_changes` is a **parsed field diff** (`pipe_class`), not a string
comparison — the tag was decomposed into its structured fields at ingest, so
the delta names the engineering property that changed.

**On that `confidence: 0.0214`** — this is honest, not a bug. Confidence is
the bipartite matcher's *margin* (best match vs. next-best). This sheet has
many structurally similar line tags, so the winning match barely beat its
runner-up, and the score says so. The change is real and correctly typed;
the low number is the engine flagging that *this particular alignment* was
contested. That signal is exactly what the confidence-calibration table in
the scorecard measures.

---

## 2 — Grounded chat

```bash
make chat A=data/samples/real_pair/a/L0.pdf B=data/samples/real_pair/b/L0.pdf
```

Four real exchanges, verbatim. Citation format is
`[source:sheet:zone:element_or_delta_id]`.

### 2a — a cited answer, and a refusal to over-claim

```
Q: What changed in the pipe class for any line tags, and does it matter?

A: The pipe class for a line tag changed from AC21 to AC31
   [delta:1:F-8:delta0009]. The provided chunks do not state why this
   change matters or its engineering significance.
```

The citation `delta0009` resolves to the exact JSON record shown above —
same `did`, same sheet, same zone. The answer is traceable end to end.

The second half is the more important part: asked *"does it matter?"*, the
model answered the part the sources support and **explicitly declined the
part they don't**, rather than producing plausible engineering commentary.

### 2b — a partial refusal that still cites what it found

```
Q: What is the service and duty of the equipment on this drawing?

REFUSED: The provided context identifies the labels "SERVICE"
   [A:1:G-2:el_8d11f4dbb571] and "DUTY kW" [A:1:G-2:el_1d82c19de43c] on the
   drawing, but does not provide the specific values or descriptions for them.
```

The datasheet *labels* were retrieved; their *values* were not. Rather than
inferring plausible values, it refuses — and cites the two elements it did
find, so the refusal itself is auditable.

### 2c — a clean refusal

```
Q: What is the recommended lubricant viscosity for the compressor bearings?

REFUSED: The provided chunks do not contain any information about
   recommended lubricant viscosity for the compressor bearings.
```

Not in either revision. No hedged guess.

### 2d — a change-summary question

```
Q: Which numbered notes were deleted or reworded between these revisions?

A: Based on the provided context, the following note change was identified:

   *   **Note 14**: This note was renumbered from note 15 and marked as
       deleted [delta:1:J-2:delta0002] [delta:1:J-2:delta0004]. The full
       content of this deleted note is not provided in the context.
```

**Honest limitation, shown rather than hidden:** this answer is incomplete.
Two real changes should have surfaced and didn't — the note-5 reword
(`delta0001`) and the removal of note 13 (`delta0003`); retrieval ranked
neither into the top-8 chunks for this phrasing. This is the BM25
lexical-matching gap documented in [`docs/findings.md`](docs/findings.md):
the question says *"reworded"* and *"deleted"*, while the delta descriptions
say *"content changed"* and *"note removed"*. Grounding held — nothing was
invented, and it flagged its own missing content — but recall did not. This
is precisely the failure mode the groundedness-vs-correctness split in the
scorecard below is designed to expose rather than average away.

---

## 3 — Observability: the trace behind exchange 2a

```bash
make trace ID=5600d700798e
```

```
trace 5600d700798e
  request [2550.9ms] OK
    pid_a=data/samples/real_pair/a/L0.pdf, pid_b=data/samples/real_pair/b/L0.pdf, mode=chat
    ingest [117.1ms] OK
      ingest_a [68.0ms] OK
        path=data/samples/real_pair/a/L0.pdf, n_elements=952
      ingest_b [49.1ms] OK
        path=data/samples/real_pair/b/L0.pdf, n_elements=956
    precheck [0.1ms] OK
      is_pair=True, reason=equipment tags match
    register [0.2ms] OK
      scale=1.0
    align [151.8ms] OK
      n_sheets=1
      align_sheet [151.8ms] OK
        sheet=1, n_matches=958
    classify [0.6ms] OK
      primary_count=11, cascade_count=0, by_kind={'modify': 3, 'remove': 2, 'add': 6}
    semantic_null [0.0ms] OK
      n_flagged=0, llm_enabled=False
    raster_diff [0.0ms] OK
      enabled=False, n_regions=0
    raster_join [0.0ms] OK
      n_residue=0
    build_index [2.0ms] OK
      n_chunks=319
    retrieval [0.3ms] OK
      question=What changed in the pipe class for any line tags, and does it matter?, n_hits=8
    chat_answer [2278.5ms] OK
      model=glm-5.2 tokens_in=566 tokens_out=42 cost_usd=None
```

One request, ingest → delta → retrieval → LLM → answer, every stage timed.
The LLM call carries model, token counts, and (where provider pricing is
configured) cost; the prompt and response are captured on the span too.
`cost_usd=None` is deliberate — no pricing is configured for this provider,
so the field stays null rather than reporting a fabricated number.

`precheck` firing on `equipment tags match` rather than a drawing number is
also real: this pair is a *crop*, so the title block didn't survive, and the
check fell through to its second tier.

Raw trace JSON: `traces/5600d700798e.json`. Structured per-span events with
correlation ids append to `traces/events.jsonl`.

---

## 4 — Markup (bonus)

```bash
make markup A=data/samples/real_pair/a/L0.pdf B=data/samples/real_pair/b/L0.pdf
```

```
trace: ea4939470890 (traces/ea4939470890.json)
wrote reports/markup_a.pdf
wrote reports/markup_b.pdf
```

Both revisions are annotated (removals are only visible on A, additions only
on B). These are **real PDF annotation objects** — they show up in Acrobat's
and Bluebeam's own markup list and can be toggled or replied to — not a
flattened image of boxes.

An interactive HTML view of the same deltas is also available:

```bash
make html-report A=data/samples/real_pair/a/L0.pdf B=data/samples/real_pair/b/L0.pdf
```

---

## 5 — Eval scorecard

```bash
make dataset   # seeded, reproducible: eval/datasets/v0
make eval
```

Verbatim output of one full run (520.67s, including ~90 live LLM calls):

```
=== delta-chat eval scorecard -- eval/datasets/v0 ===

-- edited pairs, by format level --
  [L0] overall: P=0.82 R=0.86 F1=0.84 (tp=37 fp=8 fn=6)
         add      P=1.00 R=1.00 F1=1.00 (tp=8 fp=0 fn=0)
         remove   P=0.00 R=1.00 F1=0.00 (tp=0 fp=1 fn=0)
         modify   P=1.00 R=0.84 F1=0.92 (tp=27 fp=0 fn=5)
         move     P=0.22 R=0.67 F1=0.33 (tp=2 fp=7 fn=1)
         avg primary_recall=0.8361  avg cascade_recall=1.0  avg semantic_null_emission_rate=1.0
         semantic_null flag detection (engine's own flag vs GT): P=1.00 R=0.14 F1=0.25 (tp=1 fp=0 fn=6)
         unclassified_visual_change (raster recall net, opt-in, not scored via P/R/F1): 0
         confidence calibration:
           0.0-0.5      precision=0.8222 (tp=37 fp=8 n=45)
           0.5-0.75     precision=None (tp=0 fp=0 n=0)
           0.75-0.9     precision=None (tp=0 fp=0 n=0)
           0.9-1.0      precision=None (tp=0 fp=0 n=0)
  [L2] overall: P=0.13 R=0.77 F1=0.23 (tp=33 fp=217 fn=10)
         add      P=0.07 R=1.00 F1=0.13 (tp=8 fp=111 fn=0)
         remove   P=0.00 R=1.00 F1=0.00 (tp=0 fp=61 fn=0)
         modify   P=0.35 R=0.75 F1=0.48 (tp=24 fp=44 fn=8)
         move     P=0.50 R=0.33 F1=0.40 (tp=1 fp=1 fn=2)
         avg primary_recall=0.8111  avg cascade_recall=0.9697  avg semantic_null_emission_rate=1.0
         semantic_null flag detection (engine's own flag vs GT): P=1.00 R=0.14 F1=0.25 (tp=1 fp=0 fn=6)
         unclassified_visual_change (raster recall net, opt-in, not scored via P/R/F1): 0
         confidence calibration:
           0.0-0.5      precision=0.146 (tp=33 fp=193 n=226)
           0.5-0.75     precision=0.0 (tp=0 fp=19 n=19)
           0.75-0.9     precision=0.0 (tp=0 fp=2 n=2)
           0.9-1.0      precision=0.0 (tp=0 fp=3 n=3)

-- null pairs (any delta is a false positive) --
  null_ident_900       [null_ident  ] hard_fp=0 [OK]
  null_prod_901        [null_prod   ] hard_fp=0 [OK]
  null_reword_902      [null_reword ] hard_fp=0 [OK] semantic_null_rate=1.0

-- raster recall net (raster_diff.py + raster_join.py) --
  3 pair(s), 5 raster-only-catchable GT change(s)
  recall with raster off: 0.0
  recall with raster on:  0.0
  recall lift:            0.0
  residue precision (of unclassified_visual_change hits that overlap a real GT change): None

-- not-a-pair refusal --
  not_a_pair_903       refused=True [OK] (drawing numbers differ: '0D204-PID-26-907-001' vs '0D204-PID-26-901-001')

-- chat correctness / groundedness / refusal accuracy --
  n_questions=43
  refusal accuracy: 0.907 (39/43)
  groundedness: fraction_fully_supported=0.7143 citation_support_rate=0.8772 (n_answered_with_citations=35)
  correctness (LLM-judge): accuracy=0.6977 (n_judged=43, n_unparseable=0) *** SAME BACKEND AS CHAT -- self-judging risk, treat as an upper bound ***
  judge validation vs 5 hand-checked answers: agreement=1.0 (5/5) -- n=5 is small enough that a perfect score is not strong evidence on its own

-- llm_direct baseline (3x @ temperature=0, same metrics path) --
  edited_000       F1 mean=1.0000 stdev=0.0000  n_deltas mean=5 stdev=0.0
  edited_001       F1 mean=0.7500 stdev=0.0000  n_deltas mean=5 stdev=0.0
  edited_002       F1 mean=0.8889 stdev=0.0000  n_deltas mean=5 stdev=0.0
  edited_003       F1 mean=0.8889 stdev=0.0000  n_deltas mean=6 stdev=0.0
  edited_004       F1 mean=0.8829 stdev=0.0312  n_deltas mean=17 stdev=0.0
  edited_005       F1 mean=1.0000 stdev=0.0000  n_deltas mean=7 stdev=0.0
  aggregate F1 mean=0.9018  within-pair F1 stdev (mean)=0.0052

-- vs previous run --
  [L0] overall F1: 0.8409 -> 0.8409 (+0.0)
  [L2] overall F1: 0.2253 -> 0.2253 (+0.0)

(elapsed 520.67s)
wrote eval/results/1784976305.json
```

### Reading this scorecard honestly

It is designed to make the system's weaknesses **visible**, not to flatter it.
The things worth your attention are mostly the bad numbers:

- **`remove P=0.00` on both levels.** Every `remove` the engine emits on the
  edited pairs is a false positive. Real and unfixed. The `near_miss_cost`
  guard added recently means these now report confidence `0.0` instead of
  `1.0`, so they are at least correctly *distrusted* — but they are still
  emitted.
- **L2 (scanned) `P=0.13`.** OCR noise wrecks precision — 217 false positives.
  Recall holds at 0.77, so the pipeline still *finds* real changes through
  OCR; it just can't yet tell them from OCR artifacts.
- **Confidence is not well calibrated on L2.** Read the calibration block:
  the `0.9-1.0` band scores precision `0.0` while `0.0-0.5` scores `0.146`.
  Higher confidence is currently *worse*. This was found by adding the
  calibration table, not assumed — the exact kind of thing that stays
  invisible without one.
- **`semantic_null` flag recall = 0.14.** The rule half alone catches one of
  seven. `DELTA_SEMANTIC_NULL_LLM=1` raises this materially (see
  `docs/findings.md`); it is off by default and off in this run.
- **Raster recall lift = 0.0.** The opt-in raster net didn't help on this
  seed. `docs/findings.md` explains why (the valve glyph is independently
  visible to the symbolic geometry pipeline, which suppresses the raster
  hit at the same spot) rather than quietly dropping the metric.
- **The judge shares a backend with the model it judges.** The scorecard
  prints that warning itself, live, next to the number, and says to treat
  `0.6977` as an upper bound. Judge validation is 5/5 — and the line
  immediately notes n=5 is too small to lean on.

What is genuinely good: **0 false positives on all three null pairs**
(identical, re-rendered with a different producer, and reworded-only), the
`not_a_pair` sibling drawing correctly refused rather than diffed, `add` at
`P=1.00 R=1.00` on L0, and `modify` at `P=1.00`.

**On the `llm_direct` baseline (aggregate F1 0.9018).** Handing both PDFs
straight to the LLM scores *higher* on this dataset than the deterministic
engine's 0.84 — and the honest framing is that the baseline is a real
competitor on accuracy alone. What it does not give you is any of: a stable
answer across runs (an earlier run of this same baseline swung one pair from
F1 1.00 → 0.50 → 0.93 at temperature 0), a bounding box to draw markup from,
a confidence signal, a severity ranking, or a per-stage trace when it is
wrong. The comparison is in the repo precisely so that trade-off is
measurable rather than asserted.

**Chat metrics move between runs.** These are live LLM calls; a prior run of
the identical command scored refusal accuracy 0.98 / groundedness 0.75 /
correctness 0.72 against this run's 0.907 / 0.714 / 0.698. The deterministic
half of the scorecard does not move at all — the `vs previous run` block
shows L0 and L2 F1 both at `+0.0`, which is the regression check working.

---

## Reproducing all of it

```bash
make install
make dataset
make run     A=data/samples/real_pair/a/L0.pdf B=data/samples/real_pair/b/L0.pdf
make markup  A=data/samples/real_pair/a/L0.pdf B=data/samples/real_pair/b/L0.pdf
make chat    A=data/samples/real_pair/a/L0.pdf B=data/samples/real_pair/b/L0.pdf
make eval
```

`make chat` and the chat half of `make eval` need an LLM credential — see
[`.env.example`](.env.example). Everything else (delta, markup, report,
trace, and the entire deterministic half of the scorecard) runs with **no
credential at all**.
