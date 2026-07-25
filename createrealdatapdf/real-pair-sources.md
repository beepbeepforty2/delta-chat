# Held-out real-P&ID test set — sources and procedure

## Why constructed, not found
Matched pre/post revision pairs of the same P&ID are effectively absent from the
public internet — revision history lives in controlled EDMS behind
organizational boundaries. (That absence is itself confirmation of the problem
statement.) A *found* pair would also have no labels, so you'd hand-label anyway.

Constructing gives real extraction difficulty **plus** exact ground truth:
the base document is real; only the edit is ours.

## Source ranking (by redistribution safety)

| Source | Redistributable | Notes |
|---|---|---|
| **DOE Hanford** `hmis.hanford.gov` — e.g. HNF-64103 | yes (US Gov work) | best lead; P&ID standards + embedded example drawings; also try searching the site for `H-9-` series drawings |
| **EPA** public-domain process docs | yes | the PDH course cites `epa.gov/.../wbs-ixclo4-documentation-june-2019.pdf` as public domain |
| **NRC ADAMS** (nuclear plant drawings) | yes (US Gov) | large, awkward search UI; genuine multi-revision documents exist here |
| **NASA NTRS** | yes | fluid/propellant schematics |
| PIP PIC001 (Academia/Scribd) | **no** | copyrighted practice; private use only |
| KLM Technology standard | **no** | free teaser of a paid standard ("order the complete document") |

Keep `gt/provenance.json` accurate per pair (the tool writes it). Only commit
PDFs from the top four rows; for the others, commit the spec + a download script
and let the grader fetch the base themselves.

## Procedure

```bash
pip install pymupdf

# 1. find a page that is an actual drawing (not prose)
python make_real_pair.py inspect base.pdf --page 12
python make_real_pair.py inspect base.pdf --page 12 --grep "HH|PSV|barg|GC11S"

# 2. starter spec, then edit it to match real strings on that page
python make_real_pair.py example-spec > spec.json

# 3. build the pair + ground truth
python make_real_pair.py edit base.pdf spec.json \
    --page 12 --out eval/datasets/holdout/real_001 --pair-id real_001
```

Output matches the generator's layout, so `run_eval.py` needs no new plumbing:
`a/L0.pdf b/L0.pdf gt/{elements_a,elements_b,correspondence,deltas,provenance}.json qa.jsonl`

## The four pairs to build (one per capability boundary)

| Pair | Base | Edits | Boundary tested |
|---|---|---|---|
| real_001 | Hanford example P&ID | setpoint + pipe class | symbolic path on real fonts/density |
| real_002 | EPA / NRC drawing | delete instrument, renumber cascade | add/remove + cascade grouping |
| real_003 | any | `swap_symbol` only | **graphical — raster recall net** |
| real_004 | any | `erase_region` on a line + `move_text` | geometry/connectivity gap (expected miss) |

Pairs 003–004 are expected to **fail** on the symbolic engine alone. That's the
point: they measure the raster net, and 004 documents the connectivity gap.

## Null control (do this too — it's the cheapest calibration you have)
```bash
qpdf --linearize base.pdf base_relin.pdf     # or re-save via a different producer
```
Diff `base.pdf` vs `base_relin.pdf` → ground truth is **empty**. Any delta is a
false positive, and the raster-region count on this pair is the number that tells
you whether your morphological cleanup is calibrated.

## Reporting rule
Report holdout numbers **separately** from generated-set numbers, never pooled.
The gap between them is your realism gap — the only instrument that catches
overfitting to your own edit taxonomy.

## Verified behaviour of the tool
- redaction removes original glyphs; all unchanged text preserved
- `detectable_by` field marks `symbolic` vs `raster_only` per delta
- element ids are content+position hashes (stable across re-extraction)
- the `swap_symbol` edit produces a real pixel-level change inside the glyph
  bbox (confirmed: 73 changed px at 100 dpi in an 18×18pt region) with **no**
  text change — exactly the case the symbolic engine cannot see
