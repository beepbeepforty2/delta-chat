# Real, hand-edited revision pair

Unlike the eval harness's seeded synthetic dataset (`eval/datasets/v0/`), this
is a **real** single-sheet revision pair: a genuine crop of real vendor P&ID
content (not synthetic generator output), hand-edited with actual PDF editing
tools (PyMuPDF redact + reinsert), not a re-render.

## Provenance

Cropped from `data/samples/Lift Gas compressor-P&ID.pdf` (drawing 26-KA-901,
3rd Stage HP Gas Lift Compressor, MAN Energy Solutions — see the parent
`data/samples/PROVENANCE.md` for that file's own redistribution note).
Region: columns 1-3, full height (`fitz.Rect(0, 0, 250, 822)` of the original
1191x842pt page), containing an instrument bubble cluster, ~10 line tags
with real pipe-class codes, ~13 valve tags, the full equipment datasheet
block (`TAG NUMBER` / `SERVICE` / `DUTY` / ... / `VENDOR`), and the first 15
entries of the drawing's own real numbered-notes list (5 of which — 6, 7,
10, 12, 15 — are themselves already real historical `N. DELETED.` entries).

Built via: `insert_pdf` (flattened page copy) → `set_cropbox` **before** any
redaction (critical ordering — see note below) → redact everything outside
the crop rect with `PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED` so out-of-crop
vector geometry is actually clipped, not just visually hidden → resaved with
`garbage=4, deflate=True`.

## Edits made to produce revision B (`b/L0.pdf`)

1. **Line tag pipe-class change** (zone F-8): `2"-WC-40-9014-AC21-00` →
   `...-AC31-00`. Correctly classified as `modify`/`line_tag`, `HIGH`
   severity (a mechanical rating change, not administrative).
2. **Note 5 reworded**, same meaning: `OIL CHANGE BY USING TEMPORARY
   ARRANGEMENT WITH HOSES.` → `OIL CHANGE USING TEMPORARY HOSE
   ARRANGEMENT.`. With `DELTA_SEMANTIC_NULL_LLM=1`, correctly flagged
   `semantic_null=True` by the LLM adjudication pass (verified live).
3. **Notes 13 & 14 collapsed** into a single `13-14. DELETED.` entry, and
   note 15 renumbered to 14 — the same `N. DELETED.` → `N-M. DELETED.`
   collapse pattern this drawing's own real revision history already
   contains five times over.

No vertical reflow was done after the collapse (the vacated row is left
blank rather than shifting notes 15+ up by one line) — a deliberate
simplification; harmless to extraction (an empty row yields no element).

## Two real things this crop surfaced that the synthetic dataset never would

- **A zone-row-label collision.** Note 14's text line sits close enough to
  the sheet's own "J" row-grid-label glyph that `pdf_native.py`'s span
  clustering (see its module docstring) merges them into one cluster,
  `"J 14. VENT ROUTED TO SAFE LOCATION."` — which fails `NUMBERED_NOTE_RE`
  (no longer starts with a digit) and falls through to a generic `text`
  classification instead of `note`. This is *not* a synthetic-dataset
  formatting artifact; it's a real consequence of this drawing's actual
  layout density, and it shows up in the delta output as a `remove text` /
  `add unknown` pair rather than a clean note collapse. One consequence:
  `semantic_null.py`'s rule (`kind == "modify"` on a `note_deleted`) doesn't
  fire on this particular collapse, because the real alignment resolves it
  as `remove note` + `add note_deleted` rather than a single `modify` — a
  real, narrower-than-ideal scope boundary of that rule, not a bug.
- **`insert_text` with a raw embedded CID font corrupts extracted text.**
  Re-embedding the source's own Calibri (`doc.extract_font(32)`, a
  Type0/Identity-H TrueType subset with no `ToUnicode` CMap) for visual
  fidelity looked perfect when rendered, but every `-` (U+002D) in inserted
  text silently re-extracted as `‐` (U+2010) via `get_text()` — enough to
  break `LINE_TAG_RE`/`DELETED_NOTE_RE`, both anchored on a literal hyphen.
  Switched to the base-14 `helv` font for all inserted text, which
  round-trips every character correctly. Worth remembering for any future
  PDF-editing script: a visually-perfect embedded-font edit is not
  necessarily a *text-correct* one.

## Regenerating

`tools/` has no dedicated script for this (it was built as a one-off, by
hand, from `data/samples/Lift Gas compressor-P&ID.pdf`) — these two files
are themselves the artifact and are committed directly.
