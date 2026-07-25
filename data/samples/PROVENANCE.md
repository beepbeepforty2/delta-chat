# Real sample provenance

Two real vendor P&ID PDFs, used as format exemplars and for validating the
native-PDF adapter (`src/ingest/pdf_native.py`) against real element density
and layout, not just the synthetic dataset generator's output.

| File | Drawing number | Title | Vendor |
|---|---|---|---|
| `Lift Gas compressor-P&ID.pdf` | 26-KA-901 | 3rd Stage HP Gas Lift Compressor | MAN Energy Solutions |
| `Export Gas Compressor-P&ID (1).pdf` | 26-KA-902 | 3rd Stage HP Gas Export Compressor | MAN Energy Solutions |

Source: assignment example materials. Redistribution rights confirmed.

**These two are not a revision pair.** They're sibling drawings — different
equipment (gas lift vs. gas export compressor), different duty/flow/stage
count. Do not treat them as an edited A/B pair in tests; they're useful
individually as extraction exemplars, and later as a real-world test case
for the `not_a_pair` pre-check (DESIGN.md decision #6, README Plan step 3).

Every operator in `eval/datasets/generator/ops.py` and the composite-tag
regexes in `src/canonical/tags.py` were originally modeled on edits observed
between this exact pair (see `DESIGN.md`'s Dataset section) — these files
are the ground truth those design decisions trace back to, not just
incidental samples.

Both files share an identical drafting template (confirmed by visual
inspection): A-J x 1-12 border-grid zone labels, a datasheet block bottom-
left (TAG NUMBER / SERVICE / DUTY / FLOW RATE / ... / VENDOR), and a
numbered-notes block spanning the full bottom width. Neither file's visible
page area contains a separate REV/DRAWN/CHECKED title-block stamp.
