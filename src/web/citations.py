"""Turn a chat citation back into a place on the drawing.

`answer()` returns citations as inline markers the model wrote into its
prose -- `[A:1:F-7:el_a1b2c3d4e5f6]`. In the CLI they stay as literal text,
which is fine for a terminal and useless for an engineer who wants to *see*
the thing being cited. This resolves each marker to a normalized box so the
web UI can jump to it and highlight it.

Only the trailing `id` is trustworthy. `validate_citations` checks the id
against what was actually retrieved and checks nothing else
(src/chat/citations.py:53), so a model that writes a plausible-looking but
wrong `sheet` or `zone` into the marker passes validation. Both are treated
here as display hints; every coordinate comes from the id lookup.

Deltas are resolved through the already-built payload records rather than
by re-walking id_a/id_b -> element -> bbox. That chain has a fallback
(raster-origin deltas carry bbox_a/bbox_b on the Delta itself, having no
element to point at) which `payload.build_delta_records` already implements
and tests; a second copy here would be one more place to forget it.
"""
from __future__ import annotations

from src.canonical.model import CanonicalDocument
from src.chat.citations import ParsedCitation
from src.markup.overlay import _index_elements


class CitationResolver:
    """Built once per job, reused across questions -- `_index_elements`
    walks every element in both documents, which is wasted work to repeat
    on each turn of a conversation."""

    def __init__(self, doc_a: CanonicalDocument, doc_b: CanonicalDocument,
                 delta_records: list[dict]):
        self._els = {"A": _index_elements(doc_a), "B": _index_elements(doc_b)}
        self._records = {r["did"]: r for r in delta_records}

    def resolve(self, c: ParsedCitation) -> dict | None:
        """-> {source, sheet, box_a, box_b, description} or None when the id
        matches nothing. None is returned rather than a guessed location:
        a chip that does nothing is honest, a chip that highlights the wrong
        valve is worse than no chip at all."""
        if c.source in ("A", "B"):
            el = self._els[c.source].get(c.id)
            if el is None:
                return None
            box = [el.bbox.x0, el.bbox.y0, el.bbox.x1, el.bbox.y1]
            return {
                "source": c.source,
                "sheet": el.sheet,
                # An element exists in one revision only, so the other pane
                # has nothing to highlight -- the UI shows one box, not two.
                "box_a": box if c.source == "A" else None,
                "box_b": box if c.source == "B" else None,
                "description": el.content,
            }

        if c.source == "delta":
            rec = self._records.get(c.id)
            if rec is None:
                return None
            return {
                "source": "delta",
                "sheet": rec["sheet"],
                "box_a": rec["box_a"],
                "box_b": rec["box_b"],
                "description": rec["description"],
                "did": rec["did"],
                "severity": rec["severity"],
                "kind": rec["kind"],
            }

        # Unknown source label. The regex accepts any non-":[]" run, so a
        # model can invent one; it is not a crash, just unresolvable.
        return None

    def resolve_all(self, citations: list[ParsedCitation]) -> list[dict]:
        """One entry per citation, in the order they appear in the answer,
        each carrying the literal `raw` marker so the client can substitute
        it in the prose by exact string match rather than re-parsing."""
        out = []
        for c in citations:
            resolved = self.resolve(c)
            out.append({
                "raw": c.raw,
                "source": c.source,
                "id": c.id,
                "sheet_hint": c.sheet,
                "zone_hint": c.zone,
                "resolved": resolved,
            })
        return out
