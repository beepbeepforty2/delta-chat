"""Retrieval over PID A + PID B + delta report entries. CLAUDE.md decision
#7: "retrieval over PID A + PID B + delta-report JSON entries; every chunk
carries {source, sheet, zone, element_id}."

Homegrown BM25, not a vector DB or LangChain -- consistent with the working
agreements (stdlib + pyproject deps only) and the same "zero infra, fully
understood" reasoning src/observability/tracer.py uses for its own
homegrown choice. For the 2-10 sheet documents and few-hundred-chunk corpus
this project targets, exhaustive BM25 scoring is fast and fully inspectable;
a vector index would be infrastructure this problem doesn't need.

Pure lexical matching has a real gap on natural phrasing: "trip" shares no
token with a setpoint chunk's actual printed content ("...SD HH:150
LL:120"), "spec" shares none with the field name "pipe_class". config/
domain.yaml is a curated alias table (query phrase -> literal drawing
tokens) that expands the query's token set additively before scoring --
never removes or replaces the base tokenization, so it can only help,
never hurt, an already-working query. Missing the config file entirely is
not an error (empty alias table, base lexical matching only).
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import yaml

from src.canonical.model import CanonicalDocument
from src.delta.model import Delta

_TOKEN_RE = re.compile(r"[A-Za-z0-9\"'/.\-]+")

DOMAIN_ALIASES_PATH = os.environ.get("DOMAIN_ALIASES_PATH", "config/domain.yaml")


def _load_domain_aliases(path: str) -> dict[str, list[str]]:
    """Missing file -> empty dict, not an error: retrieval still works
    with zero config (aliases are a precision/recall enhancement layered
    on top of the base lexical index, not a requirement)."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    return {str(k).lower(): [str(v).lower() for v in vs] for k, vs in data.items()}


def _expand_query(query: str, aliases: dict[str, list[str]]) -> list[str]:
    """Additive expansion tokens only -- never replaces the base
    _tokenize() output. Multi-word keys ("high high", "tag number") are
    matched as a substring of the lowercased query text, which also
    naturally covers single-word keys."""
    q_lower = query.lower()
    extra: list[str] = []
    for phrase, targets in aliases.items():
        if phrase in q_lower:
            extra.extend(targets)
    return extra


def _tokenize(text: str) -> list[str]:
    """Whole tokens plus hyphen-split sub-tokens: composite P&ID tags are
    written inconsistently across a document and a query about them (line
    tags, instrument tags like "FIT-9050" in a question vs. "FIT 9050" as
    printed on the drawing per decision #4's field parsing) -- indexing
    only the whole hyphenated token means those two spellings share no
    tokens at all and BM25 scores the match as zero. Splitting is additive
    (both forms stay indexed), so it never removes a signal the exact-token
    match already had."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    for t in tokens[:]:
        if "-" in t:
            tokens.extend(p for p in t.split("-") if p)
    return tokens


@dataclass
class Chunk:
    id: str
    source: str  # "A" | "B" | "delta"
    sheet: int
    zone: Optional[str]
    content: str
    element_type: Optional[str] = None   # set for A/B element chunks
    kind: Optional[str] = None           # set for delta chunks (add/remove/modify/move)


def build_chunks(doc_a: CanonicalDocument, doc_b: CanonicalDocument, deltas: list[Delta]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, doc in (("A", doc_a), ("B", doc_b)):
        for sheet in doc.sheets:
            for el in sheet.elements:
                if el.type in ("geometry", "zone_label") or not el.content.strip():
                    continue  # no text to retrieve on / not a meaningful citation target:
                    # zone_label is just the single border-grid letter/digit
                    # rendered at the sheet edge (decision #5's zone grid),
                    # already surfaced as every other chunk's .zone field --
                    # indexing it too means a query mentioning any zone
                    # letter spuriously dominates BM25 against it (a 1-char
                    # "document" scores enormous under length normalization).
                chunks.append(Chunk(
                    id=el.id, source=source, sheet=el.sheet, zone=el.zone,
                    content=el.content, element_type=el.type,
                ))
    for d in deltas:
        content = d.description or f"{d.kind} {d.element_type}"
        chunks.append(Chunk(
            id=d.did, source="delta", sheet=d.sheet, zone=(d.zone_b or d.zone_a),
            content=content, kind=d.kind,
        ))
    return chunks


DELTA_SOURCE_BOOST = 1.5  # BM25 systematically favors short documents under
# its own length normalization (dl/avg_len) -- a single-line static note
# like "5. DELETED." out-scores a genuinely relevant, longer delta
# description on pure term overlap alone. Live-tested against a real
# "did anything change in zone B-1?" query: at 1.0x the correct delta chunk
# ranked 19th (pushed out of any reasonable top_k) behind a run of
# unrelated short A/B element chunks that merely share zone tokens; 1.5x
# was the smallest boost that consistently surfaced it. This is a source
# (field) weighting, not a hack specific to one query -- delta chunks are
# the direct answer to "what changed" style questions, which is most of
# what this chat feature exists to answer, and there are always far fewer
# of them than raw A/B element chunks, so boosting them costs little
# precision on other query types.


class BM25Index:
    """Standard Okapi BM25 (k1=1.5, b=0.75 defaults), scored exhaustively --
    no inverted-index infra needed at this corpus size."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75,
                 aliases_path: Optional[str] = None):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._aliases = _load_domain_aliases(aliases_path or DOMAIN_ALIASES_PATH)
        # Zone is chunk metadata, not part of .content (the text shown to
        # the LLM/cited) -- but a "what changed in zone F-7?" query has
        # nothing to match against without it, so it's folded into the
        # indexed token surface only, never into .content itself.
        self._doc_tokens = [
            _tokenize(c.content) + (["zone"] + _tokenize(c.zone) if c.zone else [])
            for c in chunks
        ]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avg_len = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0
        df: Counter = Counter()
        for tokens in self._doc_tokens:
            for term in set(tokens):
                df[term] += 1
        n = len(chunks)
        self._idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str, top_k: int = 8) -> list[tuple[Chunk, float]]:
        q_tokens = _tokenize(query) + _expand_query(query, self._aliases)
        scored: list[tuple[Chunk, float]] = []
        for i, chunk in enumerate(self.chunks):
            tf = Counter(self._doc_tokens[i])
            dl = self._doc_len[i]
            score = 0.0
            for term in q_tokens:
                freq = tf.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self._avg_len) if self._avg_len else freq
                score += idf * (freq * (self.k1 + 1)) / denom if denom else 0.0
            if chunk.source == "delta":
                score *= DELTA_SOURCE_BOOST
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_k]
