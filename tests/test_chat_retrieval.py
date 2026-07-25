import pathlib

import pytest

from src.chat.retrieval import BM25Index, Chunk, build_chunks
from src.delta.model import Delta
from src.ingest.pdf_native import PdfNativeAdapter

PAIRS_DIR = pathlib.Path(__file__).parent.parent / "eval" / "datasets" / "v0" / "pairs"


def _make_chunks():
    return [
        Chunk(id="c1", source="A", sheet=1, zone="A-1", content="pipe class GC11S changed to FC11S"),
        Chunk(id="c2", source="A", sheet=1, zone="B-2", content="instrument PIT 9055 26 setpoint HH 245"),
        Chunk(id="c3", source="B", sheet=1, zone="C-3", content="valve 26BL9123 removed from the drawing"),
        Chunk(id="c4", source="delta", sheet=1, zone="A-1", content="pipe class changed: GC11S -> FC11S"),
    ]


def test_bm25_ranks_relevant_chunk_higher():
    index = BM25Index(_make_chunks())
    results = index.search("pipe class GC11S")
    assert results
    top_chunk, _score = results[0]
    assert top_chunk.id in ("c1", "c4")


def test_bm25_returns_empty_for_no_match():
    index = BM25Index(_make_chunks())
    results = index.search("xyzxyz nonsense query")
    assert results == []


def test_bm25_respects_top_k():
    index = BM25Index(_make_chunks())
    results = index.search("pipe valve instrument", top_k=2)
    assert len(results) <= 2


def test_bm25_empty_index():
    index = BM25Index([])
    assert index.search("anything") == []


def test_bm25_alias_expansion_finds_trip_setpoint_by_hh_ll():
    chunks = [
        Chunk(id="c1", source="B", sheet=1, zone="C-3", content="PDIT 9017 26 SD HH:240 LL:110"),
        Chunk(id="c2", source="A", sheet=1, zone="A-1", content="unrelated note about something else"),
    ]
    index = BM25Index(chunks)
    results = index.search("did the trip setpoint change?")
    assert results
    assert results[0][0].id == "c1"


def test_bm25_missing_alias_file_does_not_error():
    chunks = _make_chunks()
    index = BM25Index(chunks, aliases_path="config/does_not_exist.yaml")
    results = index.search("pipe class GC11S")
    assert results  # base lexical matching still works with zero alias config


def test_bm25_alias_expansion_never_removes_base_tokens():
    """A query that already matches on its own must still match with
    aliases loaded -- expansion is additive only."""
    chunks = _make_chunks()
    index = BM25Index(chunks)
    results = index.search("pipe class GC11S")
    top_chunk, _score = results[0]
    assert top_chunk.id in ("c1", "c4")


def test_expand_query_multi_word_phrase():
    from src.chat.retrieval import _expand_query
    aliases = {"tag number": ["equipment_tag"], "trip": ["sd", "hh", "ll"]}
    assert _expand_query("what is the tag number for this pump?", aliases) == ["equipment_tag"]
    assert set(_expand_query("did it trip?", aliases)) == {"sd", "hh", "ll"}
    assert _expand_query("nothing relevant here", aliases) == []


def test_build_chunks_from_real_pair():
    pair_dir = PAIRS_DIR / "edited_002"
    if not pair_dir.exists():
        pytest.skip("run `make dataset` first")
    adapter = PdfNativeAdapter()
    doc_a = adapter.ingest("A", str(pair_dir / "a" / "L0.pdf"))
    doc_b = adapter.ingest("B", str(pair_dir / "b" / "L0.pdf"))
    deltas = [Delta("d1", "add", "note", None, "x", 1, None, "A-3", {}, 1.0, "note added: test")]

    chunks = build_chunks(doc_a, doc_b, deltas)

    assert any(c.source == "A" for c in chunks)
    assert any(c.source == "B" for c in chunks)
    assert any(c.source == "delta" for c in chunks)
    assert not any(c.element_type == "geometry" for c in chunks)
    assert all(c.content.strip() for c in chunks)

    index = BM25Index(chunks)
    results = index.search("valve")
    assert isinstance(results, list)
