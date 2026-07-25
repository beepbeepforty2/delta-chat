from src.chat.citations import format_citation, parse_citations, validate_citations
from src.chat.retrieval import Chunk


def test_format_citation():
    c = Chunk(id="el_abc123", source="A", sheet=1, zone="F-7", content="x")
    assert format_citation(c) == "[A:1:F-7:el_abc123]"


def test_format_citation_missing_zone():
    c = Chunk(id="el_abc123", source="A", sheet=1, zone=None, content="x")
    assert format_citation(c) == "[A:1:-:el_abc123]"


def test_parse_citations_single():
    text = "The pipe class changed from GC11S to FC11S [A:1:F-7:el_abc123]."
    parsed = parse_citations(text)
    assert len(parsed) == 1
    assert parsed[0].source == "A"
    assert parsed[0].sheet == "1"
    assert parsed[0].zone == "F-7"
    assert parsed[0].id == "el_abc123"
    assert parsed[0].raw == "[A:1:F-7:el_abc123]"


def test_parse_citations_multiple():
    text = "Two things changed [A:1:F-7:el_1] and [delta:1:B-2:delta0003]."
    parsed = parse_citations(text)
    assert len(parsed) == 2
    assert parsed[1].source == "delta"
    assert parsed[1].id == "delta0003"


def test_parse_citations_none():
    assert parse_citations("No citations here.") == []


def test_validate_citations_all_valid():
    chunks = [Chunk(id="el_1", source="A", sheet=1, zone="F-7", content="x")]
    parsed = parse_citations("claim [A:1:F-7:el_1]")
    result = validate_citations(parsed, chunks)
    assert result.all_valid
    assert len(result.valid) == 1
    assert not result.invalid


def test_validate_citations_hallucinated_id():
    chunks = [Chunk(id="el_1", source="A", sheet=1, zone="F-7", content="x")]
    parsed = parse_citations("claim [A:1:F-7:el_999_never_retrieved]")
    result = validate_citations(parsed, chunks)
    assert not result.all_valid
    assert len(result.invalid) == 1


def test_validate_citations_id_matches_despite_wrong_display_fields():
    """The id is ground truth; a typo'd sheet/zone in the citation text
    doesn't independently invalidate it (see citations.py docstring)."""
    chunks = [Chunk(id="el_1", source="A", sheet=1, zone="F-7", content="x")]
    parsed = parse_citations("claim [A:99:Z-9:el_1]")
    result = validate_citations(parsed, chunks)
    assert result.all_valid
