from itertools import pairwise

import pytest

from fine_print.chunking import (
    DEFAULT_SECTION,
    chunk_document,
    find_sections,
    fixed_size,
    matching_fixed_size,
    mean_chunk_chars,
    section_aware,
    sliding_windows,
)
from fine_print.config import Strategy
from fine_print.models import Chunk, Document

SIZE = 200
OVERLAP = 40
STRATEGIES: tuple[Strategy, ...] = ("fixed", "section")


def numbered_words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


@pytest.fixture
def document() -> Document:
    text = (
        f"Opening words before any heading.\n\n"
        f"# Short Section\n{numbered_words('short', 10)}\n\n"
        f"## Long [Section](#long) **Title**\n{numbered_words('long', 200)}\n\n"
        f"## Heading Without A Body\n\n"
        f"## Last Section\n{numbered_words('last', 10)}"
    )
    return Document(company="acme", doc_type="terms", source="acme_terms.md", text=text)


def section_of(document: Document, title: str) -> str:
    section = next(s for s in find_sections(document.text) if s.title == title)
    return document.text[section.body_start : section.end]


def long_chunks(document: Document) -> list[Chunk]:
    chunks = section_aware(document, SIZE, OVERLAP)
    return [chunk for chunk in chunks if chunk.section == "Long Section Title"]


# Overlap


@pytest.mark.parametrize("chunker", [fixed_size, section_aware])
def test_consecutive_chunks_share_at_least_the_overlap(document: Document, chunker) -> None:
    chunks = [c for c in chunker(document, SIZE, OVERLAP) if c.section == "Long Section Title"]
    assert len(chunks) > 1
    for current, following in pairwise(chunks):
        shared = current.end_char - following.start_char
        assert OVERLAP <= shared <= 2 * OVERLAP
        assert current.text[-shared:] == following.text[:shared]


def test_zero_overlap_cuts_chunks_cleanly(document: Document) -> None:
    chunks = [c for c in section_aware(document, SIZE, 0) if c.section == "Long Section Title"]
    assert len(chunks) > 1
    for current, following in pairwise(chunks):
        assert current.end_char <= following.start_char


def test_splitting_loses_and_invents_no_words(document: Document) -> None:
    rebuilt: list[str] = []
    position = 0
    for chunk in long_chunks(document):
        rebuilt.extend(document.text[max(position, chunk.start_char) : chunk.end_char].split())
        position = chunk.end_char
    assert rebuilt == section_of(document, "Long Section Title").split()


def test_chunks_split_between_words(document: Document) -> None:
    allowed = set(section_of(document, "Long Section Title").split())
    for chunk in long_chunks(document):
        assert set(chunk.text.split()) <= allowed


def test_text_without_whitespace_is_still_cut_to_size() -> None:
    text = "x" * 1000
    spans = sliding_windows(text, 0, len(text), SIZE, OVERLAP)
    assert spans[0] == (0, SIZE)
    assert all(end - start <= SIZE for start, end in spans)
    assert all(b[0] > a[0] for a, b in pairwise(spans))
    assert spans[-1][1] == len(text)


def test_span_that_fits_is_not_split() -> None:
    assert sliding_windows("short text", 0, 10, SIZE, OVERLAP) == [(0, 10)]


@pytest.mark.parametrize(("size", "overlap"), [(10, 10), (10, 12), (10, -1), (0, 0), (-5, 0)])
def test_invalid_window_settings_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="Chunk"):
        sliding_windows("any text", 0, 8, size, overlap)


# Empty chunks and size limits


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_no_chunk_is_empty_or_oversized(document: Document, strategy: Strategy) -> None:
    chunks = chunk_document(document, strategy, SIZE, OVERLAP)
    assert chunks
    for chunk in chunks:
        assert chunk.text == chunk.text.strip()
        assert chunk.text
        assert len(chunk.text) <= SIZE


def test_heading_without_a_body_makes_no_chunk(document: Document) -> None:
    sections = {chunk.section for chunk in section_aware(document, SIZE, OVERLAP)}
    assert "Heading Without A Body" not in sections


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("text", ["", "   \n\n  "])
def test_blank_document_gives_no_chunks(text: str, strategy: Strategy) -> None:
    document = Document(company="acme", doc_type="terms", source="a_b.md", text=text)
    assert chunk_document(document, strategy, SIZE, OVERLAP) == []


def test_document_with_only_a_heading_gives_no_section_chunks() -> None:
    document = Document(company="acme", doc_type="terms", source="a_b.md", text="# Only A Heading")
    assert section_aware(document, SIZE, OVERLAP) == []


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_document_without_headings_uses_the_default_section(strategy: Strategy) -> None:
    document = Document(company="acme", doc_type="terms", source="a_b.md", text="Just a paragraph.")
    chunks = chunk_document(document, strategy, SIZE, OVERLAP)
    assert [(chunk.section, chunk.text) for chunk in chunks] == [
        (DEFAULT_SECTION, "Just a paragraph.")
    ]


# Metadata


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_metadata_is_present_and_offsets_locate_the_text(
    document: Document, strategy: Strategy
) -> None:
    for chunk in chunk_document(document, strategy, SIZE, OVERLAP):
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.metadata() == {
            "company": "acme",
            "doc_type": "terms",
            "section": chunk.section,
            "chunk_id": chunk.chunk_id,
            "strategy": strategy,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }
        assert chunk.section


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunk_ids_are_unique_and_name_their_origin(document: Document, strategy: Strategy) -> None:
    ids = [chunk.chunk_id for chunk in chunk_document(document, strategy, SIZE, OVERLAP)]
    assert len(ids) == len(set(ids))
    assert ids[0] == f"acme_terms_{strategy}_0000"


def test_chunk_ids_differ_between_strategies(document: Document) -> None:
    fixed = {chunk.chunk_id for chunk in fixed_size(document, SIZE, OVERLAP)}
    section = {chunk.chunk_id for chunk in section_aware(document, SIZE, OVERLAP)}
    assert not fixed & section


# Section boundaries


def test_section_chunks_stay_inside_their_section(document: Document) -> None:
    sections = {s.title: s for s in find_sections(document.text)}
    chunks = section_aware(document, SIZE, OVERLAP)
    assert {chunk.section for chunk in chunks} == {
        DEFAULT_SECTION,
        "Short Section",
        "Long Section Title",
        "Last Section",
    }
    for chunk in chunks:
        section = sections[chunk.section]
        assert section.body_start <= chunk.start_char
        assert chunk.end_char <= section.end


def test_section_chunks_contain_no_heading_lines(document: Document) -> None:
    for chunk in section_aware(document, SIZE, OVERLAP):
        assert "#" not in chunk.text


def test_short_sections_stay_whole_and_long_sections_are_split(document: Document) -> None:
    chunks = section_aware(document, SIZE, OVERLAP)
    short = [chunk for chunk in chunks if chunk.section == "Short Section"]
    assert [chunk.text for chunk in short] == [numbered_words("short", 10)]
    assert len(long_chunks(document)) > 1


def test_overlap_never_reaches_into_a_neighbouring_section(document: Document) -> None:
    last = next(c for c in section_aware(document, SIZE, OVERLAP) if c.section == "Last Section")
    assert last.text == numbered_words("last", 10)


def test_fixed_chunk_is_labelled_with_its_majority_section() -> None:
    text = f"# First\n{'a ' * 10}\n# Second\n{'b ' * 70}"
    document = Document(company="acme", doc_type="terms", source="a_b.md", text=text.strip())
    # "First" takes up the first 29 characters of the document.
    assert fixed_size(document, 40, 0)[0].section == "First"
    assert fixed_size(document, 150, 0)[0].section == "Second"


def test_fixed_chunks_may_cross_section_boundaries(document: Document) -> None:
    chunks = fixed_size(document, 400, OVERLAP)
    assert any("# Short Section" in chunk.text and "short0" in chunk.text for chunk in chunks)
    assert any(chunk.text.startswith("Opening words") and "#" in chunk.text for chunk in chunks)


def test_text_before_the_first_heading_gets_the_default_section() -> None:
    sections = find_sections("Intro.\n\n# Heading\nBody.")
    assert [s.title for s in sections] == [DEFAULT_SECTION, "Heading"]


def test_section_titles_lose_their_markdown() -> None:
    sections = find_sections(r"## [**1\. Intro**](#intro)" + "\nBody.")
    assert [s.title for s in sections] == ["1. Intro"]


def test_mean_chunk_chars_of_nothing_is_zero() -> None:
    assert mean_chunk_chars([], "fixed", SIZE, OVERLAP) == 0.0


def test_matching_fixed_size_gets_close_to_the_target(document: Document) -> None:
    documents = [document, document.model_copy(update={"company": "beta"})]
    target = 150.0

    size = matching_fixed_size(documents, target, OVERLAP)

    assert size > OVERLAP
    assert mean_chunk_chars(documents, "fixed", size, OVERLAP) == pytest.approx(target, abs=10)


def test_matching_fixed_size_is_closer_than_its_neighbours(document: Document) -> None:
    size = matching_fixed_size([document], 180.0, OVERLAP)
    miss = abs(mean_chunk_chars([document], "fixed", size, OVERLAP) - 180.0)
    assert miss <= abs(mean_chunk_chars([document], "fixed", size + 50, OVERLAP) - 180.0)
    assert miss <= abs(mean_chunk_chars([document], "fixed", size - 50, OVERLAP) - 180.0)


def test_unknown_strategy_is_rejected(document: Document) -> None:
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_document(document, "magic", SIZE, OVERLAP)  # ty: ignore[invalid-argument-type]
