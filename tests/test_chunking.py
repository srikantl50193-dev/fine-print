import pytest

from fine_print.chunking import chunk_document, fixed_size, section_aware, word_windows
from fine_print.ingest import Document, Section


def numbered_words(count: int) -> str:
    return " ".join(f"w{i}" for i in range(count))


@pytest.fixture
def document() -> Document:
    return Document(
        company="acme",
        doc_type="terms",
        source="acme_terms.txt",
        sections=[
            Section("Short Section", numbered_words(20)),
            Section("Long Section", numbered_words(95)),
            Section("Last Section", numbered_words(10)),
        ],
    )


def test_consecutive_fixed_chunks_share_exactly_the_overlap(document: Document) -> None:
    chunks = fixed_size(document, size=40, overlap=10)
    assert len(chunks) > 1
    for current, following in zip(chunks, chunks[1:]):
        assert current.text.split()[-10:] == following.text.split()[:10]


def test_fixed_chunks_respect_size_and_lose_no_words(document: Document) -> None:
    chunks = fixed_size(document, size=40, overlap=10)
    assert all(len(chunk.text.split()) <= 40 for chunk in chunks)

    rebuilt = chunks[0].text.split()
    for chunk in chunks[1:]:
        rebuilt.extend(chunk.text.split()[10:])
    original = [
        word
        for section in document.sections
        for word in f"{section.heading} {section.text}".split()
    ]
    assert rebuilt == original


def test_fixed_chunk_is_labelled_with_its_majority_section(document: Document) -> None:
    # The first 40 words hold 22 from "Short Section" and 18 from "Long Section".
    assert fixed_size(document, size=40, overlap=0)[0].section == "Short Section"


def test_short_sections_stay_whole_and_long_sections_are_split(document: Document) -> None:
    chunks = section_aware(document, max_size=40, overlap=10)
    counts = {
        heading: sum(chunk.section == heading for chunk in chunks)
        for heading in ("Short Section", "Long Section", "Last Section")
    }
    assert counts == {"Short Section": 1, "Long Section": 3, "Last Section": 1}


def test_every_section_chunk_starts_with_its_heading(document: Document) -> None:
    for chunk in section_aware(document, max_size=40, overlap=10):
        assert chunk.text.startswith(f"{chunk.section}\n")


def test_section_subsplits_overlap(document: Document) -> None:
    long_chunks = [
        chunk
        for chunk in section_aware(document, max_size=40, overlap=10)
        if chunk.section == "Long Section"
    ]
    bodies = [chunk.text.split("\n", 1)[1].split() for chunk in long_chunks]
    for current, following in zip(bodies, bodies[1:]):
        assert current[-10:] == following[:10]


@pytest.mark.parametrize("strategy", ["fixed", "section"])
def test_no_empty_chunks_and_all_metadata_present(document: Document, strategy: str) -> None:
    chunks = chunk_document(document, strategy)
    assert chunks
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.metadata() == {
            "company": "acme",
            "doc_type": "terms",
            "section": chunk.section,
            "chunk_id": chunk.chunk_id,
            "strategy": strategy,
        }
        assert chunk.section


@pytest.mark.parametrize("strategy", ["fixed", "section"])
def test_chunk_ids_are_unique(document: Document, strategy: str) -> None:
    ids = [chunk.chunk_id for chunk in chunk_document(document, strategy)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("strategy", ["fixed", "section"])
def test_document_without_sections_gives_no_chunks(strategy: str) -> None:
    assert chunk_document(Document("acme", "terms", "acme_terms.txt", []), strategy) == []


def test_final_window_is_not_a_redundant_tail() -> None:
    # A window ending at the last word must stop the loop, or the final words
    # would be repeated in an extra chunk that adds no new information.
    assert word_windows(10, size=10, overlap=3) == [(0, 10)]
    assert word_windows(17, size=10, overlap=3) == [(0, 10), (7, 17)]


@pytest.mark.parametrize("size, overlap", [(10, 10), (10, 12), (10, -1), (0, 0)])
def test_invalid_window_settings_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        word_windows(100, size, overlap)


def test_unknown_strategy_is_rejected(document: Document) -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        chunk_document(document, "magic")
