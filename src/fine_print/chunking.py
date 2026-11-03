"""Two ways of cutting a document into chunks, so they can be compared in eval."""

from collections import Counter
from dataclasses import asdict, dataclass

from fine_print import config
from fine_print.ingest import Document


@dataclass(frozen=True)
class Chunk:
    text: str
    company: str
    doc_type: str
    section: str
    chunk_id: str
    strategy: str

    def metadata(self) -> dict[str, str]:
        """Everything except the text, for storing beside the vector."""
        return {key: value for key, value in asdict(self).items() if key != "text"}


def word_windows(total_words: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) word positions for a sliding window over a text.

    Consecutive windows share exactly `overlap` words, so a sentence cut by one
    window boundary still appears whole in a neighbouring chunk.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")

    windows: list[tuple[int, int]] = []
    start = 0
    while start < total_words:
        end = min(start + size, total_words)
        windows.append((start, end))
        if end == total_words:
            break
        start = end - overlap
    return windows


def fixed_size(
    document: Document,
    size: int = config.CHUNK_SIZE_WORDS,
    overlap: int = config.CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Slide a window over the whole document, ignoring its structure.

    A window can span several sections, so each chunk is labelled with the
    section that contributes the most words to it.
    """
    words: list[str] = []
    word_sections: list[str] = []
    for section in document.sections:
        section_words = f"{section.heading} {section.text}".split()
        words.extend(section_words)
        word_sections.extend([section.heading] * len(section_words))

    return [
        _make_chunk(
            document,
            " ".join(words[start:end]),
            Counter(word_sections[start:end]).most_common(1)[0][0],
            position,
            "fixed",
        )
        for position, (start, end) in enumerate(word_windows(len(words), size, overlap))
    ]


def section_aware(
    document: Document,
    max_size: int = config.CHUNK_SIZE_WORDS,
    overlap: int = config.CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Make one chunk per section, sub-splitting only sections that are too long.

    Every chunk starts with its heading so the embedding captures the topic,
    even for the later pieces of a long section.
    """
    chunks: list[Chunk] = []
    for section in document.sections:
        words = section.text.split()
        for start, end in word_windows(len(words), max_size, overlap):
            text = f"{section.heading}\n{' '.join(words[start:end])}"
            chunks.append(_make_chunk(document, text, section.heading, len(chunks), "section"))
    return chunks


def chunk_document(document: Document, strategy: str) -> list[Chunk]:
    """Apply a chunking strategy by name, using the sizes from config."""
    if strategy == "fixed":
        return fixed_size(document)
    if strategy == "section":
        return section_aware(document)
    raise ValueError(f"Unknown strategy '{strategy}'. Choose from {config.STRATEGIES}.")


def chunk_documents(documents: list[Document], strategy: str) -> list[Chunk]:
    """Chunk every document with one strategy."""
    return [chunk for document in documents for chunk in chunk_document(document, strategy)]


def _make_chunk(
    document: Document, text: str, section: str, position: int, strategy: str
) -> Chunk:
    """Build a Chunk with an ID that is unique across documents and strategies."""
    return Chunk(
        text=text,
        company=document.company,
        doc_type=document.doc_type,
        section=section,
        chunk_id=f"{document.company}_{document.doc_type}_{strategy}_{position:04d}",
        strategy=strategy,
    )
