"""Two ways of cutting a document into chunks, so they can be compared in eval."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from fine_print.config import Strategy
from fine_print.models import Chunk, Document

DEFAULT_SECTION = "General"

HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)
MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


@dataclass(frozen=True)
class Section:
    """A heading and its extent: `start:end` includes the heading, `body_start:end` does not."""

    title: str
    start: int
    body_start: int
    end: int


def find_sections(text: str) -> list[Section]:
    """Split Markdown text at its `#` headings, keeping any text before the first one."""
    matches = list(HEADING.finditer(text))
    first_heading = matches[0].start() if matches else len(text)
    sections: list[Section] = []
    if text[:first_heading].strip():
        sections.append(Section(DEFAULT_SECTION, 0, 0, first_heading))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(Section(_clean_title(match.group(1)), match.start(), match.end(), end))
    return sections


def _clean_title(title: str) -> str:
    """Remove Markdown markup from a heading, since it becomes the citation label."""
    return MARKDOWN_LINK.sub(r"\1", title).replace("*", "").replace("\\", "").strip()


def _last_whitespace(text: str, low: int, high: int) -> int | None:
    """Return the position of the last whitespace character in `low:high`, if there is one."""
    return next((i for i in range(high - 1, low - 1, -1) if text[i].isspace()), None)


def sliding_windows(
    text: str, start: int, end: int, size: int, overlap: int
) -> list[tuple[int, int]]:
    """Return (start, end) spans covering `start:end`, each at most `size` characters.

    Spans break at whitespace where possible, so words stay whole. To keep that true of
    the shared text too, consecutive spans share between `overlap` and `2 * overlap`
    characters, which means a sentence cut by one boundary still appears in a neighbour.

    Args:
        text: The full text the positions refer to.
        start: Where to begin.
        end: Where to stop.
        size: The maximum length of a span.
        overlap: The minimum number of characters consecutive spans share.

    Returns:
        The spans, in order.

    Raises:
        ValueError: If `size` is not positive, or `overlap` is not in `0 <= overlap < size`.
    """
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be at least 0 and smaller than the chunk size")

    spans: list[tuple[int, int]] = []
    while start + size < end:
        # Ending past start + overlap guarantees that the next span starts further on.
        stop = _last_whitespace(text, start + overlap + 1, start + size + 1) or start + size
        spans.append((start, stop))
        next_start = stop - overlap
        floor = max(start + 1, next_start - overlap)
        whitespace = _last_whitespace(text, floor, next_start)
        start = next_start if whitespace is None else whitespace + 1
    spans.append((start, end))
    return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a span to exclude the whitespace at its edges."""
    piece = text[start:end]
    return start + len(piece) - len(piece.lstrip()), end - len(piece) + len(piece.rstrip())


def _make_chunk(
    document: Document, span: tuple[int, int], section: str, position: int, strategy: Strategy
) -> Chunk:
    """Build a Chunk with an ID that is unique across documents and strategies."""
    return Chunk(
        text=document.text[span[0] : span[1]],
        company=document.company,
        doc_type=document.doc_type,
        section=section,
        chunk_id=f"{document.company}_{document.doc_type}_{strategy}_{position:04d}",
        strategy=strategy,
        start_char=span[0],
        end_char=span[1],
    )


def _majority_section(sections: list[Section], start: int, end: int) -> str:
    """Return the title of the section that contributes the most characters to a span."""
    return max(sections, key=lambda s: min(end, s.end) - max(start, s.start)).title


def fixed_size(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Slide a window over the whole document, ignoring its structure.

    A window can span several sections, so each chunk is labelled with the section that
    contributes the most characters to it.
    """
    text = document.text
    sections = find_sections(text)
    spans = [_trim(text, *span) for span in sliding_windows(text, 0, len(text), size, overlap)]
    spans = [span for span in spans if span[0] < span[1]]
    return [
        _make_chunk(document, span, _majority_section(sections, *span), position, "fixed")
        for position, span in enumerate(spans)
    ]


def section_aware(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Make one chunk per section, sub-splitting only sections that are too long.

    Chunks never cross a section boundary. They hold only body text, because the title is
    kept in `Chunk.section` and `Chunk.embedding_text` puts it back for embedding.
    """
    text = document.text
    chunks: list[Chunk] = []
    for section in find_sections(text):
        for window in sliding_windows(text, section.body_start, section.end, size, overlap):
            span = _trim(text, *window)
            if span[0] < span[1]:
                chunks.append(_make_chunk(document, span, section.title, len(chunks), "section"))
    return chunks


def chunk_document(document: Document, strategy: Strategy, size: int, overlap: int) -> list[Chunk]:
    """Apply a chunking strategy by name.

    Raises:
        ValueError: If the strategy is unknown, or the size and overlap are invalid.
    """
    if strategy == "fixed":
        return fixed_size(document, size, overlap)
    if strategy == "section":
        return section_aware(document, size, overlap)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def mean_chunk_chars(
    documents: Sequence[Document], strategy: Strategy, size: int, overlap: int
) -> float:
    """Return the mean chunk length over some documents, or 0 if there are no chunks."""
    lengths = [
        len(chunk.text)
        for document in documents
        for chunk in chunk_document(document, strategy, size, overlap)
    ]
    return sum(lengths) / len(lengths) if lengths else 0.0


def matching_fixed_size(documents: Sequence[Document], target_mean: float, overlap: int) -> int:
    """Find the fixed chunk size whose mean chunk length is closest to a target.

    Keyword-based evaluation favours longer chunks, so strategies can only be compared
    fairly when their chunks are about the same length.

    Args:
        documents: The documents to chunk.
        target_mean: The mean chunk length to match, in characters.
        overlap: The overlap to use, which has to stay the same as the other strategy's.

    Returns:
        The chunk size, which is always larger than the overlap.
    """
    low, high = overlap + 1, max(overlap + 2, int(target_mean * 2))
    while high - low > 1:
        middle = (low + high) // 2
        if mean_chunk_chars(documents, "fixed", middle, overlap) < target_mean:
            low = middle
        else:
            high = middle
    return min(
        (low, high),
        key=lambda size: abs(mean_chunk_chars(documents, "fixed", size, overlap) - target_mean),
    )
