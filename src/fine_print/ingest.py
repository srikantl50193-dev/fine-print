"""Load raw policy files into Documents whose text is Markdown with `#` headings."""

import re
import unicodedata
from pathlib import Path

import structlog
from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from fine_print.models import Document

log = structlog.get_logger(__name__)

MARKDOWN_SUFFIXES = {".md"}
HTML_SUFFIXES = {".html", ".htm"}
SUPPORTED_SUFFIXES = {*MARKDOWN_SUFFIXES, *HTML_SUFFIXES, ".txt", ".pdf"}

MAX_HEADING_WORDS = 12
MAX_HEADING_CHARS = 200
MIN_TITLE_CASE_RATIO = 0.75
MIN_CAPS_HEADING_CHARS = 4
MIN_TITLE_CASE_WORD_CHARS = 4

NOISE_TAGS = ["title", "script", "style", "noscript", "nav", "header", "footer", "aside", "form"]
HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "table", "tr", "section", "article", "br"]

ATX_HEADING = re.compile(r"^#{1,6}\s")
SETEXT_HEADING = re.compile(rf"^([^\n]{{1,{MAX_HEADING_CHARS}}})\n(=+|-+)$", re.MULTILINE)
BOLD_LINE = re.compile(r"^\*\*([^*]+)\*\*$")
NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")
# Zero-width characters are common in text copied from web pages.
INVISIBLE_CHARS = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
# Table rows, links, lists, images and quotes are never headings, whatever their case.
MARKUP_PREFIXES = ("|", "[", "*", "-", "!", ">")


def parse_filename(path: Path) -> tuple[str, str]:
    """Split `<company>_<doctype>.<ext>` into (company, doc_type), lowercased.

    Args:
        path: The document's path. Only its name is read.

    Returns:
        The company and the document type.

    Raises:
        ValueError: If the name does not follow the pattern.
    """
    company, separator, doc_type = path.stem.lower().partition("_")
    if not (company and separator and doc_type):
        raise ValueError(
            f"Bad filename '{path.name}': expected <company>_<doctype>.<ext>, "
            "for example spotify_terms.md"
        )
    return company, doc_type


def clean_text(text: str) -> str:
    """Normalize unicode and whitespace, keeping line and paragraph breaks."""
    text = INVISIBLE_CHARS.sub("", unicodedata.normalize("NFKC", text))
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_markdown(html: str) -> str:
    """Convert HTML to text with `#` headings, dropping scripts and page furniture."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for tag in soup(HEADING_TAGS):
        level = int(tag.name[1])
        tag.replace_with(f"\n\n{'#' * level} {' '.join(tag.get_text().split())}\n\n")
    for tag in soup(BLOCK_TAGS):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    # Single newlines in HTML source are only line wrapping, so join them.
    paragraphs = re.split(r"\n\s*\n", soup.get_text())
    return "\n\n".join(" ".join(p.split()) for p in paragraphs if p.strip())


def _is_title_case(line: str) -> bool:
    """Return whether most longer words start with a capital, as in 'Data We Collect'."""
    words = [word for word in line.split() if len(word) >= MIN_TITLE_CASE_WORD_CHARS]
    capitalized = sum(word[0].isupper() for word in words)
    return bool(words) and capitalized / len(words) >= MIN_TITLE_CASE_RATIO


def _heading_from_line(line: str, *, allow_title_case: bool) -> str | None:
    """Return the heading text if this line looks like a heading, else None."""
    bold = BOLD_LINE.match(line)
    # Markdown converters escape list-like numbering, as in "1\. The Service".
    title = (bold.group(1) if bold else line).strip().replace("\\", "")
    if (
        not title
        or (not bold and line.startswith(MARKUP_PREFIXES))
        or len(title.split()) > MAX_HEADING_WORDS
        or title[-1] in ".,;:?!"
    ):
        return None
    if bold or NUMBERED_HEADING.match(title):
        return title
    if title.isupper() and len(title) >= MIN_CAPS_HEADING_CHARS:
        return title
    return title if allow_title_case and _is_title_case(title) else None


def normalize_headings(text: str, *, allow_title_case: bool = False) -> str:
    """Rewrite every heading style as a `#` heading line.

    Args:
        text: Cleaned text, with stripped lines.
        allow_title_case: Also treat a Title Case line with blank lines around it as a
            heading. This suits plain text and PDFs. Markdown marks its headings, so there
            it would only add false positives such as navigation links.

    Returns:
        The text with `#` headings.
    """
    text = SETEXT_HEADING.sub(
        lambda match: f"{'#' if match.group(2)[0] == '=' else '##'} {match.group(1)}", text
    )
    lines = text.splitlines()
    output: list[str] = []
    for index, line in enumerate(lines):
        if not line or ATX_HEADING.match(line):
            output.append(line)
            continue
        stands_alone = (index == 0 or not lines[index - 1]) and (
            index == len(lines) - 1 or not lines[index + 1]
        )
        heading = _heading_from_line(line, allow_title_case=allow_title_case and stands_alone)
        output.append(f"## {heading}" if heading else line)
    return "\n".join(output)


def read_as_markdown(path: Path) -> str:
    """Read any supported file as cleaned Markdown with `#` headings."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = PdfReader(path).pages
        raw = "\n\n".join(page.extract_text() or "" for page in pages)
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in HTML_SUFFIXES:
        return clean_text(html_to_markdown(raw))
    return normalize_headings(clean_text(raw), allow_title_case=suffix not in MARKDOWN_SUFFIXES)


def load_document(path: Path) -> Document:
    """Parse one file into a Document.

    Raises:
        ValueError: If the filename is invalid or a PDF cannot be read.
    """
    company, doc_type = parse_filename(path)
    try:
        text = read_as_markdown(path)
    except PyPdfError as error:
        raise ValueError(f"Could not read {path.name}: {error}") from error
    return Document(company=company, doc_type=doc_type, source=path.name, text=text)


def load_documents(raw_dir: Path) -> list[Document]:
    """Load every supported file in a folder, skipping any with no extractable text.

    Args:
        raw_dir: The folder of `<company>_<doctype>.<ext>` files.

    Returns:
        The documents, in filename order.

    Raises:
        FileNotFoundError: If the folder does not exist.
        ValueError: If a filename is invalid, or one document exists in two formats.
    """
    documents: list[Document] = []
    seen: dict[str, str] = {}
    paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    for path in paths:
        key = path.stem.lower()
        if key in seen:
            raise ValueError(f"Duplicate document: {seen[key]} and {path.name}. Keep one.")
        seen[key] = path.name
        document = load_document(path)
        if document.text:
            documents.append(document)
        else:
            log.warning("no_text_extracted", file=path.name)
    return documents
