"""Load raw documents and clean them into a list of titled sections."""

import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from fine_print import config

SUPPORTED_SUFFIXES = {".txt", ".html", ".htm", ".pdf"}
DEFAULT_SECTION = "General"
MAX_HEADING_WORDS = 12

NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]
HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "table", "tr", "section", "article", "br"]

MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")
# Zero-width characters are common in text copied from web pages.
INVISIBLE_CHARS = re.compile(r"[​‌‍⁠﻿]")


@dataclass(frozen=True)
class Section:
    heading: str
    text: str


@dataclass(frozen=True)
class Document:
    company: str
    doc_type: str
    source: str
    sections: list[Section]


def parse_filename(path: Path) -> tuple[str, str]:
    """Split `<company>_<doctype>.<ext>` into (company, doc_type), lowercased."""
    company, separator, doc_type = path.stem.lower().partition("_")
    if not (company and separator and doc_type):
        raise ValueError(
            f"Bad filename '{path.name}': expected <company>_<doctype>.<ext>, "
            "for example uber_terms.html"
        )
    return company, doc_type


def clean_text(text: str) -> str:
    """Normalize unicode and whitespace, keeping line and paragraph breaks."""
    text = INVISIBLE_CHARS.sub("", unicodedata.normalize("NFKC", text))
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(html: str) -> str:
    """Convert HTML to plain text, rewriting heading tags as `## Heading` lines."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for tag in soup(HEADING_TAGS):
        tag.replace_with(f"\n\n## {' '.join(tag.get_text().split())}\n\n")
    for tag in soup(BLOCK_TAGS):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    # Single newlines in HTML source are only line wrapping, so join them.
    paragraphs = re.split(r"\n\s*\n", soup.get_text())
    return "\n\n".join(" ".join(p.split()) for p in paragraphs if p.strip())


def pdf_to_text(path: Path) -> str:
    """Extract the text of every page in a PDF."""
    pages = PdfReader(path).pages
    return "\n\n".join(page.extract_text() or "" for page in pages)


def read_text(path: Path) -> str:
    """Read any supported file type as cleaned plain text."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = pdf_to_text(path)
    elif suffix in {".html", ".htm"}:
        raw = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
    return clean_text(raw)


def is_title_case(line: str) -> bool:
    """True if most longer words start with a capital ('Data We Collect')."""
    words = [word for word in line.split() if len(word) > 3]
    capitalized = sum(word[0].isupper() for word in words)
    return bool(words) and capitalized / len(words) >= 0.75


def heading_from_line(line: str, stands_alone: bool) -> str | None:
    """Return the heading text if this line looks like a heading, else None.

    `stands_alone` means blank lines surround it. Title Case alone is weak
    evidence (a wrapped line can look the same), so it also requires this.
    """
    if match := MARKDOWN_HEADING.match(line):
        return match.group(1).strip()
    if len(line.split()) > MAX_HEADING_WORDS or line[-1] in ".,;:?!":
        return None
    if NUMBERED_HEADING.match(line) or (line.isupper() and len(line) > 3):
        return line
    return line if stands_alone and is_title_case(line) else None


def split_into_sections(text: str) -> list[Section]:
    """Group lines under the most recent heading, dropping empty sections."""
    lines = text.splitlines()
    sections: list[Section] = []
    heading = DEFAULT_SECTION
    body: list[str] = []

    def flush() -> None:
        joined = " ".join(line for line in body if line)
        if joined:
            sections.append(Section(heading, joined))
        body.clear()

    for i, line in enumerate(lines):
        if not line:
            continue
        before_blank = i == 0 or not lines[i - 1]
        after_blank = i == len(lines) - 1 or not lines[i + 1]
        found = heading_from_line(line, before_blank and after_blank)
        if found:
            flush()
            heading = found
        else:
            body.append(line)
    flush()
    return sections


def ensure_sample_data(raw_dir: Path = config.RAW_DATA_DIR) -> list[Path]:
    """If raw_dir has no documents, copy in the fictional samples and say so."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if _supported_files(raw_dir):
        return []
    copied = [
        Path(shutil.copy(sample, raw_dir / sample.name))
        for sample in _supported_files(config.SAMPLE_DATA_DIR)
    ]
    names = ", ".join(path.name for path in copied)
    print(f"NOTICE: {raw_dir} was empty, so fictional sample documents were added: {names}")
    return copied


def load_document(path: Path) -> Document:
    """Parse one file into a Document."""
    company, doc_type = parse_filename(path)
    return Document(company, doc_type, path.name, split_into_sections(read_text(path)))


def load_documents(raw_dir: Path = config.RAW_DATA_DIR) -> list[Document]:
    """Load every supported file, skipping any that contain no extractable text."""
    documents: list[Document] = []
    seen: dict[str, str] = {}
    for path in _supported_files(raw_dir):
        key = path.stem.lower()
        if key in seen:
            raise ValueError(f"Duplicate document: {seen[key]} and {path.name}. Keep one.")
        seen[key] = path.name
        document = load_document(path)
        if document.sections:
            documents.append(document)
        else:
            print(f"WARNING: no text found in {path.name} (a scanned PDF?). Skipping it.")
    return documents


def _supported_files(directory: Path) -> list[Path]:
    """List the documents in a directory, in a stable order."""
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
