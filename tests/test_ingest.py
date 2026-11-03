from pathlib import Path

import pytest

from fine_print.chunking import find_sections
from fine_print.config import get_settings
from fine_print.ingest import (
    clean_text,
    html_to_markdown,
    load_document,
    load_documents,
    normalize_headings,
    parse_filename,
)

# A tiny hand-written PDF with one line of text, so the tests need no PDF-writing
# library. It has no cross-reference table, so pypdf rebuilds one when reading it.
PDF_TEXT_STREAM = b"BT /F1 12 Tf 20 100 Td (We keep your data safe.) Tj ET"
MINIMAL_PDF = b"""%%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 200]/Contents 4 0 R
/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>endobj
4 0 obj<</Length %d>>stream
%s
endstream endobj
trailer<</Root 1 0 R>>
startxref
0
%%%%EOF
""" % (len(PDF_TEXT_STREAM), PDF_TEXT_STREAM)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("spotify_privacy.md", ("spotify", "privacy")),
        ("lyft_terms.html", ("lyft", "terms")),
        ("Lyft_Terms.PDF", ("lyft", "terms")),
        ("lyft_privacy_policy.pdf", ("lyft", "privacy_policy")),
    ],
)
def test_parse_filename(filename: str, expected: tuple[str, str]) -> None:
    assert parse_filename(Path("some/folder") / filename) == expected


@pytest.mark.parametrize("filename", ["lyft.txt", "_terms.txt", "lyft_.txt", "MANIFEST.json"])
def test_parse_filename_rejects_bad_names(filename: str) -> None:
    with pytest.raises(ValueError, match="expected <company>_<doctype>"):
        parse_filename(Path(filename))


def test_clean_text_removes_invisible_characters_and_keeps_paragraphs() -> None:
    assert clean_text("a\u200bb\ufeff   c  \n\n\n\n d\xa0e ") == "ab c\n\nd e"


def test_html_keeps_headings_and_body_but_drops_page_furniture() -> None:
    html = """
    <html><head><script>var secret = 1;</script><style>p {color: red}</style></head>
    <body><nav>Home Careers</nav>
    <h2>Your
        Data</h2>
    <p>We keep it <a href="#">for a
        while</a>.</p>
    <footer>Copyright</footer></body></html>
    """
    assert html_to_markdown(html) == "## Your Data\n\nWe keep it for a while."


@pytest.mark.parametrize(
    ("line", "heading"),
    [
        ("Underlined Title\n=====", "# Underlined Title"),
        ("Underlined Subtitle\n---", "## Underlined Subtitle"),
        ("### Already A Heading", "### Already A Heading"),
        ("**2\\. ORDERING SERVICES**", "## 2. ORDERING SERVICES"),
        ("1\\. The Netflix Service", "## 1. The Netflix Service"),
        ("2.1 Numbered Heading", "## 2.1 Numbered Heading"),
        ("ALL CAPS HEADING", "## ALL CAPS HEADING"),
    ],
)
def test_every_heading_style_becomes_a_hash_heading(line: str, heading: str) -> None:
    assert normalize_headings(line) == heading


@pytest.mark.parametrize(
    "line",
    [
        "1. We collect your email address.",
        r"4\. Account Access. Your login details are personal and should be kept confidential.",
        "**Bold sentence that ends with a period.**",
        "[A Navigation Link](https://example.com)",
        "| TABLE ROW | ANOTHER CELL |",
        "*   A LIST ITEM",
        "Plain sentence without any capitals",
    ],
)
def test_ordinary_lines_are_not_mistaken_for_headings(line: str) -> None:
    assert normalize_headings(line) == line


def test_horizontal_rule_after_a_blank_line_is_not_a_heading() -> None:
    text = "Some paragraph.\n\n---\n\nNext paragraph."
    assert normalize_headings(text) == text


def test_title_case_needs_blank_lines_around_it_and_is_opt_in() -> None:
    standalone = "Before.\n\nData We Collect\n\nAfter."
    wrapped = "Before.\nData We Collect\nAfter."
    assert "## Data We Collect" in normalize_headings(standalone, allow_title_case=True)
    assert normalize_headings(standalone) == standalone
    assert normalize_headings(wrapped, allow_title_case=True) == wrapped


def test_markdown_file_is_normalized_but_its_title_case_lines_are_not(tmp_path: Path) -> None:
    path = tmp_path / "acme_terms.md"
    path.write_text("Terms\n=====\n\nHome Page Link\n\nBody.", encoding="utf-8")
    document = load_document(path)
    assert (document.company, document.doc_type, document.source) == (
        "acme",
        "terms",
        "acme_terms.md",
    )
    assert document.text == "# Terms\n\nHome Page Link\n\nBody."


def test_load_documents_reads_supported_types_in_filename_order(tmp_path: Path) -> None:
    (tmp_path / "beta_terms.html").write_text("<h1>Fees</h1><p>Five dollars.</p>")
    (tmp_path / "acme_terms.txt").write_text("Plain text terms.")
    (tmp_path / "gamma_privacy.md").write_text("# Data\nWe keep it.")
    (tmp_path / "MANIFEST.json").write_text("{}")

    documents = load_documents(tmp_path)

    assert [document.company for document in documents] == ["acme", "beta", "gamma"]
    assert find_sections(documents[1].text)[0].title == "Fees"


def test_same_document_in_two_formats_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "acme_terms.txt").write_text("One.")
    (tmp_path / "acme_terms.html").write_text("<p>Two.</p>")
    with pytest.raises(ValueError, match="Duplicate document"):
        load_documents(tmp_path)


def test_file_without_text_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "acme_terms.txt").write_text("  \n ")
    assert load_documents(tmp_path) == []


def test_pdf_text_is_extracted(tmp_path: Path) -> None:
    path = tmp_path / "acme_privacy.pdf"
    path.write_bytes(MINIMAL_PDF)
    assert "We keep your data safe." in load_document(path).text


def test_html_page_title_stays_out_of_the_text() -> None:
    html = "<html><head><title>Page Title</title></head><body><p>Body.</p></body></html>"
    assert html_to_markdown(html) == "Body."


def test_unreadable_pdf_gives_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "acme_terms.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(ValueError, match=r"Could not read acme_terms\.pdf"):
        load_document(path)


def test_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "absent")


def test_sample_html_strips_scripts_and_navigation() -> None:
    document = load_document(get_settings().samples_dir / "ridely_terms.html")
    assert "must not reach the index" not in document.text
    assert "Careers" not in document.text
    assert find_sections(document.text)[0].title == "Ridely Terms of Service"


def test_sample_text_finds_its_numbered_headings() -> None:
    document = load_document(get_settings().samples_dir / "tunely_privacy.txt")
    assert "1. Information We Collect" in [s.title for s in find_sections(document.text)]
