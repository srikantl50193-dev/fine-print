from pathlib import Path

import pytest

from fine_print.ingest import (
    ensure_sample_data,
    html_to_text,
    load_documents,
    parse_filename,
    split_into_sections,
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
    "filename, expected",
    [
        ("spotify_privacy.txt", ("spotify", "privacy")),
        ("uber_terms.html", ("uber", "terms")),
        ("Uber_Terms.PDF", ("uber", "terms")),
        ("uber_privacy_policy.pdf", ("uber", "privacy_policy")),
    ],
)
def test_parse_filename(filename: str, expected: tuple[str, str]) -> None:
    assert parse_filename(Path(filename)) == expected


@pytest.mark.parametrize("filename", ["uber.txt", "_terms.txt", "uber_.txt"])
def test_parse_filename_rejects_bad_names(filename: str) -> None:
    with pytest.raises(ValueError, match="expected <company>_<doctype>"):
        parse_filename(Path(filename))


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
    assert html_to_text(html) == "## Your Data\n\nWe keep it for a while."


def test_split_recognizes_each_heading_style() -> None:
    text = (
        "Intro text before any heading.\n\n"
        "# Markdown Heading\nBody one.\n\n"
        "2.1 Numbered Heading\nBody two.\n\n"
        "ALL CAPS HEADING\nBody three.\n\n"
        "Title Case Heading\n\nBody four."
    )
    sections = split_into_sections(text)
    assert [(s.heading, s.text) for s in sections] == [
        ("General", "Intro text before any heading."),
        ("Markdown Heading", "Body one."),
        ("2.1 Numbered Heading", "Body two."),
        ("ALL CAPS HEADING", "Body three."),
        ("Title Case Heading", "Body four."),
    ]


def test_sentences_are_not_mistaken_for_headings() -> None:
    text = "1. We collect your email address.\nThe Service Is Great.\nWrapped Title Case Line\nend"
    assert [s.heading for s in split_into_sections(text)] == ["General"]


def test_load_documents_reads_every_supported_format(tmp_path: Path) -> None:
    (tmp_path / "acme_terms.txt").write_text("Plain text terms.")
    (tmp_path / "beta_terms.html").write_text("<h1>Fees</h1><p>Five dollars.</p>")
    (tmp_path / "gamma_privacy.pdf").write_bytes(MINIMAL_PDF)
    (tmp_path / "notes.md").write_text("Ignored, because .md is not a supported type.")

    documents = {doc.company: doc for doc in load_documents(tmp_path)}

    assert set(documents) == {"acme", "beta", "gamma"}
    assert documents["beta"].sections[0].heading == "Fees"
    assert "keep your data safe" in documents["gamma"].sections[0].text


def test_same_document_in_two_formats_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "acme_terms.txt").write_text("One.")
    (tmp_path / "acme_terms.html").write_text("<p>Two.</p>")
    with pytest.raises(ValueError, match="Duplicate document"):
        load_documents(tmp_path)


def test_samples_are_only_added_when_raw_dir_is_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_dir = tmp_path / "raw"
    assert len(ensure_sample_data(raw_dir)) == 2
    assert "fictional sample" in capsys.readouterr().out
    assert ensure_sample_data(raw_dir) == []
