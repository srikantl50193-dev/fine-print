import pytest
from pydantic import ValidationError

from fine_print.models import NOT_FOUND_MESSAGE, Answer, Chunk, Citation, RetrievedChunk


def chunk_fields(**overrides: object) -> dict[str, object]:
    return {
        "text": "We keep your data for 30 days.",
        "company": "spotify",
        "doc_type": "privacy",
        "section": "Data Retention",
        "chunk_id": "spotify_privacy_section_0007",
        "strategy": "section",
        "start_char": 100,
        "end_char": 130,
    } | overrides


@pytest.mark.parametrize(
    ("company", "section", "label"),
    [
        ("spotify", "Data Retention", "[Spotify – Data Retention]"),
        ("netflix", "1. The Netflix Service", "[Netflix – 1. The Netflix Service]"),
    ],
)
def test_citation_label_format(company: str, section: str, label: str) -> None:
    fields = chunk_fields(company=company, section=section)
    chunk = RetrievedChunk.model_validate({**fields, "similarity": 0.8})
    assert chunk.label == label


def test_citation_label_is_included_when_serialized() -> None:
    chunk = RetrievedChunk.model_validate({**chunk_fields(), "similarity": 0.8})
    assert chunk.model_dump()["label"] == "[Spotify – Data Retention]"


def test_embedding_text_leads_with_the_section_title() -> None:
    chunk = Chunk.model_validate(chunk_fields())
    assert chunk.embedding_text() == "Data Retention\nWe keep your data for 30 days."


def test_metadata_is_everything_except_the_text() -> None:
    metadata = Chunk.model_validate(chunk_fields()).metadata()
    assert "text" not in metadata
    assert metadata == {key: value for key, value in chunk_fields().items() if key != "text"}


def test_models_are_immutable() -> None:
    chunk = Chunk.model_validate(chunk_fields())
    with pytest.raises(ValidationError, match="frozen"):
        chunk.text = "changed"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": ""},
        {"start_char": -1},
        {"end_char": 0},
        {"strategy": "magic"},
    ],
)
def test_invalid_chunks_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Chunk.model_validate(chunk_fields(**overrides))


def test_answer_defaults_to_a_non_refusal_without_citations() -> None:
    answer = Answer(question="Q?", text="A.")
    assert answer.citations == []
    assert answer.retrieved == []
    assert answer.refused is False
    assert answer.refusal_reason is None


def test_refused_answer_records_why() -> None:
    answer = Answer(
        question="Q?",
        text=NOT_FOUND_MESSAGE,
        refused=True,
        refusal_reason="low_similarity",
    )
    assert answer.refusal_reason == "low_similarity"
    with pytest.raises(ValidationError):
        Answer.model_validate({"question": "Q?", "text": "A.", "refusal_reason": "other"})


def test_citation_keeps_the_retrieved_chunks_label() -> None:
    chunk = RetrievedChunk.model_validate({**chunk_fields(), "similarity": 0.8})
    citation = Citation(
        label=chunk.label,
        company=chunk.company,
        doc_type=chunk.doc_type,
        section=chunk.section,
        chunk_id=chunk.chunk_id,
        similarity=chunk.similarity,
        text=chunk.text,
    )
    assert citation.label == "[Spotify – Data Retention]"
