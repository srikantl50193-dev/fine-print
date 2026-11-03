"""Pydantic domain models shared by every layer of the pipeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from fine_print.config import Strategy

NOT_FOUND_MESSAGE = "Not found in the documents"

type RefusalReason = Literal["low_similarity", "model_declined"]


class FrozenModel(BaseModel):
    """Base class: domain objects are values, so they cannot change once created."""

    model_config = ConfigDict(frozen=True)


class Document(FrozenModel):
    """One policy file, with its text normalized to Markdown with `#` headings."""

    company: str
    doc_type: str
    source: str
    text: str


class Chunk(FrozenModel):
    """A passage of a document. `start_char:end_char` locates it in `Document.text`."""

    text: str = Field(min_length=1)
    company: str
    doc_type: str
    section: str
    chunk_id: str
    strategy: Strategy
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    def embedding_text(self) -> str:
        """Return the text to embed, which leads with the section title for topical context."""
        return f"{self.section}\n{self.text}"

    def metadata(self) -> dict[str, str | int]:
        """Return everything except the text, for storing beside the vector."""
        return self.model_dump(exclude={"text"})


class RetrievedChunk(Chunk):
    """A chunk returned by a search, with its cosine similarity to the query."""

    similarity: float

    @computed_field
    @property
    def label(self) -> str:
        """Return the citation form the prompt tells the model to use."""
        return f"[{self.company.title()} – {self.section}]"


class Citation(FrozenModel):
    """A retrieved chunk that the generated answer actually cites."""

    label: str
    company: str
    doc_type: str
    section: str
    chunk_id: str
    similarity: float
    text: str


class Answer(FrozenModel):
    """The result of a question: the text, its citations and how it was produced."""

    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: RefusalReason | None = None
