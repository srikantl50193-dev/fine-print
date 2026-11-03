"""The HTTP API, which `make api` serves on localhost."""

from typing import Annotated

import httpx
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field

from fine_print.config import Strategy, get_settings
from fine_print.logging_config import configure_logging
from fine_print.models import Answer
from fine_print.rag import OllamaError, answer_question
from fine_print.store import IndexNotBuiltError, list_companies

HEALTH_TIMEOUT_SECONDS = 2.0

settings = get_settings()
configure_logging(settings.log_level, json_output=settings.log_json)

app = FastAPI(
    title="fine-print",
    summary="Citation-first answers from Terms of Service and privacy policies.",
)


class AskRequest(BaseModel):
    """A question, with optional limits on where to look for the answer."""

    question: str = Field(min_length=1, max_length=1000)
    company: str | None = None
    doc_type: str | None = None
    strategy: Strategy | None = None
    top_k: int | None = Field(default=None, gt=0, le=50)


class Health(BaseModel):
    """Whether the services that answering depends on are ready."""

    status: str
    index_ready: bool
    ollama_ready: bool


def _ollama_ready() -> bool:
    """Check that Ollama responds, without loading a model."""
    url = f"{get_settings().ollama_url}/api/tags"
    try:
        return httpx.get(url, timeout=HEALTH_TIMEOUT_SECONDS).is_success
    except httpx.HTTPError:
        return False


# These handlers are plain functions, so FastAPI runs them in a thread pool and the slow
# model call doesn't block other requests.
@app.get("/health")
def health() -> Health:
    """Report whether the index and Ollama are ready."""
    try:
        index_ready = bool(list_companies(get_settings().default_strategy))
    except IndexNotBuiltError:
        index_ready = False
    ollama_ready = _ollama_ready()
    return Health(
        status="ok" if index_ready and ollama_ready else "degraded",
        index_ready=index_ready,
        ollama_ready=ollama_ready,
    )


@app.get("/companies")
def companies(
    strategy: Annotated[Strategy | None, Query()] = None,
) -> list[str]:
    """List the companies in an index."""
    try:
        return list_companies(strategy or get_settings().default_strategy)
    except IndexNotBuiltError as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error


@app.post("/ask")
def ask(request: AskRequest) -> Answer:
    """Answer a question from the documents, with citations."""
    try:
        return answer_question(
            request.question,
            request.strategy,
            request.top_k,
            request.company,
            request.doc_type,
        )
    except (IndexNotBuiltError, OllamaError) as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
