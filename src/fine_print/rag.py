"""The RAG pipeline: retrieve chunks, build a prompt, ask Ollama, return citations."""

import re

import httpx
import structlog
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from fine_print.config import Strategy, get_settings
from fine_print.embed import embed_query
from fine_print.models import NOT_FOUND_MESSAGE, Answer, Citation, RetrievedChunk
from fine_print.prompts import CitationStyle, Prompt, load_prompt
from fine_print.store import query

log = structlog.get_logger(__name__)

# Ollama silently drops prompt text beyond num_ctx, and the reply shares that space.
RESERVED_TOKENS = 1024
# Deliberately low for English, so the estimate errs towards dropping a passage.
CHARS_PER_TOKEN = 3
HTTP_NOT_FOUND = 404
HTTP_SERVER_ERROR = 500

NUMBERED_CITATION = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
DASHES = re.compile(r"\s*[-\u2010-\u2015]\s*")


class OllamaError(RuntimeError):
    """Raised with a message the user can act on when Ollama cannot answer."""


class _RetryableOllamaError(RuntimeError):
    """Raised for a server error that may pass, so that it is retried."""


def retrieve(
    question: str,
    strategy: Strategy | None = None,
    top_k: int | None = None,
    company: str | None = None,
    doc_type: str | None = None,
) -> list[RetrievedChunk]:
    """Find the chunks most similar to a question.

    Raises:
        IndexNotBuiltError: If the strategy's index does not exist.
    """
    settings = get_settings()
    return query(
        embed_query(question),
        strategy or settings.default_strategy,
        top_k or settings.top_k,
        company,
        doc_type,
    )


def fit_to_context(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the best-ranked passages that fit in the model's context window."""
    budget = (get_settings().ollama_num_ctx - RESERVED_TOKENS) * CHARS_PER_TOKEN
    kept: list[RetrievedChunk] = []
    for chunk in chunks:
        budget -= len(chunk.label) + len(chunk.text)
        if budget < 0:
            log.warning("passages_dropped_to_fit_context", dropped=len(chunks) - len(kept))
            break
        kept.append(chunk)
    return kept


def select_passages(retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Choose what the model sees: the passages over the threshold that fit in its context."""
    threshold = get_settings().min_similarity
    return fit_to_context([c for c in retrieved if c.similarity >= threshold])


def build_prompt(
    question: str, chunks: list[RetrievedChunk], style: CitationStyle = "label"
) -> str:
    """Lay out the labelled or numbered passages, followed by the question."""
    if style == "number":
        headers = [
            f"[{position}] {chunk.company.title()} – {chunk.section}"
            for position, chunk in enumerate(chunks, start=1)
        ]
    else:
        headers = [chunk.label for chunk in chunks]
    passages = "\n\n".join(
        f"{header}\n{chunk.text}" for header, chunk in zip(headers, chunks, strict=True)
    )
    return f"Context:\n\n{passages}\n\nQuestion: {question}"


def _post_chat(prompt: str, system: str, model: str) -> str:
    """Make one request to Ollama's chat endpoint and return the reply text."""
    settings = get_settings()
    response = httpx.post(
        f"{settings.ollama_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            # Thinking output would use up the reply cap before the answer starts.
            "think": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            # Temperature 0 keeps answers repeatable and close to the context.
            "options": {
                "temperature": 0,
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": settings.ollama_num_predict,
            },
        },
        timeout=httpx.Timeout(
            settings.ollama_read_timeout, connect=settings.ollama_connect_timeout
        ),
    )
    if response.status_code == HTTP_NOT_FOUND:
        raise OllamaError(f"Model '{model}' is not installed. Run: ollama pull {model}")
    if response.status_code >= HTTP_SERVER_ERROR:
        raise _RetryableOllamaError(f"HTTP {response.status_code}: {response.text[:200]}")
    if not response.is_success:
        raise OllamaError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")
    body = response.json()
    if body.get("done_reason") == "length":
        log.warning("reply_cut_off_at_token_limit", limit=settings.ollama_num_predict)
    return body["message"]["content"].strip()


def call_ollama(prompt: str, system: str | None = None, model: str | None = None) -> str:
    """Send a prompt to Ollama, retrying failures that may pass.

    The system prompt and model default to the ones in the settings. A read timeout is not
    retried: the model has already had its full time, so another try would only make the
    caller wait longer.

    Raises:
        OllamaError: If Ollama cannot answer, with a message saying what to do.
    """
    settings = get_settings()
    system = system or load_prompt(settings.prompt_name).system
    model = model or settings.ollama_model
    retrying = Retrying(
        stop=stop_after_attempt(settings.ollama_max_attempts),
        wait=wait_exponential(multiplier=0.5, max=4),
        retry=retry_if_exception_type(
            (httpx.ConnectError, httpx.ConnectTimeout, _RetryableOllamaError)
        ),
        reraise=True,
    )
    try:
        return retrying(_post_chat, prompt, system, model)
    except (httpx.ConnectError, httpx.ConnectTimeout) as error:
        raise OllamaError(
            f"Cannot reach Ollama at {settings.ollama_url}. "
            "Start it with `ollama serve` (or open the Ollama app) and try again."
        ) from error
    except httpx.TimeoutException as error:
        raise OllamaError(
            f"Ollama did not reply within {settings.ollama_read_timeout:.0f} seconds. "
            "Try a smaller model or a lower top_k."
        ) from error
    except _RetryableOllamaError as error:
        raise OllamaError(f"Ollama failed: {error}") from error
    except httpx.HTTPError as error:
        raise OllamaError(f"Could not talk to Ollama: {error}") from error


def _normalize_citations(text: str) -> str:
    """Ignore the case, spacing and dash style that models change when copying a label."""
    return " ".join(DASHES.sub(" - ", text).lower().split())


def _as_citation(chunk: RetrievedChunk) -> Citation:
    """Keep the parts of a retrieved chunk that a citation shows."""
    return Citation(
        label=chunk.label,
        company=chunk.company,
        doc_type=chunk.doc_type,
        section=chunk.section,
        chunk_id=chunk.chunk_id,
        similarity=chunk.similarity,
        text=chunk.text,
    )


def _cited_numbers(answer: str, count: int) -> set[int]:
    """Return the passage numbers an answer cites, ignoring any that don't exist."""
    numbers = {
        int(number)
        for match in NUMBERED_CITATION.finditer(answer)
        for number in match.group(1).split(",")
    }
    return {number for number in numbers if 1 <= number <= count}


def find_citations(
    answer: str, chunks: list[RetrievedChunk], style: CitationStyle = "label"
) -> list[Citation]:
    """Return the chunks an answer cites.

    Passages from one section share a label, so citing a label counts for all of them. A
    number names one passage.
    """
    if style == "number":
        numbers = _cited_numbers(answer, len(chunks))
        return [_as_citation(c) for i, c in enumerate(chunks, start=1) if i in numbers]
    normalized = _normalize_citations(answer)
    return [_as_citation(c) for c in chunks if _normalize_citations(c.label) in normalized]


def render_numbers_as_labels(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Rewrite citations like [2] as labels, because a number means nothing to the reader."""

    def replace(match: re.Match[str]) -> str:
        numbers = [int(number) for number in match.group(1).split(",")]
        if not all(1 <= number <= len(chunks) for number in numbers):
            return match.group(0)
        labels = dict.fromkeys(chunks[number - 1].label for number in numbers)
        return " ".join(labels)

    return NUMBERED_CITATION.sub(replace, answer)


def answer_question(
    question: str,
    strategy: Strategy | None = None,
    top_k: int | None = None,
    company: str | None = None,
    doc_type: str | None = None,
    model: str | None = None,
    prompt: Prompt | None = None,
) -> Answer:
    """Run the full pipeline for one question.

    The model is only called when a passage clears `min_similarity`, and it only sees the
    passages that do. A weak match would give it material to invent an answer from.

    Args:
        question: The question to answer.
        strategy: The index to search. Defaults to the one in the settings.
        top_k: How many passages to retrieve. Defaults to the one in the settings.
        company: Only search this company.
        doc_type: Only search this document type.
        model: The Ollama model. Defaults to the one in the settings.
        prompt: The prompt variant. Defaults to the one in the settings.

    Raises:
        IndexNotBuiltError: If the strategy's index does not exist.
        OllamaError: If Ollama cannot answer.
    """
    settings = get_settings()
    prompt = prompt or load_prompt(settings.prompt_name)
    retrieved = retrieve(question, strategy, top_k, company, doc_type)
    passages = select_passages(retrieved)
    if not passages:
        return Answer(
            question=question,
            text=NOT_FOUND_MESSAGE,
            retrieved=retrieved,
            refused=True,
            refusal_reason="low_similarity",
        )

    style = prompt.citation_style
    reply = call_ollama(build_prompt(question, passages, style), prompt.system, model)
    citations = find_citations(reply, passages, style)
    lowered = reply.lower()
    not_found = NOT_FOUND_MESSAGE.lower()
    # Small models sometimes write a claim and then say that the answer wasn't found. With no
    # valid citation to back the claim, the refusal is the part to believe.
    if lowered.startswith(not_found) or (not citations and not_found in lowered):
        return Answer(
            question=question,
            text=NOT_FOUND_MESSAGE,
            retrieved=retrieved,
            refused=True,
            refusal_reason="model_declined",
        )
    text = render_numbers_as_labels(reply, passages) if style == "number" else reply
    return Answer(question=question, text=text, citations=citations, retrieved=retrieved)
