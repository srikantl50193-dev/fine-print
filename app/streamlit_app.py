"""Streamlit UI for fine-print. It talks to the API, so run both: make api, then make app."""

from collections.abc import Sequence

import httpx
import streamlit as st

from fine_print.config import get_settings
from fine_print.models import Answer, Citation, RetrievedChunk

ALL_COMPANIES = "All companies"
ALL_DOCUMENTS = "Terms and privacy"
DOC_TYPES = [ALL_DOCUMENTS, "terms", "privacy"]
API_DOWN = "Cannot reach the API at {url}. Start it in another terminal with `make api`."
# The API can wait this long on the model, so the UI has to wait longer.
EXTRA_WAIT_SECONDS = 30.0


class ApiError(RuntimeError):
    """Raised with a message the user can act on when the API cannot answer."""


def _request(method: str, path: str, **kwargs: object) -> httpx.Response:
    """Call the API, turning any failure into an ApiError."""
    settings = get_settings()
    try:
        response = httpx.request(
            method,
            f"{settings.api_url}{path}",
            timeout=settings.ollama_read_timeout + EXTRA_WAIT_SECONDS,
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
    except httpx.TransportError as error:
        raise ApiError(API_DOWN.format(url=settings.api_url)) from error
    if not response.is_success:
        try:
            detail = response.json()["detail"]
        except ValueError, KeyError, TypeError:
            detail = response.text[:200]
        raise ApiError(str(detail))
    return response


def fetch_companies() -> list[str]:
    """Return the companies in the index."""
    return _request("GET", "/companies").json()


def ask(question: str, company: str | None, doc_type: str | None) -> Answer:
    """Ask the API a question."""
    body = {"question": question, "company": company, "doc_type": doc_type}
    return Answer.model_validate(_request("POST", "/ask", json=body).json())


def show_passages(chunks: Sequence[Citation | RetrievedChunk]) -> None:
    """Show each passage in its own expander, with its similarity score."""
    for chunk in chunks:
        title = f"{chunk.label} ({chunk.doc_type}), similarity {chunk.similarity:.3f}"
        with st.expander(title):
            st.text(chunk.text)


def show_answer(answer: Answer) -> None:
    """Show the answer, its evidence, and how far to trust it."""
    st.subheader("Answer")
    st.write(answer.text)

    if answer.refusal_reason == "low_similarity":
        st.info("No passage was similar enough to the question, so the model wasn't asked.")
    elif answer.citations:
        st.subheader("Cited passages")
        show_passages(answer.citations)
    elif not answer.refused:
        st.warning(
            "The model gave no citation matching a retrieved passage, so treat this answer as "
            "unverified. Compare it with the retrieved passages below."
        )

    if answer.retrieved:
        st.subheader(f"All {len(answer.retrieved)} retrieved passages")
        show_passages(answer.retrieved)


def main() -> None:
    """Render the page."""
    settings = get_settings()
    st.set_page_config(page_title="fine-print", page_icon="📜")
    st.title("📜 fine-print")
    st.caption(
        "Ask about Terms of Service and privacy policies. Runs locally on "
        f"{settings.ollama_model}. AI answers can be wrong and aren't legal advice, so check "
        "the cited text."
    )

    try:
        companies = fetch_companies()
    except ApiError as error:
        st.error(str(error))
        st.stop()

    with st.form("ask"):
        question = st.text_input("Your question", placeholder="Can I cancel for free?")
        company = st.selectbox("Company", [ALL_COMPANIES, *companies])
        doc_type = st.selectbox("Document", DOC_TYPES)
        submitted = st.form_submit_button("Ask")

    if not (submitted and question.strip()):
        return

    try:
        with st.spinner("Searching the documents and asking the model..."):
            answer = ask(
                question.strip(),
                None if company == ALL_COMPANIES else company,
                None if doc_type == ALL_DOCUMENTS else doc_type,
            )
    except ApiError as error:
        st.error(str(error))
        st.stop()

    show_answer(answer)


if __name__ == "__main__":
    main()
