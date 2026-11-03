"""Streamlit UI for fine-print. Run with: make app"""

import streamlit as st

from fine_print import config
from fine_print.rag import OllamaError, answer_question
from fine_print.store import IndexNotBuiltError, RetrievedChunk, list_companies

ALL_COMPANIES = "All companies"


def show_chunks(chunks: list[RetrievedChunk]) -> None:
    """Show each chunk in its own expander, with its similarity score."""
    for chunk in chunks:
        with st.expander(f"{chunk.label}  (similarity {chunk.similarity:.3f})"):
            st.text(chunk.text)


def main() -> None:
    st.set_page_config(page_title="fine-print", page_icon="📜")
    st.title("📜 fine-print")
    st.caption(
        f"Ask about Terms of Service and privacy policies. Runs locally on {config.OLLAMA_MODEL}. "
        "AI answers can be wrong and are not legal advice, so check the cited text."
    )

    try:
        companies = list_companies(config.DEFAULT_STRATEGY)
    except IndexNotBuiltError:
        st.error("No index found. Run `make index` in a terminal, then reload this page.")
        st.stop()

    with st.form("ask"):
        question = st.text_input("Your question", placeholder="Can I cancel for free?")
        choice = st.selectbox("Company filter (optional)", [ALL_COMPANIES, *companies])
        submitted = st.form_submit_button("Ask")

    if not (submitted and question.strip()):
        return

    company = None if choice == ALL_COMPANIES else choice
    try:
        with st.spinner("Searching the documents and asking the model..."):
            result = answer_question(question.strip(), company=company)
    except OllamaError as error:
        st.error(str(error))
        st.stop()

    st.subheader("Answer")
    st.write(result.answer)

    is_not_found = result.answer.lower().startswith(config.NOT_FOUND_MESSAGE.lower())
    if result.citations:
        st.subheader("Cited chunks")
        show_chunks(result.citations)
    elif not is_not_found:
        st.warning(
            "The model gave no citation matching a retrieved chunk, so treat this answer as "
            "unverified. Compare it with the retrieved chunks below."
        )

    with st.expander(f"All {len(result.retrieved)} retrieved chunks"):
        for chunk in result.retrieved:
            st.markdown(f"**{chunk.label}** (similarity {chunk.similarity:.3f})")
            st.text(chunk.text)


if __name__ == "__main__":
    main()
