import pytest

from fine_print import config, rag
from fine_print.store import RetrievedChunk


def make_chunk(company: str = "acme", section: str = "Fees") -> RetrievedChunk:
    return RetrievedChunk("Some text.", company, "terms", section, f"{company}_{section}", 0.9)


def test_prompt_labels_every_passage_and_includes_the_question() -> None:
    prompt = rag.build_prompt("What are the fees?", [make_chunk(), make_chunk("beta", "Refunds")])
    assert "[acme - Fees]\nSome text." in prompt
    assert "[beta - Refunds]\nSome text." in prompt
    assert prompt.endswith("Question: What are the fees?")


def test_system_prompt_states_the_grounding_rules() -> None:
    assert "ONLY from the context" in rag.SYSTEM_PROMPT
    assert "[company - section]" in rag.SYSTEM_PROMPT
    assert "Not found in the documents" in rag.SYSTEM_PROMPT


def test_find_citations_returns_only_the_chunks_named_in_the_answer() -> None:
    cited, ignored = make_chunk(), make_chunk("beta", "Refunds")
    assert rag.find_citations("It is $5 [ACME - Fees].", [cited, ignored]) == [cited]


def test_stopped_ollama_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Port 9 (discard) has nothing listening, which is what a stopped Ollama looks like.
    monkeypatch.setattr(config, "OLLAMA_URL", "http://127.0.0.1:9")
    with pytest.raises(rag.OllamaError, match="ollama serve"):
        rag.call_ollama("hello")


def test_not_found_answers_carry_no_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [make_chunk()]
    monkeypatch.setattr(rag, "retrieve", lambda *args: chunks)
    monkeypatch.setattr(rag, "call_ollama", lambda prompt: "Not found in the documents.")
    result = rag.answer_question("Unrelated question?")
    assert result.citations == []
    assert result.retrieved == chunks


def test_no_retrieved_chunks_skips_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(prompt: str) -> str:
        raise AssertionError("the LLM should not be called")

    monkeypatch.setattr(rag, "retrieve", lambda *args: [])
    monkeypatch.setattr(rag, "call_ollama", fail)
    assert rag.answer_question("Anything?").answer == config.NOT_FOUND_MESSAGE
