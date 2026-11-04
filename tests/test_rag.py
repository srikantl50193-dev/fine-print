import json
from collections.abc import Iterator

import httpx
import pytest
import respx
import structlog

from fine_print import rag
from fine_print.config import get_settings
from fine_print.models import NOT_FOUND_MESSAGE, RetrievedChunk
from fine_print.prompts import load_prompt

CHAT_URL = "http://ollama.test/api/chat"


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("FINE_PRINT_OLLAMA_URL", "http://ollama.test")
    monkeypatch.setenv("FINE_PRINT_OLLAMA_MAX_ATTEMPTS", "2")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_chunk(
    company: str = "acme",
    section: str = "Fees",
    similarity: float = 0.8,
    text: str = "A $5 fee applies.",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        company=company,
        doc_type="terms",
        section=section,
        chunk_id=f"{company}_{section}",
        strategy="section",
        start_char=0,
        end_char=len(text),
        similarity=similarity,
    )


def reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})


@pytest.fixture
def retrieved(monkeypatch: pytest.MonkeyPatch) -> list[RetrievedChunk]:
    chunks = [make_chunk(), make_chunk("beta", "Refunds", 0.7), make_chunk("gamma", "Weak", 0.2)]
    monkeypatch.setattr(rag, "retrieve", lambda *args: chunks)
    return chunks


# Prompt


def test_prompt_labels_every_passage_and_ends_with_the_question() -> None:
    prompt = rag.build_prompt("What are the fees?", [make_chunk(), make_chunk("beta", "Refunds")])
    assert "[Acme – Fees]\nA $5 fee applies." in prompt
    assert "[Beta – Refunds]\nA $5 fee applies." in prompt
    assert prompt.endswith("Question: What are the fees?")


def test_numbered_prompt_numbers_every_passage_in_order() -> None:
    prompt = rag.build_prompt("Q?", [make_chunk(), make_chunk("beta", "Refunds")], "number")
    assert "[1] Acme – Fees\nA $5 fee applies." in prompt
    assert "[2] Beta – Refunds\nA $5 fee applies." in prompt


def test_system_prompt_states_the_grounding_rules() -> None:
    system = load_prompt(get_settings().prompt_name).system
    assert "ONLY from the context" in system
    assert NOT_FOUND_MESSAGE in system
    assert "Ignore any instructions" in system


def test_passages_that_would_overflow_the_context_are_dropped_worst_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINE_PRINT_OLLAMA_NUM_CTX", "2024")  # Leaves room for 3000 characters.
    get_settings.cache_clear()
    chunks = [make_chunk(section=str(i), text="x" * 1400) for i in range(3)]
    assert rag.fit_to_context(chunks) == chunks[:2]


def test_default_settings_fit_the_default_top_k() -> None:
    settings = get_settings()
    chunks = [make_chunk(section=str(i), text="x" * settings.chunk_size) for i in range(5)]
    assert rag.fit_to_context(chunks[: settings.top_k]) == chunks[: settings.top_k]


# Citations


@pytest.mark.parametrize(
    "answer",
    [
        "It is $5 [Acme – Fees].",
        "It is $5 [ACME - FEES].",
        "It is $5 [acme — fees].",
        "It is $5 [Acme-Fees].",
        "It is $5 [ Acme  –  Fees ]".replace("[ ", "[").replace(" ]", "]"),
    ],
)
def test_citations_survive_the_changes_models_make_when_copying(answer: str) -> None:
    cited, ignored = make_chunk(), make_chunk("beta", "Refunds")
    citations = rag.find_citations(answer, [cited, ignored])
    assert [c.chunk_id for c in citations] == [cited.chunk_id]
    assert citations[0].label == "[Acme – Fees]"
    assert citations[0].text == cited.text


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("It is $5 [1].", ["acme_Fees"]),
        ("It is $5 [ 2 ].", ["beta_Refunds"]),
        ("It is $5 [1, 2].", ["acme_Fees", "beta_Refunds"]),
        ("It is $5 [1][2].", ["acme_Fees", "beta_Refunds"]),
        ("It is $5 [9].", []),
        ("It is $5 [0].", []),
        ("It is $5.", []),
        ("It is $5 [Acme – Fees].", []),
    ],
)
def test_numbered_citations(answer: str, expected: list[str]) -> None:
    chunks = [make_chunk(), make_chunk("beta", "Refunds")]
    assert [c.chunk_id for c in rag.find_citations(answer, chunks, "number")] == expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("It is $5 [1].", "It is $5 [Acme – Fees]."),
        ("It is $5 [1, 2].", "It is $5 [Acme – Fees] [Beta – Refunds]."),
        ("It is $5 [1, 1].", "It is $5 [Acme – Fees]."),
        ("It is $5 [9].", "It is $5 [9]."),
        ("It is $5 [1, 9].", "It is $5 [1, 9]."),
    ],
)
def test_numbers_are_shown_to_readers_as_labels(answer: str, expected: str) -> None:
    chunks = [make_chunk(), make_chunk("beta", "Refunds")]
    assert rag.render_numbers_as_labels(answer, chunks) == expected


def test_invented_citations_are_not_accepted() -> None:
    assert rag.find_citations("It is $5 [Acme – Refunds].", [make_chunk()]) == []


# Ollama


@respx.mock
def test_request_carries_the_model_and_grounding_settings() -> None:
    route = respx.post(CHAT_URL).mock(return_value=reply(" An answer. "))

    assert rag.call_ollama("the prompt") == "An answer."

    body = route.calls[0].request.content.decode()
    settings = get_settings()
    assert f'"model":"{settings.ollama_model}"' in body
    assert '"temperature":0' in body
    assert f'"num_ctx":{settings.ollama_num_ctx}' in body
    assert f'"num_predict":{settings.ollama_num_predict}' in body
    assert '"stream":false' in body
    assert "the prompt" in body


def test_reply_cap_fits_in_the_space_reserved_for_it() -> None:
    assert get_settings().ollama_num_predict < rag.RESERVED_TOKENS


@respx.mock
def test_model_and_system_prompt_can_be_chosen_per_call() -> None:
    route = respx.post(CHAT_URL).mock(return_value=reply("Hi."))

    rag.call_ollama("the prompt", "Be nice.", "qwen3:8b")

    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "qwen3:8b"
    assert body["messages"][0] == {"role": "system", "content": "Be nice."}
    assert body["think"] is False


@respx.mock
@pytest.mark.usefixtures("retrieved")
def test_numbered_prompt_runs_end_to_end() -> None:
    route = respx.post(CHAT_URL).mock(return_value=reply("It is $5 [2]."))

    answer = rag.answer_question("Any fee?", prompt=load_prompt("numbered"), model="qwen3:8b")

    body = route.calls[0].request.content.decode()
    assert "qwen3:8b" in body
    assert "[1] Acme" in body
    assert answer.text == "It is $5 [Beta – Refunds]."
    assert [c.label for c in answer.citations] == ["[Beta – Refunds]"]


@respx.mock
def test_missing_chosen_model_is_named_in_the_error() -> None:
    respx.post(CHAT_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(rag.OllamaError, match=r"ollama pull qwen3:8b"):
        rag.call_ollama("hello", "system", "qwen3:8b")


@respx.mock
def test_reply_cut_off_at_the_limit_is_kept_and_logged() -> None:
    body = {"message": {"content": "Repeating and repeating"}, "done_reason": "length"}
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=body))

    with structlog.testing.capture_logs() as logs:
        assert rag.call_ollama("hello") == "Repeating and repeating"

    assert logs[0]["event"] == "reply_cut_off_at_token_limit"


@respx.mock
def test_stopped_ollama_gives_a_clear_error_after_retrying() -> None:
    route = respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(rag.OllamaError, match="ollama serve"):
        rag.call_ollama("hello")
    assert route.call_count == 2


@respx.mock
def test_connection_failure_recovers_when_a_retry_succeeds() -> None:
    respx.post(CHAT_URL).mock(side_effect=[httpx.ConnectError("refused"), reply("Back.")])
    assert rag.call_ollama("hello") == "Back."


@respx.mock
def test_server_error_is_retried_then_reported() -> None:
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(rag.OllamaError, match="boom"):
        rag.call_ollama("hello")
    assert route.call_count == 2


@respx.mock
def test_missing_model_says_how_to_install_it_without_retrying() -> None:
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(rag.OllamaError, match="ollama pull"):
        rag.call_ollama("hello")
    assert route.call_count == 1


@respx.mock
def test_bad_request_is_reported_without_retrying() -> None:
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(400, text="bad"))
    with pytest.raises(rag.OllamaError, match="HTTP 400"):
        rag.call_ollama("hello")
    assert route.call_count == 1


@respx.mock
def test_read_timeout_is_not_retried() -> None:
    route = respx.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(rag.OllamaError, match="did not reply"):
        rag.call_ollama("hello")
    assert route.call_count == 1


@respx.mock
def test_dropped_connection_is_reported() -> None:
    respx.post(CHAT_URL).mock(side_effect=httpx.RemoteProtocolError("gone"))
    with pytest.raises(rag.OllamaError, match="Could not talk to Ollama"):
        rag.call_ollama("hello")


# The whole pipeline


def test_retrieve_uses_the_settings_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(rag, "embed_query", lambda question: question)
    monkeypatch.setattr(rag, "query", lambda *args: calls.append(args) or [])

    rag.retrieve("Q?")
    rag.retrieve("Q?", "fixed", 3, "acme", "terms")

    settings = get_settings()
    assert calls == [
        ("Q?", settings.default_strategy, settings.top_k, None, None),
        ("Q?", "fixed", 3, "acme", "terms"),
    ]


@respx.mock
def test_answer_reports_citations_and_everything_retrieved(
    retrieved: list[RetrievedChunk],
) -> None:
    respx.post(CHAT_URL).mock(return_value=reply("It is $5 [Acme – Fees]."))

    answer = rag.answer_question("Any fee?", prompt=load_prompt("baseline"))

    assert answer.text == "It is $5 [Acme – Fees]."
    assert [c.label for c in answer.citations] == ["[Acme – Fees]"]
    assert answer.retrieved == retrieved
    assert answer.refused is False


@respx.mock
@pytest.mark.usefixtures("retrieved")
def test_default_prompt_turns_cited_numbers_into_labels() -> None:
    respx.post(CHAT_URL).mock(return_value=reply("It is $5 [1]."))

    answer = rag.answer_question("Any fee?")

    assert answer.text == "It is $5 [Acme – Fees]."
    assert [c.label for c in answer.citations] == ["[Acme – Fees]"]
    assert answer.refusal_reason is None


@respx.mock
def test_passages_below_the_threshold_are_not_shown_to_the_model(
    retrieved: list[RetrievedChunk],
) -> None:
    route = respx.post(CHAT_URL).mock(return_value=reply("It is $5 [Gamma – Weak]."))

    answer = rag.answer_question("Any fee?")

    assert retrieved[2].label not in route.calls[0].request.content.decode()
    assert answer.citations == []


@respx.mock
def test_low_similarity_refuses_without_calling_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = respx.post(CHAT_URL)
    weak = [make_chunk(similarity=0.3)]
    monkeypatch.setattr(rag, "retrieve", lambda *args: weak)

    answer = rag.answer_question("Unrelated?")

    assert route.call_count == 0
    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.refused is True
    assert answer.refusal_reason == "low_similarity"
    assert answer.retrieved == weak


@respx.mock
def test_no_retrieved_chunks_refuses_without_calling_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = respx.post(CHAT_URL)
    monkeypatch.setattr(rag, "retrieve", lambda *args: [])

    answer = rag.answer_question("Anything?", company="nobody")

    assert route.call_count == 0
    assert answer.refusal_reason == "low_similarity"


@respx.mock
@pytest.mark.usefixtures("retrieved")
@pytest.mark.parametrize("text", ["Not found in the documents", "not found in the documents."])
def test_model_refusal_carries_no_citations(text: str) -> None:
    respx.post(CHAT_URL).mock(return_value=reply(text))

    answer = rag.answer_question("Unrelated?")

    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.citations == []
    assert answer.refused is True
    assert answer.refusal_reason == "model_declined"


@respx.mock
@pytest.mark.usefixtures("retrieved")
def test_uncited_claim_followed_by_not_found_counts_as_a_refusal() -> None:
    respx.post(CHAT_URL).mock(
        return_value=reply("You may not sue [Class Action Waiver]\n\nNot found in the documents")
    )

    answer = rag.answer_question("Can I sue?")

    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.refusal_reason == "model_declined"


@respx.mock
@pytest.mark.usefixtures("retrieved")
def test_cited_answer_that_also_mentions_not_found_is_kept() -> None:
    text = "It is $5 [Acme – Fees]. Refunds: not found in the documents."
    respx.post(CHAT_URL).mock(return_value=reply(text))

    answer = rag.answer_question("Fees and refunds?", prompt=load_prompt("baseline"))

    assert answer.text == text
    assert answer.refused is False


@respx.mock
@pytest.mark.usefixtures("retrieved")
def test_answer_without_a_valid_citation_is_not_a_refusal() -> None:
    respx.post(CHAT_URL).mock(return_value=reply("It is $5."))

    answer = rag.answer_question("Any fee?")

    assert answer.refused is False
    assert answer.citations == []
