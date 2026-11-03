import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from fine_print import api
from fine_print.config import get_settings
from fine_print.models import NOT_FOUND_MESSAGE, Answer, RetrievedChunk
from fine_print.rag import OllamaError, find_citations
from fine_print.store import IndexNotBuiltError

client = TestClient(api.app)
TAGS_URL = f"{get_settings().ollama_url}/api/tags"
CHUNK = RetrievedChunk(
    text="A $5 fee applies.",
    company="acme",
    doc_type="terms",
    section="Fees",
    chunk_id="id",
    strategy="section",
    start_char=0,
    end_char=17,
    similarity=0.812,
)


def raise_error(error: Exception):
    def fail(*args: object) -> None:
        raise error

    return fail


def test_ask_returns_the_answer_with_citation_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_answer(*args: object) -> Answer:
        calls.append(args)
        return Answer(
            question="Any fee?",
            text="It is $5 [Acme – Fees].",
            citations=find_citations(CHUNK.label, [CHUNK]),
            retrieved=[CHUNK],
        )

    monkeypatch.setattr(api, "answer_question", fake_answer)

    response = client.post(
        "/ask",
        json={"question": "Any fee?", "company": "acme", "doc_type": "terms", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "It is $5 [Acme – Fees]."
    assert body["citations"][0]["label"] == "[Acme – Fees]"
    assert body["retrieved"][0]["label"] == "[Acme – Fees]"
    assert body["retrieved"][0]["similarity"] == 0.812
    assert body["refused"] is False
    assert calls == [("Any fee?", None, 3, "acme", "terms")]
    # A client can rebuild the domain model from the response.
    assert Answer.model_validate(body).citations[0].chunk_id == "id"


def test_refusal_is_a_normal_response(monkeypatch: pytest.MonkeyPatch) -> None:
    refusal = Answer(
        question="Off topic?",
        text=NOT_FOUND_MESSAGE,
        refused=True,
        refusal_reason="low_similarity",
    )
    monkeypatch.setattr(api, "answer_question", lambda *args: refusal)

    response = client.post("/ask", json={"question": "Off topic?"})

    assert response.status_code == 200
    assert response.json()["refusal_reason"] == "low_similarity"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "x" * 1001},
        {"question": "Q?", "strategy": "magic"},
        {"question": "Q?", "top_k": 0},
        {"question": "Q?", "top_k": 51},
    ],
)
def test_invalid_requests_are_rejected(payload: dict[str, object]) -> None:
    assert client.post("/ask", json=payload).status_code == 422


@pytest.mark.parametrize(
    "error",
    [OllamaError("Start it with `ollama serve`."), IndexNotBuiltError("Run: make index")],
)
def test_unavailable_dependencies_say_what_to_do(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(api, "answer_question", raise_error(error))

    response = client.post("/ask", json={"question": "Any fee?"})

    assert response.status_code == 503
    assert response.json()["detail"] == str(error)


def test_companies(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = []

    def fake_list(strategy: str) -> list[str]:
        requested.append(strategy)
        return ["acme", "beta"]

    monkeypatch.setattr(api, "list_companies", fake_list)

    assert client.get("/companies").json() == ["acme", "beta"]
    assert client.get("/companies", params={"strategy": "fixed"}).json() == ["acme", "beta"]
    assert requested == [get_settings().default_strategy, "fixed"]


def test_companies_without_an_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "list_companies", raise_error(IndexNotBuiltError("Run: make index")))

    response = client.get("/companies")

    assert response.status_code == 503
    assert "make index" in response.json()["detail"]


@respx.mock
@pytest.mark.parametrize(
    ("index_ok", "ollama", "expected"),
    [
        (True, httpx.Response(200, json={"models": []}), ("ok", True, True)),
        (True, httpx.ConnectError("refused"), ("degraded", True, False)),
        (True, httpx.Response(500), ("degraded", True, False)),
        (False, httpx.Response(200, json={"models": []}), ("degraded", False, True)),
    ],
)
def test_health_reports_each_dependency(
    monkeypatch: pytest.MonkeyPatch,
    index_ok: bool,
    ollama: httpx.Response | Exception,
    expected: tuple[str, bool, bool],
) -> None:
    respx.get(TAGS_URL).mock(side_effect=[ollama])
    monkeypatch.setattr(
        api,
        "list_companies",
        (lambda strategy: ["acme"]) if index_ok else raise_error(IndexNotBuiltError("none")),
    )

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["index_ready"], body["ollama_ready"]) == expected
