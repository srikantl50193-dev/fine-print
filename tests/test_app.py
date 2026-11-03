import json
from pathlib import Path

import httpx
import pytest
import respx
from streamlit.testing.v1 import AppTest

from fine_print.config import get_settings
from fine_print.models import NOT_FOUND_MESSAGE, Answer, RetrievedChunk
from fine_print.rag import find_citations

APP_PATH = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")
API_URL = get_settings().api_url
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


def answer_json(**fields: object) -> dict[str, object]:
    answer = Answer.model_validate(
        {"question": "Any fee?", "text": "It is $5.", "retrieved": [CHUNK], **fields}
    )
    return answer.model_dump(mode="json")


@pytest.fixture
def mock_api():
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{API_URL}/companies").mock(
            return_value=httpx.Response(200, json=["acme", "beta"])
        )
        yield router


@pytest.fixture
def app(mock_api: respx.MockRouter) -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


def ask(app: AppTest, question: str = "Any fee?") -> AppTest:
    app.text_input[0].set_value(question)
    return app.button[0].click().run()


def test_filters_list_the_companies_and_document_types(app: AppTest) -> None:
    assert not app.exception
    assert app.selectbox[0].options == ["All companies", "acme", "beta"]
    assert app.selectbox[1].options == ["Terms and privacy", "terms", "privacy"]


def test_page_warns_that_answers_can_be_wrong(app: AppTest) -> None:
    assert "aren't legal advice" in app.caption[0].value


def test_answer_shows_cited_passage_and_score(app: AppTest, mock_api: respx.MockRouter) -> None:
    citations = find_citations(CHUNK.label, [CHUNK])
    mock_api.post(f"{API_URL}/ask").mock(
        return_value=httpx.Response(
            200,
            json=answer_json(text="It is $5 [Acme – Fees].", citations=citations),
        )
    )

    ask(app)

    assert not app.exception
    assert "It is $5 [Acme – Fees]." in [m.value for m in app.markdown]
    assert [s.value for s in app.subheader] == [
        "Answer",
        "Cited passages",
        "All 1 retrieved passages",
    ]
    assert any("similarity 0.812" in e.label for e in app.expander)
    assert not app.warning


def test_filters_are_sent_to_the_api(app: AppTest, mock_api: respx.MockRouter) -> None:
    route = mock_api.post(f"{API_URL}/ask").mock(
        return_value=httpx.Response(200, json=answer_json())
    )

    app.selectbox[0].set_value("beta")
    app.selectbox[1].set_value("privacy")
    ask(app, "  Any fee?  ")

    assert json.loads(route.calls[0].request.content) == {
        "question": "Any fee?",
        "company": "beta",
        "doc_type": "privacy",
    }


def test_no_filter_sends_none(app: AppTest, mock_api: respx.MockRouter) -> None:
    route = mock_api.post(f"{API_URL}/ask").mock(
        return_value=httpx.Response(200, json=answer_json())
    )

    ask(app)

    body = json.loads(route.calls[0].request.content)
    assert body["company"] is None
    assert body["doc_type"] is None


def test_blank_question_does_not_call_the_api(app: AppTest, mock_api: respx.MockRouter) -> None:
    route = mock_api.post(f"{API_URL}/ask")
    ask(app, "   ")
    assert route.call_count == 0
    assert not app.subheader


def test_answer_without_a_valid_citation_shows_a_warning(
    app: AppTest, mock_api: respx.MockRouter
) -> None:
    mock_api.post(f"{API_URL}/ask").mock(return_value=httpx.Response(200, json=answer_json()))

    ask(app)

    assert "unverified" in app.warning[0].value


def test_low_similarity_refusal_says_the_model_was_not_asked(
    app: AppTest, mock_api: respx.MockRouter
) -> None:
    mock_api.post(f"{API_URL}/ask").mock(
        return_value=httpx.Response(
            200,
            json=answer_json(
                text=NOT_FOUND_MESSAGE, refused=True, refusal_reason="low_similarity", retrieved=[]
            ),
        )
    )

    ask(app)

    assert NOT_FOUND_MESSAGE in [m.value for m in app.markdown]
    assert "wasn't asked" in app.info[0].value
    assert not app.warning


def test_model_refusal_shows_what_was_retrieved_without_a_warning(
    app: AppTest, mock_api: respx.MockRouter
) -> None:
    mock_api.post(f"{API_URL}/ask").mock(
        return_value=httpx.Response(
            200,
            json=answer_json(text=NOT_FOUND_MESSAGE, refused=True, refusal_reason="model_declined"),
        )
    )

    ask(app)

    assert not app.warning
    assert not app.info
    assert "All 1 retrieved passages" in [s.value for s in app.subheader]


def test_api_error_shows_its_detail(app: AppTest, mock_api: respx.MockRouter) -> None:
    mock_api.post(f"{API_URL}/ask").mock(
        return_value=httpx.Response(503, json={"detail": "Start it with `ollama serve`."})
    )

    ask(app)

    assert "ollama serve" in app.error[0].value
    assert not app.exception


def test_non_json_api_error_is_still_shown(app: AppTest, mock_api: respx.MockRouter) -> None:
    mock_api.post(f"{API_URL}/ask").mock(return_value=httpx.Response(502, text="Bad gateway"))

    ask(app)

    assert "Bad gateway" in app.error[0].value


def test_stopped_api_says_how_to_start_it() -> None:
    with respx.mock() as router:
        router.get(f"{API_URL}/companies").mock(side_effect=httpx.ConnectError("refused"))
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert "make api" in app.error[0].value
    assert not app.exception
    assert not app.button


def test_missing_index_says_how_to_build_it() -> None:
    with respx.mock() as router:
        router.get(f"{API_URL}/companies").mock(
            return_value=httpx.Response(503, json={"detail": "No index found. Run: make index"})
        )
        app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert "make index" in app.error[0].value
