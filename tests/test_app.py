from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fine_print import rag, store
from fine_print.store import RetrievedChunk

APP_PATH = str(Path(__file__).parents[1] / "app.py")
CHUNK = RetrievedChunk("A $5 fee applies.", "acme", "terms", "Fees", "id", 0.812)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """The app with a fake index, so this test needs no vector store or model."""
    monkeypatch.setattr(store, "list_companies", lambda strategy: ["acme", "beta"])
    monkeypatch.setattr(rag, "retrieve", lambda *args: [CHUNK])
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


def ask(app: AppTest, question: str = "Any fee?") -> AppTest:
    app.text_input[0].set_value(question)
    return app.button[0].click().run()


def test_filter_lists_all_companies(app: AppTest) -> None:
    assert app.selectbox[0].options == ["All companies", "acme", "beta"]


def test_answer_shows_cited_chunk_and_score(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag, "call_ollama", lambda prompt: "It is $5 [acme - Fees].")
    ask(app)
    assert not app.exception
    assert "It is $5 [acme - Fees]." in [m.value for m in app.markdown]
    assert any("similarity 0.812" in e.label for e in app.expander)


def test_answer_without_valid_citation_shows_a_warning(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag, "call_ollama", lambda prompt: "It is $5.")
    ask(app)
    assert "unverified" in app.warning[0].value


def test_ollama_failure_shows_the_error(app: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(prompt: str) -> str:
        raise rag.OllamaError("Cannot reach Ollama. Start it with `ollama serve`.")

    monkeypatch.setattr(rag, "call_ollama", fail)
    ask(app)
    assert "ollama serve" in app.error[0].value
