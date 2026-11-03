from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fine_print import embed, rag, store
from fine_print import evaluate as ev
from fine_print.cli import app
from fine_print.config import Strategy, get_settings
from fine_print.models import NOT_FOUND_MESSAGE, Answer, FrozenModel, RetrievedChunk

runner = CliRunner()


@pytest.fixture(autouse=True)
def fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FINE_PRINT_RAW_DIR", str(tmp_path))
    return tmp_path


def test_ingest_reports_counts_on_stdout(raw_dir: Path) -> None:
    (raw_dir / "acme_terms.md").write_text("# Fees\nFive dollars.\n\n# Refunds\nNone at all.")

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 0
    row = next(line for line in result.stdout.splitlines() if line.startswith("acme_terms.md"))
    assert row.split()[-2:] == ["1", "2"]
    assert "1 documents" in result.stdout


def test_ingest_on_the_fictional_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINE_PRINT_RAW_DIR", str(get_settings().samples_dir))
    get_settings.cache_clear()

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 0
    assert "ridely_terms.html" in result.stdout
    assert "tunely_privacy.txt" in result.stdout


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda path: path.rmdir(), "does not exist"),
        (lambda path: None, "No documents"),
        (lambda path: (path / "badname.md").write_text("text"), "Bad filename"),
    ],
)
def test_ingest_failures_exit_cleanly(raw_dir: Path, setup, message: str) -> None:
    setup(raw_dir)

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 1
    assert message in result.stderr
    assert isinstance(result.exception, SystemExit)


def test_ingest_reports_an_invalid_overlap(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (raw_dir / "acme_terms.md").write_text("Some text.")
    monkeypatch.setenv("FINE_PRINT_CHUNK_OVERLAP", "5000")

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 1
    assert "overlap" in result.stderr


def test_invalid_settings_exit_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINE_PRINT_TOP_K", "not-a-number")

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 1
    assert "top_k" in result.stderr
    assert isinstance(result.exception, SystemExit)


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embed,
        "embed_chunks",
        lambda chunks: np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(chunks), 1)),
    )


@pytest.mark.usefixtures("fake_embeddings")
def test_index_builds_both_strategies(
    raw_dir: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    chroma_dir = tmp_path_factory.mktemp("chroma")
    monkeypatch.setenv("FINE_PRINT_CHROMA_DIR", str(chroma_dir))
    (raw_dir / "acme_terms.md").write_text("# Fees\nFive dollars.\n\n# Refunds\nNone at all.")

    result = runner.invoke(app, ["index"])

    assert result.exit_code == 0, result.stderr
    assert "Indexed 1 chunks for the 'fixed' strategy" in result.stdout
    assert "Indexed 2 chunks for the 'section' strategy" in result.stdout
    assert store.list_companies("section") == ["acme"]
    assert store.list_companies("fixed") == ["acme"]


@pytest.mark.usefixtures("fake_embeddings")
def test_index_can_build_one_strategy(
    raw_dir: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_PRINT_CHROMA_DIR", str(tmp_path_factory.mktemp("chroma")))
    (raw_dir / "acme_terms.md").write_text("# Fees\nFive dollars.")

    result = runner.invoke(app, ["index", "--strategy", "section"])

    assert result.exit_code == 0, result.stderr
    assert "'fixed'" not in result.stdout
    assert "Indexed 1 chunks for the 'section' strategy" in result.stdout


def test_index_rejects_an_unknown_strategy() -> None:
    assert runner.invoke(app, ["index", "--strategy", "magic"]).exit_code == 2


def test_index_without_documents_fails_cleanly(raw_dir: Path) -> None:
    result = runner.invoke(app, ["index"])

    assert result.exit_code == 1
    assert "make data" in result.stderr


def make_retrieved() -> RetrievedChunk:
    return RetrievedChunk(
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


def make_answer(**fields: object) -> Answer:
    return Answer.model_validate({"question": "Any fee?", "text": "It is $5.", **fields})


def test_ask_prints_the_answer_and_its_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = make_retrieved()
    citation = rag.find_citations(chunk.label, [chunk])
    calls = []

    def fake_answer(*args: object) -> Answer:
        calls.append(args)
        return make_answer(citations=citation)

    monkeypatch.setattr(rag, "answer_question", fake_answer)

    result = runner.invoke(
        app,
        ["ask", "Any fee?", "--company", "acme", "--doc-type", "terms", "--top-k", "3"],
    )

    assert result.exit_code == 0, result.stderr
    assert "It is $5." in result.stdout
    assert "0.812  [Acme – Fees] (terms)" in result.stdout
    assert calls == [("Any fee?", None, 3, "acme", "terms")]


def test_ask_warns_when_no_citation_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "answer_question", lambda *args: make_answer())

    result = runner.invoke(app, ["ask", "Any fee?"])

    assert result.exit_code == 0
    assert "unverified" in result.stderr


def test_ask_explains_a_low_similarity_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    refusal = make_answer(text=NOT_FOUND_MESSAGE, refused=True, refusal_reason="low_similarity")
    monkeypatch.setattr(rag, "answer_question", lambda *args: refusal)

    result = runner.invoke(app, ["ask", "Unrelated?"])

    assert result.exit_code == 0
    assert NOT_FOUND_MESSAGE in result.stdout
    assert "similar enough" in result.stdout


def test_ask_prints_a_model_refusal_without_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    refusal = make_answer(text=NOT_FOUND_MESSAGE, refused=True, refusal_reason="model_declined")
    monkeypatch.setattr(rag, "answer_question", lambda *args: refusal)

    result = runner.invoke(app, ["ask", "Unrelated?"])

    assert result.exit_code == 0
    assert result.stdout.strip() == NOT_FOUND_MESSAGE
    assert result.stderr == ""


def test_ask_retrieve_only_skips_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "retrieve", lambda *args: [make_retrieved()])

    result = runner.invoke(app, ["ask", "Any fee?", "--retrieve-only"])

    assert result.exit_code == 0
    assert "[Acme – Fees]" in result.stdout


@pytest.mark.parametrize(
    "error",
    [rag.OllamaError("Start it with `ollama serve`."), store.IndexNotBuiltError("Run: make index")],
)
def test_ask_failures_exit_cleanly(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def fail(*args: object) -> Answer:
        raise error

    monkeypatch.setattr(rag, "answer_question", fail)

    result = runner.invoke(app, ["ask", "Any fee?"])

    assert result.exit_code == 1
    assert str(error) in result.stderr
    assert isinstance(result.exception, SystemExit)


@pytest.fixture
def fake_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    def result(strategy: str, questions: object) -> ev.StrategyResult:
        return ev.StrategyResult(
            strategy="section" if strategy == "section" else "fixed",
            num_chunks=7,
            mean_chunk_chars=300.0,
            recall_at_k=dict.fromkeys(get_settings().eval_k_values, 0.5),
            mrr=0.5,
            misses=[f"Missed {strategy}?"],
        )

    monkeypatch.setattr(ev, "evaluate_strategy", result)
    monkeypatch.setattr(
        ev,
        "evaluate_threshold",
        lambda *args: ev.ThresholdResult(
            threshold=0.6,
            answerable_kept=0.9,
            unanswerable_refused=0.4,
            wrongly_kept=["Kept?"],
            wrongly_refused=["Refused?"],
        ),
    )


@pytest.mark.usefixtures("fake_evaluation")
def test_eval_prints_the_scores_and_the_mistakes() -> None:
    result = runner.invoke(app, ["eval"])

    assert result.exit_code == 0, result.stderr
    assert "| section | 7 | 300 |" in result.stdout
    assert "Missed by 'fixed': Missed fixed?" in result.stdout
    assert "keeps 90%" in result.stdout
    assert "refuses 40%" in result.stdout
    assert "Passed to the model, but unanswerable: Kept?" in result.stdout
    assert "Refused, but answerable: Refused?" in result.stdout
    assert "Model answers" not in result.stdout


@pytest.mark.usefixtures("fake_evaluation")
def test_eval_answers_reports_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ev,
        "evaluate_answers",
        lambda *args: ev.AnswerResult(
            answered=0.8,
            cited_any=0.7,
            cited_relevant=0.6,
            unanswerable_refused=0.5,
            wrongly_refused=["Declined?"],
            wrongly_answered=["Invented?"],
        ),
    )

    result = runner.invoke(app, ["eval", "--answers"])

    assert result.exit_code == 0, result.stderr
    assert "answered: 80%" in result.stdout
    assert "with a valid citation: 70%" in result.stdout
    assert "with the expected text: 60%" in result.stdout
    assert "Model refused, but answerable: Declined?" in result.stdout
    assert "Model answered, but unanswerable: Invented?" in result.stdout


def test_eval_labels_the_tuning_set(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = ev.load_jsonl

    def spy(path: Path, model: type[FrozenModel]) -> list[FrozenModel]:
        calls.append(path)
        return original(path, model)

    monkeypatch.setattr(ev, "load_jsonl", spy)
    monkeypatch.setattr(
        ev, "evaluate_strategy", lambda *args: (_ for _ in ()).throw(FileNotFoundError("stop"))
    )

    result = runner.invoke(app, ["eval"])

    assert "Question set: tuning" in result.stdout
    assert calls == [get_settings().questions_path, get_settings().out_of_scope_path]


@pytest.mark.usefixtures("fake_evaluation")
def test_eval_holdout_reads_only_held_out_files_and_scores_only_the_default_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = []
    strategies = []
    original_load = ev.load_jsonl
    original_strategy = ev.evaluate_strategy

    def spy_load(path: Path, model: type[FrozenModel]) -> list[FrozenModel]:
        paths.append(path)
        return original_load(path, model)

    def spy_strategy(strategy: Strategy, questions: list[ev.EvalQuestion]) -> ev.StrategyResult:
        strategies.append(strategy)
        return original_strategy(strategy, questions)

    monkeypatch.setattr(ev, "load_jsonl", spy_load)
    monkeypatch.setattr(ev, "evaluate_strategy", spy_strategy)

    result = runner.invoke(app, ["eval", "--holdout"])

    settings = get_settings()
    assert result.exit_code == 0, result.stderr
    assert "Question set: HELD-OUT" in result.stdout
    assert paths == [settings.holdout_questions_path, settings.holdout_out_of_scope_path]
    assert strategies == [settings.default_strategy]


def test_eval_without_an_index_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_PRINT_CHROMA_DIR", str(tmp_path))
    monkeypatch.setattr(embed, "embed_query", lambda question: np.array([1.0, 0.0]))
    monkeypatch.setattr(rag, "embed_query", lambda question: np.array([1.0, 0.0]))

    result = runner.invoke(app, ["eval"])

    assert result.exit_code == 1
    assert "make index" in result.stderr
    assert isinstance(result.exception, SystemExit)


def test_eval_without_questions_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_PRINT_QUESTIONS_PATH", str(tmp_path / "absent.jsonl"))

    result = runner.invoke(app, ["eval"])

    assert result.exit_code == 1
    assert "absent.jsonl" in result.stderr


def test_bench_prints_the_table_and_saves_the_chart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_PRINT_DOCS_DIR", str(tmp_path))
    monkeypatch.setenv("FINE_PRINT_BENCH_LATENCY_REPEATS", "1")
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(50, 8)).astype(np.float32)
    monkeypatch.setattr(store, "get_all_embeddings", lambda strategy: vectors)
    monkeypatch.setattr(embed, "embed_query", lambda question: rng.normal(size=8))

    result = runner.invoke(app, ["bench", "--strategy", "fixed"])

    assert result.exit_code == 0, result.stderr
    assert "50 chunks ('fixed' strategy)" in result.stdout
    assert "| Flat | - | 1.000 |" in result.stdout
    assert (tmp_path / "benchmark.png").exists()


def test_bench_without_an_index_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINE_PRINT_CHROMA_DIR", str(tmp_path))

    result = runner.invoke(app, ["bench"])

    assert result.exit_code == 1
    assert "make index" in result.stderr
    assert isinstance(result.exception, SystemExit)


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("ingest", "index", "ask", "eval", "bench"):
        assert command in result.stdout
