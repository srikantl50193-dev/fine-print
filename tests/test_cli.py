from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fine_print.cli import app
from fine_print.config import get_settings

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


@pytest.mark.parametrize(
    "arguments",
    [["index"], ["ask", "Can I get a refund?"], ["eval"], ["bench"]],
)
def test_unbuilt_commands_say_so_and_fail(arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)

    assert result.exit_code == 1
    assert "not implemented yet" in result.stderr
    assert f"fine-print {arguments[0]}" in result.stderr
    assert isinstance(result.exception, SystemExit)
    assert result.stdout == ""


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("ingest", "index", "ask", "eval", "bench"):
        assert command in result.stdout
