"""The `fine-print` command line."""

from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError

from fine_print.chunking import chunk_document
from fine_print.config import STRATEGIES, get_settings
from fine_print.ingest import load_documents
from fine_print.logging_config import configure_logging

app = typer.Typer(
    help="Local, citation-first RAG over Terms of Service and privacy policies.",
    no_args_is_help=True,
    add_completion=False,
)


def _fail(message: str) -> NoReturn:
    """Print an error to stderr and exit with a non-zero code, without a traceback."""
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _not_implemented(command: str, missing: str) -> NoReturn:
    """Exit for a command whose pipeline stage is not built yet."""
    _fail(f"`fine-print {command}` is not implemented yet (still to build: {missing}).")


@app.callback()
def main() -> None:
    """Local, citation-first RAG over Terms of Service and privacy policies."""
    try:
        settings = get_settings()
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()
        )
        _fail(f"Invalid FINE_PRINT_* settings: {problems}")
    configure_logging(settings.log_level, json_output=settings.log_json)


@app.command()
def ingest() -> None:
    """Parse and chunk every document in the raw folder, and report the counts."""
    settings = get_settings()
    try:
        documents = load_documents(settings.raw_dir)
        counts = [
            [
                len(chunk_document(document, strategy, settings.chunk_size, settings.chunk_overlap))
                for strategy in STRATEGIES
            ]
            for document in documents
        ]
    except FileNotFoundError:
        _fail(f"{settings.raw_dir} does not exist. Run scripts/download_data.py first.")
    except ValueError as error:
        _fail(str(error))
    if not documents:
        _fail(f"No documents in {settings.raw_dir}. Run scripts/download_data.py first.")

    typer.echo(f"{'document':<28}{'characters':>12}" + "".join(f"{s:>10}" for s in STRATEGIES))
    for document, row in zip(documents, counts, strict=True):
        cells = "".join(f"{count:>10}" for count in row)
        typer.echo(f"{document.source:<28}{len(document.text):>12}{cells}")
    totals = "".join(f"{sum(column):>10}" for column in zip(*counts, strict=True))
    typer.echo(f"{f'{len(documents)} documents':<28}{'':>12}{totals}")


@app.command()
def index() -> None:
    """Embed the chunks and build the vector index."""
    _not_implemented("index", "the embedding and vector store layer")


@app.command()
def ask(question: Annotated[str, typer.Argument(help="The question to answer.")]) -> None:
    """Answer a question from the documents, with citations."""
    _not_implemented("ask", "retrieval and the Ollama answer step")


@app.command(name="eval")
def evaluate() -> None:
    """Measure retrieval quality for each chunking strategy."""
    _not_implemented("eval", "the evaluation harness")


@app.command()
def bench() -> None:
    """Compare vector index types on search speed and recall."""
    _not_implemented("bench", "the index benchmark")
