"""The `fine-print` command line."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import numpy as np
import typer
from pydantic import ValidationError

from fine_print.chunking import chunk_document
from fine_print.config import STRATEGIES, get_settings
from fine_print.ingest import load_documents
from fine_print.logging_config import configure_logging
from fine_print.models import Citation, Document, RetrievedChunk

# Typer reads the choices from this at runtime, which it can't do through `Strategy`,
# because that is a `type` statement.
StrategyChoice = Literal["fixed", "section"]

app = typer.Typer(
    help="Local, citation-first RAG over Terms of Service and privacy policies.",
    no_args_is_help=True,
    add_completion=False,
)


def _fail(message: str) -> NoReturn:
    """Print an error to stderr and exit with a non-zero code, without a traceback."""
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


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


def _load_documents_or_fail() -> list[Document]:
    """Load the raw documents, turning any problem into a clean exit."""
    raw_dir = get_settings().raw_dir
    try:
        documents = load_documents(raw_dir)
    except FileNotFoundError:
        _fail(f"{raw_dir} does not exist. Run: make data")
    except ValueError as error:
        _fail(str(error))
    if not documents:
        _fail(f"No documents in {raw_dir}. Run: make data")
    return documents


@app.command()
def ingest() -> None:
    """Parse and chunk every document in the raw folder, and report the counts."""
    settings = get_settings()
    documents = _load_documents_or_fail()
    try:
        counts = [
            [
                len(chunk_document(document, strategy, settings.chunk_size, settings.chunk_overlap))
                for strategy in STRATEGIES
            ]
            for document in documents
        ]
    except ValueError as error:
        _fail(str(error))

    typer.echo(f"{'document':<28}{'characters':>12}" + "".join(f"{s:>10}" for s in STRATEGIES))
    for document, row in zip(documents, counts, strict=True):
        cells = "".join(f"{count:>10}" for count in row)
        typer.echo(f"{document.source:<28}{len(document.text):>12}{cells}")
    totals = "".join(f"{sum(column):>10}" for column in zip(*counts, strict=True))
    typer.echo(f"{f'{len(documents)} documents':<28}{'':>12}{totals}")


@app.command()
def index(
    strategy: Annotated[
        StrategyChoice | None,
        typer.Option(help="Build only this strategy's index. The default is both."),
    ] = None,
) -> None:
    """Embed the chunks and build the vector index."""
    # Imported here because these load torch and Chroma, which most commands don't need.
    from fine_print.embed import embed_chunks  # noqa: PLC0415
    from fine_print.store import build_collection  # noqa: PLC0415

    settings = get_settings()
    documents = _load_documents_or_fail()
    typer.echo(f"Loaded {len(documents)} documents from {settings.raw_dir}")
    for name in [strategy] if strategy else STRATEGIES:
        try:
            chunks = [
                chunk
                for document in documents
                for chunk in chunk_document(
                    document, name, settings.chunk_size, settings.chunk_overlap
                )
            ]
        except ValueError as error:
            _fail(str(error))
        count = build_collection(chunks, embed_chunks(chunks), name)
        typer.echo(f"Indexed {count} chunks for the '{name}' strategy")
    typer.echo(f"Index saved in {settings.chroma_dir}")


def _print_passages(chunks: Sequence[RetrievedChunk | Citation]) -> None:
    """Print one line per passage, with its score, label and a preview of its text."""
    for chunk in chunks:
        preview = " ".join(chunk.text.split())[:90]
        typer.echo(
            f"  {chunk.similarity:.3f}  {chunk.label} ({chunk.doc_type})\n         {preview}"
        )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    company: Annotated[str | None, typer.Option(help="Only search this company.")] = None,
    doc_type: Annotated[
        str | None, typer.Option(help="Only search this document type: terms or privacy.")
    ] = None,
    strategy: Annotated[StrategyChoice | None, typer.Option(help="The index to search.")] = None,
    top_k: Annotated[int | None, typer.Option(min=1, help="How many passages to retrieve.")] = None,
    retrieve_only: Annotated[
        bool, typer.Option("--retrieve-only", help="Show the passages and skip the model.")
    ] = False,
) -> None:
    """Answer a question from the documents, with citations."""
    # Imported here because these load torch and Chroma, which most commands don't need.
    from fine_print.rag import OllamaError, answer_question, retrieve  # noqa: PLC0415
    from fine_print.store import IndexNotBuiltError  # noqa: PLC0415

    try:
        if retrieve_only:
            _print_passages(retrieve(question, strategy, top_k, company, doc_type))
            return
        answer = answer_question(question, strategy, top_k, company, doc_type)
    except (IndexNotBuiltError, OllamaError) as error:
        _fail(str(error))

    typer.echo(answer.text)
    if answer.refusal_reason == "low_similarity":
        typer.echo("\nNo passage was similar enough to the question to ask the model.")
    elif answer.citations:
        typer.echo("\nCited passages:")
        _print_passages(answer.citations)
    elif not answer.refused:
        typer.echo(
            "\nWarning: the answer cites no retrieved passage, so treat it as unverified.",
            err=True,
        )


@app.command(name="eval")
def evaluate(
    answers: Annotated[
        bool,
        typer.Option(
            "--answers", help="Also ask the model every question. Slow, and needs Ollama."
        ),
    ] = False,
    holdout: Annotated[
        bool,
        typer.Option(
            "--holdout",
            help="Report on the held-out questions, using only the default strategy. "
            "For a frozen configuration: never tune anything from this.",
        ),
    ] = False,
    model: Annotated[
        str | None, typer.Option(help="The Ollama model for --answers. Defaults to the settings.")
    ] = None,
    prompt: Annotated[
        str | None, typer.Option(help="The prompt variant for --answers. Defaults to the settings.")
    ] = None,
    save: Annotated[
        Path | None, typer.Option(help="Write every question's result from --answers to this file.")
    ] = None,
) -> None:
    """Measure retrieval quality for each chunking strategy."""
    # Imported here because these load torch and Chroma, which most commands don't need.
    from fine_print import evaluate as ev  # noqa: PLC0415
    from fine_print.prompts import load_prompt  # noqa: PLC0415
    from fine_print.rag import OllamaError  # noqa: PLC0415
    from fine_print.store import IndexNotBuiltError  # noqa: PLC0415

    settings = get_settings()
    questions_path = settings.holdout_questions_path if holdout else settings.questions_path
    off_topic_path = settings.holdout_out_of_scope_path if holdout else settings.out_of_scope_path
    # Scoring every strategy here would invite choosing between them on held-out data.
    strategies = (settings.default_strategy,) if holdout else STRATEGIES
    typer.echo(f"Question set: {'HELD-OUT' if holdout else 'tuning'}")
    try:
        questions = ev.load_jsonl(questions_path, ev.EvalQuestion)
        off_topic = ev.load_jsonl(off_topic_path, ev.OutOfScopeQuestion)
        results = [ev.evaluate_strategy(strategy, questions) for strategy in strategies]
        threshold = ev.evaluate_threshold(settings.default_strategy, questions, off_topic)
        answer_result = (
            ev.evaluate_answers(
                settings.default_strategy,
                questions,
                off_topic,
                model,
                load_prompt(prompt or settings.prompt_name),
            )
            if answers
            else None
        )
    except (
        FileNotFoundError,
        ValueError,
        IndexNotBuiltError,
        OllamaError,
    ) as error:
        _fail(str(error))

    typer.echo(f"Evaluated {len(questions)} questions\n\n{ev.format_table(results)}")
    for result in results:
        for question in result.misses:
            typer.echo(f"Missed by '{result.strategy}': {question}")

    typer.echo(
        f"\nRefusal threshold {threshold.threshold} ({settings.default_strategy} index): "
        f"keeps {threshold.answerable_kept:.0%} of {len(questions)} answerable questions, "
        f"refuses {threshold.unanswerable_refused:.0%} of {len(off_topic)} unanswerable ones"
    )
    for question in threshold.wrongly_refused:
        typer.echo(f"Refused, but answerable: {question}")
    for question in threshold.wrongly_kept:
        typer.echo(f"Passed to the model, but unanswerable: {question}")

    if answer_result:
        typer.echo(
            f"\nModel answers ({answer_result.model}, prompt '{answer_result.prompt}'):\n"
            f"  answered: {answer_result.answered:.0%} of answerable questions\n"
            f"  with a valid citation: {answer_result.cited_any:.0%}\n"
            f"  citing a passage with the expected text: {answer_result.cited_relevant:.0%}\n"
            f"  refused: {answer_result.unanswerable_refused:.0%} of unanswerable questions\n"
            f"  wrong refusals: {len(answer_result.wrongly_refused)}\n"
            f"  p95 latency: {answer_result.p95_latency_s:.1f}s\n"
            f"  failures: {answer_result.failures}"
        )
        for question in answer_result.wrongly_refused:
            typer.echo(f"Model refused, but answerable: {question}")
        for question in answer_result.wrongly_answered:
            typer.echo(f"Model answered, but unanswerable: {question}")
        if save:
            save.parent.mkdir(parents=True, exist_ok=True)
            save.write_text(answer_result.model_dump_json(indent=1) + "\n", encoding="utf-8")
            typer.echo(f"Results saved to {save}")


@app.command()
def bench(
    strategy: Annotated[StrategyChoice | None, typer.Option(help="The vectors to use.")] = None,
) -> None:
    """Compare vector index types on search speed and recall."""
    # Imported here because these load torch, FAISS and Chroma, which most commands don't need.
    from fine_print import benchmark  # noqa: PLC0415
    from fine_print.embed import embed_query  # noqa: PLC0415
    from fine_print.evaluate import EvalQuestion, load_jsonl  # noqa: PLC0415
    from fine_print.store import IndexNotBuiltError, get_all_embeddings  # noqa: PLC0415

    settings = get_settings()
    name = strategy or settings.default_strategy
    try:
        vectors = get_all_embeddings(name)
        questions = load_jsonl(settings.questions_path, EvalQuestion)
    except (FileNotFoundError, ValidationError, IndexNotBuiltError) as error:
        _fail(str(error))

    queries = np.stack([embed_query(q.question) for q in questions]).astype(np.float32)
    rows, k = benchmark.run_benchmark(vectors, queries)
    typer.echo(f"{len(vectors)} chunks ('{name}' strategy), {len(queries)} queries, k={k}\n")
    typer.echo(benchmark.format_table(rows, k))

    chart = settings.docs_dir / "benchmark.png"
    benchmark.save_chart(rows, k, chart)
    typer.echo(f"\nChart saved to {chart}")
