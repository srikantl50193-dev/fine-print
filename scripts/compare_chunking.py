"""Compare the chunking strategies fairly, with fixed chunks sized to match section chunks.

The keyword test favours longer chunks, and fixed chunks at the default size are longer than
section chunks. So this also scores fixed chunks at the size whose mean length matches.

Each run builds its own temporary index and leaves the real one alone. It uses the tuning
questions only, because its purpose is to choose between strategies.
"""

import os
import tempfile

from fine_print.chunking import chunk_document, matching_fixed_size, mean_chunk_chars
from fine_print.config import Strategy, get_settings
from fine_print.embed import embed_chunks
from fine_print.evaluate import EvalQuestion, evaluate_strategy, load_jsonl
from fine_print.ingest import load_documents
from fine_print.models import Document
from fine_print.store import build_collection


def score(
    label: str, documents: list[Document], strategy: Strategy, size: int, overlap: int
) -> str:
    questions = load_jsonl(get_settings().questions_path, EvalQuestion)
    chunks = [c for d in documents for c in chunk_document(d, strategy, size, overlap)]
    with tempfile.TemporaryDirectory() as folder:
        os.environ["FINE_PRINT_CHROMA_DIR"] = folder
        get_settings.cache_clear()
        build_collection(chunks, embed_chunks(chunks), strategy)
        result = evaluate_strategy(strategy, questions)
    del os.environ["FINE_PRINT_CHROMA_DIR"]
    get_settings.cache_clear()

    recalls = " | ".join(f"{value:.2f}" for value in result.recall_at_k.values())
    cells = f"{size} | {result.num_chunks} | {result.mean_chunk_chars:.0f} | {recalls}"
    return f"| {label} | {cells} | {result.mrr:.3f} |"


def main() -> None:
    settings = get_settings()
    size, overlap = settings.chunk_size, settings.chunk_overlap
    documents = load_documents(settings.raw_dir)

    section_mean = mean_chunk_chars(documents, "section", size, overlap)
    matched = matching_fixed_size(documents, section_mean, overlap)
    print(f"Section mean: {section_mean:.0f} characters. Matching fixed size: {matched}.\n")

    recall_headers = " | ".join(f"Recall@{k}" for k in settings.eval_k_values)
    print(f"| Run | Size limit | Chunks | Mean chars | {recall_headers} | MRR |")
    print("|" + " --- |" * (5 + len(settings.eval_k_values)))
    print(score("section", documents, "section", size, overlap))
    print(score("fixed, default size", documents, "fixed", size, overlap))
    print(score("fixed, matched length", documents, "fixed", matched, overlap))


if __name__ == "__main__":
    main()
