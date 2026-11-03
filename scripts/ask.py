"""Ask a question from the command line.

Usage:
  uv run python scripts/ask.py "Can I cancel for free?" --company ridely
  uv run python scripts/ask.py "Can I cancel for free?" --retrieve-only
"""

import argparse

from fine_print import config
from fine_print.rag import OllamaError, answer_question, retrieve
from fine_print.store import IndexNotBuiltError, RetrievedChunk


def print_chunks(chunks: list[RetrievedChunk]) -> None:
    for chunk in chunks:
        preview = " ".join(chunk.text.split())[:110]
        print(f"  {chunk.similarity:.3f}  {chunk.label}\n         {preview}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--company", help="only search this company's documents")
    parser.add_argument("--strategy", choices=config.STRATEGIES, default=config.DEFAULT_STRATEGY)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--retrieve-only", action="store_true", help="skip the LLM")
    args = parser.parse_args()

    try:
        if args.retrieve_only:
            print_chunks(retrieve(args.question, args.strategy, args.top_k, args.company))
            return
        result = answer_question(args.question, args.strategy, args.top_k, args.company)
    except (IndexNotBuiltError, OllamaError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"\n{result.answer}\n\nCited chunks:")
    print_chunks(result.citations)


if __name__ == "__main__":
    main()
