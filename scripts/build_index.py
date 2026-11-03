"""Build the Chroma index for one chunking strategy.

Usage: uv run python scripts/build_index.py --strategy section
"""

import argparse

from fine_print import config
from fine_print.chunking import chunk_documents
from fine_print.embed import embed_texts
from fine_print.ingest import ensure_sample_data, load_documents
from fine_print.store import build_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=config.STRATEGIES, default=config.DEFAULT_STRATEGY)
    strategy = parser.parse_args().strategy

    ensure_sample_data()
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = chunk_documents(documents, strategy)
    if not chunks:
        raise SystemExit(f"No chunks produced. Check the files in {config.RAW_DATA_DIR}.")
    print(f"Created {len(chunks)} chunks with the '{strategy}' strategy")

    embeddings = embed_texts([chunk.text for chunk in chunks])
    print(f"Embedded them into {embeddings.shape[1]}-dimensional vectors")

    collection = build_collection(chunks, embeddings, strategy)
    print(f"Saved collection '{collection.name}' with {collection.count()} chunks")


if __name__ == "__main__":
    main()
