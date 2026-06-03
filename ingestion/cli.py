"""
ingestion/cli.py
────────────────
Command-line interface for the ingestion pipeline.

Usage
─────
# Ingest all supported files in a directory:
    python -m ingestion.cli ingest ./docs/

# Ingest a single file:
    python -m ingestion.cli ingest ./docs/my_report.pdf

# Ingest a web URL:
    python -m ingestion.cli ingest https://example.com/article

# Query the knowledge base (quick sanity check):
    python -m ingestion.cli query "What is RAG?"

# Check collection stats:
    python -m ingestion.cli stats

Options
───────
--chunk-size    Max characters per chunk (default: 1000)
--chunk-overlap Overlap between chunks (default: 150)
-k, --top-k     Number of results to return for 'query' (default: 4)
--collection    Chroma collection name (default: insightagent_kb)
"""
from __future__ import annotations

import argparse
import sys
import os
import logging

# Ensure project root is on sys.path when invoked as `python -m ingestion.cli`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.logging_config import setup_logging
from app.config import get_settings


def cmd_ingest(args: argparse.Namespace) -> None:
    """Load, split, embed, and store the given source."""
    from ingestion.loaders import load_source, load_directory
    from ingestion.splitter import split_documents
    from ingestion.vector_store import ingest_documents

    source = args.source
    logger = logging.getLogger(__name__)

    print(f"\n{'─'*60}")
    print(f"  InsightAgent — Ingestion Pipeline")
    print(f"  Source : {source}")
    print(f"{'─'*60}\n")

    # Step 1: Load
    print("📄  Step 1/3 — Loading documents…")
    is_dir = os.path.isdir(source)
    if is_dir:
        docs = load_directory(source)
    else:
        docs = load_source(source)

    print(f"    Loaded {len(docs)} document(s).")

    if not docs:
        print("⚠️  No documents loaded — check your source path/URL.")
        sys.exit(1)

    # Step 2: Split
    print(f"\n✂️   Step 2/3 — Splitting into chunks "
          f"(size={args.chunk_size}, overlap={args.chunk_overlap})…")
    chunks = split_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"    Produced {len(chunks)} chunk(s).")

    # Step 3: Embed + Store
    print(f"\n🔢  Step 3/3 — Embedding and storing in Chroma…")
    store = ingest_documents(chunks, collection_name=args.collection)

    print(f"\n{'─'*60}")
    print(f"  ✅  Ingestion complete!")
    print(f"  Collection : {args.collection}")
    print(f"  Chunks stored : {len(chunks)}")
    print(f"{'─'*60}\n")


def cmd_query(args: argparse.Namespace) -> None:
    """Run a similarity search and print results."""
    from ingestion.vector_store import similarity_search

    print(f"\n🔍  Querying knowledge base: '{args.query}'")
    print(f"    Top-{args.top_k} results from collection '{args.collection}'\n")

    results = similarity_search(
        query=args.query,
        k=args.top_k,
        collection_name=args.collection,
    )

    if not results:
        print("  No results found — has anything been ingested yet?")
        return

    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        file_type = doc.metadata.get("file_type", "?")
        print(f"  ── Result {i} ─────────────────────────────────────────")
        print(f"  Source     : {source}")
        print(f"  File type  : {file_type}")
        print(f"  Chunk index: {chunk_idx}")
        print(f"  Content    :\n")
        # Print first 400 chars of the chunk
        preview = doc.page_content[:400].replace("\n", " ")
        print(f"    {preview}…\n")


def cmd_stats(args: argparse.Namespace) -> None:
    """Print statistics about the current collection."""
    from langchain_chroma import Chroma
    from ingestion.embeddings import get_embeddings
    from app.config import get_settings

    settings = get_settings()
    embeddings = get_embeddings()

    store = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    # Chroma exposes the underlying collection via ._collection
    count = store._collection.count()  # type: ignore[attr-defined]
    print(f"\n📊  Collection: '{args.collection}'")
    print(f"    Document chunks stored: {count}")
    print(f"    Persist directory: {settings.chroma_persist_dir}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.cli",
        description="InsightAgent — Ingestion CLI",
    )
    parser.add_argument(
        "--collection",
        default="insightagent_kb",
        help="Chroma collection name (default: insightagent_kb)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ────────────────────────────────────────────────────────────────
    ingest_p = sub.add_parser("ingest", help="Load, split, embed, and store a source.")
    ingest_p.add_argument("source", help="File path, directory, or https:// URL.")
    ingest_p.add_argument(
        "--chunk-size", type=int, default=1000, metavar="N",
        help="Max characters per chunk (default: 1000)"
    )
    ingest_p.add_argument(
        "--chunk-overlap", type=int, default=150, metavar="N",
        help="Overlap characters between chunks (default: 150)"
    )

    # ── query ─────────────────────────────────────────────────────────────────
    query_p = sub.add_parser("query", help="Run a similarity search (sanity check).")
    query_p.add_argument("query", help="Natural-language query string.")
    query_p.add_argument(
        "-k", "--top-k", type=int, default=4,
        help="Number of results to return (default: 4)"
    )

    # ── stats ─────────────────────────────────────────────────────────────────
    sub.add_parser("stats", help="Show collection statistics.")

    return parser


def main() -> None:
    setup_logging(get_settings().log_level)
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "stats": cmd_stats,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
