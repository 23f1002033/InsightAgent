"""
scripts/test_rag.py
────────────────────
Phase 3 smoke test — end-to-end RAG pipeline.

Workflow:
  1. Creates a small in-memory knowledge base from a temp file.
  2. Runs ask() against it.
  3. Prints the answer + inline citations.
  4. Demonstrates streaming with ask_stream().

Run:
    python scripts/test_rag.py
    python scripts/test_rag.py --stream
    python scripts/test_rag.py --question "What is LangGraph?"
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.logging_config import setup_logging
from app.config import get_settings


SAMPLE_DOCS = [
    (
        "rag_overview.txt",
        """\
Retrieval-Augmented Generation (RAG) is a technique that enhances large language
models by grounding their responses in retrieved external documents. Instead of
relying solely on parametric knowledge baked in during training, RAG dynamically
fetches relevant passages from a knowledge base at query time.

The two core components are the retriever and the generator. The retriever uses
dense vector similarity (e.g., cosine distance in embedding space) to find the
most relevant document chunks. The generator (an LLM) then conditions its
answer on those retrieved passages, dramatically reducing hallucinations.

RAG was introduced by Lewis et al. (2020) in "Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks" and has since become the dominant paradigm
for enterprise knowledge assistants.
""",
    ),
    (
        "langgraph_overview.txt",
        """\
LangGraph is a library built on top of LangChain for constructing stateful,
multi-actor applications with LLMs. It models agent workflows as explicit
StateGraphs — directed graphs where nodes represent processing steps and edges
represent control flow.

Unlike simple sequential LCEL chains, LangGraph supports cycles (loops), which
are essential for agentic behaviour: an agent can decide to call a tool, observe
the result, and then decide to call another tool before finally answering.

The state is a typed dictionary (or Pydantic model) that persists across all
nodes in the graph. LangGraph's checkpointing system (MemorySaver, PostgresSaver)
automatically snapshots the state after every node, enabling resumable workflows
and conversational memory keyed by thread_id.
""",
    ),
    (
        "vector_stores.txt",
        """\
A vector store (also called a vector database) stores document embeddings and
supports efficient approximate nearest-neighbour (ANN) search.

Chroma is an open-source, Python-native vector store that persists data to disk.
It is ideal for local development: zero infrastructure, fast enough for tens of
thousands of documents, and trivially swappable with Pinecone or Qdrant in
production.

Pinecone is a fully managed cloud vector database with horizontal scalability,
metadata filtering, and multi-tenant namespaces. It is the go-to choice for
production deployments that need sub-100 ms latency at scale.

The LangChain VectorStore abstract interface ensures that switching from Chroma
to Pinecone requires changing only the backend construction line — all retrieval
and ingestion code remains identical.
""",
    ),
]


def build_temp_kb(tmp_dir: str) -> None:
    """Write sample docs to tmp_dir and ingest them into a temp Chroma DB."""
    from ingestion.loaders import load_source
    from ingestion.splitter import split_documents
    from ingestion.vector_store import ingest_documents
    from ingestion.embeddings import get_embeddings

    all_docs = []
    for filename, content in SAMPLE_DOCS:
        path = os.path.join(tmp_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        all_docs.extend(load_source(path))

    chunks = split_documents(all_docs, chunk_size=400, chunk_overlap=60)
    embeddings = get_embeddings()
    ingest_documents(
        chunks,
        embeddings=embeddings,
        persist_directory=os.path.join(tmp_dir, "chroma"),
        collection_name="test_rag_phase3",
    )
    return embeddings, os.path.join(tmp_dir, "chroma")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 RAG smoke test")
    parser.add_argument(
        "--question",
        default="What is RAG and how does it reduce hallucinations?",
        help="Question to ask the knowledge base",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream the answer token-by-token",
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of chunks to retrieve (default: 3)",
    )
    args = parser.parse_args()

    setup_logging(get_settings().log_level)

    print(f"\n{'─'*64}")
    print(f"  InsightAgent — Phase 3 RAG Smoke Test")
    print(f"  Provider   : {get_settings().llm_provider}")
    print(f"  Model      : {get_settings().active_model()}")
    print(f"  Embeddings : {get_settings().embedding_provider}")
    print(f"  Question   : {args.question}")
    print(f"{'─'*64}\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("📄 Building temporary knowledge base…")
        embeddings, chroma_dir = build_temp_kb(tmp_dir)
        print(f"   Ingested {len(SAMPLE_DOCS)} documents\n")

        from ingestion.vector_store import get_vector_store
        from app.llm import get_llm
        from agent.rag_chain import ask, ask_stream

        store = get_vector_store(
            embeddings=embeddings,
            persist_directory=chroma_dir,
            collection_name="test_rag_phase3",
        )
        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": args.k},
        )
        llm = get_llm(temperature=0.0)

        if args.stream:
            print("💬 Streaming answer:\n")
            for token in ask_stream(args.question, llm=llm, retriever=retriever):
                print(token, end="", flush=True)
            print("\n")
        else:
            print("🔍 Retrieving and generating answer…\n")
            response = ask(args.question, llm=llm, retriever=retriever)

            print("💬 Answer:\n")
            print(response.answer)

            print(f"\n{'─'*64}")
            print(f"📚 Citations ({len(response.citations)} source(s)):\n")
            for i, cit in enumerate(response.citations, start=1):
                print(f"  [{i}] {cit.source}  (chunk {cit.chunk_index}, {cit.file_type})")
                print(f"      \"{cit.snippet[:100]}…\"\n")

    print(f"{'─'*64}")
    print("  ✅  Phase 3 complete — RAG chain is working!")
    print(f"{'─'*64}\n")


if __name__ == "__main__":
    main()
