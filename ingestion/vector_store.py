"""
ingestion/vector_store.py
─────────────────────────
Vector store abstraction for InsightAgent.

Concept — VectorStore interface
────────────────────────────────
LangChain's VectorStore base class exposes a uniform API regardless of backend:
  • add_documents(docs)              → store embeddings
  • similarity_search(query, k)      → List[Document] (nearest neighbours)
  • similarity_search_with_score(…)  → List[Tuple[Document, float]]
  • as_retriever(**kwargs)           → VectorStoreRetriever (for chain composition)

We use Chroma locally because it:
  • persists to disk (no external server to run)
  • is fast enough for thousands of documents
  • is trivially swappable — see get_vector_store() below

Swapping to Pinecone / Qdrant
──────────────────────────────
Replace the Chroma lines in get_vector_store() with:
  from langchain_pinecone import PineconeVectorStore
  return PineconeVectorStore(index_name=..., embedding=embeddings)

Everything downstream (retriever, agent tools) stays identical.

Trade-offs
──────────
• Chroma with persist_directory: data survives restarts but is single-node.
  For multi-instance deployments, use a client/server mode or Qdrant/Pinecone.
• collection_name: use different names per knowledge base / tenant to isolate
  document sets without spinning up separate Chroma instances.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "insightagent_kb"


def get_vector_store(
    embeddings: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
) -> VectorStore:
    """
    Return a Chroma vector store (creates or loads from disk).

    Parameters
    ----------
    embeddings:
        LangChain Embeddings instance.  If None, calls get_embeddings().
    persist_directory:
        Path where Chroma persists data.  Defaults to settings.chroma_persist_dir.
    collection_name:
        Chroma collection name.  Useful for multi-tenant setups.

    Returns
    -------
    VectorStore
        A Chroma instance implementing the LangChain VectorStore interface.

    Design note
    ───────────
    The return type is annotated as VectorStore (the abstract base) not Chroma.
    This is intentional: nothing downstream should import Chroma directly;
    they only depend on the VectorStore interface.  Swap backends here only.
    """
    from langchain_chroma import Chroma

    if embeddings is None:
        from ingestion.embeddings import get_embeddings
        embeddings = get_embeddings()

    settings = get_settings()
    persist_dir = persist_directory or settings.chroma_persist_dir

    # Ensure directory exists
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    logger.debug(
        "Opening Chroma collection '%s' at %s", collection_name, persist_dir
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def ingest_documents(
    documents: List[Document],
    embeddings: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
) -> VectorStore:
    """
    Embed and store documents in Chroma.

    This is the main entry point for the ingestion pipeline.  It:
      1. Opens (or creates) the Chroma collection.
      2. Adds all documents (Chroma deduplicates by document id if provided).
      3. Returns the store so callers can immediately run test queries.

    Parameters
    ----------
    documents:
        Chunked documents from split_documents().
    embeddings:
        Optional pre-built Embeddings instance.
    persist_directory:
        Override the default Chroma persist path.
    collection_name:
        Chroma collection to write into.

    Returns
    -------
    VectorStore
        The populated Chroma instance.
    """
    if not documents:
        logger.warning("ingest_documents called with an empty list — nothing to store.")
        return get_vector_store(
            embeddings=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    store = get_vector_store(
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    logger.info("Embedding and storing %d chunk(s)…", len(documents))
    store.add_documents(documents)
    logger.info("✅ Ingestion complete — %d chunks in collection '%s'.",
                len(documents), collection_name)
    return store


def similarity_search(
    query: str,
    k: int = 4,
    embeddings: Optional[Embeddings] = None,
    persist_directory: Optional[str] = None,
    collection_name: str = COLLECTION_NAME,
) -> List[Document]:
    """
    Run a similarity search against the stored knowledge base.

    Parameters
    ----------
    query:
        Natural-language query string.
    k:
        Number of documents to return.
    embeddings, persist_directory, collection_name:
        Forwarded to get_vector_store().

    Returns
    -------
    List[Document]
        The k most similar documents, ordered by descending similarity.
    """
    store = get_vector_store(
        embeddings=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return store.similarity_search(query, k=k)
