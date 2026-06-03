"""
ingestion
─────────
Public API for the InsightAgent ingestion pipeline.

Typical usage:
    from ingestion import load_source, split_documents, ingest_documents
"""
from ingestion.loaders import load_directory, load_source
from ingestion.splitter import split_documents
from ingestion.embeddings import get_embeddings
from ingestion.vector_store import get_vector_store, ingest_documents, similarity_search

__all__ = [
    "load_source",
    "load_directory",
    "split_documents",
    "get_embeddings",
    "get_vector_store",
    "ingest_documents",
    "similarity_search",
]
