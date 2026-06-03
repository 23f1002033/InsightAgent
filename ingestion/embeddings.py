"""
ingestion/embeddings.py
───────────────────────
Embedding model factory for InsightAgent.

Concept — LangChain Embeddings interface
─────────────────────────────────────────
Every LangChain embedding class inherits from langchain_core.embeddings.Embeddings
and exposes:
  • embed_documents(texts: List[str]) → List[List[float]]
  • embed_query(text: str)            → List[float]

The distinction matters:
  • embed_documents is for ingestion (batch, may be cached/parallelised).
  • embed_query is for retrieval (single vector, used at query time).

The vector store only ever sees the Embeddings interface, so swapping models
(e.g., OpenAI → Gemini) never touches vector_store.py.

Trade-offs
──────────
• OpenAIEmbeddings ("text-embedding-3-small")
    Pro:  Best quality, 1536-dim, no local GPU needed.
    Con:  API cost per ingestion; data leaves your environment.
• GoogleGenerativeAIEmbeddings ("models/text-embedding-004")
    Pro:  Free tier available, 768-dim, same API key as Gemini LLM.
    Con:  Data goes to Google; 768-dim vs 1536-dim for OpenAI.
• HuggingFaceEmbeddings ("all-MiniLM-L6-v2")
    Pro:  Runs locally, free, data stays on-prem.
    Con:  Lower quality for domain-specific text; first call downloads model (~90 MB).
• Dimension consistency: once you embed documents with model A, ALL future
  queries must also use model A.  Changing models requires re-ingesting.
"""
from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_embeddings() -> Embeddings:
    """
    Return the configured embedding model.

    Reads EMBEDDING_PROVIDER from settings:
      "openai"       → OpenAIEmbeddings (text-embedding-3-small by default)
      "gemini"       → GoogleGenerativeAIEmbeddings (text-embedding-004 by default)
      "huggingface"  → HuggingFaceEmbeddings (all-MiniLM-L6-v2 by default)

    Returns
    -------
    Embeddings
        A ready-to-use LangChain Embeddings instance.

    Raises
    ------
    ValueError
        If EMBEDDING_PROVIDER is not recognized or the required API key is missing.
    """
    settings = get_settings()
    provider = settings.embedding_provider
    logger.debug("Building embeddings: provider=%s", provider)

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set — required for OpenAI embeddings."
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set — required for Gemini embeddings.\n"
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,  # type: ignore[arg-type]
        )

    if provider == "huggingface":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "HuggingFace embeddings require 'sentence-transformers'. "
                "Run: pip install sentence-transformers"
            ) from exc

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("Loading HuggingFace model '%s' (may download ~90 MB).", model_name)
        return HuggingFaceEmbeddings(model_name=model_name)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER='{provider}'. "
        "Must be 'openai', 'gemini', or 'huggingface'."
    )
