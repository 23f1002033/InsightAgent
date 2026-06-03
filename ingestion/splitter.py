"""
ingestion/splitter.py
─────────────────────
Text splitting layer for InsightAgent.

Concept — RecursiveCharacterTextSplitter
─────────────────────────────────────────
LangChain's RecursiveCharacterTextSplitter tries to split on a priority-ordered
list of separators:  ["\\n\\n", "\\n", " ", ""]

  1. First it splits on paragraph breaks (\\n\\n) — preserving semantic units.
  2. If a chunk is still too large, it falls back to single newlines.
  3. Then spaces (word boundaries).
  4. Finally, hard character splits (last resort).

This hierarchy means we get the largest meaningful chunks possible, rather than
mechanically cutting every N characters.

Key parameters
──────────────
chunk_size:     Maximum characters per chunk.
chunk_overlap:  Characters shared between adjacent chunks.
                Prevents a sentence that straddles a boundary from being lost.

Trade-offs
──────────
• Larger chunk_size → richer context per retrieval, but higher token cost and
  more chance of irrelevant content diluting the answer.
• Smaller chunk_size → more precise retrieval, but the model may miss context
  that spans multiple paragraphs.
• chunk_overlap of 10–15 % of chunk_size is a common sweet spot.
• For code files, use Language.PYTHON / Language.MARKDOWN splitter variants.
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Defaults tuned for typical knowledge-base prose.
# Override via kwargs in split_documents() or set in config.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def get_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter with the given parameters.

    Returns a ready-to-use splitter.  Factored out so tests and callers
    can easily create custom splitters without reimplementing the logic.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks suitable for embedding.

    Each output chunk inherits the parent document's metadata plus two new keys:
      chunk_index:  0-based position of this chunk within the source document.
      chunk_count:  Total number of chunks produced from that source document.

    These metadata fields survive into the vector store and surface as
    citation detail in the final answer.

    Parameters
    ----------
    documents:
        Raw documents from the loading layer.
    chunk_size:
        Maximum characters per chunk (default 1000).
    chunk_overlap:
        Overlap between adjacent chunks (default 150 chars ≈ 15 %).

    Returns
    -------
    List[Document]
        Chunked documents ready to embed.
    """
    if not documents:
        logger.warning("split_documents called with an empty document list.")
        return []

    splitter = get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Enrich metadata: add chunk_index relative to source
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = source_counters.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        source_counters[src] = idx + 1

    # Second pass: fill in total chunk count per source
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_count"] = source_counters[src]

    logger.info(
        "Split %d document(s) → %d chunk(s) "
        "(chunk_size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
