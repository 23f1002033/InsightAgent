"""
agent/rag_chain.py
──────────────────
Phase 3 — Basic RAG chain built with LangChain Expression Language (LCEL).

Concept — LCEL and the pipe operator
──────────────────────────────────────
LangChain Expression Language (LCEL) lets you compose Runnables using `|`
(the pipe operator), just like Unix shell pipes:

    chain = retriever | format_context | prompt | llm | output_parser

Each component is a Runnable with:
  .invoke(input)              → single synchronous call
  .stream(input)              → token-by-token generator
  .batch(inputs)              → parallel calls
  .ainvoke / .astream         → async variants

Trade-off: LCEL is elegant for linear chains but becomes harder to read for
branching logic.  That's exactly why Phase 4 upgrades to LangGraph StateGraph —
explicit nodes and edges make complex routing legible.

Data flow in this chain
────────────────────────
  question (str)
      │
      ▼
  retriever.invoke(question) → List[Document]   # vector similarity search
      │
      ▼
  format_context(docs) → str                   # labelled context blocks
      │
      ▼ (merged with original question via RunnablePassthrough)
  RAG_PROMPT.invoke({context, question})        # ChatPromptTemplate
      │
      ▼
  llm.invoke(messages) → AIMessage              # Gemini / OpenAI / Anthropic
      │
      ▼
  StrOutputParser()    → str                    # extract .content

Public API
──────────
  build_rag_chain(llm, retriever) → Runnable
  ask(question, k=4)              → RAGResponse
  ask_stream(question, k=4)       → Iterator[str]
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from agent.prompts import RAG_PROMPT
from app.config import get_settings

logger = logging.getLogger(__name__)


# ── Response schema ────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """
    A single source reference extracted from the agent's answer.

    Attributes
    ----------
    source:      File path or URL of the document.
    chunk_index: Position of this chunk within the source document.
    file_type:   'pdf', 'txt', 'md', 'url', etc.
    snippet:     First 200 chars of the chunk content (for UI preview).
    """
    source: str
    chunk_index: int
    file_type: str
    snippet: str


@dataclass
class RAGResponse:
    """
    Structured result from a RAG chain invocation.

    Attributes
    ----------
    answer:         The LLM's answer string (with inline [source: …] markers).
    citations:      Deduplicated list of Citation objects derived from
                    the retrieved documents (not parsed from the answer text —
                    we trust the retriever's metadata, not the LLM's output).
    retrieved_docs: The raw Document objects returned by the retriever,
                    in case the caller wants full chunk text.
    """
    answer: str
    citations: List[Citation]
    retrieved_docs: List[Document] = field(default_factory=list)


# ── Context formatting ─────────────────────────────────────────────────────────

def format_context(docs: List[Document]) -> str:
    """
    Convert a list of retrieved Documents into a labelled context string.

    Each block is prefixed with its metadata so the LLM knows which source
    to cite for each passage.

    Example output:
        [Context 1] source=intro.pdf, chunk=2, type=pdf
        RAG combines retrieval with generation...

        ---

        [Context 2] source=https://example.com, chunk=0, type=url
        Large language models can hallucinate...

    Design note: we include the chunk_index in the label so the LLM can
    produce citations like [source: intro.pdf, chunk: 2] that we can later
    map back to the exact Document.
    """
    if not docs:
        return "No relevant context found in the knowledge base."

    parts: List[str] = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        ftype = doc.metadata.get("file_type", "?")
        header = f"[Context {i}] source={src}, chunk={chunk_idx}, type={ftype}"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def _docs_to_citations(docs: List[Document]) -> List[Citation]:
    """
    Build Citation objects from retrieved Document metadata.

    We derive citations from the retriever's metadata (ground truth) rather
    than parsing the LLM's answer text.  This is more reliable because:
      • The LLM might slightly mis-quote the source name.
      • Parsing regex can fail on unusual filenames.
      • We always know exactly which chunks were retrieved.
    """
    seen: set[tuple[str, int]] = set()
    citations: List[Citation] = []

    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        key = (src, chunk_idx)

        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    source=src,
                    chunk_index=chunk_idx,
                    file_type=doc.metadata.get("file_type", "?"),
                    snippet=doc.page_content[:200].replace("\n", " "),
                )
            )

    return citations


# ── Chain builder ──────────────────────────────────────────────────────────────

def build_rag_chain(
    llm: BaseChatModel,
    retriever: BaseRetriever,
) -> object:
    """
    Compose and return a basic RAG chain using LCEL.

    The chain signature:
      Input:  str  (the user's question)
      Output: str  (the LLM's grounded answer)

    Parameters
    ----------
    llm:
        Any LangChain BaseChatModel (Gemini, OpenAI, Anthropic).
    retriever:
        A VectorStoreRetriever (or any BaseRetriever).

    Returns
    -------
    Runnable[str, str]
        A composable LCEL chain.

    Concept — RunnablePassthrough
    ──────────────────────────────
    When a chain expects a dict but the input is a plain string, we use
    RunnablePassthrough to pass the original input through unchanged to
    one key while another key is computed by a different branch:

        {
            "context": retriever | format_context,  # retrieves + formats
            "question": RunnablePassthrough(),       # passes the string as-is
        }

    Both branches receive the same input (the question string).
    Their outputs are merged into {"context": "...", "question": "..."} which
    feeds the prompt template's {context} and {question} placeholders.
    """
    chain = (
        {
            "context": retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ── High-level API ─────────────────────────────────────────────────────────────

def ask(
    question: str,
    k: int = 4,
    llm: Optional[BaseChatModel] = None,
    retriever: Optional[BaseRetriever] = None,
) -> RAGResponse:
    """
    Ask a question against the knowledge base and return a structured response.

    This is the main entry point for Phase 3.  It:
      1. Retrieves the top-k most relevant chunks.
      2. Formats them into a labelled context string.
      3. Calls the LLM with the grounding prompt.
      4. Returns the answer string + structured Citation objects.

    Parameters
    ----------
    question:
        The user's natural-language question.
    k:
        Number of chunks to retrieve (default 4).
    llm:
        Optional pre-built LLM.  If None, calls get_llm() with defaults.
    retriever:
        Optional pre-built retriever.  If None, builds from the default
        Chroma collection.

    Returns
    -------
    RAGResponse
        Contains the answer string, citations list, and raw retrieved docs.
    """
    from app.llm import get_llm
    from ingestion.embeddings import get_embeddings
    from ingestion.vector_store import get_vector_store

    if llm is None:
        llm = get_llm(temperature=0.0)

    if retriever is None:
        embeddings = get_embeddings()
        store = get_vector_store(embeddings=embeddings)
        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    logger.info("RAG query: '%s' (k=%d)", question[:80], k)

    # Retrieve documents first so we can build citations independently
    retrieved_docs: List[Document] = retriever.invoke(question)
    logger.info("Retrieved %d chunk(s)", len(retrieved_docs))

    # Build the chain and invoke it
    chain = build_rag_chain(llm, retriever)
    answer: str = chain.invoke(question)

    citations = _docs_to_citations(retrieved_docs)

    logger.debug("Answer length: %d chars, citations: %d", len(answer), len(citations))

    return RAGResponse(
        answer=answer,
        citations=citations,
        retrieved_docs=retrieved_docs,
    )


def ask_stream(
    question: str,
    k: int = 4,
    llm: Optional[BaseChatModel] = None,
    retriever: Optional[BaseRetriever] = None,
) -> Iterator[str]:
    """
    Stream the answer token-by-token.

    Concept — LCEL streaming
    ─────────────────────────
    Every LCEL chain automatically supports .stream() — it propagates
    through all components.  The LLM yields AIMessageChunks; StrOutputParser
    extracts the .content string from each chunk.

    Trade-off: streaming returns just the text tokens, not the citations.
    The caller should call ask() first to retrieve citations, then call
    ask_stream() for the streaming UI experience — or retrieve docs manually.

    Yields
    ------
    str
        Partial token strings as they arrive from the LLM.
    """
    from app.llm import get_llm
    from ingestion.embeddings import get_embeddings
    from ingestion.vector_store import get_vector_store

    if llm is None:
        llm = get_llm(temperature=0.0)

    if retriever is None:
        embeddings = get_embeddings()
        store = get_vector_store(embeddings=embeddings)
        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    chain = build_rag_chain(llm, retriever)

    for chunk in chain.stream(question):
        yield chunk
