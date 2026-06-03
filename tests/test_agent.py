"""
tests/test_agent.py
───────────────────
Phase 3 tests for the RAG chain.

Strategy
────────
• Every test mocks the LLM and the retriever so no API calls are made.
• We test each sub-component (format_context, _docs_to_citations) in
  isolation before testing the full chain.
• One integration-marked test runs the full chain against a temp Chroma DB
  with real Gemini embeddings + LLM (requires GEMINI_API_KEY in .env).

Run unit tests only (no API key):
    pytest tests/test_agent.py -v -m "not integration"

Run with real LLM (requires GEMINI_API_KEY):
    pytest tests/test_agent.py -v -m integration
"""
from __future__ import annotations

from typing import Iterator, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage


# ── Helpers ───────────────────────────────────────────────────────────────────

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class _FakeLLM(BaseChatModel):
    """Minimal BaseChatModel that always returns a fixed answer string."""
    answer: str = "The answer [source: doc.txt, chunk: 0]."

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.answer))]
        )

    @property
    def _llm_type(self) -> str:
        return "fake"


class _FakeRetriever(BaseRetriever):
    """BaseRetriever that returns a fixed list of Documents."""
    docs: List[Document] = []

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return self.docs


def _make_docs(
    texts: list[str] | None = None,
    source: str = "doc.txt",
    file_type: str = "txt",
) -> List[Document]:
    texts = texts or ["This is chunk zero.", "This is chunk one.", "Chunk two here."]
    return [
        Document(
            page_content=t,
            metadata={
                "source": source,
                "chunk_index": i,
                "chunk_count": len(texts),
                "file_type": file_type,
            },
        )
        for i, t in enumerate(texts)
    ]


def _fake_llm(answer: str = "The answer [source: doc.txt, chunk: 0].") -> _FakeLLM:
    return _FakeLLM(answer=answer)


def _fake_retriever(docs: List[Document]) -> _FakeRetriever:
    return _FakeRetriever(docs=docs)


# ── format_context ────────────────────────────────────────────────────────────

class TestFormatContext:
    def test_labels_each_chunk(self) -> None:
        """Each doc gets a [Context N] header with its metadata."""
        from agent.rag_chain import format_context

        docs = _make_docs(["Alpha text.", "Beta text."], source="paper.pdf")
        result = format_context(docs)

        assert "[Context 1]" in result
        assert "[Context 2]" in result
        assert "source=paper.pdf" in result
        assert "Alpha text." in result
        assert "Beta text." in result

    def test_empty_docs_returns_fallback(self) -> None:
        """Empty list should return the 'no context' fallback message."""
        from agent.rag_chain import format_context

        result = format_context([])
        assert "No relevant context" in result

    def test_chunk_index_in_header(self) -> None:
        """chunk_index from metadata must appear in the formatted header."""
        from agent.rag_chain import format_context

        docs = [
            Document(
                page_content="content",
                metadata={"source": "x.txt", "chunk_index": 7, "file_type": "txt"},
            )
        ]
        result = format_context(docs)
        assert "chunk=7" in result

    def test_separator_between_chunks(self) -> None:
        """Multiple chunks should be separated by '---'."""
        from agent.rag_chain import format_context

        docs = _make_docs(["A", "B", "C"])
        result = format_context(docs)
        assert "---" in result


# ── _docs_to_citations ────────────────────────────────────────────────────────

class TestDocsToCitations:
    def test_creates_citation_per_unique_chunk(self) -> None:
        """One Citation per unique (source, chunk_index) pair."""
        from agent.rag_chain import _docs_to_citations

        docs = _make_docs(["a", "b", "c"], source="file.pdf", file_type="pdf")
        citations = _docs_to_citations(docs)

        assert len(citations) == 3
        assert citations[0].source == "file.pdf"
        assert citations[0].chunk_index == 0
        assert citations[1].chunk_index == 1

    def test_deduplicates_same_chunk(self) -> None:
        """Duplicate (source, chunk_index) should appear only once."""
        from agent.rag_chain import _docs_to_citations

        doc = Document(
            page_content="repeated",
            metadata={"source": "a.txt", "chunk_index": 0, "file_type": "txt"},
        )
        citations = _docs_to_citations([doc, doc, doc])
        assert len(citations) == 1

    def test_snippet_truncated(self) -> None:
        """Snippet should be at most 200 characters."""
        from agent.rag_chain import _docs_to_citations

        long_text = "word " * 100  # 500 chars
        doc = Document(
            page_content=long_text,
            metadata={"source": "big.txt", "chunk_index": 0, "file_type": "txt"},
        )
        citations = _docs_to_citations([doc])
        assert len(citations[0].snippet) <= 200

    def test_url_source_preserved(self) -> None:
        """URL sources should be stored verbatim in the Citation."""
        from agent.rag_chain import _docs_to_citations

        doc = Document(
            page_content="web content",
            metadata={
                "source": "https://example.com/article",
                "chunk_index": 0,
                "file_type": "url",
            },
        )
        cit = _docs_to_citations([doc])[0]
        assert cit.source == "https://example.com/article"
        assert cit.file_type == "url"

    def test_empty_docs_returns_empty_list(self) -> None:
        from agent.rag_chain import _docs_to_citations
        assert _docs_to_citations([]) == []


# ── build_rag_chain ───────────────────────────────────────────────────────────

class TestBuildRagChain:
    def test_chain_invokes_retriever_and_llm(self) -> None:
        """build_rag_chain().invoke() should call retriever and llm."""
        from agent.rag_chain import build_rag_chain

        docs = _make_docs(["RAG is a technique."], source="intro.txt")
        chain = build_rag_chain(_fake_llm(), _fake_retriever(docs))
        result = chain.invoke("What is RAG?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_returns_string(self) -> None:
        """The output of the chain must be a plain string."""
        from agent.rag_chain import build_rag_chain

        docs = _make_docs(["Some content."])
        chain = build_rag_chain(
            llm=_fake_llm("The answer."),
            retriever=_fake_retriever(docs),
        )
        result = chain.invoke("test question")
        assert isinstance(result, str)


# ── ask() ─────────────────────────────────────────────────────────────────────

class TestAsk:
    def test_returns_rag_response(self) -> None:
        """ask() should return a RAGResponse with answer + citations."""
        from agent.rag_chain import ask, RAGResponse

        docs = _make_docs(
            ["RAG grounds LLM answers in documents."],
            source="rag.txt",
        )
        response = ask(
            "What is RAG?",
            llm=_fake_llm("RAG grounds answers [source: rag.txt, chunk: 0]."),
            retriever=_fake_retriever(docs),
        )

        assert isinstance(response, RAGResponse)
        assert "RAG" in response.answer
        assert len(response.citations) == 1
        assert response.citations[0].source == "rag.txt"
        assert len(response.retrieved_docs) == 1

    def test_empty_kb_returns_response_with_no_citations(self) -> None:
        """When the retriever returns nothing, citations should be empty."""
        from agent.rag_chain import ask

        response = ask(
            "unknown topic",
            llm=_fake_llm("I don't have enough information."),
            retriever=_fake_retriever([]),
        )

        assert isinstance(response.answer, str)
        assert response.citations == []
        assert response.retrieved_docs == []

    def test_multiple_sources_produce_multiple_citations(self) -> None:
        """Docs from different sources should produce one citation each."""
        from agent.rag_chain import ask

        docs = [
            Document(
                page_content="Content from A.",
                metadata={"source": "a.pdf", "chunk_index": 0, "file_type": "pdf"},
            ),
            Document(
                page_content="Content from B.",
                metadata={"source": "b.pdf", "chunk_index": 0, "file_type": "pdf"},
            ),
        ]
        response = ask(
            "Tell me about A and B",
            llm=_fake_llm("Answer with both sources."),
            retriever=_fake_retriever(docs),
        )

        sources = {c.source for c in response.citations}
        assert "a.pdf" in sources
        assert "b.pdf" in sources


# ── ask_stream() ──────────────────────────────────────────────────────────────

class TestAskStream:
    def test_stream_yields_strings(self) -> None:
        """ask_stream() should yield non-empty string chunks."""
        from agent.rag_chain import ask_stream

        docs = _make_docs(["streaming content"])

        with patch("agent.rag_chain.build_rag_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.stream.return_value = iter(["Hello", " world", "!"])
            mock_build.return_value = mock_chain

            tokens = list(
                ask_stream(
                    "test",
                    llm=_fake_llm(),
                    retriever=_fake_retriever(docs),
                )
            )

        assert tokens == ["Hello", " world", "!"]
        assert all(isinstance(t, str) for t in tokens)


# ── Prompt template ───────────────────────────────────────────────────────────

class TestRAGPrompt:
    def test_prompt_renders_context_and_question(self) -> None:
        """RAG_PROMPT should format {context} and {question} placeholders."""
        from agent.prompts import RAG_PROMPT

        messages = RAG_PROMPT.format_messages(
            context="This is the retrieved context.",
            question="What does the context say?",
        )

        # Should produce at least two messages: system + human
        assert len(messages) >= 2

        full_text = " ".join(m.content for m in messages)
        assert "This is the retrieved context." in full_text
        assert "What does the context say?" in full_text

    def test_prompt_contains_citation_instructions(self) -> None:
        """System message must include the citation format instructions."""
        from agent.prompts import RAG_PROMPT

        messages = RAG_PROMPT.format_messages(
            context="ctx", question="q"
        )
        system_content = messages[0].content
        assert "[source:" in system_content
        assert "chunk:" in system_content


# ── Integration (real Gemini + Chroma) ───────────────────────────────────────

@pytest.mark.integration
class TestRAGIntegration:
    """
    Full end-to-end: ingest docs → ask() → get grounded answer.
    Requires GEMINI_API_KEY in .env.

    Run: pytest -m integration
    """

    def test_full_rag_pipeline(self, tmp_path) -> None:
        import os

        # Write a small knowledge base
        doc = tmp_path / "ml_basics.txt"
        doc.write_text(
            "Retrieval-Augmented Generation (RAG) is a framework that combines "
            "dense retrieval with sequence-to-sequence generation. "
            "It was proposed by Lewis et al. in 2020 at Facebook AI Research. "
            "RAG retrieves the top-k relevant passages from an external knowledge "
            "base and conditions the language model on those passages, reducing "
            "hallucinations compared to open-domain question answering."
            * 3,  # repeat to produce multiple chunks
            encoding="utf-8",
        )

        from ingestion.loaders import load_source
        from ingestion.splitter import split_documents
        from ingestion.vector_store import ingest_documents, get_vector_store
        from ingestion.embeddings import get_embeddings
        from app.llm import get_llm
        from agent.rag_chain import ask

        embeddings = get_embeddings()
        chunks = split_documents(load_source(str(doc)), chunk_size=300, chunk_overlap=30)
        chroma_dir = str(tmp_path / "chroma")
        ingest_documents(chunks, embeddings=embeddings, persist_directory=chroma_dir,
                         collection_name="integration_test")

        store = get_vector_store(embeddings=embeddings, persist_directory=chroma_dir,
                                 collection_name="integration_test")
        retriever = store.as_retriever(search_kwargs={"k": 3})
        llm = get_llm(temperature=0.0)

        response = ask("Who proposed RAG and when?", llm=llm, retriever=retriever)

        assert response.answer
        assert len(response.citations) > 0
        # The answer should mention Lewis or 2020
        assert any(word in response.answer for word in ["Lewis", "2020", "Facebook"])
