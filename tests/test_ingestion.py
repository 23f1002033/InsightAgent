"""
tests/test_ingestion.py
───────────────────────
Phase 2 tests for the ingestion pipeline.

Strategy
────────
• We test each layer in isolation (unit tests) and one end-to-end flow.
• Chroma and embedding calls are mocked so tests run without API keys
  and without writing to disk.
• We write one real integration test that uses HuggingFace local embeddings
  with a temporary Chroma directory — mark it with @pytest.mark.integration
  so it's skipped in CI unless explicitly opted in.

Run all tests (mocked):
    pytest tests/test_ingestion.py -v

Run including integration (needs no API key — uses local HF embeddings):
    pytest tests/test_ingestion.py -v -m integration
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_docs(n: int = 3, source: str = "test.txt") -> List[Document]:
    """Create n dummy Documents with the given source."""
    return [
        Document(
            page_content=f"Paragraph {i}: " + ("word " * 50),
            metadata={"source": source},
        )
        for i in range(n)
    ]


# ── Loaders ───────────────────────────────────────────────────────────────────

class TestLoaders:
    def test_load_txt_file(self, tmp_path: Path) -> None:
        """load_source() should return one Document per .txt file."""
        f = tmp_path / "sample.txt"
        f.write_text("Hello InsightAgent!\nThis is a test document.", encoding="utf-8")

        from ingestion.loaders import load_source
        docs = load_source(str(f))

        assert len(docs) == 1
        assert "InsightAgent" in docs[0].page_content
        assert docs[0].metadata["source"] == str(f)
        assert docs[0].metadata["file_type"] == "txt"

    def test_load_md_file(self, tmp_path: Path) -> None:
        """load_source() should accept .md files."""
        f = tmp_path / "README.md"
        f.write_text("# Title\n\nSome content here.", encoding="utf-8")

        from ingestion.loaders import load_source
        docs = load_source(str(f))

        assert len(docs) >= 1
        assert docs[0].metadata["file_type"] == "md"

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Unsupported file types should raise ValueError."""
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3", encoding="utf-8")

        from ingestion.loaders import load_source
        with pytest.raises(ValueError, match="Unsupported"):
            load_source(str(f))

    def test_missing_file_raises(self) -> None:
        """Non-existent paths should raise FileNotFoundError."""
        from ingestion.loaders import load_source
        with pytest.raises(FileNotFoundError):
            load_source("/non/existent/file.txt")

    def test_load_directory(self, tmp_path: Path) -> None:
        """load_directory() should recursively find all supported files."""
        (tmp_path / "a.txt").write_text("Document A", encoding="utf-8")
        (tmp_path / "b.md").write_text("Document B", encoding="utf-8")
        (tmp_path / "ignore.csv").write_text("not loaded", encoding="utf-8")

        from ingestion.loaders import load_directory
        docs = load_directory(str(tmp_path))

        assert len(docs) == 2
        sources = {doc.metadata["source"] for doc in docs}
        assert any("a.txt" in s for s in sources)
        assert any("b.md" in s for s in sources)

    def test_load_url_mocked(self) -> None:
        """load_url() should call WebBaseLoader and tag metadata correctly."""
        fake_doc = Document(
            page_content="Page content from a website.",
            metadata={"source": "https://example.com"},
        )
        with patch(
            "langchain_community.document_loaders.WebBaseLoader.load",
            return_value=[fake_doc],
        ):
            from ingestion.loaders import load_url
            docs = load_url("https://example.com")

        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "url"


# ── Splitter ──────────────────────────────────────────────────────────────────

class TestSplitter:
    def test_basic_split(self) -> None:
        """Large docs should be split into multiple chunks."""
        from ingestion.splitter import split_documents

        # 5000-char doc will exceed chunk_size=1000
        big_doc = Document(
            page_content="word " * 1000,  # ~5000 chars
            metadata={"source": "big.txt"},
        )
        chunks = split_documents([big_doc], chunk_size=1000, chunk_overlap=100)
        assert len(chunks) > 1

    def test_chunk_metadata_injected(self) -> None:
        """Each chunk should have chunk_index and chunk_count metadata."""
        from ingestion.splitter import split_documents

        doc = Document(page_content="word " * 500, metadata={"source": "test.txt"})
        chunks = split_documents([doc], chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "chunk_count" in chunk.metadata

    def test_chunk_indices_are_sequential(self) -> None:
        """chunk_index values for the same source should be sequential from 0."""
        from ingestion.splitter import split_documents

        doc = Document(page_content="sentence. " * 300, metadata={"source": "seq.txt"})
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_input_returns_empty(self) -> None:
        """split_documents([]) should return [] without raising."""
        from ingestion.splitter import split_documents
        result = split_documents([])
        assert result == []

    def test_small_doc_not_split(self) -> None:
        """A doc smaller than chunk_size should produce exactly one chunk."""
        from ingestion.splitter import split_documents

        doc = Document(page_content="Short content.", metadata={"source": "tiny.txt"})
        chunks = split_documents([doc], chunk_size=1000, chunk_overlap=0)
        assert len(chunks) == 1

    def test_original_metadata_preserved(self) -> None:
        """Source metadata from the original doc must survive splitting."""
        from ingestion.splitter import split_documents

        doc = Document(
            page_content="word " * 500,
            metadata={"source": "orig.pdf", "page": 3, "file_type": "pdf"},
        )
        chunks = split_documents([doc], chunk_size=300, chunk_overlap=30)
        for chunk in chunks:
            assert chunk.metadata["source"] == "orig.pdf"
            assert chunk.metadata["page"] == 3


# ── Embeddings ────────────────────────────────────────────────────────────────

class TestEmbeddings:
    def test_openai_embeddings_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_embeddings() with provider=openai should return OpenAIEmbeddings."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        from app.config import get_settings
        get_settings.cache_clear()

        from ingestion.embeddings import get_embeddings
        from langchain_openai import OpenAIEmbeddings

        emb = get_embeddings()
        assert isinstance(emb, OpenAIEmbeddings)
        get_settings.cache_clear()

    def test_missing_openai_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing OPENAI_API_KEY with provider=openai should raise ValueError."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "")

        from app.config import get_settings
        get_settings.cache_clear()

        from ingestion.embeddings import get_embeddings
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_embeddings()
        get_settings.cache_clear()

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unknown EMBEDDING_PROVIDER should raise ValueError from get_embeddings."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        from app.config import get_settings
        get_settings.cache_clear()

        from ingestion import embeddings as emb_module
        with patch.object(
            get_settings(), "embedding_provider", "cohere"
        ):
            with pytest.raises(ValueError):
                emb_module.get_embeddings()
        get_settings.cache_clear()


# ── Vector Store ──────────────────────────────────────────────────────────────

class TestVectorStore:
    def test_ingest_and_search_mocked(self) -> None:
        """
        End-to-end ingestion → search with mocked Chroma.

        We patch Chroma so no actual embeddings are computed and no disk
        writes occur.  This makes the test fast and API-key-free.
        """
        from ingestion.vector_store import ingest_documents, similarity_search

        docs = _make_docs(3)
        expected_result = [docs[0]]

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = expected_result

        with patch("ingestion.vector_store.get_vector_store", return_value=mock_store):
            store = ingest_documents(docs)
            mock_store.add_documents.assert_called_once_with(docs)

    def test_empty_ingest_does_not_crash(self) -> None:
        """ingest_documents([]) should not raise."""
        from ingestion.vector_store import ingest_documents

        mock_store = MagicMock()
        with patch("ingestion.vector_store.get_vector_store", return_value=mock_store):
            ingest_documents([])
            mock_store.add_documents.assert_not_called()

    def test_similarity_search_mocked(self) -> None:
        """similarity_search() should call vector store's similarity_search."""
        from ingestion.vector_store import similarity_search

        expected = [Document(page_content="relevant chunk", metadata={"source": "x.txt"})]
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = expected

        with patch("ingestion.vector_store.get_vector_store", return_value=mock_store):
            results = similarity_search("test query", k=2)

        mock_store.similarity_search.assert_called_once_with("test query", k=2)
        assert results == expected


# ── Integration Test (skipped by default) ─────────────────────────────────────

@pytest.mark.integration
class TestIntegration:
    """
    Full pipeline with real HuggingFace embeddings and a temp Chroma dir.
    No API key needed — runs locally.

    Run with: pytest -m integration
    """

    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Load → split → embed → store → retrieve."""
        # Write a small test document
        doc_path = tmp_path / "rag_intro.txt"
        doc_path.write_text(
            "Retrieval-Augmented Generation (RAG) combines a retrieval "
            "system with a language model to produce grounded answers. "
            "The retrieval step fetches relevant context from a knowledge base. "
            "The generation step uses that context to produce an accurate response. "
            "RAG reduces hallucinations by anchoring the model in real documents. "
            * 5,  # repeat to have enough text for splitting
            encoding="utf-8",
        )

        from ingestion.loaders import load_source
        from ingestion.splitter import split_documents
        from ingestion.vector_store import ingest_documents, similarity_search
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        docs = load_source(str(doc_path))
        chunks = split_documents(docs, chunk_size=300, chunk_overlap=30)
        assert len(chunks) > 1

        ingest_documents(
            chunks,
            embeddings=embeddings,
            persist_directory=str(tmp_path / "chroma"),
        )

        results = similarity_search(
            "What is RAG and how does it reduce hallucinations?",
            k=2,
            embeddings=embeddings,
            persist_directory=str(tmp_path / "chroma"),
        )
        assert len(results) > 0
        assert any("RAG" in r.page_content for r in results)
