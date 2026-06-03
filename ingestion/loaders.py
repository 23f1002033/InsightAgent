"""
ingestion/loaders.py
────────────────────
Document loading layer for InsightAgent.

Concept — LangChain BaseLoader / BaseDocumentLoader
────────────────────────────────────────────────────
Every loader in LangChain inherits from BaseLoader and exposes two methods:
  • load()       → List[Document]   (load everything at once)
  • lazy_load()  → Iterator[Document]  (stream large files)

A Document is simply:
  Document(page_content: str, metadata: dict)

The metadata dict is where we track provenance: source filename, page number,
URL, etc.  These travel all the way to the final answer and become the
inline citations shown to the user.

Trade-offs
──────────
• PyPDFLoader vs UnstructuredPDFLoader:
    PyPDF is fast, pure-Python, splits by page.
    Unstructured handles complex layouts (tables, columns) but is slower and
    has more C dependencies.  We default to PyPDF; swap by changing one line.
• WebBaseLoader uses requests + BeautifulSoup — fine for public pages.
  For JavaScript-heavy sites, consider PlaywrightURLLoader instead.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── Individual loaders ────────────────────────────────────────────────────────

def load_pdf(path: Path) -> List[Document]:
    """
    Load a PDF file, one Document per page.

    Each document's metadata will contain:
      source: absolute path string
      page:   0-indexed page number
    """
    from langchain_community.document_loaders import PyPDFLoader

    logger.debug("Loading PDF: %s", path)
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    # Normalise source to a consistent key
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata["file_type"] = "pdf"
    return docs


def load_text(path: Path) -> List[Document]:
    """
    Load a plain-text (.txt) or Markdown (.md) file as a single Document.

    Trade-off: TextLoader loads the whole file at once — fine for typical
    knowledge-base documents (<1 MB).  For multi-GB logs, use lazy_load().
    """
    from langchain_community.document_loaders import TextLoader

    logger.debug("Loading text/md: %s", path)
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata["file_type"] = path.suffix.lstrip(".") or "txt"
    return docs


def load_url(url: str) -> List[Document]:
    """
    Fetch and parse a web page.

    Uses WebBaseLoader (requests + BeautifulSoup).  Only the visible text
    is extracted, so navigation chrome and scripts are stripped.

    Trade-off: Some sites block headless requests.  If you hit 403s, switch to
    SeleniumURLLoader or PlaywrightURLLoader (requires extra deps).
    """
    from langchain_community.document_loaders import WebBaseLoader

    logger.debug("Loading URL: %s", url)
    loader = WebBaseLoader(web_paths=[url])
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", url)
        doc.metadata["file_type"] = "url"
    return docs


# ── Dispatcher ────────────────────────────────────────────────────────────────

def load_source(source: str) -> List[Document]:
    """
    Unified entry point: accepts a file path or a URL string.

    Parameters
    ----------
    source:
        Absolute or relative path to a .pdf, .txt, or .md file,
        OR an http(s):// URL.

    Returns
    -------
    List[Document]
        Loaded documents with populated metadata.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If a local path does not exist.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source)

    path = Path(source).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in (".txt", ".md", ".markdown"):
        return load_text(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}' for {path}. "
            "Supported: .pdf, .txt, .md"
        )


def load_directory(directory: str, glob: str = "**/*") -> List[Document]:
    """
    Recursively load all supported files from a directory.

    Parameters
    ----------
    directory:
        Path to a folder containing documents.
    glob:
        Glob pattern relative to the directory.  Default loads everything
        recursively; use '*.pdf' to limit to PDFs.

    Returns
    -------
    List[Document]
        All documents from all matched files.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    supported = {".pdf", ".txt", ".md", ".markdown"}
    docs: List[Document] = []

    for file_path in sorted(root.glob(glob)):
        if file_path.is_file() and file_path.suffix.lower() in supported:
            try:
                docs.extend(load_source(str(file_path)))
                logger.info("Loaded %s", file_path.name)
            except Exception as exc:
                logger.warning("Skipping %s: %s", file_path, exc)

    logger.info("Loaded %d document(s) from %s", len(docs), root)
    return docs
