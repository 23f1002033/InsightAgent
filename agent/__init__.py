"""
agent/__init__.py
─────────────────
Public API for the InsightAgent agentic layer.

Phase 3 exports:
  ask()        — single-turn RAG query → RAGResponse
  ask_stream() — streaming token generator

Phase 4+ will add:
  run_agent()  — LangGraph agentic loop with web_search
"""
from agent.rag_chain import RAGResponse, Citation, ask, ask_stream

__all__ = [
    "RAGResponse",
    "Citation",
    "ask",
    "ask_stream",
]
