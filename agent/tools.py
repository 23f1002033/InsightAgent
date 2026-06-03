"""
agent/tools.py
──────────────
LangGraph tool definitions for InsightAgent (Phase 4).

Tools the agent can call:
  retrieve_documents(query) — vector similarity search over the user's KB
  web_search(query)         — live Tavily search for info not in the KB

Each tool is decorated with @tool so LangChain can bind it to the LLM
and parse structured arguments automatically.
"""
from __future__ import annotations

# Phase 4 implementation — placeholder
