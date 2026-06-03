"""
agent/graph.py
──────────────
LangGraph StateGraph definition for InsightAgent (Phase 4).

The graph implements the classic ReAct (Reason + Act) loop:
  agent node  — calls LLM with bound tools; decides what to do next
  tools node  — executes the chosen tool (retrieve_documents / web_search)
  loop        — continues until the LLM produces a final answer (no tool call)

State schema:
  messages   — conversation history (HumanMessage, AIMessage, ToolMessage)
  citations  — accumulated source metadata for the current turn
"""
from __future__ import annotations

# Phase 4 implementation — placeholder
