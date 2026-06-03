"""
agent/memory.py
───────────────
Conversation memory / checkpointer factory (Phase 5).

LangGraph checkpointers persist conversation state between turns.
The state is keyed by thread_id, so each user session is independent.

Dev  → MemorySaver   (in-process, lost on restart — fine for development)
Prod → PostgresSaver (persistent, survives restarts — requires a Postgres URL)
"""
from __future__ import annotations

# Phase 5 implementation — placeholder
