"""
scripts/hello_llm.py
─────────────────────
Phase 1 smoke-test: Proves the LLM factory works end-to-end.

Run:
    python scripts/hello_llm.py
    python scripts/hello_llm.py --provider anthropic
    python scripts/hello_llm.py --stream

Concepts demonstrated
─────────────────────
* HumanMessage / AIMessage — LangChain's universal message schema.
  Every chat model speaks this protocol, so your code is provider-agnostic.
* BaseChatModel.invoke()   — synchronous single-turn call.
* BaseChatModel.stream()   — token-by-token streaming via Python generator.
"""
from __future__ import annotations

import argparse
import os
import sys

# ── make sure the project root is on the path ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.llm import get_llm
from app.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="InsightAgent Phase-1 smoke test")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default=None,
        help="Override LLM_PROVIDER from .env",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming output instead of a single response",
    )
    args = parser.parse_args()

    # Override provider via CLI without touching .env
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
        get_settings.cache_clear()  # invalidate cached settings singleton

    setup_logging(get_settings().log_level)

    settings = get_settings()
    print(f"\n{'─'*60}")
    print(f"  InsightAgent — Phase 1 Smoke Test")
    print(f"  Provider : {settings.llm_provider}")
    print(f"  Model    : {settings.active_model()}")
    print(f"{'─'*60}\n")

    llm = get_llm(temperature=0.0)

    messages = [
        SystemMessage(
            content=(
                "You are InsightAgent, a concise knowledge assistant. "
                "Respond in 2–3 sentences max."
            )
        ),
        HumanMessage(
            content=(
                "What is Retrieval-Augmented Generation (RAG) and why is it "
                "useful for enterprise knowledge bases?"
            )
        ),
    ]

    if args.stream:
        print("Streaming response:\n")
        # stream() returns an iterator of AIMessageChunk objects.
        # Each chunk has a .content attribute with the partial token string.
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
        print("\n")
    else:
        # invoke() returns a single AIMessage with the full response.
        response = llm.invoke(messages)
        print("Response:\n")
        print(response.content)
        print()
        # LangChain attaches usage metadata when the provider supports it.
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata
            tokens = meta.get("token_usage") or meta.get("usage", {})
            if tokens:
                print(f"Token usage: {tokens}")

    print(f"\n{'─'*60}")
    print("  ✅  Phase 1 complete — LLM factory is working!")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
