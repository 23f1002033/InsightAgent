"""
app/llm.py
──────────
Provider-agnostic LLM factory.

Concept — LangChain's BaseChatModel interface
─────────────────────────────────────────────
Every LangChain chat model (OpenAI, Anthropic, Gemini, Ollama, …) inherits from
BaseChatModel and exposes the same .invoke() / .stream() / .bind_tools() API.
This means the rest of InsightAgent never imports from langchain_openai or
langchain_anthropic directly — it just calls get_llm() and treats the result
as a BaseChatModel.  Swapping providers = one env var change.

Trade-off: the factory pattern adds a tiny indirection layer, but it makes
testing trivially easy (mock get_llm to return a fake model) and future
provider additions non-breaking.

Supported providers
───────────────────
  openai    → ChatOpenAI (gpt-4o-mini by default)
  anthropic → ChatAnthropic (claude-3-5-haiku by default)
  gemini    → ChatGoogleGenerativeAI (gemini-2.0-flash by default)
"""
from __future__ import annotations

import logging
import os

from langchain_core.language_models import BaseChatModel

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Instantiate and return the configured chat LLM.

    Parameters
    ----------
    temperature:
        Sampling temperature.  0.0 → deterministic / reproducible, which
        is recommended for RAG so the model stays grounded in retrieved text.

    Returns
    -------
    BaseChatModel
        A ready-to-use LangChain chat model.

    Raises
    ------
    ValueError
        If the configured provider is unknown or the API key is missing.
    """
    settings = get_settings()

    # Optionally propagate LangSmith tracing env vars
    if settings.langchain_tracing_v2:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)
        logger.info("LangSmith tracing ENABLED (project=%s)", settings.langchain_project)

    provider = settings.llm_provider
    logger.debug("Building LLM: provider=%s model=%s", provider, settings.active_model())

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=temperature,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=temperature,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        )

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file.\n"
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=temperature,
            google_api_key=settings.gemini_api_key,  # type: ignore[arg-type]
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. Must be 'openai', 'anthropic', or 'gemini'."
    )
