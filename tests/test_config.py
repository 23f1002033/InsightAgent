"""
tests/test_config.py
────────────────────
Phase 1 tests: validate that Settings loads correctly and the LLM factory
raises meaningful errors when keys are missing.

These tests use monkeypatch so they never touch your real .env file.
"""
from __future__ import annotations

import pytest

from app.config import Settings, get_settings


class TestSettings:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should resolve sensible defaults without any env vars."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        get_settings.cache_clear()

        s = Settings(_env_file=None)  # type: ignore[call-arg]  # skip .env entirely
        assert s.llm_provider == "openai"
        assert s.openai_model == "gpt-4o-mini"
        assert s.chroma_persist_dir == "./data/chroma"
        assert s.langchain_tracing_v2 is False
        get_settings.cache_clear()

    def test_provider_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting LLM_PROVIDER=anthropic should be reflected in Settings."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        get_settings.cache_clear()

        s = Settings()
        assert s.llm_provider == "anthropic"
        assert s.active_llm_key() == "test-key"

    def test_invalid_provider(self) -> None:
        """An unrecognised provider should fail pydantic validation."""
        with pytest.raises(Exception):
            Settings(llm_provider="cohere")  # type: ignore[arg-type]

    def test_active_model_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        get_settings.cache_clear()
        s = Settings()
        assert s.active_model() == "gpt-4o"

    def test_active_model_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-5")
        get_settings.cache_clear()
        s = Settings()
        assert s.active_model() == "claude-opus-4-5"


class TestLLMFactory:
    def test_missing_openai_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_llm() should raise ValueError when the API key is absent."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        get_settings.cache_clear()

        from app.llm import get_llm

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_llm()

    def test_missing_anthropic_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        get_settings.cache_clear()

        from app.llm import get_llm

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            get_llm()

    def test_missing_gemini_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_llm() should raise ValueError when GEMINI_API_KEY is absent."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        get_settings.cache_clear()

        from app.llm import get_llm

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            get_llm()

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_llm() should raise ValueError for an unknown provider string."""
        # Bypass pydantic validation by patching settings post-construction
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "dummy")
        get_settings.cache_clear()

        from unittest.mock import patch

        from app import llm as llm_module

        with patch.object(
            llm_module.get_settings(), "llm_provider", "cohere"
        ):
            # Re-import to pick up patch
            with pytest.raises((ValueError, AttributeError)):
                llm_module.get_llm()


class TestGeminiSettings:
    def test_gemini_provider_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should load gemini fields correctly."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
        monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        get_settings.cache_clear()

        s = Settings()
        assert s.llm_provider == "gemini"
        assert s.gemini_api_key == "AIza-test-key"
        assert s.gemini_model == "gemini-2.0-flash"
        assert s.active_llm_key() == "AIza-test-key"
        assert s.active_model() == "gemini-2.0-flash"
        get_settings.cache_clear()

    def test_gemini_embedding_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """embedding_provider=gemini should be accepted by Settings."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-key")
        get_settings.cache_clear()

        s = Settings()
        assert s.embedding_provider == "gemini"
        get_settings.cache_clear()
