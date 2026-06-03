"""
app/logging_config.py
─────────────────────
Configure Python's standard logging once at import time.
All modules use logging.getLogger(__name__) — never print().
"""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger with a concise format.

    Call this once at application startup (main.py, FastAPI lifespan, etc.).
    Subsequent calls are idempotent because basicConfig is a no-op if handlers
    are already attached.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
