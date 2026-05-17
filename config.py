"""
Centralised application configuration for the Multi-Agent Research Assistant.

Loads environment variables from a ``.env`` file (via ``python-dotenv``),
defines application-wide constants, and re-exports the LLM / tool factory
functions for convenience so that agent modules can do::

    from config import get_llm, get_search_tool, settings

All tuneable parameters live here so operators can adjust behaviour
without touching agent logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env at import time ─────────────────────────────────────────────────
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings.

    All values are derived from environment variables or sensible
    defaults.  The ``frozen=True`` flag prevents accidental mutation
    after initialisation.

    Attributes:
        groq_api_key: API key for the Groq Cloud LLM service.
        tavily_api_key: API key for the Tavily search service.
        fast_model: Model identifier for low-latency tasks
            (Clarity Agent, Validator Agent).
        reasoning_model: Model identifier for high-capability tasks
            (Research Agent, Synthesis Agent).
        default_temperature: Default sampling temperature for LLM calls.
        max_retries: Number of retry attempts on transient API errors.
        request_timeout: Per-request timeout in seconds.
        max_research_attempts: Maximum number of research → validate
            cycles before forcing synthesis.
        confidence_threshold: Minimum ``confidence_score`` (0-10) for
            the Research Agent to bypass validation and proceed
            directly to synthesis.
        max_conversation_messages: Soft cap on conversation history
            length.  When exceeded, older messages are trimmed to
            prevent context-window overflow.
        tavily_max_results: Number of search results per Tavily query.
    """

    # ── API Keys ─────────────────────────────────────────────────────────
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    tavily_api_key: str = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", "")
    )

    # ── Model Selection ──────────────────────────────────────────────────
    fast_model: str = "llama-3.1-8b-instant"
    reasoning_model: str = "llama-3.3-70b-versatile"

    # ── LLM Parameters ──────────────────────────────────────────────────
    default_temperature: float = 0.0
    max_retries: int = 3
    request_timeout: float = 60.0

    # ── Orchestration Thresholds ─────────────────────────────────────────
    max_research_attempts: int = 3
    confidence_threshold: float = 6.0
    max_conversation_messages: int = 20

    # ── Tavily Settings ──────────────────────────────────────────────────
    tavily_max_results: int = 5


# ── Singleton instance ───────────────────────────────────────────────────────
settings = Settings()

# ── Re-exports for convenience ───────────────────────────────────────────────
from utils.llm import get_llm, get_search_tool  # noqa: E402, F401

__all__: list[str] = [
    "settings",
    "get_llm",
    "get_search_tool",
]
