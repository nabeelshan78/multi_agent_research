"""
LLM client factory and search tool instantiation.

Provides two factory functions that centralise all provider configuration
so that agent modules never construct clients directly.  This makes it
trivial to swap providers, adjust retry policies, or add observability
hooks in a single place.

Dual-Model Strategy:
    - **fast**:      ``llama-3.1-8b-instant`` — low-latency, cost-efficient.
                     Used by Clarity and Validator agents (binary classification).
    - **reasoning**: ``llama-3.3-70b-versatile`` — high-capability, deeper reasoning.
                     Used by Research and Synthesis agents.
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

from utils.errors import LLMAPIError, SearchToolError


# ── Model registry ───────────────────────────────────────────────────────────
_MODEL_REGISTRY: dict[str, str] = {
    "fast": "llama-3.1-8b-instant",
    "reasoning": "llama-3.3-70b-versatile",
}

# Type alias for the two supported model tiers.
ModelTier = Literal["fast", "reasoning"]


def _validate_env_var(name: str) -> str:
    """Return the value of an environment variable or raise clearly.

    Args:
        name: The environment variable name to look up.

    Returns:
        The non-empty string value of the variable.

    Raises:
        LLMAPIError: If the variable is missing or empty.
    """
    value = os.getenv(name, "").strip()
    if not value:
        raise LLMAPIError(
            f"Environment variable '{name}' is not set. "
            f"Please add it to your .env file."
        )
    return value


def get_llm(
    tier: ModelTier = "reasoning",
    *,
    temperature: float = 0.0,
    max_retries: int = 3,
    timeout: float | None = 60.0,
) -> ChatGroq:
    """Create and return a configured ``ChatGroq`` instance.

    This factory selects the appropriate Groq-hosted model based on
    the requested *tier* and wires up retry / timeout settings
    suitable for production workloads.

    Args:
        tier: Model tier to use.

            - ``"fast"`` → ``llama-3.1-8b-instant`` (Clarity & Validator).
            - ``"reasoning"`` → ``llama-3.3-70b-versatile`` (Research & Synthesis).

        temperature: Sampling temperature.  Defaults to ``0.0`` for
            deterministic output (important for routing decisions).
        max_retries: Number of automatic retries on transient failures
            (e.g. 429 rate-limit, 5xx server errors).
        timeout: Request timeout in seconds.  ``None`` disables the
            timeout entirely.

    Returns:
        A fully configured ``ChatGroq`` chat model instance.

    Raises:
        LLMAPIError: If the ``GROQ_API_KEY`` environment variable is
            missing or empty.

    Example::

        llm_fast = get_llm("fast")
        llm_reasoning = get_llm("reasoning", temperature=0.3)
    """
    api_key = _validate_env_var("GROQ_API_KEY")

    model_name = _MODEL_REGISTRY[tier]

    try:
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            api_key=api_key,
        )
    except Exception as exc:
        raise LLMAPIError(
            f"Failed to instantiate ChatGroq with model '{model_name}': {exc}",
            cause=exc,
        ) from exc


def get_search_tool(max_results: int = 5) -> TavilySearchResults:
    """Create and return a configured Tavily search tool.

    The tool is designed to be bound to a ``ChatGroq`` instance via
    ``.bind_tools()`` for the Research Agent's tool-calling workflow.

    Args:
        max_results: Maximum number of search results to return per
            query.  Defaults to ``5`` to balance breadth vs. token cost.

    Returns:
        A ``TavilySearchResults`` tool instance ready for LangChain
        tool-calling integration.

    Raises:
        SearchToolError: If the ``TAVILY_API_KEY`` environment variable
            is missing or empty, or if tool instantiation fails.

    Example::

        tool = get_search_tool(max_results=3)
        llm_with_tools = get_llm("reasoning").bind_tools([tool])
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise SearchToolError(
            "Environment variable 'TAVILY_API_KEY' is not set. "
            "Please add it to your .env file."
        )

    try:
        return TavilySearchResults(
            max_results=max_results,
            tavily_api_key=api_key,
        )
    except Exception as exc:
        raise SearchToolError(
            f"Failed to instantiate TavilySearchResults: {exc}",
            cause=exc,
        ) from exc
