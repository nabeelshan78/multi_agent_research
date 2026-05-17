"""
Custom exception hierarchy for the Multi-Agent Business Research Assistant.

Provides granular, typed exceptions for each failure domain so that calling
code can catch and handle errors precisely rather than relying on broad
``Exception`` catches.

Exception Tree::

    ResearchAssistantError (base)
    ├── LLMAPIError          — Groq Cloud API failures
    ├── SearchToolError      — Tavily search failures
    └── StateValidationError — Invalid state transitions or missing fields
"""

from __future__ import annotations


class ResearchAssistantError(Exception):
    """Base exception for all research assistant errors.

    All custom exceptions in this package inherit from this class,
    allowing callers to catch the entire family with a single handler
    when broad error handling is acceptable.
    """

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        """Initialize with a human-readable message and optional root cause.

        Args:
            message: Descriptive error message.
            cause: The original exception that triggered this error, if any.
        """
        super().__init__(message)
        self.cause = cause


class LLMAPIError(ResearchAssistantError):
    """Raised when a call to the Groq Cloud LLM API fails.

    This covers network timeouts, rate-limit (429) responses, malformed
    responses, and authentication failures.  The ``cause`` attribute
    preserves the original provider-level exception for debugging.

    Example::

        try:
            response = llm.invoke(messages)
        except Exception as exc:
            raise LLMAPIError(
                "Groq API call failed after retries",
                cause=exc,
            ) from exc
    """


class SearchToolError(ResearchAssistantError):
    """Raised when the Tavily search tool fails to return results.

    Covers network errors, invalid API keys, quota exhaustion, and
    unexpected response shapes from the Tavily API.

    Example::

        try:
            results = tavily_tool.invoke(query)
        except Exception as exc:
            raise SearchToolError(
                "Tavily search failed",
                cause=exc,
            ) from exc
    """


class StateValidationError(ResearchAssistantError):
    """Raised when the graph state contains invalid or inconsistent data.

    Examples include missing required fields, out-of-range confidence
    scores, or unexpected routing flag values.  This exception is
    primarily used as a guardrail during development and testing.

    Example::

        if state["confidence_score"] < 0 or state["confidence_score"] > 10:
            raise StateValidationError(
                f"confidence_score must be 0-10, got {state['confidence_score']}"
            )
    """
