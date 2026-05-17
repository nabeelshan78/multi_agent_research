"""
Validator Agent — Research quality critic and gatekeeper node.

Uses the **fast** model tier (``llama-3.1-8b-instant``) to evaluate
whether the accumulated research sufficiently answers the user's query.
Acts purely as a critic — does NOT perform any research itself.

Routing Logic (handled by graph.py):
    - "insufficient" AND attempts < max → loop back to Research.
    - "sufficient" OR attempts >= max → proceed to Synthesis.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import settings
from state import AgentState, ResearchFinding
from utils.errors import LLMAPIError
from utils.llm import get_llm

logger = logging.getLogger(__name__)


class ValidationAssessment(BaseModel):
    """Structured output for the Validator Agent's decision."""

    validation_result: str = Field(
        description=(
            'Must be exactly "sufficient" or "insufficient".'
        ),
    )
    reasoning: str = Field(
        default="",
        description="Detailed justification for the decision.",
    )
    missing_aspects: str = Field(
        default="",
        description="Specific topics/data points still missing.",
    )


_VALIDATOR_SYSTEM_PROMPT = """\
You are a Research Quality Validator in a multi-agent business research system.

Evaluate whether gathered research data sufficiently answers the user's query.

Research is **SUFFICIENT** if:
- The core question is answered with evidence from multiple sources.
- Key data points are present and cited.
- There is enough material for a coherent summary.

Research is **INSUFFICIENT** if:
- The core question remains unanswered or only tangentially addressed.
- Critical data points are missing.
- Only one source covers the topic.

Be CRITICAL but FAIR. Consider practical utility for a business professional.
"""


def validator_agent(state: AgentState) -> dict[str, Any]:
    """Evaluate research quality and decide whether to loop or proceed.

    Calls the fast model with structured output to classify research
    as sufficient or insufficient. Increments research_attempts.

    Args:
        state: Current graph state with user_query, research_data,
            and research_attempts.

    Returns:
        Dict with validation_result, validation_reasoning,
        incremented research_attempts, and status messages.

    Raises:
        LLMAPIError: If the Groq API call fails after retries.
    """
    logger.info("Validator Agent: Evaluating research quality.")

    user_query = state["user_query"]
    research_data: list[ResearchFinding] = state.get("research_data", [])
    current_attempts = state.get("research_attempts", 0)
    new_attempts = current_attempts + 1

    messages = [SystemMessage(content=_VALIDATOR_SYSTEM_PROMPT)]

    if research_data:
        findings_text = "\n\n".join(
            f"**Source {i+1}**: {f['source']}\n"
            f"**Query**: {f['query']}\n"
            f"**Content**: {f['content'][:400]}"
            for i, f in enumerate(research_data[:12])
        )
    else:
        findings_text = "(No research data collected.)"

    messages.append(
        HumanMessage(
            content=(
                f'## Original Question\n"{user_query}"\n\n'
                f"## Attempt {new_attempts}/{settings.max_research_attempts}\n\n"
                f"## Research Data ({len(research_data)} sources)\n\n"
                f"{findings_text}\n\n"
                f"## Confidence Score: "
                f"{state.get('confidence_score', 'N/A')}/10\n\n"
                f"Validate whether this research is sufficient."
            )
        )
    )

    try:
        llm = get_llm("fast")
        structured_llm = llm.with_structured_output(ValidationAssessment)
        assessment: ValidationAssessment = structured_llm.invoke(messages)
    except Exception as exc:
        raise LLMAPIError(
            "Validator Agent: Failed to validate research.",
            cause=exc,
        ) from exc

    logger.info(
        "Validator Agent: result=%s (attempt %d/%d)",
        assessment.validation_result,
        new_attempts,
        settings.max_research_attempts,
    )

    if assessment.validation_result == "sufficient":
        status_msg = (
            f"✅ Validation (attempt {new_attempts}/"
            f"{settings.max_research_attempts}): Research SUFFICIENT."
        )
    else:
        if new_attempts >= settings.max_research_attempts:
            status_msg = (
                f"⚠️ Validation (attempt {new_attempts}/"
                f"{settings.max_research_attempts}): INSUFFICIENT but "
                f"max attempts reached. Proceeding with available data."
            )
        else:
            status_msg = (
                f"🔄 Validation (attempt {new_attempts}/"
                f"{settings.max_research_attempts}): INSUFFICIENT — "
                f"gaps: {assessment.missing_aspects or 'unspecified'}. "
                f"Routing back to Research."
            )

    return {
        "validation_result": assessment.validation_result,
        "validation_reasoning": assessment.reasoning,
        "research_attempts": new_attempts,
        "messages": [AIMessage(content=status_msg)],
    }
