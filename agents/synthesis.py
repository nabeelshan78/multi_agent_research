"""
Synthesis Agent — Final report generation node.

Uses the **reasoning** model tier (``llama-3.3-70b-versatile``) to consume all
accumulated research data and conversation history, then produce a
structured, highly readable business research report.

Output Structure:
    - Executive Summary
    - Key Findings (with source citations)
    - Detailed Analysis
    - Recommendations / Next Steps
    - Sources

This is the terminal node — its output is the primary deliverable
presented to the user.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import settings
from state import AgentState, ResearchFinding
from utils.errors import LLMAPIError
from utils.llm import get_llm

logger = logging.getLogger(__name__)


_SYNTHESIS_SYSTEM_PROMPT = """\
You are a Senior Business Research Analyst producing a final research report.

## Report Structure

Produce a well-formatted report with the following sections:

### 📋 Executive Summary
A 2-3 sentence overview answering the user's core question.

### 🔍 Key Findings
Bullet points of the most important discoveries, each with a source citation \
in [brackets].

### 📊 Detailed Analysis
A coherent narrative synthesizing the research. Connect data points, identify \
patterns, and provide context. Use subheadings if covering multiple aspects.

### 💡 Recommendations
Actionable next steps or considerations based on the findings.

### 📚 Sources
A numbered list of all sources referenced.

## Guidelines
- Write in a professional, clear tone suitable for business stakeholders.
- Always cite sources using [Source N] notation.
- Highlight data points: numbers, percentages, dates.
- If data is incomplete or contradictory, acknowledge limitations honestly.
- Use markdown formatting for readability.
- Keep the report concise but comprehensive (aim for 400-800 words).
"""


def synthesis_agent(state: AgentState) -> dict[str, Any]:
    """Generate the final structured research report.

    Consumes all research data, conversation history, and validation
    feedback to produce a comprehensive business research report.

    Args:
        state: The current graph state with user_query,
            research_data, messages, and validation context.

    Returns:
        Dict with final_report string and appended messages.

    Raises:
        LLMAPIError: If the Groq API call fails after retries.
    """
    logger.info("Synthesis Agent: Generating final report.")

    user_query = state["user_query"]
    research_data: list[ResearchFinding] = state.get("research_data", [])
    attempts = state.get("research_attempts", 0)
    confidence = state.get("confidence_score", 0.0)

    # ── Build source material ────────────────────────────────────────
    if research_data:
        sources_text = "\n\n".join(
            f"**[Source {i+1}]** {f['source']}\n"
            f"Search Query: {f['query']}\n"
            f"Content: {f['content'][:600]}"
            for i, f in enumerate(research_data[:15])
        )
        source_list = "\n".join(
            f"{i+1}. {f['source']}"
            for i, f in enumerate(research_data[:15])
        )
    else:
        sources_text = "(No research data available.)"
        source_list = "(No sources.)"

    # ── Build synthesis prompt ───────────────────────────────────────
    messages = [SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPT)]

    # Include trimmed conversation for multi-turn context
    history = state.get("messages", [])
    if len(history) > settings.max_conversation_messages:
        history = history[-settings.max_conversation_messages:]
    messages.extend(history)

    messages.append(
        HumanMessage(
            content=(
                f"## Research Question\n"
                f'"{user_query}"\n\n'
                f"## Research Metadata\n"
                f"- Research cycles: {attempts}\n"
                f"- Confidence score: {confidence:.1f}/10\n"
                f"- Total sources: {len(research_data)}\n\n"
                f"## Available Source Material\n\n"
                f"{sources_text}\n\n"
                f"## Source List for Citations\n"
                f"{source_list}\n\n"
                f"Please synthesize a comprehensive research report."
            )
        )
    )

    # ── Call reasoning LLM ───────────────────────────────────────────
    try:
        llm = get_llm("reasoning", temperature=0.3)
        response = llm.invoke(messages)
        report = response.content
    except Exception as exc:
        raise LLMAPIError(
            "Synthesis Agent: Failed to generate report.",
            cause=exc,
        ) from exc

    logger.info(
        "Synthesis Agent: Report generated (%d chars).", len(report)
    )

    return {
        "final_report": report,
        "messages": [AIMessage(content=report)],
    }
