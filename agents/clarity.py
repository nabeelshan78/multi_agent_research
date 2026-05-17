"""
Clarity Agent — Query comprehension and disambiguation node.

Uses the **fast** model tier (``llama-3.1-8b-instant``) to perform a
binary classification on the user's query:

    - **"clear"**: The query is specific enough to proceed with research.
    - **"needs_clarification"**: The query is vague, ambiguous, or
      missing critical context.

When clarification is needed, the agent triggers a LangGraph
``interrupt()`` to pause the graph and yield control to the user
via the CLI.  Upon resumption, the user's clarification is merged
into the conversation history and the clarity check re-runs.

Cost Rationale:
    This is a classification task — no deep reasoning required.
    Using ``llama-3.1-8b-instant`` on Groq Cloud keeps latency and cost minimal.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from config import settings
from state import AgentState
from utils.errors import LLMAPIError
from utils.llm import get_llm

logger = logging.getLogger(__name__)


# ── Structured output schema ────────────────────────────────────────────────

class ClarityAssessment(BaseModel):
    """Structured output schema for the Clarity Agent's decision.

    Attributes:
        clarity_status: Whether the query is ``"clear"`` (ready for
            research) or ``"needs_clarification"`` (requires user input).
        clarification_question: A specific, actionable question to ask
            the user.  Empty string when ``clarity_status`` is ``"clear"``.
        reasoning: Brief internal reasoning for the decision, used
            for logging and debugging.
    """

    clarity_status: str = Field(
        description=(
            'Must be exactly "clear" or "needs_clarification". '
            '"clear" means the query is specific enough for business research. '
            '"needs_clarification" means the query is vague or ambiguous.'
        ),
    )
    clarification_question: str = Field(
        default="",
        description=(
            "A specific, helpful question to ask the user to clarify their "
            "query. Leave empty if clarity_status is 'clear'."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Brief internal reasoning for the clarity decision.",
    )


# ── System prompt ────────────────────────────────────────────────────────────

_CLARITY_SYSTEM_PROMPT = """\
You are a Clarity Assessment Specialist in a multi-agent business research system.

Your SOLE job is to evaluate whether the user's query is clear and specific enough \
to conduct meaningful business research.

## Evaluation Criteria

A query is **CLEAR** if it:
- Identifies a specific company, industry, market, or business topic
- Has a discernible research objective (e.g., competitive analysis, market sizing, \
financial performance, trend analysis)
- Provides enough context to formulate targeted search queries

A query **NEEDS CLARIFICATION** if it:
- Is extremely vague (e.g., "tell me about business")
- References unnamed entities (e.g., "that company", "the stock")
- Could refer to multiple unrelated topics without disambiguation
- Lacks any actionable research direction

## Important Guidelines

- Be GENEROUS in your assessment. If the query has a clear subject and implied \
research goal, mark it as "clear" even if it could be more specific.
- Only flag "needs_clarification" when the query is genuinely too vague to research.
- When asking for clarification, be specific about WHAT information you need.
- Consider the full conversation history for context — a short follow-up like \
"what about their competitors?" is CLEAR if the prior context identifies the subject.
"""


# ── Node function ────────────────────────────────────────────────────────────

def clarity_agent(state: AgentState) -> dict[str, Any] | Command:
    """Evaluate whether the user's query is clear enough for research.

    This is a LangGraph node function that:

    1. Builds a prompt from the conversation history + current query.
    2. Calls the **fast** Groq model with structured output to get a
       binary classification.
    3. If ``"clear"`` → returns updated state fields to proceed.
    4. If ``"needs_clarification"`` → calls ``interrupt()`` to pause
       the graph.  When the user provides input and the graph resumes,
       the clarification is injected and the node re-evaluates.

    Args:
        state: The current graph state containing ``messages``,
            ``user_query``, and conversation history.

    Returns:
        A dict of state updates (when clear) or a ``Command`` (after
        interrupt resumption).

    Raises:
        LLMAPIError: If the Groq API call fails after all retries.
    """
    logger.info("Clarity Agent: Evaluating query clarity.")

    # ── Build message payload ────────────────────────────────────────────
    messages = [SystemMessage(content=_CLARITY_SYSTEM_PROMPT)]

    # Include trimmed conversation history for context
    history = state.get("messages", [])
    if len(history) > settings.max_conversation_messages:
        history = history[-settings.max_conversation_messages :]

    messages.extend(history)

    # Append the current query as explicit evaluation target
    messages.append(
        HumanMessage(
            content=(
                f"Please evaluate the following user query for clarity:\n\n"
                f'"{state["user_query"]}"'
            )
        )
    )

    # ── Call LLM with structured output ──────────────────────────────────
    try:
        llm = get_llm("fast")
        structured_llm = llm.with_structured_output(ClarityAssessment)
        assessment: ClarityAssessment = structured_llm.invoke(messages)
    except Exception as exc:
        raise LLMAPIError(
            "Clarity Agent: Failed to evaluate query clarity.",
            cause=exc,
        ) from exc

    logger.info(
        "Clarity Agent: status=%s, reasoning=%s",
        assessment.clarity_status,
        assessment.reasoning,
    )

    # ── Route based on assessment ────────────────────────────────────────
    if assessment.clarity_status == "clear":
        return {
            "clarity_status": "clear",
            "clarification_question": "",
            "messages": [
                AIMessage(
                    content=(
                        f"✅ Query assessed as clear. "
                        f"Proceeding with research on: {state['user_query']}"
                    )
                )
            ],
        }

    # ── Needs clarification → interrupt ──────────────────────────────────
    logger.info(
        "Clarity Agent: Requesting clarification — %s",
        assessment.clarification_question,
    )

    # Pause the graph and surface the question to the user.
    # The CLI will catch this interrupt, display the question,
    # collect user input, and resume with Command(resume=user_input).
    user_clarification: str = interrupt(
        {
            "type": "clarification_needed",
            "question": assessment.clarification_question,
            "reasoning": assessment.reasoning,
        }
    )

    # ── Resumed after interrupt — update state with clarification ────────
    logger.info("Clarity Agent: Resumed with user clarification.")

    # Build an enriched query that combines original + clarification
    enriched_query = (
        f"{state['user_query']} — Additional context: {user_clarification}"
    )

    return {
        "clarity_status": "clear",
        "clarification_question": "",
        "user_query": enriched_query,
        "messages": [
            HumanMessage(content=f"Clarification: {user_clarification}"),
            AIMessage(
                content=(
                    f"✅ Clarification received. Updated research query: "
                    f"{enriched_query}"
                )
            ),
        ],
    }
