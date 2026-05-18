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

def clarity_agent(state: AgentState) -> dict[str, Any]:
    """Evaluate whether the user's query is clear enough for research.

    Runs an iterative clarification loop capped at
    ``MAX_CLARIFICATION_ATTEMPTS`` rounds:

    1. Builds a prompt from conversation history + the current query.
    2. Calls the **fast** Groq model with structured output to classify
       the query as ``"clear"`` or ``"needs_clarification"``.
    3. If ``"clear"`` → exits the loop immediately and returns.
    4. If ``"needs_clarification"`` → calls ``interrupt()`` to pause
       the graph, surfaces the question to the user via the CLI, merges
       their response into the running query, then **loops back to
       step 1** for a fresh LLM re-evaluation.
    5. If the cap is reached without a clear assessment, the loop breaks
       and proceeds with the best enriched query available, preventing
       an infinite interrupt cycle.

    Args:
        state: The current graph state containing ``messages``,
            ``user_query``, and conversation history.

    Returns:
        A dict of state updates with ``clarity_status``, ``user_query``,
        and ``messages``.

    Raises:
        LLMAPIError: If the Groq API call fails after all retries.
    """
    # Maximum number of clarification rounds before forcing a proceed.
    MAX_CLARIFICATION_ATTEMPTS: int = 3

    # Working copy of the query — updated in-place on each loop iteration.
    current_query: str = state["user_query"]

    # Accumulate all messages produced during the loop so they are
    # returned as a single batch (the add_messages reducer merges them).
    outbound_messages: list = []

    llm = get_llm("fast")
    structured_llm = llm.with_structured_output(ClarityAssessment)

    for attempt in range(1, MAX_CLARIFICATION_ATTEMPTS + 1):
        logger.info(
            "Clarity Agent: Re-evaluation attempt %d/%d — query: %r",
            attempt,
            MAX_CLARIFICATION_ATTEMPTS,
            current_query,
        )

        # ── Build message payload ────────────────────────────────────
        messages = [SystemMessage(content=_CLARITY_SYSTEM_PROMPT)]

        # Include trimmed conversation history for full context.
        history = state.get("messages", [])
        if len(history) > settings.max_conversation_messages:
            history = history[-settings.max_conversation_messages:]
        messages.extend(history)

        # Any clarification messages accumulated within this loop also go
        # in so the model sees the full in-loop dialogue.
        messages.extend(outbound_messages)

        # Append the current (possibly enriched) query as the evaluation target.
        messages.append(
            HumanMessage(
                content=(
                    f"Please evaluate the following user query for clarity:\n\n"
                    f'"{current_query}"'
                )
            )
        )

        # ── Call LLM with structured output ──────────────────────────
        try:
            assessment: ClarityAssessment = structured_llm.invoke(messages)
        except Exception as exc:
            raise LLMAPIError(
                f"Clarity Agent: Failed to evaluate query clarity "
                f"(attempt {attempt}/{MAX_CLARIFICATION_ATTEMPTS}).",
                cause=exc,
            ) from exc

        logger.info(
            "Clarity Agent: attempt=%d status=%s reasoning=%s",
            attempt,
            assessment.clarity_status,
            assessment.reasoning,
        )

        # ── Clear → exit loop immediately ────────────────────────────
        if assessment.clarity_status == "clear":
            outbound_messages.append(
                AIMessage(
                    content=(
                        f"✅ Query assessed as clear "
                        f"(attempt {attempt}/{MAX_CLARIFICATION_ATTEMPTS}). "
                        f"Proceeding with research on: {current_query}"
                    )
                )
            )
            return {
                "clarity_status": "clear",
                "clarification_question": "",
                "user_query": current_query,
                "messages": outbound_messages,
            }

        # ── Needs clarification — have we hit the cap? ────────────────
        if attempt == MAX_CLARIFICATION_ATTEMPTS:
            logger.warning(
                "Clarity Agent: Max clarification attempts (%d) reached. "
                "Proceeding with enriched query: %r",
                MAX_CLARIFICATION_ATTEMPTS,
                current_query,
            )
            outbound_messages.append(
                AIMessage(
                    content=(
                        f"⚠️ Maximum clarification attempts "
                        f"({MAX_CLARIFICATION_ATTEMPTS}) reached. "
                        f"Proceeding with the best available query: {current_query}"
                    )
                )
            )
            break

        # ── Needs clarification — interrupt and wait for user ─────────
        logger.info(
            "Clarity Agent: Requesting clarification (attempt %d/%d) — %s",
            attempt,
            MAX_CLARIFICATION_ATTEMPTS,
            assessment.clarification_question,
        )

        # Pause the graph. The CLI detects this interrupt, displays the
        # question, collects user input, and resumes with
        # Command(resume=user_input). The return value of interrupt()
        # is whatever string the CLI passes back.
        user_clarification: str = interrupt(
            {
                "type": "clarification_needed",
                "question": assessment.clarification_question,
                "reasoning": assessment.reasoning,
            }
        )

        logger.info(
            "Clarity Agent: Resumed after attempt %d with clarification: %r",
            attempt,
            user_clarification,
        )

        # Merge the clarification into the running query, then loop back
        # for a fresh LLM re-evaluation — do NOT assume it is now clear.
        current_query = (
            f"{current_query} — Additional context: {user_clarification}"
        )
        outbound_messages.extend([
            HumanMessage(content=f"Clarification: {user_clarification}"),
            AIMessage(
                content=(
                    f"🔄 Clarification received (attempt {attempt}/"
                    f"{MAX_CLARIFICATION_ATTEMPTS}). "
                    f"Re-evaluating updated query: {current_query}"
                )
            ),
        ])

    # ── Exited loop via cap — proceed with whatever query we have ─────
    return {
        "clarity_status": "clear",
        "clarification_question": "",
        "user_query": current_query,
        "messages": outbound_messages,
    }
