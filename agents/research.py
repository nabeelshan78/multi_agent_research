"""
Research Agent — Tool-calling node with Tavily search integration.

Uses the **reasoning** model tier (``llama-3.3-70b-versatile``) to:

1. Analyse the user's query and conversation history.
2. Generate targeted search queries.
3. Execute searches via the Tavily tool.
4. Collect and deduplicate findings.
5. Self-evaluate a ``confidence_score`` (0–10) reflecting how well
   the gathered data answers the user's question.

The agent runs a manual tool-calling loop (rather than
``create_react_agent``) for full control over:
    - The number of search iterations
    - Result deduplication
    - Confidence self-evaluation as a separate structured call

Cost Rationale:
    This agent needs deep reasoning to formulate good search queries
    and synthesise findings, so it uses ``llama-3.3-70b-versatile`` on Groq Cloud.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from config import settings
from state import AgentState, ResearchFinding
from utils.errors import LLMAPIError, SearchToolError
from utils.llm import get_llm, get_search_tool

logger = logging.getLogger(__name__)


# ── Structured output schemas ───────────────────────────────────────────────

class SearchQueryPlan(BaseModel):
    """LLM-generated plan of search queries to execute.

    Attributes:
        queries: A list of 2-4 targeted search query strings designed
            to cover different facets of the user's research question.
        reasoning: Brief explanation of why these queries were chosen.
    """

    queries: list[str] = Field(
        description=(
            "A list of 2-4 specific, targeted search queries to execute. "
            "Each query should cover a different angle of the user's question. "
            "Use professional business/financial terminology."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the search strategy.",
    )


class ConfidenceEvaluation(BaseModel):
    """Self-evaluation of research quality.

    Attributes:
        confidence_score: A score from 0 to 10 indicating how well
            the gathered research answers the user's question.
        reasoning: Explanation of the score.
        gaps: Identified information gaps, if any.
    """

    confidence_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Score from 0-10 indicating research completeness. "
            "0 = no relevant data found, "
            "5 = partial coverage with gaps, "
            "10 = comprehensive coverage of the topic."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Explanation of the confidence score.",
    )
    gaps: str = Field(
        default="",
        description="Identified gaps in the research, if any.",
    )


# ── System prompts ──────────────────────────────────────────────────────────

_QUERY_PLANNING_PROMPT = """\
You are a Senior Business Research Analyst. Your task is to generate precise, \
targeted search queries for a business research question.

## Guidelines
- Generate 2-4 search queries that cover DIFFERENT angles of the question.
- Use specific business terminology, company names, and industry jargon.
- Include queries for recent data (financials, news, market trends).
- If the conversation history provides context, use it to refine queries.
- Avoid redundant or overly broad queries.
"""

_CONFIDENCE_EVAL_PROMPT = """\
You are a Research Quality Evaluator. Assess how well the gathered research \
data answers the user's original question.

## Scoring Guidelines
- **0-2**: Almost no relevant information found.
- **3-4**: Some relevant information but major gaps remain.
- **5-6**: Moderate coverage — core question partially answered.
- **7-8**: Good coverage — most aspects addressed with sources.
- **9-10**: Comprehensive — the research thoroughly answers the question \
with multiple supporting sources.

Be HONEST and CRITICAL in your assessment. Do not inflate scores.
"""


# ── Node function ────────────────────────────────────────────────────────────

def research_agent(state: AgentState) -> dict[str, Any]:
    """Execute research via Tavily search and self-evaluate confidence.

    This is a LangGraph node function that performs a multi-step
    research workflow:

    1. **Query Planning**: Uses the reasoning LLM to generate 2-4
       targeted search queries.
    2. **Search Execution**: Runs each query through the Tavily tool,
       collecting results and deduplicating by URL.
    3. **Confidence Evaluation**: Uses a separate structured LLM call
       to score the research quality on a 0-10 scale.

    The results are appended to (not replacing) the existing
    ``research_data`` in state, supporting iterative deepening across
    multiple research cycles.

    Args:
        state: The current graph state.  Must contain ``user_query``
            and ``messages``.

    Returns:
        A dict of state updates including ``research_data``,
        ``confidence_score``, and appended ``messages``.

    Raises:
        LLMAPIError: If the Groq API call fails after retries.
        SearchToolError: If Tavily search execution fails.
    """
    logger.info("Research Agent: Starting research cycle.")

    user_query = state["user_query"]
    existing_data: list[ResearchFinding] = state.get("research_data", [])
    attempt_num = state.get("research_attempts", 0) + 1

    # ── Step 1: Generate search queries ──────────────────────────────────
    query_plan = _generate_search_queries(state)
    logger.info(
        "Research Agent: Generated %d queries — %s",
        len(query_plan.queries),
        query_plan.reasoning,
    )

    # ── Step 2: Execute searches ─────────────────────────────────────────
    new_findings = _execute_searches(query_plan.queries)
    logger.info(
        "Research Agent: Collected %d new findings.", len(new_findings)
    )

    # Merge with existing data, deduplicating by source URL
    seen_sources = {item["source"] for item in existing_data}
    for finding in new_findings:
        if finding["source"] not in seen_sources:
            existing_data.append(finding)
            seen_sources.add(finding["source"])

    # ── Step 3: Self-evaluate confidence ─────────────────────────────────
    evaluation = _evaluate_confidence(user_query, existing_data)
    logger.info(
        "Research Agent: Confidence = %.1f/10 — %s",
        evaluation.confidence_score,
        evaluation.reasoning,
    )

    # ── Build status message ─────────────────────────────────────────────
    status_msg = (
        f"🔍 Research cycle {attempt_num} complete. "
        f"Found {len(new_findings)} new sources "
        f"({len(existing_data)} total). "
        f"Confidence: {evaluation.confidence_score:.1f}/10."
    )
    if evaluation.gaps:
        status_msg += f"\n   Gaps identified: {evaluation.gaps}"

    return {
        "research_data": existing_data,
        "confidence_score": evaluation.confidence_score,
        "messages": [AIMessage(content=status_msg)],
    }


# ── Internal helpers ─────────────────────────────────────────────────────────

def _generate_search_queries(state: AgentState) -> SearchQueryPlan:
    """Use the reasoning LLM to plan search queries.

    Args:
        state: Current graph state with query and conversation history.

    Returns:
        A ``SearchQueryPlan`` with 2-4 targeted queries.

    Raises:
        LLMAPIError: On API failure.
    """
    messages = [SystemMessage(content=_QUERY_PLANNING_PROMPT)]

    # Add trimmed history for context
    history = state.get("messages", [])
    if len(history) > settings.max_conversation_messages:
        history = history[-settings.max_conversation_messages :]
    messages.extend(history)

    # Add existing research summary to avoid redundant queries
    existing_data = state.get("research_data", [])
    if existing_data:
        existing_summary = "\n".join(
            f"- [{f['source']}]: {f['content'][:150]}..."
            for f in existing_data[:10]
        )
        messages.append(
            HumanMessage(
                content=(
                    f"I already have the following research. "
                    f"Generate NEW queries that fill gaps:\n\n"
                    f"{existing_summary}"
                )
            )
        )

    messages.append(
        HumanMessage(
            content=(
                f"Generate targeted search queries for this research question:\n\n"
                f'"{state["user_query"]}"'
            )
        )
    )

    try:
        llm = get_llm("reasoning")
        structured_llm = llm.with_structured_output(SearchQueryPlan)
        return structured_llm.invoke(messages)
    except Exception as exc:
        raise LLMAPIError(
            "Research Agent: Failed to generate search queries.",
            cause=exc,
        ) from exc


def _execute_searches(queries: list[str]) -> list[ResearchFinding]:
    """Execute Tavily searches for each query and collect findings.

    Args:
        queries: List of search query strings.

    Returns:
        A list of ``ResearchFinding`` dicts with source, content, and
        query fields.

    Note:
        Individual query failures are logged and skipped rather than
        crashing the entire research cycle.  This ensures partial
        results are still useful.
    """
    findings: list[ResearchFinding] = []

    try:
        search_tool = get_search_tool(max_results=settings.tavily_max_results)
    except SearchToolError:
        logger.error("Research Agent: Failed to initialise Tavily tool.")
        raise

    for query in queries:
        try:
            logger.info("Research Agent: Searching — '%s'", query)
            raw_results = search_tool.invoke(query)

            # TavilySearchResults returns a list of dicts or a JSON string
            if isinstance(raw_results, str):
                try:
                    raw_results = json.loads(raw_results)
                except json.JSONDecodeError:
                    # If it's plain text, wrap it as a single result
                    raw_results = [{"url": "tavily_result", "content": raw_results}]

            if isinstance(raw_results, list):
                for result in raw_results:
                    source = result.get("url", result.get("source", "unknown"))
                    content = result.get("content", str(result))
                    findings.append(
                        ResearchFinding(
                            source=source,
                            content=content,
                            query=query,
                        )
                    )
            else:
                logger.warning(
                    "Research Agent: Unexpected result type: %s",
                    type(raw_results),
                )

        except Exception as exc:
            logger.warning(
                "Research Agent: Search failed for query '%s': %s",
                query,
                exc,
            )
            # Continue with remaining queries — partial results are
            # better than no results.
            continue

    return findings


def _evaluate_confidence(
    user_query: str,
    research_data: list[ResearchFinding],
) -> ConfidenceEvaluation:
    """Self-evaluate the quality and completeness of gathered research.

    Uses a separate LLM call (reasoning tier) to critically assess
    whether the accumulated findings adequately answer the user's
    question.

    Args:
        user_query: The original research question.
        research_data: All accumulated findings across cycles.

    Returns:
        A ``ConfidenceEvaluation`` with score, reasoning, and gaps.

    Raises:
        LLMAPIError: On API failure.
    """
    # Build a condensed summary of findings for evaluation
    if not research_data:
        return ConfidenceEvaluation(
            confidence_score=0.0,
            reasoning="No research data collected.",
            gaps="All information is missing.",
        )

    findings_summary = "\n\n".join(
        f"**Source**: {f['source']}\n"
        f"**Query**: {f['query']}\n"
        f"**Content**: {f['content'][:500]}"
        for f in research_data[:15]  # Cap to avoid token overflow
    )

    messages = [
        SystemMessage(content=_CONFIDENCE_EVAL_PROMPT),
        HumanMessage(
            content=(
                f"## Original Research Question\n"
                f'"{user_query}"\n\n'
                f"## Gathered Research Data\n"
                f"{findings_summary}\n\n"
                f"Please evaluate the confidence score for this research."
            )
        ),
    ]

    try:
        llm = get_llm("reasoning", temperature=0.0)
        structured_llm = llm.with_structured_output(ConfidenceEvaluation)
        return structured_llm.invoke(messages)
    except Exception as exc:
        raise LLMAPIError(
            "Research Agent: Failed to evaluate confidence.",
            cause=exc,
        ) from exc
