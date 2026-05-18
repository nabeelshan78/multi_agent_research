"""
Interactive CLI entry point for the Multi-Agent Research Assistant.

Provides a terminal-based interface that:

1. Initialises the LangGraph state graph with checkpointing.
2. Manages multi-turn conversation sessions via thread IDs.
3. Handles human-in-the-loop interrupts gracefully — detecting
   graph pauses, prompting the user for clarification, and
   resuming with ``Command(resume=...)``.
4. Displays agent status updates and final reports with formatted
   terminal output.

Usage::

    python main.py
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import build_graph

# ── Logging configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
# Suppress noisy HTTP logs from httpx / httpcore
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("main")


# ── Terminal formatting helpers ──────────────────────────────────────────────

_SEPARATOR = "═" * 72
_THIN_SEP = "─" * 72

# ANSI colour codes for terminal output
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _print_banner() -> None:
    """Print the application banner on startup."""
    print(f"\n{_CYAN}{_BOLD}{_SEPARATOR}")
    print("   🔬  Multi-Agent Business Research Assistant")
    print(f"   Powered by Groq Cloud + Tavily + LangGraph")
    print(f"{_SEPARATOR}{_RESET}")
    print(
        f"{_DIM}   Type your research query and press Enter."
    )
    print(f"   Type 'quit' or 'exit' to end the session.")
    print(f"   Type 'new' to start a new research thread.{_RESET}\n")


def _print_agent_event(node_name: str, data: dict[str, Any]) -> None:
    """Print a formatted agent status update.

    Args:
        node_name: Name of the agent node that produced this event.
        data: The state update dict from the node.
    """
    # Extract message content if present
    messages = data.get("messages", [])
    for msg in messages:
        content = getattr(msg, "content", str(msg))
        colour = {
            "clarity_agent": _CYAN,
            "research_agent": _MAGENTA,
            "validator_agent": _YELLOW,
            "synthesis_agent": _GREEN,
        }.get(node_name, _RESET)

        label = node_name.replace("_", " ").title()
        print(f"\n{colour}{_BOLD}[{label}]{_RESET}")
        print(f"{colour}{content}{_RESET}")


def _print_final_report(report: str) -> None:
    """Print the final research report with formatting.

    Args:
        report: The synthesised report text.
    """
    print(f"\n{_GREEN}{_BOLD}{_SEPARATOR}")
    print("   📊  FINAL RESEARCH REPORT")
    print(f"{_SEPARATOR}{_RESET}\n")
    print(report)
    print(f"\n{_GREEN}{_THIN_SEP}{_RESET}")


def _print_error(message: str) -> None:
    """Print an error message in red.

    Args:
        message: The error message to display.
    """
    print(f"\n{_RED}{_BOLD}[Error]{_RESET} {_RED}{message}{_RESET}")


# ── Core execution logic ────────────────────────────────────────────────────

def _create_initial_state(user_query: str) -> dict[str, Any]:
    """Create the initial state dict for a new graph invocation.

    Args:
        user_query: The user's research question.

    Returns:
        A dict matching the ``AgentState`` schema with all fields
        initialised to sensible defaults.
    """
    return {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "clarity_status": "",
        "clarification_question": "",
        "research_data": [],
        "confidence_score": 0.0,
        "research_attempts": 0,
        "validation_result": "",
        "validation_reasoning": "",
        "final_report": "",
    }


def _run_graph_with_interrupt_handling(
    graph: Any,
    input_data: dict[str, Any] | Command | None,
    config: dict[str, Any],
) -> str | None:
    """Stream the graph, handle interrupts, and return the final report.

    Implements a reliable interrupt detection strategy:

    1. Stream graph events, printing agent updates as they arrive.
    2. When an ``__interrupt__`` event key appears **during the stream**,
       capture its payload immediately into ``captured_interrupt``.
       This works on every round of the clarity agent's clarification
       loop — not just the first — because the payload is read from the
       live stream rather than from ``graph_state.tasks`` after the fact.
    3. After the stream exhausts, use ``captured_interrupt`` (primary) or
       fall back to ``graph_state.tasks`` inspection (secondary safety net)
       to resolve the interrupt payload.
    4. Display the clarification prompt, collect user input, and resume
       with ``Command(resume=clarification)``.
    5. Repeat until the graph completes or the user exits.

    Args:
        graph: The compiled LangGraph state graph.
        input_data: Initial state dict, a ``Command`` for resumption,
            or ``None`` to continue from the last checkpoint.
        config: LangGraph config dict with ``thread_id``.

    Returns:
        The final report string if the graph completed, or ``None``
        if the user interrupted the session.
    """
    while True:
        # ── Stream events ────────────────────────────────────────────
        final_report = None
        # Reset on every pass — populated inline from the __interrupt__ key.
        # This is the primary detection mechanism and works for all rounds
        # of the clarity agent's iterative clarification loop.
        captured_interrupt: dict | None = None

        try:
            for event in graph.stream(input_data, config, stream_mode="updates"):
                # Each event is {node_name: state_update_value}
                for node_name, node_data in event.items():

                    if node_name == "__interrupt__":
                        # In stream_mode="updates", node_data for __interrupt__
                        # is a tuple/list of Interrupt objects.  Pull the dict
                        # payload from the first element's .value attribute.
                        try:
                            if node_data:
                                first = node_data[0]
                                payload = (
                                    first.value
                                    if hasattr(first, "value")
                                    else first
                                )
                                if isinstance(payload, dict):
                                    captured_interrupt = payload
                                    logger.debug(
                                        "Interrupt captured from stream: %s",
                                        captured_interrupt,
                                    )
                        except (IndexError, TypeError, AttributeError) as exc:
                            logger.warning(
                                "Could not parse __interrupt__ payload: %s", exc
                            )
                        # Never pass interrupt events to _print_agent_event.
                        continue

                    _print_agent_event(node_name, node_data)

                    # Capture the final report when synthesis completes.
                    if node_name == "synthesis_agent":
                        report = node_data.get("final_report", "")
                        if report:
                            final_report = report

        except Exception as exc:
            _print_error(f"Graph execution error: {exc}")
            logger.exception("Graph execution failed.")
            return None

        # ── Resolve interrupt payload ─────────────────────────────────
        # Primary: payload captured live from the stream above.
        interrupt_data: dict | None = captured_interrupt

        # Always fetch graph_state — needed both for the fallback and for
        # the completion check (graph_state.next == () means finished).
        graph_state = graph.get_state(config)

        # Fallback: inspect tasks only when the live stream missed the payload.
        # This covers edge cases (e.g. future LangGraph version changes).
        if interrupt_data is None and graph_state.next:
            if hasattr(graph_state, "tasks") and graph_state.tasks:
                for task in graph_state.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        for intr in task.interrupts:
                            val = intr.value if hasattr(intr, "value") else intr
                            if isinstance(val, dict):
                                interrupt_data = val
                                break
                    if interrupt_data:
                        break

            if interrupt_data:
                logger.debug(
                    "Interrupt payload resolved via graph_state.tasks fallback."
                )

        # ── Handle interrupt or completion ────────────────────────────
        if interrupt_data and isinstance(interrupt_data, dict):
            question = interrupt_data.get(
                "question", "Could you please clarify your query?"
            )
            reasoning = interrupt_data.get("reasoning", "")

            print(f"\n{_YELLOW}{_BOLD}{_THIN_SEP}")
            print(f"   CLARIFICATION NEEDED")
            print(f"{_THIN_SEP}{_RESET}")
            if reasoning:
                print(f"{_DIM}   Reason: {reasoning}{_RESET}")
            print(f"\n{_YELLOW}   {question}{_RESET}\n")

            try:
                clarification = input(
                    f"{_BOLD}Your clarification ▶ {_RESET}"
                ).strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{_DIM}Session ended.{_RESET}")
                return None

            if not clarification:
                print(f"{_DIM}No input provided. Using original query.{_RESET}")
                clarification = "Please proceed with the original query."

            # Resume the graph — this string becomes the return value of
            # interrupt() inside the clarity agent's loop.
            input_data = Command(resume=clarification)
            continue  # Loop back to stream the resumed graph

        elif graph_state.next:
            # Graph is paused but we found no interrupt payload anywhere —
            # unexpected state; nudge the graph forward rather than hanging.
            logger.warning(
                "Graph paused at %s but no interrupt payload found. Nudging forward.",
                graph_state.next,
            )
            input_data = Command(resume="Please proceed.")
            continue

        else:
            # Graph has completed normally.
            if final_report:
                _print_final_report(final_report)
            return final_report


# ── Main interaction loop ────────────────────────────────────────────────────

def main() -> None:
    """Run the interactive CLI session.

    Initialises the graph, manages multi-turn conversation threads,
    and handles user input in a loop until the user exits.
    """
    _print_banner()

    # ── Build graph ──────────────────────────────────────────────────
    try:
        graph = build_graph()
        logger.info("Graph initialised successfully.")
    except Exception as exc:
        _print_error(f"Failed to build graph: {exc}")
        logger.exception("Graph build failed.")
        sys.exit(1)

    # ── Session management ───────────────────────────────────────────
    thread_id = str(uuid.uuid4())
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id}
    }

    print(
        f"{_DIM}   Session ID: {thread_id[:8]}...{_RESET}\n"
    )

    # ── Interactive loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                f"{_CYAN}{_BOLD}Research Query ▶ {_RESET}"
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{_DIM}Goodbye! 👋{_RESET}\n")
            break

        # ── Handle meta-commands ─────────────────────────────────────
        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print(f"\n{_DIM}Goodbye! 👋{_RESET}\n")
            break

        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print(
                f"\n{_CYAN}🆕 New session started. "
                f"ID: {thread_id[:8]}...{_RESET}\n"
            )
            continue

        # ── Execute research pipeline ────────────────────────────────
        print(f"\n{_DIM}{_THIN_SEP}")
        print(f"   Starting research pipeline...")
        print(f"{_THIN_SEP}{_RESET}")

        initial_state = _create_initial_state(user_input)
        _run_graph_with_interrupt_handling(graph, initial_state, config)

        print()  # Spacing between queries


if __name__ == "__main__":
    main()
