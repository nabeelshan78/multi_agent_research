# Multi-Agent Business Research Assistant

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.4+-FF6B6B?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq_Cloud-LLM-F55036?style=for-the-badge)
![Tavily](https://img.shields.io/badge/Tavily-Search-4A90D9?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade, cyclical multi-agent system that researches, validates, and synthesises business intelligence — autonomously.**

[Features](#-features) · [Architecture](#-architecture) · [Quickstart](#-quickstart) · [How It Works](#-how-it-works) · [Configuration](#-configuration)

</div>

---

## What Is This?

This is not a chatbot wrapper. This is a **stateful, cyclical AI pipeline** built on LangGraph where four specialised agents collaborate to answer business research queries — each with a single responsibility, communicating through a shared typed state object.

The system accepts a natural language query, evaluates its clarity, executes targeted web searches, runs the results through an independent quality critic, and only synthesises a final report when the research meets a confidence threshold — or forces a graceful conclusion after exhausting its retry budget.

```
User Query → Clarity Check → Research → Validate → (loop if needed) → Final Report
```

---

## Features

| Feature | Description |
|---|---|
| **4 Specialised Agents** | Clarity, Research, Validator, Synthesis — each owns one job |
| **Cyclical Validation Loop** | Research → Validate → Research again until quality threshold is met |
| **Human-in-the-Loop** | Graph pauses mid-execution to ask for clarification, then resumes exactly where it left off |
| **Iterative Clarification** | Re-evaluates query clarity after each user response — never blindly proceeds |
| **Dual-Model Cost Strategy** | Fast model for classification, reasoning model for research and synthesis |
| **Tavily Web Search** | Real-time business data via targeted multi-query search execution |
| **Checkpointed State** | Full MemorySaver checkpointing — interrupt/resume across any graph node |
| **Multi-turn Memory** | Conversation history flows through every agent with automatic context trimming |
| **Production Structure** | Typed state schema, custom exception hierarchy, factory pattern for all clients |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentState (Shared)                      │
│  messages · user_query · clarity_status · research_data         │
│  confidence_score · research_attempts · validation_result       │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
               ┌────│   Clarity Agent   │────┐
               │    │  (fast model)     │    │
               │    └───────────────────┘    │
               │                             │
          needs_clarification              clear
               │                             │
         ┌─────▼──────┐             ┌───────▼────────┐
         │  INTERRUPT  │             │ Research Agent │
         │  (HITL)     │             │ (reasoning)    │
         └─────┬──────┘             └───────┬────────┘
               │                            │
          user input               confidence score
               │                    ┌───────┴────────┐
          re-evaluates           ≥ 6.0             < 6.0
                                    │                 │
                          ┌─────────▼──┐    ┌────────▼────────┐
                          │ Synthesis  │    │ Validator Agent │
                          │  Agent     │    │ (fast model)    │
                          │(reasoning) │    └────────┬────────┘
                          └─────────┬──┘             │
                                    │      ┌──────────┴──────────┐
                                   END  sufficient /          insufficient
                                        attempts>=3          & attempts<3
                                            │                     │
                                     Synthesis Agent ←── Research Agent
                                          (forced)              (loop)
```

### Project Structure

```
multi_agent_system/
│
├── agents/
│   ├── __init__.py          # Package exports
│   ├── clarity.py           # Query disambiguation + HITL interrupt loop
│   ├── research.py          # Tavily search + confidence self-evaluation
│   ├── validator.py         # Research quality critic
│   └── synthesis.py         # Final report generation
│
├── utils/
│   ├── __init__.py
│   ├── llm.py               # get_llm() / get_search_tool() factory
│   └── errors.py            # LLMAPIError, SearchToolError, StateValidationError
│
├── state.py                 # AgentState TypedDict + add_messages reducer
├── graph.py                 # StateGraph construction + conditional routing
├── config.py                # Immutable Settings dataclass + env loading
├── main.py                  # Interactive CLI + interrupt handling loop
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/nabeelshan78/multi_agent_system.git
cd multi_agent_system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

> **Get your keys:**
> - Groq Cloud → https://console.groq.com
> - Tavily → https://app.tavily.com

### 3. Run

```bash
python main.py
```

---

## 🖥️ CLI Preview

```
════════════════════════════════════════════════════════════════════════
   Multi-Agent Business Research Assistant
   Powered by Groq Cloud + Tavily + LangGraph
════════════════════════════════════════════════════════════════════════
   Session ID: a3f8c21b...

Research Query ▶ What is Nvidia's competitive position in the AI chip market?

────────────────────────────────────────────────────────────────────────
   Starting research pipeline...
────────────────────────────────────────────────────────────────────────

[Clarity Agent]
Query assessed as clear (attempt 1/3). Proceeding with research...

[Research Agent]
Research cycle 1 complete. Found 18 new sources (18 total). Confidence: 8.2/10.

════════════════════════════════════════════════════════════════════════
   FINAL RESEARCH REPORT
════════════════════════════════════════════════════════════════════════

### Executive Summary
Nvidia dominates the AI accelerator market with an estimated 70-95% market
share in data center GPUs...
```

### Built-in Commands

| Command | Action |
|---|---|
| Any query | Start a new research pipeline |
| `new` | Start a fresh conversation thread |
| `quit` / `exit` / `q` | Exit the application |

---

## How It Works

### The Execution Flow

**1. Clarity Agent** *(llama-3.1-8b-instant)*
Evaluates whether the query is specific enough to research. Uses structured Pydantic output for type-safe routing. If vague, it calls `interrupt()` — pausing the graph, prompting the user, and re-evaluating the enriched query. This loop repeats up to 3 times before forcing progression.

**2. Research Agent** *(llama-3.3-70b-versatile)*
Runs a two-step pipeline:
- **Query Planning** — LLM generates 2-4 targeted search queries, aware of what's already been found
- **Search Execution** — Each query hits Tavily independently (failures isolated, not raised)
- **Confidence Evaluation** — Separate LLM call scores research completeness 0-10

**3. Validator Agent** *(llama-3.1-8b-instant)*
Independent critic. Reads all gathered research and decides `sufficient` or `insufficient`. Owns and increments the `research_attempts` counter. If insufficient and under the 3-attempt cap, routes back to Research for another cycle.

**4. Synthesis Agent** *(llama-3.3-70b-versatile)*
Terminal node. Consumes all research data, conversation history, and metadata to produce a structured markdown report with Executive Summary, Key Findings, Detailed Analysis, Recommendations, and cited Sources.

### The Interrupt/Resume Mechanism

```python
# Inside clarity_agent — graph pauses here
user_clarification: str = interrupt({
    "type": "clarification_needed",
    "question": assessment.clarification_question,
    "reasoning": assessment.reasoning,
})

# Execution resumes here when Command(resume=user_input) is called
# user_clarification now contains the user's response
```

The graph is checkpointed at every interrupt. `main.py` captures the interrupt payload live from the stream's `__interrupt__` event (not post-hoc from `graph_state.tasks`), ensuring reliable detection across multiple sequential interrupts.

---

## Configuration

All tuning knobs live in `config.py`. No agent code needs to change.

```python
@dataclass(frozen=True, slots=True)
class Settings:
    fast_model: str = "llama-3.1-8b-instant"       # Clarity + Validator
    reasoning_model: str = "llama-3.3-70b-versatile" # Research + Synthesis
    
    confidence_threshold: float = 6.0   # Skip validator if score >= this
    max_research_attempts: int = 3       # Max validator → research cycles
    max_conversation_messages: int = 20  # Context window trim cap
    tavily_max_results: int = 5          # Results per Tavily query
    default_temperature: float = 0.0    # Deterministic routing decisions
```

| Parameter | Default | Effect |
|---|---|---|
| `confidence_threshold` | `6.0` | Lower = more validation cycles. Set to `10.0` to always validate |
| `max_research_attempts` | `3` | Max research→validate loops before forcing synthesis |
| `tavily_max_results` | `5` | More results = better coverage, higher token cost |

---

## State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # append reducer
    user_query: str
    clarity_status: str           # "clear" | "needs_clarification"
    clarification_question: str
    research_data: list[ResearchFinding]
    confidence_score: float       # 0.0 – 10.0
    research_attempts: int        # owned by Validator
    validation_result: str        # "sufficient" | "insufficient"
    validation_reasoning: str
    final_report: str
```

The `add_messages` reducer on the `messages` field means every agent **appends** to conversation history — no overwrites, no lost context.

---

## Execution Paths

| Scenario | Flow | LLM Calls |
|---|---|---|
| Clear query, high confidence | Clarity → Research → Synthesis | 4 |
| Clear query, low confidence, valid data | Clarity → Research → Validator → Synthesis | 5 |
| Low confidence, one retry needed | Clarity → Research → Validator → Research → Synthesis | 7 |
| Max retries exhausted | Clarity → Research → Validator ×3 → Synthesis (forced) | 9 |
| Vague query, one clarification | Clarity (interrupt) → Clarity → Research → Synthesis | 5+ |
| Vague query, cap hit | Clarity (interrupt ×2) → Clarity (cap) → Research → Synthesis | 6+ |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph 0.4+ — StateGraph with MemorySaver checkpointing |
| **LLM Provider** | Groq Cloud — llama-3.1-8b-instant + llama-3.3-70b-versatile |
| **Search** | Tavily Search API — real-time web retrieval |
| **LLM Framework** | LangChain — ChatGroq, structured output, message types |
| **Validation** | Pydantic v2 — type-safe structured outputs for all routing decisions |
| **Language** | Python 3.11+ with full type annotations |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by [Nabeel Shan](https://github.com/nabeelshan78)

</div>
