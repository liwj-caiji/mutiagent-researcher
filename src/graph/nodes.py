"""LangGraph node implementations for the research pipeline.

Each node calls the appropriate OpenManus-based agent via agent.run(request).
Agents are reset (state → IDLE, memory cleared) before each invocation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from src._framework import AgentState as OM_AgentState

from src.agents.specialized import (
    AnalystAgent,
    CriticAgent,
    PlannerAgent,
    SynthesizerAgent,
    WriterAgent,
)
from src.graph.state import ResearchState

if TYPE_CHECKING:
    from src.utils.progress import ProgressTracker

logger = logging.getLogger(__name__)


def _reset_agent(agent) -> None:
    """Reset an OpenManus agent to IDLE, ready for a new run."""
    agent.state = OM_AgentState.IDLE
    agent.current_step = 0
    agent.memory.messages = []


# ── Agent runner with progress & timeout ─────────────────────────────────────

async def _run_agent_with_progress(
    agent,
    request: str,
    agent_name: str,
    progress: ProgressTracker | None = None,
    timeout: float = 600,
) -> str:
    """Run an agent with progress polling and timeout protection."""
    _reset_agent(agent)
    max_steps = getattr(agent, "max_steps", 10)

    if progress:
        progress.agent_started(agent_name, max_steps)

    async def _poll():
        """Poll agent state every 500ms while run() executes."""
        try:
            while True:
                await asyncio.sleep(0.5)
                if progress is None:
                    continue
                step = getattr(agent, "current_step", 0)
                state_val = agent.state
                state_str = state_val.value if hasattr(state_val, "value") else str(state_val)
                detail = f"step {step}/{max_steps} — {state_str}"
                progress.agent_step_update(agent_name, step, max_steps, detail)
        except asyncio.CancelledError:
            pass

    try:
        async with asyncio.timeout(timeout):
            poll_task = asyncio.create_task(_poll())
            try:
                result = await agent.run(request)
            finally:
                poll_task.cancel()
                try:
                    await poll_task
                except asyncio.CancelledError:
                    pass
        if progress:
            progress.agent_finished(agent_name)
        return result
    except asyncio.TimeoutError:
        agent.state = OM_AgentState.FINISHED
        if progress:
            progress.agent_timeout(agent_name, timeout)
        logger.error(f"[{agent_name}] Timed out after {timeout:.0f}s")
        raise RuntimeError(
            f"Agent '{agent_name}' timed out after {timeout:.0f}s"
        ) from None
    except Exception:
        if progress:
            progress.agent_error(agent_name, "failed")
        raise


# ── Planner Node ─────────────────────────────────────────────────────────────

async def planner_node(
    state: ResearchState,
    planner: PlannerAgent,
    progress: ProgressTracker | None = None,
    timeout: float = 300,
) -> dict:
    """Plan the research: decompose topic into outline and search queries."""
    round_num = state.get("research_round", 1)
    logger.info(f"[Planner] Round {round_num} — planning...")

    topic = state["topic"]
    gaps = state.get("gaps", [])

    if round_num == 1:
        request = (
            f"Research topic: {topic}\n\n"
            "Create a comprehensive research plan. Break down this topic into sub-topics, "
            "generate specific search queries, and create a structured outline for the final report."
        )
    else:
        gaps_text = "\n".join(f"- {g}" for g in gaps) if gaps else "No specific gaps identified."
        request = (
            f"Research topic: {topic}\n\n"
            f"Current gaps to address:\n{gaps_text}\n\n"
            "Based on these gaps, generate additional search queries. Focus only on filling identified gaps."
        )

    await _run_agent_with_progress(planner, request, "planner", progress, timeout)
    plan = planner.parse_plan()

    queries = []
    for st in plan.get("sub_topics", []):
        queries.extend(st.get("search_queries", []))

    if not queries:
        queries = [topic]

    return {
        "outline": plan.get("outline", []),
        "search_queries": queries,
        "information_needs": plan.get("information_needs", []),
        "current_phase": "planned",
        "research_round": round_num,
    }


# ── Searcher Node ────────────────────────────────────────────────────────────

async def searcher_node(
    state: ResearchState,
    searcher_factory,  # Callable[[], SearcherAgent] — creates a fresh agent per query
    progress: ProgressTracker | None = None,
    timeout: float = 600,
    max_parallel: int = 3,
) -> dict:
    """Execute searches for assigned queries in parallel.

    Each query gets its own SearcherAgent instance, isolated from others.
    An asyncio.Semaphore limits concurrency to respect API rate limits.
    Results are returned as a list and merged into ResearchState.search_results
    via the operator.add reducer.
    """
    queries = state.get("search_queries", [])
    query_count = min(len(queries), 10)
    max_parallel = max(1, min(max_parallel, query_count))
    per_query_timeout = timeout / max_parallel if max_parallel > 0 else timeout
    semaphore = asyncio.Semaphore(max_parallel)

    logger.info(f"[Searcher] Searching {query_count} queries (max parallel: {max_parallel})...")

    if progress:
        progress.agent_started("searcher", query_count)

    async def _search_one(query: str, index: int) -> dict:
        """Run a single search query on its own agent instance."""
        async with semaphore:
            agent = searcher_factory()
            request = (
                f"Search for the following query. Use the search tools available to you.\n\n"
                f"Query: {query}\n\n"
                "For each result, record: title, source URL, key findings. Call Terminate when done."
            )
            try:
                async with asyncio.timeout(per_query_timeout):
                    await agent.run(request)
            except asyncio.TimeoutError:
                logger.warning(f"[Searcher] Query '{query[:60]}...' timed out after {per_query_timeout:.0f}s")
                return {"query": query, "content": f"Search timed out after {per_query_timeout:.0f}s"}

            assistant_msgs = [m for m in agent.messages if m.role == "assistant" and m.content]
            content = assistant_msgs[-1].content if assistant_msgs else "No results"

            if progress:
                progress.agent_step_update("searcher", index + 1, query_count, f"query {index + 1}/{query_count}: {query[:40]}...")

            return {"query": query, "content": content}

    try:
        async with asyncio.timeout(timeout):
            tasks = [_search_one(query, i) for i, query in enumerate(queries[:query_count])]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unwrap exceptions from gather
        resolved: list[dict] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"[Searcher] Query {i + 1} failed: {r}")
                resolved.append({"query": queries[i], "content": f"Error: {r}"})
            else:
                resolved.append(r)

        if progress:
            progress.agent_finished("searcher")

        return {
            "search_results": resolved,
            "current_phase": "searched",
        }
    except asyncio.TimeoutError:
        if progress:
            progress.agent_timeout("searcher", timeout)
        logger.error(f"[Searcher] Timed out after {timeout:.0f}s")
        raise RuntimeError(f"Agent 'searcher' timed out after {timeout:.0f}s") from None


# ── Analyst Node ─────────────────────────────────────────────────────────────

async def analyst_node(
    state: ResearchState,
    analyst: AnalystAgent,
    progress: ProgressTracker | None = None,
    timeout: float = 600,
) -> dict:
    """Analyze search results and extract key findings."""
    search_results = state.get("search_results", [])
    logger.info(f"[Analyst] Analyzing {len(search_results)} search result sets...")

    all_content = ""
    for i, sr in enumerate(search_results):
        all_content += f"\n### Source Group {i+1}: {sr.get('query', 'Unknown')}\n"
        all_content += sr.get("content", "")[:5000]

    request = (
        f"Research topic: {state['topic']}\n\n"
        f"Please critically analyze these search results:\n{all_content[:30000]}\n\n"
        "Provide: key findings, credibility assessment, contradictions between sources, "
        "information gaps, and recommended conclusions. Call Terminate when done."
    )

    await _run_agent_with_progress(analyst, request, "analyst", progress, timeout)
    assistant_msgs = [m for m in analyst.messages if m.role == "assistant" and m.content]
    result = assistant_msgs[-1].content if assistant_msgs else ""

    return {
        "analyses": [{"content": result, "timestamp": ""}],
        "current_phase": "analyzed",
    }


# ── Synthesizer Node ────────────────────────────────────────────────────────

async def synthesizer_node(
    state: ResearchState,
    synthesizer: SynthesizerAgent,
    progress: ProgressTracker | None = None,
    timeout: float = 600,
) -> dict:
    """Synthesize all analyses into a unified framework."""
    analyses = state.get("analyses", [])
    outline = state.get("outline", [])
    logger.info(f"[Synthesizer] Synthesizing {len(analyses)} analyses...")

    analyses_text = "\n\n".join(a.get("content", "")[:5000] for a in analyses)[:30000]
    outline_text = json.dumps(outline, ensure_ascii=False, indent=2)

    request = (
        f"Research topic: {state['topic']}\n\n"
        f"Outline:\n{outline_text}\n\n"
        f"Analysis findings:\n{analyses_text}\n\n"
        "Synthesize all findings into a cohesive framework following the outline. "
        "Resolve contradictions. Identify best-supported conclusions. "
        "Write in a structured format the Writer can directly use. Call Terminate when done."
    )

    await _run_agent_with_progress(synthesizer, request, "synthesizer", progress, timeout)
    assistant_msgs = [m for m in synthesizer.messages if m.role == "assistant" and m.content]
    result = assistant_msgs[-1].content if assistant_msgs else ""

    return {
        "synthesized_findings": result,
        "current_phase": "synthesized",
    }


# ── Writer Node ──────────────────────────────────────────────────────────────

async def writer_node(
    state: ResearchState,
    writer: WriterAgent,
    progress: ProgressTracker | None = None,
    timeout: float = 900,
) -> dict:
    """Generate the full research report."""
    logger.info(f"[Writer] Writing report (round {state.get('research_round', 1)})...")

    synthesis = state.get("synthesized_findings", "")
    outline = state.get("outline", [])
    outline_text = json.dumps(outline, ensure_ascii=False, indent=2)

    request = (
        f"Research topic: {state['topic']}\n\n"
        f"Outline:\n{outline_text}\n\n"
        f"Synthesized findings:\n{synthesis[:20000]}\n\n"
        "Write a comprehensive research report. Include executive summary, all outline sections, "
        "and references. Use in-text citations [1], [2]. Write substantive, detailed content. "
        "Output in Markdown format. Call Terminate when done."
    )

    await _run_agent_with_progress(writer, request, "writer", progress, timeout)
    assistant_msgs = [m for m in writer.messages if m.role == "assistant" and m.content]
    result = assistant_msgs[-1].content if assistant_msgs else ""

    citations = []
    for sr in state.get("search_results", []):
        citations.append({"source": sr.get("query", "Unknown"), "raw_content": sr.get("content", "")[:200]})

    return {
        "draft_report": result,
        "citations": citations,
        "current_phase": "drafted",
    }


# ── Critic Node ──────────────────────────────────────────────────────────────

async def critic_node(
    state: ResearchState,
    critic: CriticAgent,
    progress: ProgressTracker | None = None,
    timeout: float = 300,
) -> dict:
    """Review report quality and determine if another round is needed."""
    draft = state.get("draft_report", "")
    logger.info(f"[Critic] Reviewing report (round {state.get('research_round', 1)})...")

    request = (
        f"Research topic: {state['topic']}\n\n"
        f"Report to review:\n{draft[:20000]}\n\n"
        "Evaluate this report. Output scores (0-100) for completeness, accuracy, structure, "
        "depth, credibility, clarity. Identify gaps. Recommend accept or revise. Call Terminate when done."
    )

    await _run_agent_with_progress(critic, request, "critic", progress, timeout)
    assistant_msgs = [m for m in critic.messages if m.role == "assistant" and m.content]
    result_text = assistant_msgs[-1].content if assistant_msgs else ""
    review = critic.parse_review(result_text)

    overall = review.get("overall_score", 70)
    gaps = review.get("gaps", [])
    current_round = state.get("research_round", 1)

    return {
        "quality_score": overall,
        "critique": review,
        "gaps": gaps,
        "current_phase": "reviewed",
        "overall_score": overall,
        "research_round": current_round + 1,
    }
