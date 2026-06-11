"""LangGraph node implementations for the research pipeline.

Each node calls the appropriate OpenManus-based agent via agent.run(request).
Agents are reset (state → IDLE, memory cleared) before each invocation.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from loguru import logger

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
        """Poll agent state every 1.5s while run() executes."""
        try:
            while True:
                await asyncio.sleep(1.5)
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
    """Plan the research: decompose topic into outline and search queries.

    In round 1, plans from scratch. In subsequent rounds, uses accumulated
    knowledge, round history, and search feedback to focus on filling gaps."""
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

        # ── Accumulated knowledge summary ──
        accumulated = state.get("accumulated_knowledge", [])
        knowledge_text = ""
        if accumulated:
            sorted_knowledge = sorted(accumulated, key=lambda k: k.get("confidence", 0), reverse=True)
            knowledge_text = "\n".join(
                f"- [conf={k.get('confidence', '?')}, r{k.get('round', '?')}] {k.get('claim', '')}"
                for k in sorted_knowledge[:20]
            ) if sorted_knowledge else "No verified facts yet."
        if not knowledge_text:
            knowledge_text = "No accumulated knowledge yet."

        # ── Round history summary ──
        round_history = state.get("round_history", [])
        history_text = ""
        if round_history:
            for rh in round_history:
                r = rh.get("round", "?")
                scores = rh.get("critic_scores", {})
                score_summary = ", ".join(f"{k}={v}" for k, v in scores.items())
                str_text = "; ".join(rh.get("strengths", [])[:2]) or "none"
                gap_text = "; ".join(rh.get("gaps", [])[:3]) or "none"
                history_text += (
                    f"Round {r}: scores {{{score_summary}}}, "
                    f"strengths: {str_text}, gaps: {gap_text}\n"
                )
        if not history_text:
            history_text = "No previous rounds."

        # ── Search effectiveness ──
        search_feedback = state.get("search_feedback", [])
        feedback_text = ""
        if search_feedback:
            productive = [sf for sf in search_feedback if sf.get("productive")]
            unproductive = [sf for sf in search_feedback if not sf.get("productive")]
            if productive:
                feedback_text += "\n".join(
                    f"  [+ productive] {sf.get('query', '?')}" for sf in productive[:5]
                ) + "\n"
            if unproductive:
                feedback_text += "\n".join(
                    f"  [- unproductive] {sf.get('query', '?')}: {sf.get('reason', '')}"
                    for sf in unproductive[:5]
                )
        if not feedback_text:
            feedback_text = "No search effectiveness data yet."

        request = (
            f"Research topic: {topic}\n\n"
            f"=== ACCUMULATED KNOWLEDGE (verified facts, do NOT re-search) ===\n"
            f"{knowledge_text}\n\n"
            f"=== ROUND HISTORY ===\n"
            f"{history_text}\n\n"
            f"=== SEARCH EFFECTIVENESS ===\n"
            f"{feedback_text}\n\n"
            f"=== CURRENT GAPS ===\n{gaps_text}\n\n"
            "Using the accumulated knowledge and search effectiveness data, generate "
            "NEW search queries. DO NOT re-search topics that already have high-confidence "
            "(confidence >= 0.8) verified findings. Avoid query directions that were "
            "previously unproductive. Focus only on filling remaining gaps."
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
    """Analyze search results and extract key findings with structured knowledge."""
    search_results = state.get("search_results", [])
    round_num = state.get("research_round", 1)
    logger.info(f"[Analyst] Analyzing {len(search_results)} search result sets (round {round_num})...")

    all_content = ""
    for i, sr in enumerate(search_results):
        all_content += f"\n### Source Group {i+1}: {sr.get('query', 'Unknown')}\n"
        all_content += sr.get("content", "")[:5000]

    # Build context from previous rounds' accumulated knowledge
    accumulated = state.get("accumulated_knowledge", [])
    knowledge_context = ""
    if accumulated and round_num > 1:
        sorted_knowledge = sorted(accumulated, key=lambda k: k.get("confidence", 0), reverse=True)
        knowledge_context = "\n\n=== Previously Verified Facts (do not re-verify, reference by claim) ===\n"
        for k in sorted_knowledge[:15]:
            knowledge_context += (
                f"- [{k.get('confidence', '?')}] {k.get('claim', '')} "
                f"(source: {k.get('source', 'unknown')})\n"
            )

    request = (
        f"Research topic: {state['topic']}\n\n"
        f"{knowledge_context}"
        f"Please critically analyze these search results:\n{all_content[:30000]}\n\n"
        "If previously verified facts are provided, reference them where relevant "
        "instead of re-validating. Focus on NEW findings.\n"
        "Output the structured JSON block with verified_facts as specified in your system prompt. "
        "Call Terminate when done."
    )

    await _run_agent_with_progress(analyst, request, "analyst", progress, timeout)
    assistant_msgs = [m for m in analyst.messages if m.role == "assistant" and m.content]
    result = assistant_msgs[-1].content if assistant_msgs else ""

    parsed = analyst.parse_analysis(result)
    verified_facts = parsed.get("verified_facts", [])
    for fact in verified_facts:
        fact["round"] = round_num

    return {
        "analyses": [{"content": result, "timestamp": ""}],
        "accumulated_knowledge": verified_facts,
        "current_phase": "analyzed",
    }


# ── Synthesizer Node ────────────────────────────────────────────────────────

async def synthesizer_node(
    state: ResearchState,
    synthesizer: SynthesizerAgent,
    progress: ProgressTracker | None = None,
    timeout: float = 600,
) -> dict:
    """Synthesize all analyses into a unified framework, building on prior knowledge."""
    analyses = state.get("analyses", [])
    outline = state.get("outline", [])
    round_num = state.get("research_round", 1)
    logger.info(f"[Synthesizer] Synthesizing {len(analyses)} analyses (round {round_num})...")

    analyses_text = "\n\n".join(a.get("content", "")[:5000] for a in analyses)[:25000]
    outline_text = json.dumps(outline, ensure_ascii=False, indent=2)

    # Build accumulated knowledge context for cross-round continuity
    accumulated = state.get("accumulated_knowledge", [])
    knowledge_context = ""
    if accumulated and round_num > 1:
        sorted_knowledge = sorted(accumulated, key=lambda k: k.get("confidence", 0), reverse=True)
        knowledge_context = "\n\n=== Accumulated Knowledge (verified facts from all rounds) ===\n"
        for k in sorted_knowledge[:20]:
            knowledge_context += (
                f"- [conf={k.get('confidence', '?')}, r{k.get('round', '?')}] "
                f"{k.get('claim', '')}\n"
            )

    request = (
        f"Research topic: {state['topic']}\n\n"
        f"Outline:\n{outline_text}\n\n"
        f"Analysis findings:\n{analyses_text}\n"
        f"{knowledge_context}\n\n"
        "Synthesize all findings into a cohesive framework following the outline. "
        "Resolve contradictions. Identify best-supported conclusions. "
        "Integrate accumulated knowledge from previous rounds. "
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
    """Generate the research report — fresh in round 1, revision in later rounds."""
    round_num = state.get("research_round", 1)
    logger.info(f"[Writer] Writing report (round {round_num})...")

    synthesis = state.get("synthesized_findings", "")
    outline = state.get("outline", [])
    outline_text = json.dumps(outline, ensure_ascii=False, indent=2)
    previous_draft = state.get("previous_draft", "")
    gaps = state.get("gaps", [])
    critique = state.get("critique", {})

    if previous_draft and round_num > 1:
        # Revision mode: improve the previous draft with new findings
        weaknesses = critique.get("weaknesses", [])
        weaknesses_text = "\n".join(f"- {w}" for w in weaknesses) if weaknesses else "none noted"
        gaps_text = "\n".join(f"- {g}" for g in gaps) if gaps else "none noted"

        request = (
            f"Research topic: {state['topic']}\n\n"
            f"=== PREVIOUS DRAFT (to revise) ===\n{previous_draft[:15000]}\n\n"
            f"=== CRITIC FEEDBACK ===\n"
            f"Weaknesses to fix:\n{weaknesses_text}\n"
            f"Gaps to fill:\n{gaps_text}\n\n"
            f"=== NEW FINDINGS TO INCORPORATE ===\n{synthesis[:10000]}\n\n"
            "REVISE the previous draft into an improved report. "
            "Fix the identified weaknesses, fill gaps with new findings, "
            "improve depth and credibility. Preserve well-written sections. "
            "Output the COMPLETE revised report in Markdown. "
            "Include executive summary, all sections, in-text citations [1][2], "
            "and references. Call Terminate when done."
        )
    else:
        # Fresh report (round 1)
        request = (
            f"Research topic: {state['topic']}\n\n"
            f"Outline:\n{outline_text}\n\n"
            f"Synthesized findings:\n{synthesis[:20000]}\n\n"
            "Write a comprehensive research report. Include executive summary, "
            "all outline sections, and references. Use in-text citations [1], [2]. "
            "Write substantive, detailed content. "
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
    """Review report quality, generate round summary and search feedback."""
    draft = state.get("draft_report", "")
    current_round = state.get("research_round", 1)
    search_queries = state.get("search_queries", [])
    logger.info(f"[Critic] Reviewing report (round {current_round})...")

    # Include search queries context for the Critic to evaluate search effectiveness
    queries_context = ""
    if search_queries:
        queries_context = f"\n\nSearch queries used this round:\n" + "\n".join(
            f"- {q}" for q in search_queries[:10]
        )

    request = (
        f"Research topic: {state['topic']}\n\n"
        f"Report to review:\n{draft[:20000]}\n"
        f"{queries_context}\n\n"
        "CRITICAL: You MUST write your full evaluation (including the JSON with scores "
        "and search_feedback) in your response BEFORE you call Terminate. "
        "Do NOT call Terminate until you have output the complete review JSON.\n\n"
        "Evaluate this report. Output scores (0-100) for completeness, accuracy, structure, "
        "depth, credibility, clarity. For each search query, assess whether it produced "
        "productive results (search_feedback field). Output as JSON with overall_score. "
        "Identify gaps. Recommend accept or revise. Call Terminate when done."
    )

    await _run_agent_with_progress(critic, request, "critic", progress, timeout)
    all_msgs = [m for m in critic.messages if m.role == "assistant" and m.content]
    result_text = ""
    for m in reversed(all_msgs):
        if len(m.content or "") > 50:
            result_text = m.content
            break
    if not result_text and all_msgs:
        result_text = all_msgs[-1].content or ""
    review = critic.parse_review(result_text)

    overall = review.get("overall_score", 70)
    gaps = review.get("gaps", [])

    return {
        "quality_score": overall,
        "critique": review,
        "gaps": gaps,
        "current_phase": "reviewed",
        "overall_score": overall,
        "research_round": current_round + 1,
        "previous_draft": state.get("draft_report", ""),
        "round_history": [{
            "round": current_round,
            "critic_scores": review.get("scores", {}),
            "strengths": review.get("strengths", []),
            "gaps": gaps,
            "overall_score": overall,
        }],
        "search_feedback": review.get("search_feedback", []),
        "_critic_raw_result": result_text[:2000],
        "_critic_msg_count": len(all_msgs),
    }


# ── Human Review Node ────────────────────────────────────────────────────────

async def human_review_node(
    state: ResearchState,
    review_dir: str = "./reviews",
) -> dict:
    """Save the full draft report to file and pause for human review.

    Uses LangGraph interrupt() to suspend execution. The caller (main.py)
    presents the review prompt, gets user input, and resumes with Command(resume=...).

    User decisions:
        approve — proceed to formatter, output final report
        revise  — return to planner with user feedback as new gaps
        abort   — end the pipeline, save current draft as-is

    The interrupt is only triggered when human_in_the_loop is enabled in config.
    """
    from datetime import datetime
    from pathlib import Path

    from langgraph.types import interrupt

    draft = state.get("draft_report", "")
    round_num = state.get("research_round", 1)
    quality = state.get("quality_score", 0)
    gaps = state.get("gaps", [])
    topic = state.get("topic", "research")
    critique = state.get("critique", {})

    # Save full report to file for offline review
    reviews_dir = Path(review_dir)
    reviews_dir.mkdir(parents=True, exist_ok=True)
    safe_topic = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic)[:40]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    review_path = reviews_dir / f"round-{round_num}-draft-{safe_topic}-{timestamp}.md"
    review_path.write_text(draft, encoding="utf-8")

    logger.info(f"[HumanReview] Round {round_num} draft saved to {review_path}, waiting for human...")

    # Build critique summary for the terminal prompt
    scores = critique.get("scores", {})
    score_lines = "\n".join(f"  {k}: {v}/100" for k, v in scores.items()) if scores else "N/A"
    strengths = critique.get("strengths", [])
    weaknesses = critique.get("weaknesses", [])

    # Pause execution — the caller sees this data and prompts the user
    decision = interrupt({
        "report_file": str(review_path),
        "round": round_num,
        "quality_score": quality,
        "scores": score_lines,
        "strengths": strengths[:3],
        "weaknesses": weaknesses[:3],
        "gaps": gaps,
        "prompt": "输入决策: approve / revise: <反馈> / abort",
    })

    # Parse user decision (returned by Command(resume=...))
    if isinstance(decision, str):
        d = decision.strip().lower()
        if d.startswith("approve"):
            return {
                "human_decision": "approve",
                "review_path": str(review_path),
                "current_phase": "awaiting_human",
            }
        elif d.startswith("revise"):
            feedback = decision[6:].strip().lstrip(":").strip()
            new_gaps = [feedback] if feedback else gaps
            logger.info(f"[HumanReview] User requested revision: {feedback}")
            return {
                "human_decision": "revise",
                "gaps": new_gaps,
                "review_path": str(review_path),
                "current_phase": "awaiting_human",
            }
        else:
            logger.info("[HumanReview] User aborted")
            return {
                "human_decision": "abort",
                "review_path": str(review_path),
                "current_phase": "awaiting_human",
            }

    # Non-string input (e.g. dict from a WebUI) — treat as abort
    return {
        "human_decision": "abort",
        "review_path": str(review_path),
        "current_phase": "awaiting_human",
    }
