"""LangGraph workflow builder — assembles the multi-agent research pipeline.

The workflow follows a Supervisor pattern:
    Planner → Searcher → Analyst → Synthesizer → Writer → Critic
        ↑                                                    │
        └───────── (if score < threshold & rounds remain) ───┘
                                     │
                                (if passed)
                                     ▼
                                Formatter → END

All agents inherit from OpenManus ToolCallAgent (Pydantic BaseModel).
Each agent gets an LLMProvider that bridges OpenManus to native provider SDKs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, StateGraph

from src._framework import Terminate, ToolCollection

from src.agents.specialized import (
    AnalystAgent,
    CriticAgent,
    PlannerAgent,
    SearcherAgent,
    SynthesizerAgent,
    WriterAgent,
)
from src.agents.llm_adapter import LLMProvider
from src.graph.nodes import (
    analyst_node,
    critic_node,
    planner_node,
    searcher_node,
    synthesizer_node,
    writer_node,
)
from src.graph.state import ResearchState
from src.llm.config import AgentLLMConfig

if TYPE_CHECKING:
    from src.utils.progress import ProgressTracker

logger = logging.getLogger(__name__)


def _create_agent(agent_cls, llm_config: AgentLLMConfig, tools: ToolCollection, max_steps: int | None = None):
    """Create an OpenManus agent with LLMProvider.

    OpenManus's BaseAgent.model_validator replaces the llm field with an app.llm.LLM
    instance if it's not already one. We bypass this by setting llm via __dict__
    after Pydantic initialization.
    """
    agent = agent_cls(available_tools=tools)
    provider = LLMProvider(llm_config)
    agent.__dict__["llm"] = provider
    if max_steps is not None:
        agent.max_steps = max_steps
    return agent


def build_workflow(
    agent_configs: dict[str, AgentLLMConfig],
    tools: dict[str, ToolCollection] | None = None,
    use_checkpointer: bool = False,
    checkpointer_path: str = "./data/checkpoints.sqlite",
    max_agent_turns: dict[str, int] | None = None,
    progress: ProgressTracker | None = None,
    agent_timeouts: dict[str, float] | None = None,
) -> StateGraph:
    """Build and compile the research workflow StateGraph.

    Args:
        agent_configs: Dict mapping agent name -> AgentLLMConfig.
        tools: Dict mapping agent name -> OpenManus ToolCollection.
        use_checkpointer: Whether to use SQLite checkpointing.
        checkpointer_path: Path to SQLite database.
        max_agent_turns: Per-agent max step overrides (from config).
        progress: Optional ProgressTracker for real-time display.
        agent_timeouts: Per-agent wall-clock timeout in seconds.

    Returns:
        A compiled LangGraph StateGraph.
    """
    tools = tools or {}
    _terminate = ToolCollection(Terminate())
    default_cfg = agent_configs.get("default", list(agent_configs.values())[0] if agent_configs else None)
    max_agent_turns = max_agent_turns or {}
    agent_timeouts = agent_timeouts or {}

    planner = _create_agent(PlannerAgent, agent_configs.get("planner", default_cfg), tools.get("planner", _terminate), max_steps=max_agent_turns.get("planner"))
    searcher = _create_agent(SearcherAgent, agent_configs.get("searcher", default_cfg), tools.get("searcher", _terminate), max_steps=max_agent_turns.get("searcher"))
    analyst = _create_agent(AnalystAgent, agent_configs.get("analyst", default_cfg), tools.get("analyst", _terminate), max_steps=max_agent_turns.get("analyst"))
    synthesizer = _create_agent(SynthesizerAgent, agent_configs.get("synthesizer", default_cfg), tools.get("synthesizer", _terminate), max_steps=max_agent_turns.get("synthesizer"))
    writer = _create_agent(WriterAgent, agent_configs.get("writer", default_cfg), tools.get("writer", _terminate), max_steps=max_agent_turns.get("writer"))
    critic = _create_agent(CriticAgent, agent_configs.get("critic", default_cfg), tools.get("critic", _terminate), max_steps=max_agent_turns.get("critic"))

    # ── Build graph ──────────────────────────────────────────────────────
    workflow = StateGraph(ResearchState)

    t_pl = agent_timeouts.get("planner", 300)
    t_se = agent_timeouts.get("searcher", 600)
    t_an = agent_timeouts.get("analyst", 600)
    t_sy = agent_timeouts.get("synthesizer", 600)
    t_wr = agent_timeouts.get("writer", 900)
    t_cr = agent_timeouts.get("critic", 300)

    async def _planner_node(s): return await planner_node(s, planner, progress=progress, timeout=t_pl)
    async def _searcher_node(s): return await searcher_node(s, searcher, progress=progress, timeout=t_se)
    async def _analyst_node(s): return await analyst_node(s, analyst, progress=progress, timeout=t_an)
    async def _synthesizer_node(s): return await synthesizer_node(s, synthesizer, progress=progress, timeout=t_sy)
    async def _writer_node(s): return await writer_node(s, writer, progress=progress, timeout=t_wr)
    async def _critic_node(s): return await critic_node(s, critic, progress=progress, timeout=t_cr)

    workflow.add_node("planner", _planner_node)
    workflow.add_node("searcher", _searcher_node)
    workflow.add_node("analyst", _analyst_node)
    workflow.add_node("synthesizer", _synthesizer_node)
    workflow.add_node("writer", _writer_node)
    workflow.add_node("critic", _critic_node)
    workflow.add_node("formatter", _formatter_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "analyst")
    workflow.add_edge("analyst", "synthesizer")
    workflow.add_edge("synthesizer", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_conditional_edges(
        "critic",
        _supervisor_router,
        {"planner": "planner", "formatter": "formatter", "end": END},
    )
    workflow.add_edge("formatter", END)

    if use_checkpointer:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import os
        os.makedirs(os.path.dirname(checkpointer_path) or ".", exist_ok=True)
        checkpointer = SqliteSaver.from_conn_string(checkpointer_path)
        return workflow.compile(checkpointer=checkpointer)

    return workflow.compile()


def _supervisor_router(state: ResearchState) -> Literal["planner", "formatter", "end"]:
    score = state.get("quality_score", 0)
    threshold = state.get("quality_threshold", 75)
    current_round = state.get("research_round", 1)
    max_rounds = state.get("max_rounds", 3)

    if score >= threshold or current_round > max_rounds:
        logger.info(f"[Supervisor] Quality {score} >= {threshold} or round {current_round} > {max_rounds} → formatting")
        return "formatter"

    logger.info(f"[Supervisor] Quality {score} < {threshold}, round {current_round}/{max_rounds} → re-planning")
    return "planner"


async def _formatter_node(state: ResearchState) -> dict:
    draft = state.get("draft_report", "")
    topic = state["topic"]
    quality = state.get("quality_score", 0)
    round_num = state.get("research_round", 1)

    header = (
        f"# {topic}\n\n"
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"> Research rounds: {round_num}\n"
        f"> Quality score: {quality:.0f}/100\n\n"
        "---\n\n"
    )

    return {
        "final_report": header + draft,
        "current_phase": "formatted",
    }
