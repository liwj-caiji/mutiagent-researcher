"""Main entry point — CLI for the multi-agent research assistant.

Usage:
    cd C:/Users/liwenjie/Desktop/muti_agent
    uv run python -m src.main "Research topic here"
    uv run python -m src.main "研究主题" --language zh
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from langgraph.types import Command
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src._framework import Terminate, ToolCollection

from src.utils.progress import ProgressTracker

load_dotenv()

from src.agents.specialized import PlannerAgent
from src.graph.workflow import build_workflow
from src.graph.state import ResearchState
from src.llm.config import AgentLLMConfig
from src.tools.mcp import MCPManager
from src.tools.search import (
    ArxivSearchTool,
    BraveSearchTool,
    DuckDuckGoTool,
    JinaReaderTool,
    TavilySearchTool,
    WebScraperTool,
    WikipediaSearchTool,
)
from src.tools.export import CitationFormatterTool
from src.tools.analysis import PythonExecuteTool

app = typer.Typer(no_args_is_help=True)
console = Console()


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent_configs(agents_config: dict) -> dict[str, AgentLLMConfig]:
    """Parse agent YAML config into AgentLLMConfig objects."""
    configs = {}
    default = agents_config.get("default", {})

    for agent_name in ["planner", "searcher", "analyst", "synthesizer", "writer", "critic"]:
        agent_cfg = agents_config.get(agent_name, default)
        if not agent_cfg:
            agent_cfg = default

        provider = agent_cfg.get("provider", "anthropic")
        model = agent_cfg.get("model", "claude-sonnet-4-6")

        api_key = ""
        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        elif provider in ("openai", "deepseek"):
            api_key = os.getenv(f"{provider.upper()}_API_KEY", os.getenv("OPENAI_API_KEY", ""))

        base_url = None
        if provider == "anthropic":
            base_url = os.getenv("ANTHROPIC_BASE_URL", None)
        else:
            base_url = os.getenv(f"{provider.upper()}_BASE_URL", None)

        configs[agent_name] = AgentLLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=agent_cfg.get("temperature", 0.5),
            max_tokens=agent_cfg.get("max_tokens", 4096),
        )

    return configs


async def build_tool_collections(research_cfg: dict) -> tuple[dict[str, ToolCollection], MCPManager | None]:
    """Build tool collections for each agent, optionally using MCP servers.

    Returns (tool_dict, mcp_manager). Caller must call mcp_manager.disconnect_all()
    after the workflow completes.
    """
    mcp_cfg = research_cfg.get("mcp", {})
    mcp_manager = MCPManager(mcp_cfg)
    terminate = Terminate()

    if not mcp_manager.enabled:
        # In-process fallback (original behavior)
        search_tools = [
            BraveSearchTool(), TavilySearchTool(),
            DuckDuckGoTool(),
            ArxivSearchTool(), WikipediaSearchTool(),
            JinaReaderTool(), WebScraperTool(),
            terminate,
        ]
        writer_tools = [CitationFormatterTool(), terminate]
        analyst_tools = [PythonExecuteTool(), terminate]
        return {
            "planner": ToolCollection(terminate),
            "searcher": ToolCollection(*search_tools),
            "analyst": ToolCollection(*analyst_tools),
            "synthesizer": ToolCollection(terminate),
            "writer": ToolCollection(*writer_tools),
            "critic": ToolCollection(terminate),
        }, None

    # MCP mode: connect to remote tool servers
    console.print("[dim]Connecting to MCP tool servers...[/dim]")
    tool_dict = {
        "planner": await mcp_manager.create_tool_collection([], [terminate]),
        "searcher": await mcp_manager.create_tool_collection(["search"], [terminate]),
        "analyst": await mcp_manager.create_tool_collection([], [PythonExecuteTool(), terminate]),
        "synthesizer": await mcp_manager.create_tool_collection([], [terminate]),
        "writer": await mcp_manager.create_tool_collection([], [CitationFormatterTool(), terminate]),
        "critic": await mcp_manager.create_tool_collection([], [terminate]),
    }
    console.print("[green]MCP tool servers connected.[/green]")
    return tool_dict, mcp_manager


async def run_research(topic: str, agents_config_path: str, research_config_path: str):
    """Execute the multi-agent research pipeline."""
    agents_cfg = load_yaml(agents_config_path)
    research_cfg = load_yaml(research_config_path)

    agent_configs = build_agent_configs(agents_cfg)
    tool_collections, mcp_manager = await build_tool_collections(research_cfg)

    max_agent_turns: dict[str, int] = research_cfg.get("max_agent_turns", {})
    agent_timeouts: dict[str, float] = research_cfg.get("agent_timeouts", {})

    hitl_cfg = research_cfg.get("human_in_the_loop", {})
    hitl_enabled = bool(hitl_cfg.get("enabled", False))
    hitl_review_dir = hitl_cfg.get("review_dir", "./reviews")
    use_checkpointer = research_cfg.get("use_checkpointer", False)
    checkpointer_path = research_cfg.get("checkpointer_path", "./data/checkpoints.sqlite")

    console.print(Panel(f"Building research workflow for: [bold]{topic}[/bold]", style="blue"))
    if hitl_enabled:
        console.print("[yellow]Human-in-the-loop enabled — you will review drafts before finalization.[/yellow]")

    tracker = ProgressTracker(console, max_agent_turns)

    # Create async checkpointer if needed (HITL or crash recovery).
    # AsyncSqliteSaver requires aiosqlite.Connection, created asynchronously.
    checkpointer = None
    _checkpointer_conn = None
    if hitl_enabled or use_checkpointer:
        import aiosqlite as _aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as _AsyncSqliteSaver
        import os as _os
        _os.makedirs(_os.path.dirname(checkpointer_path) or ".", exist_ok=True)
        _checkpointer_conn = await _aiosqlite.connect(checkpointer_path)
        checkpointer = _AsyncSqliteSaver(_checkpointer_conn)

    workflow = build_workflow(
        agent_configs=agent_configs,
        tools=tool_collections,
        max_agent_turns=max_agent_turns,
        progress=tracker,
        agent_timeouts=agent_timeouts,
        max_parallel_searches=research_cfg.get("max_parallel_searches", 3),
        human_in_the_loop=hitl_enabled,
        review_dir=hitl_review_dir,
        checkpointer=checkpointer,
        debug_dir=research_cfg.get("debug_dir"),
        topic=topic,
    )

    initial_state: ResearchState = {
        "topic": topic,
        "language": research_cfg.get("language", "zh"),
        "search_sources": research_cfg.get("search_sources", ["web", "arxiv", "wikipedia"]),
        "search_results": [],
        "analyses": [],
        "gaps": [],
        "citations": [],
        "research_round": 1,
        "max_rounds": research_cfg.get("max_rounds", 3),
        "quality_threshold": research_cfg.get("quality_threshold", 75),
        "current_phase": "init",
        "outline": [],
        "search_queries": [],
        "information_needs": [],
        "synthesized_findings": "",
        "draft_report": "",
        "final_report": "",
        "quality_score": 0.0,
        "overall_score": 0.0,
        "critique": {},
    }

    # Config with thread_id for checkpointing (required by interrupt())
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    console.print("[bold green]Starting research pipeline...[/bold green]\n")

    final_state: dict = {}
    with tracker:
        try:
            # First invocation — may stop at human_review interrupt
            final_state = await workflow.ainvoke(initial_state, config)

            # Handle HITL resume loop
            if hitl_enabled:
                gs = await workflow.aget_state(config)
                while gs and gs.interrupts:
                    interrupt_data = gs.interrupts[0].value

                    # Show review summary
                    console.print(Panel(
                        f"[bold]Human Review Required[/bold]\n\n"
                        f"Round: {interrupt_data.get('round', '?')}\n"
                        f"Quality Score: {interrupt_data.get('quality_score', '?')}/100\n"
                        f"Gaps: {interrupt_data.get('gaps', [])}\n\n"
                        f"[bold]Critique Scores:[/bold]\n{interrupt_data.get('scores', 'N/A')}\n\n"
                        f"[green]完整报告: {interrupt_data.get('report_file', 'N/A')}[/green]\n\n"
                        f"[dim]输入 approve / revise: <反馈> / abort[/dim]",
                        style="yellow",
                        title="Human-in-the-Loop",
                    ))

                    # Get user decision
                    decision = typer.prompt("决策").strip()
                    console.print(f"[dim]Resuming with: {decision}[/dim]\n")

                    # Resume workflow
                    final_state = await workflow.ainvoke(Command(resume=decision), config)
                    gs = await workflow.aget_state(config)

        except Exception as e:
            logger.exception("Research pipeline failed")
            console.print(f"\n[red]Pipeline error: {e}[/red]")
            raise
        finally:
            if mcp_manager:
                await mcp_manager.disconnect_all()
            if _checkpointer_conn is not None:
                await _checkpointer_conn.close()

    report = final_state.get("final_report", final_state.get("draft_report", "No report generated."))
    quality = final_state.get("quality_score", 0)
    rounds = final_state.get("research_round", 1)

    console.print(f"\n[bold]Quality Score:[/bold] {quality:.0f}/100 | [bold]Rounds:[/bold] {rounds}")

    # Save report
    output_dir = Path(research_cfg.get("output_dir", "./reports"))
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_topic = "".join(c if c.isalnum() or c in '-_' else '_' for c in topic)[:50]
    report_path = output_dir / f"report-{safe_topic}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    report_path.write_text(report, encoding="utf-8")

    console.print(f"\n[green]Report saved to:[/green] {report_path}")
    console.print("\n[bold]Report Preview:[/bold]\n")
    preview = report[:1000] + ("..." if len(report) > 1000 else "")
    console.print(Markdown(preview))


@app.command()
def research(
    topic: str = typer.Argument(..., help="Research topic to investigate"),
    config: str = typer.Option("config/research.yaml", help="Path to research config YAML"),
    agents: str = typer.Option("config/agents.yaml", help="Path to agent LLM config YAML"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Run the multi-agent research pipeline on a topic."""
    # Configure loguru: remove default handler, then add handlers based on verbosity
    logger.remove()
    if verbose:
        logger.add(
            sys.stderr,
            level="INFO",
            format="<level>[{name}]</level> {message}",
        )
    else:
        logger.add(sys.stderr, level="ERROR")

    # Also suppress noisy stdlib logging from libraries (mcp uses stdlib, not loguru)
    if not verbose:
        import logging as _logging
        # Configure root logger before any library calls basicConfig
        _logging.basicConfig(level=_logging.WARNING, format="%(message)s", force=True)

    asyncio.run(run_research(topic, agents, config))


if __name__ == "__main__":
    app()
