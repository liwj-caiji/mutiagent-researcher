"""Main entry point — CLI for the multi-agent research assistant.

Usage:
    cd C:/Users/liwenjie/Desktop/muti_agent
    uv run python -m src.main "Research topic here"
    uv run python -m src.main "研究主题" --language zh
"""

from __future__ import annotations

import asyncio
import json
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


# ── Run record persistence (for crash recovery) ──────────────────────────────

RUNS_FILE = Path("./data/runs.json")


def save_run_record(thread_id: str, topic: str, status: str = "running", **extra) -> None:
    """Save or update a run record for crash recovery tracking."""
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if RUNS_FILE.exists():
        try:
            records = json.loads(RUNS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    now = datetime.now().isoformat()
    for r in records:
        if r.get("thread_id") == thread_id:
            r.update({"status": status, "updated_at": now, **extra})
            break
    else:
        records.append({
            "thread_id": thread_id,
            "topic": topic,
            "status": status,
            "started_at": now,
            "updated_at": now,
            **extra,
        })

    RUNS_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def load_runs(status_filter: list[str] | None = None) -> list[dict]:
    """Load run records, optionally filtered by status."""
    if not RUNS_FILE.exists():
        return []
    try:
        records = json.loads(RUNS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        return []
    if status_filter:
        records = [r for r in records if r.get("status") in status_filter]
    return sorted(records, key=lambda r: r.get("updated_at", ""), reverse=True)


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


def _show_hitl_prompt(interrupt_data: dict) -> None:
    """Display the HITL review prompt to the user."""
    console.print(Panel(
        f"[bold]Human Review Required[/bold]\n\n"
        f"Round: {interrupt_data.get('round', '?')}\n"
        f"Quality Score: {interrupt_data.get('quality_score', '?')}/100\n"
        f"Gaps: {interrupt_data.get('gaps', [])}\n\n"
        f"[bold]Critique Scores:[/bold]\n{interrupt_data.get('scores', 'N/A')}\n\n"
        f"[green]Full report: {interrupt_data.get('report_file', 'N/A')}[/green]",
        style="yellow",
        title="Human-in-the-Loop",
    ))


async def _handle_hitl_loop(workflow, config: dict, tracker: ProgressTracker) -> dict:
    """Process HITL interrupts in a loop. Returns the final state after all reviews."""
    final_state: dict = {}
    gs = await workflow.aget_state(config)
    while gs and gs.interrupts:
        interrupt_data = gs.interrupts[0].value
        _show_hitl_prompt(interrupt_data)

        tracker.pause()
        try:
            console.print()
            decision = console.input(
                "[yellow]👉 决策[/yellow] "
                "[dim](approve / revise: <意见> / abort)[/dim]: "
            ).strip()
        finally:
            tracker.resume()
        console.print(f"\n[dim]已收到: {decision}[/dim]\n")

        final_state = await workflow.ainvoke(Command(resume=decision), config)
        gs = await workflow.aget_state(config)
    return final_state


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

    # Generate run ID for checkpointing and crash recovery
    thread_id = str(uuid.uuid4())
    console.print(f"[bold cyan]Run ID:[/bold cyan] {thread_id}")
    console.print(f"[dim]If interrupted, resume with: uv run python -m src.main resume --last[/dim]")
    save_run_record(thread_id, topic, status="running")

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
        "accumulated_knowledge": [],
        "round_history": [],
        "search_feedback": [],
    }

    # Config with the run ID for checkpointing (required by interrupt())
    config = {"configurable": {"thread_id": thread_id}}

    console.print("[bold green]Starting research pipeline...[/bold green]\n")

    final_state: dict = {}
    with tracker:
        try:
            # First invocation — may stop at human_review interrupt
            final_state = await workflow.ainvoke(initial_state, config)

            # Handle HITL resume loop
            if hitl_enabled:
                hitl_result = await _handle_hitl_loop(workflow, config, tracker)
                if hitl_result:
                    final_state = hitl_result

        except Exception as e:
            logger.exception("Research pipeline failed")
            save_run_record(
                thread_id, topic, status="crashed",
                current_phase=final_state.get("current_phase", "unknown") if final_state else "unknown",
                error=str(e)[:300],
            )
            console.print(f"\n[red]Pipeline error: {e}[/red]")
            console.print(f"[yellow]Resume with: uv run python -m src.main resume --last[/yellow]")
            raise
        finally:
            try:
                if mcp_manager:
                    await mcp_manager.disconnect_all()
            except RuntimeError:
                pass  # event loop closing, ignore cleanup noise
            try:
                if _checkpointer_conn is not None:
                    await _checkpointer_conn.close()
            except RuntimeError:
                pass

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

    save_run_record(
        thread_id, topic, status="completed",
        current_phase="formatted",
        quality_score=quality,
        research_round=rounds,
    )


async def _resume_run(
    thread_id: str, agents_config_path: str, research_config_path: str
) -> None:
    """Resume a previously interrupted research run from its last checkpoint."""
    agents_cfg = load_yaml(agents_config_path)
    research_cfg = load_yaml(research_config_path)

    agent_configs = build_agent_configs(agents_cfg)
    tool_collections, mcp_manager = await build_tool_collections(research_cfg)

    max_agent_turns = research_cfg.get("max_agent_turns", {})
    agent_timeouts = research_cfg.get("agent_timeouts", {})

    hitl_cfg = research_cfg.get("human_in_the_loop", {})
    hitl_enabled = bool(hitl_cfg.get("enabled", False))
    hitl_review_dir = hitl_cfg.get("review_dir", "./reviews")
    checkpointer_path = research_cfg.get("checkpointer_path", "./data/checkpoints.sqlite")

    # Ensure checkpointer is used (required for resume)
    import aiosqlite as _aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as _AsyncSqliteSaver

    os.makedirs(os.path.dirname(checkpointer_path) or ".", exist_ok=True)
    _checkpointer_conn = await _aiosqlite.connect(checkpointer_path)
    checkpointer = _AsyncSqliteSaver(_checkpointer_conn)

    config = {"configurable": {"thread_id": thread_id}}

    tracker = ProgressTracker(console, max_agent_turns)

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
        topic="(resuming)",
    )

    # Load state from checkpointer
    current_state = await workflow.aget_state(config)

    if current_state is None or not current_state.values:
        console.print(f"[red]No checkpoint found for run {thread_id}[/red]")
        console.print("[dim]The run may have completed, been cleaned up, or the ID is incorrect.[/dim]")
        return

    state_values = current_state.values
    topic = state_values.get("topic", "unknown")
    current_phase = state_values.get("current_phase", "unknown")
    research_round = state_values.get("research_round", 1)

    console.print(Panel(
        f"Resuming: [bold]{topic}[/bold]\n"
        f"Run ID: {thread_id}\n"
        f"Last phase: {current_phase} | Round: {research_round}",
        style="cyan",
    ))

    save_run_record(thread_id, topic, status="running", current_phase=current_phase)

    final_state: dict = {}
    with tracker:
        try:
            if current_state.interrupts:
                # Pending HITL review — handle it first
                interrupt_data = current_state.interrupts[0].value
                _show_hitl_prompt(interrupt_data)
                tracker.pause()
                try:
                    decision = console.input(
                        "[yellow]👉 决策[/yellow] "
                        "[dim](approve / revise: <意见> / abort)[/dim]: "
                    ).strip()
                finally:
                    tracker.resume()
                console.print(f"\n[dim]已收到: {decision}[/dim]\n")
                final_state = await workflow.ainvoke(Command(resume=decision), config)

                # Continue with any follow-up HITL interrupts
                if hitl_enabled:
                    hitl_result = await _handle_hitl_loop(workflow, config, tracker)
                    if hitl_result:
                        final_state = hitl_result
            else:
                # Continue from last checkpoint (crash recovery)
                final_state = await workflow.ainvoke(None, config)

                # Handle any HITL interrupts that appear during continued execution
                if hitl_enabled:
                    hitl_result = await _handle_hitl_loop(workflow, config, tracker)
                    if hitl_result:
                        final_state = hitl_result

        except Exception as e:
            logger.exception("Research pipeline failed during resume")
            save_run_record(
                thread_id, topic, status="crashed",
                current_phase=final_state.get("current_phase", "unknown") if final_state else current_phase,
                error=str(e)[:300],
            )
            console.print(f"\n[red]Pipeline error: {e}[/red]")
            console.print(f"[yellow]Resume again with: uv run python -m src.main resume {thread_id}[/yellow]")
            raise
        finally:
            try:
                if mcp_manager:
                    await mcp_manager.disconnect_all()
            except RuntimeError:
                pass
            try:
                if _checkpointer_conn is not None:
                    await _checkpointer_conn.close()
            except RuntimeError:
                pass

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

    save_run_record(
        thread_id, topic, status="completed",
        current_phase="formatted",
        quality_score=quality,
        research_round=rounds,
    )


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


@app.command()
def resume(
    thread_id: str = typer.Argument(None, help="Run ID to resume (omit to list all interrupted runs)"),
    last: bool = typer.Option(False, "--last", "-l", help="Resume the most recently interrupted run"),
    config: str = typer.Option("config/research.yaml", help="Path to research config YAML"),
    agents: str = typer.Option("config/agents.yaml", help="Path to agent LLM config YAML"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Resume a crashed or interrupted research run.

    Examples:
        uv run python -m src.main resume --last    # resume the latest interrupted run
        uv run python -m src.main resume <run-id>  # resume a specific run
        uv run python -m src.main resume           # list all interrupted runs
    """
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="INFO", format="<level>[{name}]</level> {message}")
    else:
        logger.add(sys.stderr, level="ERROR")

    if not verbose:
        import logging as _logging
        _logging.basicConfig(level=_logging.WARNING, format="%(message)s", force=True)

    # --last: pick the most recent interrupted run
    if last:
        runs = load_runs(status_filter=["running", "crashed"])
        if not runs:
            console.print("[dim]No interrupted runs found.[/dim]")
            return
        thread_id = runs[0]["thread_id"]  # load_runs returns newest first
        console.print(f"[dim]Resuming latest run: {thread_id} (topic: {runs[0].get('topic', '?')[:40]})[/dim]")

    if thread_id is None:
        # List interrupted runs
        runs = load_runs(status_filter=["running", "crashed"])
        if not runs:
            console.print("[dim]No interrupted runs found.[/dim]")
            console.print("[dim]Run IDs are displayed when starting a new research task.[/dim]")
            return

        console.print("\n[bold]Interrupted runs:[/bold]\n")
        for r in runs:
            status_color = "red" if r.get("status") == "crashed" else "yellow"
            started = r.get("started_at", "?")[:16]
            console.print(
                f"  [{status_color}]{r['thread_id']}[/{status_color}]\n"
                f"    topic: {r.get('topic', '?')[:50]}\n"
                f"    started: {started} | phase: {r.get('current_phase', '?')} | "
                f"status: {r.get('status', '?')}"
            )
            if r.get("error"):
                console.print(f"    [red]error: {r['error'][:120]}[/red]")
            console.print()
        console.print(f"[dim]Resume latest: uv run python -m src.main resume --last[/dim]")
        console.print(f"[dim]Resume specific: uv run python -m src.main resume <run-id>[/dim]")
        return

    asyncio.run(_resume_run(thread_id, agents, config))


if __name__ == "__main__":
    app()
