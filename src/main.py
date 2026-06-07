"""Main entry point — CLI for the multi-agent research assistant.

Usage:
    cd C:/Users/liwenjie/Desktop/muti_agent
    uv run python -m src.main "Research topic here"
    uv run python -m src.main "研究主题" --language zh
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
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
from src.tools.search.tools import ArxivSearchTool, WebScraperTool, WebSearchTool, WikipediaSearchTool
from src.tools.export.tools import CitationFormatterTool, ReportSaverTool

app = typer.Typer(no_args_is_help=True)
console = Console()
logger = logging.getLogger(__name__)


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
            WebSearchTool(), ArxivSearchTool(),
            WikipediaSearchTool(), WebScraperTool(), terminate,
        ]
        writer_tools = [CitationFormatterTool(), terminate]
        return {
            "planner": ToolCollection(terminate),
            "searcher": ToolCollection(*search_tools),
            "analyst": ToolCollection(terminate),
            "synthesizer": ToolCollection(terminate),
            "writer": ToolCollection(*writer_tools),
            "critic": ToolCollection(terminate),
        }, None

    # MCP mode: connect to remote tool servers
    console.print("[dim]Connecting to MCP tool servers...[/dim]")
    tool_dict = {
        "planner": await mcp_manager.create_tool_collection([], [terminate]),
        "searcher": await mcp_manager.create_tool_collection(["search"], [terminate]),
        "analyst": await mcp_manager.create_tool_collection([], [terminate]),
        "synthesizer": await mcp_manager.create_tool_collection([], [terminate]),
        "writer": await mcp_manager.create_tool_collection(["export"], [terminate]),
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

    console.print(Panel(f"Building research workflow for: [bold]{topic}[/bold]", style="blue"))

    tracker = ProgressTracker(console, max_agent_turns)

    workflow = build_workflow(
        agent_configs=agent_configs,
        tools=tool_collections,
        use_checkpointer=research_cfg.get("use_checkpointer", False),
        checkpointer_path=research_cfg.get("checkpointer_path", "./data/checkpoints.sqlite"),
        max_agent_turns=max_agent_turns,
        progress=tracker,
        agent_timeouts=agent_timeouts,
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

    console.print("[bold green]Starting research pipeline...[/bold green]\n")

    with tracker:
        try:
            final_state = await workflow.ainvoke(initial_state)
        except Exception as e:
            logger.exception("Research pipeline failed")
            console.print(f"\n[red]Pipeline error: {e}[/red]")
            raise
        finally:
            if mcp_manager:
                await mcp_manager.disconnect_all()

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
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    asyncio.run(run_research(topic, agents, config))


if __name__ == "__main__":
    app()
