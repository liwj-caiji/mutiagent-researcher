"""ProgressTracker — Rich Live display for multi-agent research pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import ClassVar

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text


@dataclass
class _AgentSlot:
    name: str
    status: str = "pending"  # pending | running | finished | error | timeout
    current_step: int = 0
    max_steps: int = 0
    start_time: float | None = None
    detail: str = ""

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time


class ProgressTracker:
    """Real-time multi-agent progress display using Rich Live.

    Usage as context manager:
        tracker = ProgressTracker(console, max_agent_turns)
        with tracker:
            tracker.agent_started("planner", 3)
            ...
            tracker.agent_finished("planner")
    """

    _STATUS_ICONS: ClassVar[dict[str, str]] = {
        "pending":  "[dim]○[/dim]",
        "running":  "[yellow]◎[/yellow]",
        "finished": "[green]●[/green]",
        "error":    "[red]✕[/red]",
        "timeout":  "[red]⏱[/red]",
    }

    _STATUS_LABELS: ClassVar[dict[str, str]] = {
        "pending":  "[dim]Pending[/dim]",
        "running":  "[yellow]Running[/yellow]",
        "finished": "[green]Done[/green]",
        "error":    "[red]Error[/red]",
        "timeout":  "[red]Timeout[/red]",
    }

    _AGENT_ORDER: ClassVar[list[str]] = [
        "planner", "searcher", "analyst", "synthesizer", "writer", "critic",
    ]

    def __init__(self, console: Console, max_agent_turns: dict[str, int] | None = None):
        self._console = console
        self._agents: dict[str, _AgentSlot] = {}
        self._pipeline_start = 0.0
        self._live: Live | None = None

        for name in self._AGENT_ORDER:
            max_s = (max_agent_turns or {}).get(name, 0)
            self._agents[name] = _AgentSlot(name=name, max_steps=max_s)

    # ── Public API ──────────────────────────────────────────────────────

    def agent_started(self, name: str, max_steps: int) -> None:
        a = self._agents.get(name)
        if a is None:
            return
        a.status = "running"
        a.start_time = time.monotonic()
        a.max_steps = max_steps
        a.current_step = 0
        a.detail = "starting..."
        self._refresh()

    def agent_step_update(self, name: str, step: int, _max_steps: int, detail: str) -> None:
        a = self._agents.get(name)
        if a is None or a.status != "running":
            return
        a.current_step = step
        a.detail = detail
        self._refresh()

    def agent_finished(self, name: str) -> None:
        a = self._agents.get(name)
        if a is None:
            return
        a.status = "finished"
        a.detail = ""
        self._refresh()

    def agent_timeout(self, name: str, timeout_s: float) -> None:
        a = self._agents.get(name)
        if a is None:
            return
        a.status = "timeout"
        a.detail = f"exceeded {timeout_s:.0f}s"
        self._refresh()

    def agent_error(self, name: str, msg: str) -> None:
        a = self._agents.get(name)
        if a is None:
            return
        a.status = "error"
        a.detail = msg[:40]
        self._refresh()

    # ── Context manager ─────────────────────────────────────────────────

    def __enter__(self) -> ProgressTracker:
        self._pipeline_start = time.monotonic()
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()
        return self

    def __exit__(self, *args) -> None:
        # Final render
        if self._live:
            self._live.update(self._render(final=True))
            self._live.stop()
            self._live = None
        # Leave the final table visible
        self._console.print(self._render(final=True))

    # ── Internal ────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _render(self, final: bool = False) -> Table:
        table = Table(
            title="[bold]Multi-Agent Research Pipeline[/bold]",
            title_justify="left",
            expand=True,
            show_header=True,
            header_style="bold dim",
        )
        table.add_column("Agent", style="cyan", width=14, no_wrap=True)
        table.add_column("Status", width=12)
        table.add_column("Steps", width=8, justify="right")
        table.add_column("Elapsed", width=10, justify="right")
        table.add_column("Detail", style="dim", width=28)

        finished_count = 0
        error_count = 0
        total = len(self._agents)

        for name in self._AGENT_ORDER:
            a = self._agents[name]
            icon = self._STATUS_ICONS.get(a.status, " ")
            label = self._STATUS_LABELS.get(a.status, a.status)
            step_str = f"{a.current_step}/{a.max_steps}" if a.status != "pending" else "-"
            elapsed_str = f"{a.elapsed:.0f}s" if a.start_time else "-"
            table.add_row(
                f"{icon} {name}",
                label,
                step_str,
                elapsed_str,
                a.detail,
            )

            if a.status == "finished":
                finished_count += 1
            elif a.status in ("error", "timeout"):
                error_count += 1

        # Summary footer
        total_elapsed = time.monotonic() - self._pipeline_start
        if final:
            summary = f"[bold]Pipeline complete[/bold] — {finished_count}/{total} agents finished"
        else:
            summary = f"{finished_count}/{total} completed"
        if error_count:
            summary += f", {error_count} failed"
        summary += f" | [dim]{total_elapsed:.0f}s elapsed[/dim]"

        table.add_section()
        table.add_row(
            Text(summary, style="bold" if final else ""),
            "", "", "", "",
        )

        return table
