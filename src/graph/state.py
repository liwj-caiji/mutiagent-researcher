"""ResearchState — shared state across the LangGraph workflow.

Uses TypedDict with Annotated reducers to support parallel agent execution
where results are automatically merged.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    """Shared state across all nodes in the research pipeline.

    Some fields use Annotated reducers:
        - operator.add: appends new items to the list (for parallel agents)
        - add_messages: LangGraph's message list reducer
    """

    # ── Input ──
    topic: str
    language: str  # "zh" or "en"
    search_sources: list[str]  # ["web", "arxiv", "wikipedia"]

    # ── Planner outputs ──
    outline: list[dict]  # [{section, key_points, subsections}]
    search_queries: list[str]  # Generated search queries
    information_needs: list[str]

    # ── Searcher outputs (Annotated for parallel fan-out) ──
    search_results: Annotated[list[dict], operator.add]

    # ── Analyst outputs (Annotated for parallel fan-out) ──
    analyses: Annotated[list[dict], operator.add]

    # ── Synthesizer outputs ──
    synthesized_findings: str

    # ── Writer outputs ──
    draft_report: str
    final_report: str
    citations: list[dict]

    # ── Critic outputs ──
    quality_score: float
    critique: dict  # {scores, strengths, weaknesses, gaps, recommendation}
    gaps: list[str]

    # ── Flow control ──
    current_phase: str  # init | planned | searched | analyzed | synthesized | drafted | reviewed | awaiting_human | formatted
    research_round: int
    max_rounds: int
    quality_threshold: float
    overall_score: float

    # ── Human-in-the-loop ──
    human_decision: str  # "approve" | "revise" | "abort"
    review_path: str  # file path to the draft report for human review

    # ── Messages (for LangGraph checkpointing) ──
    messages: Annotated[list, add_messages]
