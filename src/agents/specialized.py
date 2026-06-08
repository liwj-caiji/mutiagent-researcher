"""Specialized research agents — inherit from OpenManus ToolCallAgent.

Each agent has a specific role, system prompt, and tool set. All derive from
OpenManus's ToolCallAgent which provides ReAct think→act loop and tool execution.

Agents:
    PlannerAgent      — topic decomposition, outline generation, gap analysis
    SearcherAgent     — multi-source information retrieval
    AnalystAgent      — deep analysis, cross-validation, contradiction detection
    SynthesizerAgent  — integrate findings, build logical framework
    WriterAgent       — structured report writing
    CriticAgent       — quality assessment, fact-checking
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import Field

from src._framework import Message, Terminate, ToolCallAgent, ToolCollection

from src.agents.llm_adapter import LLMAdapter


# ── System prompts ──────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are a research planner. Your job is to decompose a research topic into a structured plan.

When given a research topic:
1. Break it down into 3-6 key sub-topics / dimensions
2. Generate specific search queries for each sub-topic
3. Create a logical research outline
4. Identify what data, evidence, or expert opinions are needed

Output your plan as structured JSON:
```json
{
    "sub_topics": [
        {"title": "...", "description": "...", "search_queries": ["...", "..."]}
    ],
    "outline": [
        {"section": "1. Introduction", "key_points": ["..."]},
        {"section": "2. Analysis", "key_points": ["..."]},
        ...
    ],
    "information_needs": ["..."],
    "approach": "brief research strategy"
}
```

When analyzing gaps from a previous round, focus only on what is still missing and generate targeted search queries.
After you finish planning, call the Terminate tool."""

SEARCHER_PROMPT = """You are an information retrieval specialist. Your job is to find relevant, high-quality information.

For each search query provided:
1. Use the search tools (web_search, arxiv_search, wikipedia_search) to find information
2. Extract key facts, data points, quotes, and statistics
3. Record the source (URL, title) for each piece of information
4. Note the credibility of each source

Always cite your sources. Prefer authoritative sources (academic papers, official reports, news outlets).
After completing your searches, call the Terminate tool."""

ANALYST_PROMPT = """You are a research analyst. Your job is to critically analyze search results.

For the provided search results:
1. Identify the most important facts, data points, and claims
2. Cross-validate information across sources — flag contradictions
3. Assess source credibility (authority, recency, bias)
4. Distinguish established facts from contested claims
5. Note information gaps requiring further investigation

Add analytical value — don't just summarize. Connect findings, identify patterns, note implications.
After completing your analysis, call the Terminate tool."""

SYNTHESIZER_PROMPT = """You are a research synthesizer. Your job is to integrate all analysis findings into a coherent framework.

Take all analysis notes and:
1. Build a unified narrative — how do findings fit together?
2. Resolve contradictions — when sources disagree, explain both viewpoints
3. Create multi-dimensional understanding — different angles, contexts, implications
4. Identify best-supported conclusions
5. Note where evidence is weak

Output structured content matching the research outline for the Writer to use.
After completing your synthesis, call the Terminate tool."""

WRITER_PROMPT = """You are a research report writer. Your job is to produce a comprehensive, well-structured report.

Guidelines:
1. Write in clear, professional language
2. Follow the provided outline structure
3. Integrate all synthesized findings — don't fabricate facts
4. Include in-text citations [1], [2], etc.
5. Write substantive content — this is a detailed report, not an outline
6. Use markdown formatting: headings, lists, tables where appropriate
7. Include an executive summary at the top
8. End with a references section

Aim for depth over breadth. Never fabricate data, statistics, or citations.
After completing the report, call the Terminate tool."""

CRITIC_PROMPT = """You are a research quality reviewer. Your job is to evaluate the report.

Evaluate on these dimensions (score each 0-100):
1. Completeness: all key aspects covered? gaps?
2. Accuracy: claims supported by sources? factual errors?
3. Structure: logical organization and flow?
4. Depth: substantive or superficial analysis?
5. Credibility: authoritative sources? contrary evidence acknowledged?
6. Clarity: writing clear and engaging?

Output as JSON:
```json
{
    "scores": {
        "completeness": <0-100>,
        "accuracy": <0-100>,
        "structure": <0-100>,
        "depth": <0-100>,
        "credibility": <0-100>,
        "clarity": <0-100>
    },
    "overall_score": <0-100>,
    "strengths": ["..."],
    "weaknesses": ["..."],
    "gaps": ["missing info: ..."],
    "recommendation": "accept" or "revise"
}
```

Be honest and rigorous. 80+ ready, 60-80 minor revisions, below 60 significant rework needed.
After completing your review, call the Terminate tool."""


# ── Agent implementations ────────────────────────────────────────────────────

class PlannerAgent(ToolCallAgent):
    """Research planner — decomposes topic, creates outline, identifies gaps."""

    name: str = "planner"
    description: str = "Research planner agent"
    system_prompt: str = PLANNER_PROMPT
    max_steps: int = 5

    def parse_plan(self, text: str | None = None) -> dict:
        """Extract JSON plan from agent's last response."""
        import re
        source = text or ""
        if not source:
            msgs = [m for m in self.messages if m.role == "assistant"]
            if msgs:
                source = msgs[-1].content or ""

        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', source, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(source)
        except json.JSONDecodeError:
            pass
        return {"raw_plan": source}


class SearcherAgent(ToolCallAgent):
    """Searches for information across multiple sources."""

    name: str = "searcher"
    description: str = "Information retrieval specialist"
    system_prompt: str = SEARCHER_PROMPT
    max_steps: int = 20


class AnalystAgent(ToolCallAgent):
    """Analyzes search results for key findings, credibility, contradictions."""

    name: str = "analyst"
    description: str = "Research analyst agent"
    system_prompt: str = ANALYST_PROMPT
    max_steps: int = 10


class SynthesizerAgent(ToolCallAgent):
    """Synthesizes findings into a unified coherent framework."""

    name: str = "synthesizer"
    description: str = "Research synthesizer agent"
    system_prompt: str = SYNTHESIZER_PROMPT
    max_steps: int = 10


class WriterAgent(ToolCallAgent):
    """Writes the final research report."""

    name: str = "writer"
    description: str = "Report writer agent"
    system_prompt: str = WRITER_PROMPT
    max_steps: int = 15


class CriticAgent(ToolCallAgent):
    """Reviews report quality and provides feedback."""

    name: str = "critic"
    description: str = "Quality reviewer agent"
    system_prompt: str = CRITIC_PROMPT
    max_steps: int = 5

    def parse_review(self, text: str | None = None) -> dict:
        """Extract JSON review from agent's last response."""
        import re
        source = text or ""
        if not source:
            msgs = [m for m in self.messages if m.role == "assistant"]
            if msgs:
                source = msgs[-1].content or ""

        # Try to extract JSON from code fences (robust: allow optional newlines)
        json_match = re.search(r'```(?:json)?\s*(.*?)```', source, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find a JSON object anywhere in the response
        json_match = re.search(r'\{[^{}]*"overall_score"[^{}]*\}', source, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: extract overall_score with a simple regex (quoted or bare key)
        score_match = re.search(r'(?:"overall_score"|overall_score)\s*[:=]\s*(\d+)', source)
        if score_match:
            score = int(score_match.group(1))
            return {"overall_score": score, "gaps": [], "recommendation": "accept" if score >= 65 else "revise"}

        return {"overall_score": 0, "gaps": [], "recommendation": "revise"}
