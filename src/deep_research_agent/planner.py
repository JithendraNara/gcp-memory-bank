"""Deterministic planner for Deep Research Agent v0."""

from __future__ import annotations

import re

from .models import ResearchPlan, ResearchStep

_STOPWORDS = {
    "about",
    "after",
    "against",
    "between",
    "compare",
    "could",
    "does",
    "from",
    "have",
    "into",
    "latest",
    "should",
    "that",
    "their",
    "there",
    "this",
    "with",
    "would",
}


def keyword_terms(question: str, *, max_terms: int = 8) -> list[str]:
    """Extract stable query terms without external dependencies."""

    words = re.findall(r"[A-Za-z][A-Za-z0-9_.-]+", question.lower())
    terms: list[str] = []
    seen: set[str] = set()
    for word in words:
        if len(word) < 3 or word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        terms.append(word)
        if len(terms) >= max_terms:
            break
    return terms


def build_plan(question: str) -> ResearchPlan:
    """Build a compact fan-out/fan-in research plan for a question."""

    normalized = " ".join(question.split()).strip()
    if not normalized:
        raise ValueError("question must not be empty")

    terms = keyword_terms(normalized)
    core = " ".join(terms[:5]) or normalized
    steps = (
        ResearchStep(id="scope", query=normalized, intent="Define the exact question and decision context."),
        ResearchStep(id="current", query=f"{core} current evidence", intent="Find current high-signal evidence."),
        ResearchStep(id="tradeoffs", query=f"{core} tradeoffs limitations risks", intent="Collect trade-offs, limitations, and risks."),
    )
    return ResearchPlan(question=normalized, steps=steps)
