"""Typed models for Deep Research Agent v0."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ResearchStep:
    """A single research work item derived from the user's question."""

    id: str
    query: str
    intent: str


@dataclass(frozen=True)
class ResearchPlan:
    """Deterministic plan for a research run."""

    question: str
    steps: tuple[ResearchStep, ...]


@dataclass(frozen=True)
class SearchHit:
    """A result returned by a search provider."""

    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """A normalized evidence item used for synthesis and citations."""

    step_id: str
    title: str
    url: str
    quote: str
    confidence: float = 0.5

    def citation(self) -> str:
        return f"[Source: {self.url}]"


@dataclass(frozen=True)
class ResearchReport:
    """Final report plus provenance."""

    question: str
    answer: str
    evidence: tuple[Evidence, ...]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def citations(self) -> tuple[str, ...]:
        """Return unique source URLs in evidence order."""

        seen: set[str] = set()
        ordered: list[str] = []
        for item in self.evidence:
            if item.url not in seen:
                seen.add(item.url)
                ordered.append(item.url)
        return tuple(ordered)
