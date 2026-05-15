"""Provider interfaces for Deep Research Agent v0."""

from __future__ import annotations

from typing import Protocol

from .models import Evidence, ResearchPlan, ResearchReport, SearchHit


class SearchProvider(Protocol):
    """Search backend contract.

    Implementations can wrap Google Search grounding, Gemini Deep Research,
    browser agents, MiniMax search, or deterministic test fakes.
    """

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        """Return ranked search hits for a query."""


class MemoryStore(Protocol):
    """Long-term memory contract used by research runs."""

    def recall(self, query: str, *, top_k: int = 5) -> list[str]:
        """Return relevant durable facts for the query."""

    def store_report(self, report: ResearchReport) -> None:
        """Persist durable learnings from a completed report."""


class Synthesizer(Protocol):
    """Report synthesis contract."""

    def synthesize(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        evidence: list[Evidence],
        memories: list[str],
    ) -> str:
        """Create a cited answer from evidence and recalled memories."""
