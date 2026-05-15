"""Memory adapters for Deep Research Agent v0."""

from __future__ import annotations

from collections.abc import Callable

from .models import ResearchReport

ProfileFactsFn = Callable[[], list[str]]


class NullMemoryStore:
    """No-op memory store for tests, local demos, and offline runs."""

    def recall(self, query: str, *, top_k: int = 5) -> list[str]:
        return []

    def store_report(self, report: ResearchReport) -> None:
        return None




class ProfileAugmentedMemoryStore:
    """MemoryStore decorator that prepends structured profile context to recall.

    Use this around a normal MemoryStore when the host runtime can provide GCP
    Memory Bank `memory_profiles` / `retrieve_profiles(...)` output. Profile
    facts are read-only runtime context; report persistence delegates to the
    inner store unchanged.
    """

    def __init__(self, *, inner, profile_facts_fn: ProfileFactsFn) -> None:
        self.inner = inner
        self._profile_facts_fn = profile_facts_fn

    def recall(self, query: str, *, top_k: int = 5) -> list[str]:
        profile_facts = list(self._profile_facts_fn())
        recalled = list(self.inner.recall(query, top_k=top_k))
        return profile_facts + recalled

    def store_report(self, report: ResearchReport) -> None:
        self.inner.store_report(report)

class InMemoryStore:
    """Small in-process memory store used by tests and demos."""

    def __init__(self, facts: list[str] | None = None) -> None:
        self.facts = list(facts or [])
        self.reports: list[ResearchReport] = []

    def recall(self, query: str, *, top_k: int = 5) -> list[str]:
        words = {w.lower().strip(".,:;!?()[]{}") for w in query.split() if len(w) > 2}
        scored: list[tuple[int, str]] = []
        for fact in self.facts:
            score = sum(1 for word in words if word in fact.lower())
            if score:
                scored.append((score, fact))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [fact for _, fact in scored[:top_k]]

    def store_report(self, report: ResearchReport) -> None:
        self.reports.append(report)
        if report.answer:
            self.facts.append(report.answer)
