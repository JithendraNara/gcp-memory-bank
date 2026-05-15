"""Memory promotion rules for Deep Research Agent reports."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ResearchReport


@dataclass(frozen=True)
class MemoryPromotionPolicy:
    """Decide whether a report is durable enough for long-term memory.

    The default policy avoids storing weak, uncited, or boilerplate failure
    reports.  It is deliberately small and deterministic so provider-backed
    code can apply it before calling GCP Memory Bank.
    """

    min_citations: int = 1
    min_answer_chars: int = 80
    blocked_phrases: tuple[str, ...] = (
        "couldn't find enough evidence",
        "no evidence collected",
    )

    def should_store(self, report: ResearchReport) -> bool:
        if len(report.citations) < self.min_citations:
            return False
        normalized_answer = report.answer.strip().lower()
        if len(normalized_answer) < self.min_answer_chars:
            return False
        return not any(phrase in normalized_answer for phrase in self.blocked_phrases)


class PromotingMemoryStore:
    """MemoryStore decorator that applies promotion rules before persistence."""

    def __init__(self, *, inner, policy: MemoryPromotionPolicy | None = None) -> None:
        self.inner = inner
        self.policy = policy or MemoryPromotionPolicy()

    def recall(self, query: str, *, top_k: int = 5) -> list[str]:
        return list(self.inner.recall(query, top_k=top_k))

    def store_report(self, report: ResearchReport) -> None:
        if self.policy.should_store(report):
            self.inner.store_report(report)
