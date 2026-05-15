"""Core orchestration for Deep Research Agent v0."""

from __future__ import annotations

from dataclasses import dataclass

from .interfaces import MemoryStore, SearchProvider, Synthesizer
from .memory import NullMemoryStore
from .models import Evidence, ResearchPlan, ResearchReport, SearchHit
from .planner import build_plan
from .synthesis import ExtractiveSynthesizer


@dataclass
class DeepResearchAgent:
    """Small testable research orchestrator.

    The agent intentionally separates planning, search, memory recall, evidence
    normalization, and synthesis.  Provider integrations can be swapped without
    changing the orchestration contract.
    """

    search_provider: SearchProvider
    memory_store: MemoryStore | None = None
    synthesizer: Synthesizer | None = None
    search_limit_per_step: int = 3

    def __post_init__(self) -> None:
        if self.search_limit_per_step < 1:
            raise ValueError("search_limit_per_step must be >= 1")
        if self.memory_store is None:
            self.memory_store = NullMemoryStore()
        if self.synthesizer is None:
            self.synthesizer = ExtractiveSynthesizer()

    def plan(self, question: str) -> ResearchPlan:
        return build_plan(question)

    def run(self, question: str, *, persist: bool = True) -> ResearchReport:
        plan = self.plan(question)
        memories = self.memory_store.recall(plan.question, top_k=5) if self.memory_store else []
        evidence = self.collect_evidence(plan)
        answer = self.synthesizer.synthesize(
            question=plan.question,
            plan=plan,
            evidence=evidence,
            memories=memories,
        )
        report = ResearchReport(question=plan.question, answer=answer, evidence=tuple(evidence))
        if persist and self.memory_store is not None:
            self.memory_store.store_report(report)
        return report

    def collect_evidence(self, plan: ResearchPlan) -> list[Evidence]:
        evidence: list[Evidence] = []
        seen_urls: set[str] = set()
        for step in plan.steps:
            hits = self.search_provider.search(step.query, limit=self.search_limit_per_step)
            for hit in hits:
                normalized = self._hit_to_evidence(step_id=step.id, hit=hit)
                if normalized.url in seen_urls:
                    continue
                seen_urls.add(normalized.url)
                evidence.append(normalized)
        return evidence

    @staticmethod
    def _hit_to_evidence(*, step_id: str, hit: SearchHit) -> Evidence:
        quote = hit.snippet.strip() or hit.title.strip()
        if not hit.url:
            raise ValueError("search hit url must not be empty")
        return Evidence(
            step_id=step_id,
            title=hit.title.strip() or hit.url,
            url=hit.url,
            quote=quote,
            confidence=float(hit.metadata.get("confidence", 0.5)) if hit.metadata else 0.5,
        )
