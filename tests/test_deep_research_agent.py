"""Tests for Deep Research Agent v0 scaffold."""

from __future__ import annotations

import pytest

from deep_research_agent import (
    CallableMemoryStore,
    DeepResearchAgent,
    SearchHit,
    normalize_memory_search_result,
    report_to_memory_fact,
)
from deep_research_agent.memory import InMemoryStore
from deep_research_agent.planner import build_plan, keyword_terms


class FakeSearchProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        self.queries.append(query)
        return [
            SearchHit(
                title=f"Result for {query}",
                url=f"https://example.com/{len(self.queries)}",
                snippet=f"Evidence for {query}",
                metadata={"confidence": 0.9},
            )
        ][:limit]


class DuplicateSearchProvider:
    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        return [
            SearchHit(title="Duplicate", url="https://example.com/same", snippet=f"First {query}"),
            SearchHit(title="Duplicate", url="https://example.com/same", snippet=f"Second {query}"),
        ]


def test_keyword_terms_removes_stopwords_and_duplicates() -> None:
    assert keyword_terms("Compare Memory Bank with Memory Graph tradeoffs") == [
        "memory",
        "bank",
        "graph",
        "tradeoffs",
    ]


def test_build_plan_has_three_named_steps() -> None:
    plan = build_plan("Should we build a Deep Research Agent on GCP Memory Bank?")
    assert plan.question == "Should we build a Deep Research Agent on GCP Memory Bank?"
    assert [step.id for step in plan.steps] == ["scope", "current", "tradeoffs"]


def test_empty_question_rejected() -> None:
    with pytest.raises(ValueError, match="question"):
        build_plan("   ")


def test_agent_runs_plan_search_synthesis_and_memory_persistence() -> None:
    search = FakeSearchProvider()
    memory = InMemoryStore(["Deep Research Agent work must use exhaustive verification before production."])
    agent = DeepResearchAgent(search_provider=search, memory_store=memory)

    report = agent.run("Build a Deep Research Agent with GCP Memory Bank")

    assert len(search.queries) == 3
    assert len(report.evidence) == 3
    assert report.citations == (
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    )
    assert "Relevant memory" in report.answer
    assert "[Source: https://example.com/1]" in report.answer
    assert memory.reports == [report]


def test_agent_deduplicates_evidence_by_url() -> None:
    agent = DeepResearchAgent(search_provider=DuplicateSearchProvider())
    report = agent.run("Deduplicate sources", persist=False)
    assert len(report.evidence) == 1
    assert report.citations == ("https://example.com/same",)


def test_search_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="search_limit"):
        DeepResearchAgent(search_provider=FakeSearchProvider(), search_limit_per_step=0)


def test_callable_memory_store_recalls_and_stores_report() -> None:
    stored = []

    def recall_fn(query: str, top_k: int) -> list[str]:
        assert "Deep Research" in query
        assert top_k == 5
        return ["Deep Research Agent should preserve citations."]

    def store_fn(report) -> None:
        stored.append(report_to_memory_fact(report))

    agent = DeepResearchAgent(
        search_provider=FakeSearchProvider(),
        memory_store=CallableMemoryStore(recall_fn=recall_fn, store_fn=store_fn),
    )
    report = agent.run("Deep Research Agent memory adapter")

    assert "Relevant memory" in report.answer
    assert stored and "Sources: https://example.com/1" in stored[0]


def test_normalize_memory_search_result_shapes() -> None:
    assert normalize_memory_search_result(None) == []
    assert normalize_memory_search_result("- one\n- two") == ["one", "two"]
    assert normalize_memory_search_result({"memories": [{"fact": "fact one"}]}) == ["fact one"]
    assert normalize_memory_search_result([{"text": "fact two"}, "fact three"]) == [
        "fact two",
        "fact three",
    ]
