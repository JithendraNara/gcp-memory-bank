"""Runtime adapter tests for live Deep Research Agent wiring."""

from __future__ import annotations

from deep_research_agent import DeepResearchAgent, SearchHit
from deep_research_agent.adapters import CallableSearchProvider, normalize_profile_result
from deep_research_agent.memory import InMemoryStore, ProfileAugmentedMemoryStore
from deep_research_agent.promotion import MemoryPromotionPolicy, PromotingMemoryStore
from deep_research_agent.synthesis import CitationPreservingSynthesizer


class CapturingStore:
    def __init__(self) -> None:
        self.reports = []

    def recall(self, query: str, *, top_k: int = 5) -> list[str]:
        return []

    def store_report(self, report) -> None:
        self.reports.append(report)


def test_callable_search_provider_normalizes_minimax_web_search_shape() -> None:
    calls = []

    def search_fn(query: str, limit: int):
        calls.append((query, limit))
        return {
            "organic": [
                {
                    "title": "Result A",
                    "link": "https://example.com/a",
                    "snippet": "Snippet A",
                    "date": "2026-05-05",
                },
                {
                    "title": "Result B",
                    "link": "https://example.com/b",
                    "snippet": "Snippet B",
                },
            ]
        }

    provider = CallableSearchProvider(search_fn=search_fn)

    hits = provider.search("deep research memory", limit=1)

    assert calls == [("deep research memory", 1)]
    assert hits == [
        SearchHit(
            title="Result A",
            url="https://example.com/a",
            snippet="Snippet A",
            published_at="2026-05-05",
            metadata={},
        )
    ]


def test_callable_search_provider_normalizes_common_result_shapes() -> None:
    provider = CallableSearchProvider(
        search_fn=lambda query, limit: [
            {"title": "One", "url": "https://example.com/one", "content": "One content"},
            "https://example.com/two",
        ]
    )

    hits = provider.search("anything", limit=5)

    assert [hit.url for hit in hits] == ["https://example.com/one", "https://example.com/two"]
    assert hits[0].snippet == "One content"
    assert hits[1].title == "https://example.com/two"


def test_normalize_profile_result_flattens_memory_bank_structured_profiles() -> None:
    raw = {
        "profiles": {
            "hermes-profile": {
                "profile": {
                    "communication_style": "direct, no fluff",
                    "technical_stack": ["Hermes", "GCP Memory Bank"],
                    "operational_preferences": {"grounding": "inline citations"},
                }
            }
        }
    }

    facts = normalize_profile_result(raw)

    assert "Profile communication_style: direct, no fluff" in facts
    assert "Profile technical_stack: Hermes; GCP Memory Bank" in facts
    assert "Profile operational_preferences.grounding: inline citations" in facts


def test_profile_augmented_memory_store_prepends_profile_context_to_recall() -> None:
    raw = {"profiles": {"hermes-profile": {"profile": {"communication_style": "direct"}}}}
    inner = InMemoryStore(["Deep Research Agent should preserve citations."])
    store = ProfileAugmentedMemoryStore(
        inner=inner,
        profile_facts_fn=lambda: normalize_profile_result(raw),
    )

    recalled = store.recall("Deep Research Agent", top_k=5)

    assert recalled[0] == "Profile communication_style: direct"
    assert "Deep Research Agent should preserve citations." in recalled


def test_agent_injects_structured_profile_context_into_synthesis() -> None:
    captured = {}

    def synthesize_fn(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Answer with cited source. [Source: https://example.com/one]"

    raw = {"profiles": {"hermes-profile": {"profile": {"communication_style": "direct"}}}}
    agent = DeepResearchAgent(
        search_provider=CallableSearchProvider(
            search_fn=lambda query, limit: [
                {"title": "One", "url": "https://example.com/one", "snippet": "One quote"},
            ]
        ),
        memory_store=ProfileAugmentedMemoryStore(
            inner=InMemoryStore(),
            profile_facts_fn=lambda: normalize_profile_result(raw),
        ),
        synthesizer=CitationPreservingSynthesizer(synthesize_fn=synthesize_fn),
    )

    agent.run("profile-aware research", persist=False)

    assert "Profile communication_style: direct" in captured["prompt"]


def test_citation_preserving_synthesizer_appends_missing_sources() -> None:
    def synthesize_fn(prompt: str) -> str:
        assert "https://example.com/one" in prompt
        return "Provider answer mentioned only one source. [Source: https://example.com/one]"

    agent = DeepResearchAgent(
        search_provider=CallableSearchProvider(
            search_fn=lambda query, limit: [
                {"title": "One", "url": "https://example.com/one", "snippet": "One quote"},
                {"title": "Two", "url": "https://example.com/two", "snippet": "Two quote"},
            ]
        ),
        synthesizer=CitationPreservingSynthesizer(synthesize_fn=synthesize_fn),
    )

    report = agent.run("citation preservation", persist=False)

    assert "[Source: https://example.com/one]" in report.answer
    assert "[Source: https://example.com/two]" in report.answer


def test_memory_promotion_policy_blocks_weak_or_empty_reports() -> None:
    weak_agent = DeepResearchAgent(
        search_provider=CallableSearchProvider(search_fn=lambda query, limit: []),
        memory_store=CapturingStore(),
    )
    weak_report = weak_agent.run("no evidence", persist=False)

    policy = MemoryPromotionPolicy(min_citations=1, min_answer_chars=20)

    assert not policy.should_store(weak_report)


def test_promoting_memory_store_only_persists_policy_approved_reports() -> None:
    inner = CapturingStore()
    store = PromotingMemoryStore(inner=inner, policy=MemoryPromotionPolicy(min_citations=2, min_answer_chars=20))
    agent = DeepResearchAgent(
        search_provider=CallableSearchProvider(
            search_fn=lambda query, limit: [
                {"title": "One", "url": "https://example.com/one", "snippet": "One quote"},
                {"title": "Two", "url": "https://example.com/two", "snippet": "Two quote"},
            ]
        ),
        memory_store=store,
    )

    report = agent.run("store durable citation-backed report")

    assert inner.reports == [report]
