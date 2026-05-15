"""Offline demo for Deep Research Agent v0.

Run from repo root:
    python examples/deep_research_agent_demo.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deep_research_agent import DeepResearchAgent, SearchHit, normalize_profile_result  # noqa: E402
from deep_research_agent.memory import InMemoryStore, ProfileAugmentedMemoryStore  # noqa: E402


class StaticSearchProvider:
    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        hits = [
            SearchHit(
                title="Memory-guided research orchestration",
                url="https://example.com/memory-guided-research",
                snippet=f"A research agent should preserve decisions and citations while answering: {query}",
            ),
            SearchHit(
                title="Verification-first autonomous agents",
                url="https://example.com/verification-first-agents",
                snippet="Autonomous research systems need tests, audit trails, and rollback points.",
            ),
        ]
        return hits[:limit]


def main() -> None:
    raw_profile = {
        "profiles": {
            "hermes-profile": {
                "profile": {
                    "communication_style": "direct, no fluff",
                    "operational_preferences": {"verification": "exhaustive cross-check before production"},
                }
            }
        }
    }
    memory = ProfileAugmentedMemoryStore(
        inner=InMemoryStore(["User prefers exhaustive cross-verification before production."]),
        profile_facts_fn=lambda: normalize_profile_result(raw_profile),
    )
    agent = DeepResearchAgent(search_provider=StaticSearchProvider(), memory_store=memory)
    report = agent.run("How should we build Deep Research Agent v0 on GCP Memory Bank?")
    print(report.answer)


if __name__ == "__main__":
    main()
