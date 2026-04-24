"""
Mock Memory Injection Test — verifies the retrieval + prompt injection pattern
without needing live GCP credentials. This simulates how Hermes would use Memory Bank.
"""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


# ── Mock Memory Bank (in-memory stand-in for GCP) ─────────────────────────

@dataclass
class MockMemory:
    fact: str
    topic: str = "general"
    created_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0


class MockMemoryStore:
    """In-memory store that mimics MemoryBankServiceClient behavior."""

    def __init__(self):
        self._memories: Dict[str, List[MockMemory]] = {}  # user_id -> memories

    def create_memory(self, user_id: str, fact: str, topic: str = "general") -> MockMemory:
        mem = MockMemory(fact=fact, topic=topic)
        self._memories.setdefault(user_id, []).append(mem)
        return mem

    def search_memory(self, user_id: str, query: str, top_k: int = 5) -> List[MockMemory]:
        """Simple keyword-based retrieval (mock for semantic search)."""
        all_mems = self._memories.get(user_id, [])
        query_lower = query.lower()
        # Score by keyword overlap
        scored = []
        for mem in all_mems:
            score = sum(1 for word in query_lower.split() if word in mem.fact.lower())
            if score > 0:
                scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def list_memories(self, user_id: str) -> List[MockMemory]:
        return list(self._memories.get(user_id, []))

    def purge_memories(self, user_id: str) -> int:
        count = len(self._memories.get(user_id, []))
        self._memories[user_id] = []
        return count


# ── Memory Injector (the Hermes integration pattern) ───────────────────────

@dataclass
class InjectedContext:
    system_prefix: str
    memory_count: int
    memories: List[str]


class MemoryInjector:
    """
    Handles retrieval + formatting of memories into agent system prompt.
    This is the exact pattern Hermes would use.
    """

    def __init__(self, store: MockMemoryStore, max_memories: int = 10):
        self.store = store
        self.max_memories = max_memories

    def inject(self, user_id: str, base_instruction: str) -> InjectedContext:
        memories = self.store.list_memories(user_id)
        if not memories:
            return InjectedContext(
                system_prefix=base_instruction,
                memory_count=0,
                memories=[],
            )

        memory_block = "\n".join(
            f"- {mem.fact}" for mem in memories[:self.max_memories]
        )

        injected = (
            f"{base_instruction}\n\n"
            f"## User Context (from long-term memory)\n"
            f"{memory_block}\n\n"
            f"Use the above context to personalize your responses."
        )

        return InjectedContext(
            system_prefix=injected,
            memory_count=len(memories),
            memories=[mem.fact for mem in memories[:self.max_memories]],
        )


# ── Mock Agent (simulates LLM with memory context) ─────────────────────────

class MockAgent:
    """Simulates an LLM agent that uses injected memory context."""

    def __init__(self, injector: MemoryInjector, user_id: str):
        self.injector = injector
        self.user_id = user_id
        self.instruction = "You are a helpful assistant."
        self.context = self.injector.inject(user_id, self.instruction)

    def ask(self, query: str) -> str:
        """Simple keyword-matching response based on memory context."""
        query_lower = query.lower()

        # Check if query asks about known facts
        for fact in self.context.memories:
            fact_lower = fact.lower()
            if "name" in query_lower and "name" in fact_lower:
                return fact
            if "live" in query_lower or "where" in query_lower:
                if "live" in fact_lower or "fort wayne" in fact_lower:
                    return fact
            if "enjoy" in query_lower or "like" in query_lower or "hobby" in query_lower:
                if "hiking" in fact_lower or "dunes" in fact_lower:
                    return fact

        return "I don't have that information in my memory."


# ── Tests ─────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    return MockMemoryStore()


@pytest.fixture
def injector(store):
    return MemoryInjector(store)


class TestMockInjection:
    """Verify the memory injection pattern works correctly."""

    def test_empty_memory_no_injection(self, injector):
        """When no memories exist, system prompt should be unchanged."""
        ctx = injector.inject("new_user", "You are helpful.")
        assert ctx.memory_count == 0
        assert ctx.system_prefix == "You are helpful."
        assert ctx.memories == []

    def test_memory_injected_into_prompt(self, store, injector):
        """Memories should be formatted and appended to system prompt."""
        store.create_memory("user_1", "User's name is Jithendra.")
        store.create_memory("user_1", "User lives in Fort Wayne, Indiana.")

        ctx = injector.inject("user_1", "You are helpful.")

        assert ctx.memory_count == 2
        assert "User's name is Jithendra." in ctx.system_prefix
        assert "Fort Wayne, Indiana." in ctx.system_prefix
        assert "## User Context" in ctx.system_prefix

    def test_user_isolation(self, store, injector):
        """Memories from user A should not appear in user B's context."""
        store.create_memory("alice", "Alice is allergic to peanuts.")
        store.create_memory("bob", "Bob loves peanuts.")

        alice_ctx = injector.inject("alice", "You are helpful.")
        bob_ctx = injector.inject("bob", "You are helpful.")

        assert "Alice" in alice_ctx.system_prefix
        assert "Bob" in bob_ctx.system_prefix
        assert "Alice" not in bob_ctx.system_prefix
        assert "Bob" not in alice_ctx.system_prefix

    def test_max_memory_limit(self, store, injector):
        """Injector should respect max_memories limit."""
        injector.max_memories = 2
        for i in range(5):
            store.create_memory("user_x", f"Fact {i}.")

        ctx = injector.inject("user_x", "You are helpful.")
        assert ctx.memory_count == 5  # Total stored
        assert len(ctx.memories) == 2  # But only 2 injected

    def test_agent_uses_injected_memory(self, store, injector):
        """MockAgent should answer correctly when memories are injected."""
        store.create_memory("jithendra", "User's name is Jithendra.")
        store.create_memory("jithendra", "User lives in Fort Wayne, Indiana.")
        store.create_memory("jithendra", "User enjoys hiking at Indiana Dunes.")

        agent = MockAgent(injector, "jithendra")

        assert "Jithendra" in agent.ask("What is my name?")
        assert "Fort Wayne" in agent.ask("Where do I live?")
        assert "hiking" in agent.ask("What do I enjoy doing?")

    def test_agent_no_memory_for_unknown_user(self, injector):
        """Agent without memories should not hallucinate."""
        agent = MockAgent(injector, "stranger")
        resp = agent.ask("What is my name?")
        assert "don't have" in resp or "I don't know" in resp

    def test_memory_search_ranking(self, store, injector):
        """Search should return most relevant memories first."""
        store.create_memory("user_y", "User's name is Alice.")
        store.create_memory("user_y", "User works as a software engineer.")
        store.create_memory("user_y", "User enjoys hiking on weekends.")

        results = store.search_memory("user_y", "job career work", top_k=2)
        assert len(results) >= 1
        assert "software engineer" in results[0].fact

    def test_memory_purge(self, store, injector):
        """Purge should clear all memories for a user."""
        store.create_memory("user_z", "Fact 1.")
        store.create_memory("user_z", "Fact 2.")

        assert len(store.list_memories("user_z")) == 2

        count = store.purge_memories("user_z")
        assert count == 2
        assert len(store.list_memories("user_z")) == 0

        ctx = injector.inject("user_z", "You are helpful.")
        assert ctx.memory_count == 0


class TestEndToEndScenario:
    """Simulate a full Hermes-style conversation flow."""

    def test_full_session_flow(self, store, injector):
        """
        Simulate:
        1. User starts chat
        2. Hermes retrieves memories
        3. Injects into system prompt
        4. Agent responds personally
        5. New fact learned
        6. Fact stored
        7. Next session uses updated memory
        """
        user_id = "jithsss_telegram"

        # --- Session 1: Initial facts ---
        store.create_memory(user_id, "User's name is Jithendra.")
        store.create_memory(user_id, "User lives in Fort Wayne, Indiana.")

        agent = MockAgent(injector, user_id)
        resp1 = agent.ask("What is my name?")
        assert "Jithendra" in resp1

        # Agent learns something new
        store.create_memory(user_id, "User prefers dark mode in all applications.")

        # --- Session 2: New agent instance, same user ---
        agent2 = MockAgent(injector, user_id)
        assert "dark mode" in agent2.context.system_prefix
        resp2 = agent2.ask("What is my name?")
        assert "Jithendra" in resp2

        # Verify all 3 memories are present
        assert agent2.context.memory_count == 3


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
