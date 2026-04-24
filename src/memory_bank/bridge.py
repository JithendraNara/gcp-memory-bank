"""
Bridge interface for integrating Memory Bank into external systems.

Provides abstract base class and concrete examples for Hermes and OpenClaw.
NOT wired into either system — these are reference implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from memory_bank.client import MemoryBankClient
from memory_bank.config import HERMES_MEMORY_CONFIG, OPENCLAW_MEMORY_CONFIG, MemoryBankConfig
from memory_bank.memory import MemoryManager
from memory_bank.models import MemoryScope
from memory_bank.retrieval import HybridRetrievalStrategy, SimilaritySearchStrategy
from memory_bank.sessions import SessionManager
from memory_bank.utils import build_scope, format_memories_for_prompt


class BaseMemoryBridge(ABC):
    """
    Abstract bridge for connecting Memory Bank to an agent system.

    Implement this to wire Memory Bank into Hermes, OpenClaw, or any
    other agent framework.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
        engine_id: Optional[str] = None,
    ):
        self.project = project
        self.location = location
        self.engine_id = engine_id
        self._client: Optional[MemoryBankClient] = None

    async def connect(self) -> MemoryBankClient:
        """Initialize the client. Idempotent."""
        if self._client is None:
            self._client = MemoryBankClient(
                project=self.project,
                location=self.location,
                engine_id=self.engine_id,
            )
            await self._client._init()
        return self._client

    @abstractmethod
    async def on_session_start(self, user_id: str, **context: Any) -> str:
        """
        Called when a new session begins.

        Should retrieve relevant memories and inject into agent context.
        Returns the system prompt augmentation.
        """
        raise NotImplementedError

    @abstractmethod
    async def on_session_end(
        self,
        user_id: str,
        session_events: List[Dict[str, str]],
        **context: Any,
    ) -> None:
        """
        Called when a session ends.

        Should generate memories from the session's conversation history.
        """
        raise NotImplementedError

    @abstractmethod
    async def on_agent_fact(self, user_id: str, fact: str, **context: Any) -> None:
        """
        Called when the agent wants to store a pre-extracted fact.

        Bypasses LLM extraction — direct agent-controlled write.
        """
        raise NotImplementedError

    @abstractmethod
    async def recall(
        self,
        user_id: str,
        query: str,
        **context: Any,
    ) -> List[Dict[str, Any]]:
        """
        Called when the agent needs to retrieve memories.

        Returns a list of memory dicts.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Example: Hermes Bridge (reference implementation, NOT wired)
# ---------------------------------------------------------------------------

class HermesBridgeExample(BaseMemoryBridge):
    """
    Example bridge showing how Hermes COULD integrate Memory Bank.

    This is NOT wired into Hermes. Copy/adapt this when ready.
    """

    DEFAULT_CONFIG = HERMES_MEMORY_CONFIG

    async def on_session_start(self, user_id: str, **context: Any) -> str:
        client = await self.connect()
        manager = MemoryManager(client)

        scope = build_scope(user_id=user_id, agent="hermes")

        # Retrieve general user context
        memories = await manager.retrieve_by_scope(scope)

        # Also search for task-relevant memories if we know the current task
        task = context.get("current_task", "")
        if task:
            relevant = await manager.search(task, scope=scope, top_k=5)
            memories.extend([r.memory for r in relevant])

        return format_memories_for_prompt(memories)

    async def on_session_end(
        self,
        user_id: str,
        session_events: List[Dict[str, str]],
        **context: Any,
    ) -> None:
        client = await self.connect()
        manager = MemoryManager(client)

        scope = build_scope(user_id=user_id, agent="hermes")

        # Generate memories from the full conversation
        generated = await manager.generate_from_events(session_events, scope=scope)

        # Log what was learned
        for mem in generated:
            print(f"  [{mem.action.value}] {mem.memory.fact[:80]}...")

    async def on_agent_fact(self, user_id: str, fact: str, **context: Any) -> None:
        client = await self.connect()
        manager = MemoryManager(client)
        scope = build_scope(user_id=user_id, agent="hermes")
        await manager.generate_from_facts([fact], scope=scope)

    async def recall(
        self,
        user_id: str,
        query: str,
        **context: Any,
    ) -> List[Dict[str, Any]]:
        client = await self.connect()
        manager = MemoryManager(client)
        scope = build_scope(user_id=user_id, agent="hermes")
        results = await manager.search(query, scope=scope, top_k=10)
        return [
            {"fact": r.memory.fact, "distance": r.distance}
            for r in results
        ]


# ---------------------------------------------------------------------------
# Example: OpenClaw Bridge (reference implementation, NOT wired)
# ---------------------------------------------------------------------------

class OpenClawBridgeExample(BaseMemoryBridge):
    """
    Example bridge showing how OpenClaw COULD integrate Memory Bank.

    This is NOT wired into OpenClaw. Copy/adapt this when ready.
    """

    DEFAULT_CONFIG = OPENCLAW_MEMORY_CONFIG

    async def on_session_start(self, user_id: str, **context: Any) -> str:
        client = await self.connect()
        manager = MemoryManager(client)

        scope = build_scope(user_id=user_id, agent="openclaw")

        # OpenClaw: retrieve model preferences and gateway rules
        hybrid = HybridRetrievalStrategy([
            SimilaritySearchStrategy(manager, top_k=3),
        ])
        results = await hybrid.retrieve("model preferences gateway rules", scope=scope)

        return format_memories_for_prompt([r.memory for r in results])

    async def on_session_end(
        self,
        user_id: str,
        session_events: List[Dict[str, str]],
        **context: Any,
    ) -> None:
        client = await self.connect()
        manager = MemoryManager(client)
        scope = build_scope(user_id=user_id, agent="openclaw")
        await manager.generate_from_events(session_events, scope=scope)

    async def on_agent_fact(self, user_id: str, fact: str, **context: Any) -> None:
        client = await self.connect()
        manager = MemoryManager(client)
        scope = build_scope(user_id=user_id, agent="openclaw")
        await manager.generate_from_facts([fact], scope=scope)

    async def recall(
        self,
        user_id: str,
        query: str,
        **context: Any,
    ) -> List[Dict[str, Any]]:
        client = await self.connect()
        manager = MemoryManager(client)
        scope = build_scope(user_id=user_id, agent="openclaw")
        results = await manager.search(query, scope=scope, top_k=5)
        return [
            {"fact": r.memory.fact, "distance": r.distance}
            for r in results
        ]
