"""
Advanced retrieval strategies for Memory Bank.

Implements multi-strategy retrieval: similarity search, scope-based,
filtered queries, and hybrid ranking.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

from memory_bank.client import MemoryBankClient
from memory_bank.models import Memory, MemoryScope, RetrievedMemory
from memory_bank.memory import MemoryManager


class RetrievalStrategy:
    """Base class for pluggable retrieval strategies."""

    async def retrieve(
        self,
        query: str,
        scope: Optional[MemoryScope] = None,
        **kwargs: Any,
    ) -> List[RetrievedMemory]:
        raise NotImplementedError


class SimilaritySearchStrategy(RetrievalStrategy):
    """Embedding-based similarity search (default)."""

    def __init__(self, manager: MemoryManager, top_k: int = 5):
        self.manager = manager
        self.top_k = top_k

    async def retrieve(
        self,
        query: str,
        scope: Optional[MemoryScope] = None,
        **kwargs: Any,
    ) -> List[RetrievedMemory]:
        return await self.manager.search(query, scope=scope, top_k=self.top_k)


class ScopeRetrievalStrategy(RetrievalStrategy):
    """Retrieve all memories for an exact scope."""

    def __init__(self, manager: MemoryManager):
        self.manager = manager

    async def retrieve(
        self,
        query: str,
        scope: Optional[MemoryScope] = None,
        **kwargs: Any,
    ) -> List[RetrievedMemory]:
        memories = await self.manager.retrieve_by_scope(scope)
        return [RetrievedMemory(memory=m, distance=None) for m in memories]


class HybridRetrievalStrategy(RetrievalStrategy):
    """
    Multi-strategy retrieval with fusion ranking.

    Runs multiple strategies in parallel, deduplicates, and reranks.
    """

    def __init__(
        self,
        strategies: List[RetrievalStrategy],
        reranker: Optional[Callable[[List[RetrievedMemory]], List[RetrievedMemory]]] = None,
    ):
        self.strategies = strategies
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        scope: Optional[MemoryScope] = None,
        **kwargs: Any,
    ) -> List[RetrievedMemory]:
        results = await asyncio.gather(
            *[s.retrieve(query, scope, **kwargs) for s in self.strategies]
        )

        # Deduplicate by memory name
        seen: Dict[str, RetrievedMemory] = {}
        for batch in results:
            for r in batch:
                if r.memory.name not in seen:
                    seen[r.memory.name] = r
                elif r.distance is not None and (
                    seen[r.memory.name].distance is None
                    or r.distance < seen[r.memory.name].distance  # type: ignore[operator]
                ):
                    seen[r.memory.name] = r

        merged = list(seen.values())

        if self.reranker:
            merged = self.reranker(merged)

        return merged


class MultiScopeRetrieval:
    """
    Retrieve across multiple scope levels simultaneously.

    Example: retrieve from both broad (user_id) and narrow (user_id+project).
    """

    def __init__(self, manager: MemoryManager, top_k_per_scope: int = 3):
        self.manager = manager
        self.top_k = top_k_per_scope

    async def retrieve(
        self,
        query: str,
        scopes: List[MemoryScope],
    ) -> Dict[str, List[RetrievedMemory]]:
        """Retrieve for each scope, returning results keyed by scope string."""
        tasks = [self.manager.search(query, scope=s, top_k=self.top_k) for s in scopes]
        results = await asyncio.gather(*tasks)
        return {
            str(sorted(s.to_dict().items())): r for s, r in zip(scopes, results)
        }

    async def retrieve_flat(
        self,
        query: str,
        scopes: List[MemoryScope],
        deduplicate: bool = True,
    ) -> List[RetrievedMemory]:
        """Retrieve across scopes and flatten to a single list."""
        by_scope = await self.retrieve(query, scopes)
        flat: List[RetrievedMemory] = []
        for batch in by_scope.values():
            flat.extend(batch)

        if deduplicate:
            seen: Dict[str, RetrievedMemory] = {}
            for r in flat:
                if r.memory.name not in seen:
                    seen[r.memory.name] = r
            flat = list(seen.values())

        # Sort by distance if available
        flat.sort(key=lambda r: r.distance if r.distance is not None else float("inf"))
        return flat
