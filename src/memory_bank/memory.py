"""
Memory CRUD operations.

Create, read, update, delete, and purge memories with type-safe
interfaces over the raw Google Cloud SDK.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import structlog
from google.genai import types as genai_types

from memory_bank.client import MemoryBankClient
from memory_bank.models import (
    ConsolidationAction,
    GeneratedMemory,
    Memory,
    MemoryFilter,
    MemoryMetadata,
    MemoryScope,
    RetrievedMemory,
    SimilaritySearchParams,
)
from memory_bank.utils import format_retrieved_for_context

logger = structlog.get_logger(__name__)


class MemoryManager:
    """High-level manager for Memory CRUD."""

    def __init__(self, client: MemoryBankClient):
        self.client = client

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def create(
        self,
        fact: str,
        scope: Optional[MemoryScope] = None,
        metadata: Optional[MemoryMetadata] = None,
    ) -> Memory:
        """
        Directly create a memory (NO extraction/consolidation).

        Use when you already have a refined fact and want raw storage.
        Caution: May create duplicates if similar facts already exist.
        """
        scope = scope or MemoryScope()
        meta_api = metadata.to_api_dict() if metadata else None

        loop = asyncio.get_event_loop()
        op = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.create(
                name=self.client.engine_name,
                fact=fact,
                scope=scope.to_dict(),
                metadata=meta_api,
            ),
        )

        result = await self.client._poll_lro(op)
        memory = Memory(
            name=result.response.name,
            fact=result.response.fact,
            scope=scope,
            create_time=result.response.create_time,
            update_time=result.response.update_time,
        )
        logger.info("memory.created", name=memory.name, fact_preview=fact[:60])
        return memory

    async def generate_from_session(
        self,
        session_name: str,
        scope: Optional[MemoryScope] = None,
        allowed_topics: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[MemoryMetadata] = None,
    ) -> List[GeneratedMemory]:
        """
        Generate memories from an Agent Platform Session.

        The LLM extracts facts from the session's conversation history,
        then consolidates them with existing memories for the same scope.
        """
        scope = scope or MemoryScope()
        config: Dict[str, Any] = {}
        if metadata:
            config["metadata"] = metadata.to_api_dict()
        if allowed_topics:
            config["allowed_topics"] = allowed_topics

        loop = asyncio.get_event_loop()
        op = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.generate(
                name=self.client.engine_name,
                vertex_session_source={"session": session_name},
                scope=scope.to_dict(),
                config=config if config else None,
            ),
        )

        result = await self.client._poll_lro(op)
        generated = [
            GeneratedMemory(
                memory=Memory(
                    name=m.memory.name,
                    fact=m.memory.fact,
                    scope=scope,
                ),
                action=ConsolidationAction(m.action),
            )
            for m in result.response.generated_memories
        ]

        actions = {g.action.value for g in generated}
        logger.info(
            "memory.generated_from_session",
            session=session_name,
            count=len(generated),
            actions=list(actions),
        )
        return generated

    async def generate_from_events(
        self,
        events: List[Dict[str, str]],
        scope: Optional[MemoryScope] = None,
        allowed_topics: Optional[List[Dict[str, Any]]] = None,
    ) -> List[GeneratedMemory]:
        """
        Generate memories from raw conversation events.

        events: [{"role": "user", "content": "..."}, ...]
        """
        scope = scope or MemoryScope()
        genai_events = []
        for ev in events:
            genai_events.append(
                genai_types.Content(
                    role=ev["role"],
                    parts=[genai_types.Part.from_text(text=ev["content"])],
                )
            )

        config: Dict[str, Any] = {}
        if allowed_topics:
            config["allowed_topics"] = allowed_topics

        loop = asyncio.get_event_loop()
        op = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.generate(
                name=self.client.engine_name,
                direct_contents_source={"events": genai_events},
                scope=scope.to_dict(),
                config=config if config else None,
            ),
        )

        result = await self.client._poll_lro(op)
        generated = [
            GeneratedMemory(
                memory=Memory(
                    name=m.memory.name,
                    fact=m.memory.fact,
                    scope=scope,
                ),
                action=ConsolidationAction(m.action),
            )
            for m in result.response.generated_memories
        ]

        logger.info(
            "memory.generated_from_events",
            count=len(generated),
            actions=list({g.action.value for g in generated}),
        )
        return generated

    async def generate_from_facts(
        self,
        facts: List[str],
        scope: Optional[MemoryScope] = None,
        metadata: Optional[MemoryMetadata] = None,
    ) -> List[GeneratedMemory]:
        """
        Agent-controlled memory writes with consolidation.

        You provide pre-extracted facts; Memory Bank handles consolidation.
        Best for when your agent has already decided what to remember.
        """
        scope = scope or MemoryScope()
        config: Dict[str, Any] = {}
        if metadata:
            config["metadata"] = metadata.to_api_dict()

        loop = asyncio.get_event_loop()
        op = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.generate(
                name=self.client.engine_name,
                direct_memories_source={
                    "direct_memories": [{"fact": f} for f in facts]
                },
                scope=scope.to_dict(),
                config=config if config else None,
            ),
        )

        result = await self.client._poll_lro(op)
        generated = [
            GeneratedMemory(
                memory=Memory(
                    name=m.memory.name,
                    fact=m.memory.fact,
                    scope=scope,
                ),
                action=ConsolidationAction(m.action),
            )
            for m in result.response.generated_memories
        ]

        logger.info(
            "memory.generated_from_facts",
            count=len(generated),
            facts=facts,
        )
        return generated

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get(self, memory_name: str) -> Memory:
        """Fetch a single memory by fully-qualified resource name."""
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.get(name=memory_name),
        )
        return Memory(
            name=raw.name,
            fact=raw.fact,
            scope=MemoryScope(**raw.scope),
            create_time=raw.create_time,
            update_time=raw.update_time,
            expire_time=getattr(raw, "expire_time", None),
            topics=getattr(raw, "topics", None),
        )

    async def list_all(self) -> List[Memory]:
        """List all memories in the instance (paginated)."""
        loop = asyncio.get_event_loop()
        pager = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.list(
                name=self.client.engine_name
            ),
        )

        memories: List[Memory] = []
        for page in pager:
            for raw in page.memories if hasattr(page, "memories") else [page]:
                memories.append(
                    Memory(
                        name=raw.name,
                        fact=raw.fact,
                        scope=MemoryScope(**raw.scope),
                        create_time=getattr(raw, "create_time", None),
                        update_time=getattr(raw, "update_time", None),
                    )
                )
        return memories

    async def retrieve_by_scope(
        self,
        scope: Optional[MemoryScope] = None,
    ) -> List[Memory]:
        """Retrieve all memories matching an exact scope."""
        scope = scope or MemoryScope()
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.retrieve(
                name=self.client.engine_name,
                scope=scope.to_dict(),
            ),
        )

        return [
            Memory(
                name=r.memory.name,
                fact=r.memory.fact,
                scope=scope,
            )
            for r in results
        ]

    async def search(
        self,
        query: str,
        scope: Optional[MemoryScope] = None,
        top_k: int = 5,
    ) -> List[RetrievedMemory]:
        """
        Similarity search within a scope.

        Compares embedding vectors between the query and stored facts.
        Returns results sorted by Euclidean distance (ascending).
        """
        scope = scope or MemoryScope()
        params = SimilaritySearchParams(search_query=query, top_k=top_k)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.retrieve(
                name=self.client.engine_name,
                scope=scope.to_dict(),
                similarity_search_params={
                    "search_query": params.search_query,
                    "top_k": params.top_k,
                },
            ),
        )

        retrieved = [
            RetrievedMemory(
                memory=Memory(
                    name=r.memory.name,
                    fact=r.memory.fact,
                    scope=scope,
                ),
                distance=r.distance,
            )
            for r in results
        ]

        logger.info(
            "memory.searched",
            query=query[:40],
            scope=scope.to_dict(),
            results=len(retrieved),
        )
        return retrieved

    async def search_to_prompt(
        self,
        query: str,
        scope: Optional[MemoryScope] = None,
        top_k: int = 5,
    ) -> str:
        """Search and format results as a prompt-ready string."""
        results = await self.search(query, scope=scope, top_k=top_k)
        return format_retrieved_for_context(results)

    # ------------------------------------------------------------------
    # Update / Delete
    # ------------------------------------------------------------------

    async def update_metadata(
        self,
        memory_name: str,
        metadata: MemoryMetadata,
    ) -> None:
        """Update metadata on an existing memory."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.update(
                name=memory_name,
                metadata=metadata.to_api_dict(),
            ),
        )
        logger.info("memory.metadata_updated", name=memory_name)

    async def delete(self, memory_name: str, wait: bool = True) -> None:
        """Delete a memory by resource name."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.delete(
                name=memory_name,
                config={"wait_for_completion": wait},
            ),
        )
        logger.info("memory.deleted", name=memory_name)

    async def purge(
        self,
        filter_string: Optional[str] = None,
        filter_groups: Optional[List[Dict[str, Any]]] = None,
        force: bool = True,
        dry_run: bool = False,
    ) -> int:
        """
        Bulk delete memories matching criteria.

        EBNF filter_string: 'scope.user_id="123" AND fact=~".*allergies.*"'
        filter_groups: DNF metadata filters.

        Set dry_run=True to preview count without deleting.
        """
        if not filter_string and not filter_groups:
            raise ValueError("Must provide filter_string or filter_groups")

        loop = asyncio.get_event_loop()
        op = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.purge(
                name=self.client.engine_name,
                filter=filter_string,
                filter_groups=filter_groups,
                force=force and not dry_run,
                config={"wait_for_completion": True},
            ),
        )

        result = await self.client._poll_lro(op)
        count = result.response.purge_count
        logger.warning(
            "memory.purged",
            filter=filter_string,
            dry_run=dry_run,
            count=count,
        )
        return count

    async def purge_scope(self, scope: MemoryScope, dry_run: bool = False) -> int:
        """Nuclear option: delete ALL memories for a given scope."""
        from memory_bank.utils import scope_to_filter

        return await self.purge(
            filter_string=scope_to_filter(scope),
            dry_run=dry_run,
        )
