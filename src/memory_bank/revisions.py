"""
Memory Revisions / Audit Trail.

Every memory mutation creates an immutable revision.
Enables rollback, forensic analysis, and trend detection.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import structlog

from memory_bank.client import MemoryBankClient
from memory_bank.models import MemoryRevision

logger = structlog.get_logger(__name__)


class RevisionManager:
    """Manages memory revision history and audit trails."""

    def __init__(self, client: MemoryBankClient):
        self.client = client

    async def list_revisions(self, memory_name: str) -> List[MemoryRevision]:
        """List all revisions for a memory, newest first."""
        loop = asyncio.get_event_loop()
        pager = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.revisions.list(
                name=memory_name
            ),
        )

        revisions: List[MemoryRevision] = []
        for page in pager:
            for raw in page.revisions if hasattr(page, "revisions") else [page]:
                extracted = [
                    {"fact": e.fact}
                    for e in getattr(raw, "extracted_memories", [])
                ]
                revisions.append(
                    MemoryRevision(
                        name=raw.name,
                        fact=raw.fact,
                        create_time=getattr(raw, "create_time", None),
                        expire_time=getattr(raw, "expire_time", None),
                        extracted_memories=extracted,  # type: ignore[arg-type]
                    )
                )

        # Sort by creation time descending
        revisions.sort(key=lambda r: r.create_time or asyncio.get_event_loop().time(), reverse=True)
        return revisions

    async def get_latest_extraction(self, memory_name: str) -> Optional[str]:
        """
        Get the most recent raw extraction (before consolidation).

        Useful for understanding WHY a memory was updated.
        """
        revisions = await self.list_revisions(memory_name)
        if not revisions:
            return None
        latest = revisions[0]
        if latest.extracted_memories:
            return latest.extracted_memories[0].fact
        return None

    async def diff_revisions(
        self,
        memory_name: str,
        rev_index_a: int = -1,
        rev_index_b: int = 0,
    ) -> Dict[str, Any]:
        """
        Compare two revisions of a memory.

        rev_index: 0 = oldest, -1 = latest
        """
        revisions = await self.list_revisions(memory_name)
        if len(revisions) < 2:
            return {"error": "Need at least 2 revisions to diff"}

        a = revisions[rev_index_a]
        b = revisions[rev_index_b]

        return {
            "older": {"fact": a.fact, "time": a.create_time},
            "newer": {"fact": b.fact, "time": b.create_time},
            "changed": a.fact != b.fact,
        }

    async def audit_memory(self, memory_name: str) -> Dict[str, Any]:
        """Full audit report for a memory."""
        revisions = await self.list_revisions(memory_name)
        return {
            "memory_name": memory_name,
            "revision_count": len(revisions),
            "created": revisions[-1].create_time if revisions else None,
            "latest": revisions[0].fact if revisions else None,
            "history": [
                {
                    "time": r.create_time,
                    "fact": r.fact,
                    "extracted_from": [e.fact for e in r.extracted_memories],
                }
                for r in revisions
            ],
        }
