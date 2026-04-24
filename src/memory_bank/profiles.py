"""
Structured Memory Profiles (Preview).

Static JSON schemas populated and updated by LLMs. Optimized for
low-latency retrieval — curation happens at write-time.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import structlog

from memory_bank.client import MemoryBankClient
from memory_bank.models import Memory, MemoryScope

logger = structlog.get_logger(__name__)


class ProfileManager:
    """Manager for structured memory profiles."""

    def __init__(self, client: MemoryBankClient):
        self.client = client

    async def get_profile(
        self,
        schema_id: str,
        scope: Optional[MemoryScope] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a structured profile by schema ID.

        Returns None if no profile exists for the schema + scope combo.
        """
        scope = scope or MemoryScope()
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.retrieve(
                name=self.client.engine_name,
                scope=scope.to_dict(),
            ),
        )

        for r in results:
            raw = r.memory
            if getattr(raw, "memory_type", None) == "STRUCTURED_PROFILE":
                structured = getattr(raw, "structured_content", None)
                if structured and structured.get("schema_id") == schema_id:
                    return structured.get("data")

        return None

    async def list_profiles(
        self,
        scope: Optional[MemoryScope] = None,
    ) -> List[Dict[str, Any]]:
        """List all structured profiles for a scope."""
        scope = scope or MemoryScope()
        memories = await self._list_scope_memories(scope)
        profiles = []
        for m in memories:
            if m.memory_type == "STRUCTURED_PROFILE" and m.structured_content:
                profiles.append({
                    "schema_id": m.structured_content.get("schema_id"),
                    "data": m.structured_content.get("data"),
                    "name": m.name,
                })
        return profiles

    async def _list_scope_memories(
        self,
        scope: MemoryScope,
    ) -> List[Memory]:
        """Helper: retrieve all memories for a scope."""
        from memory_bank.memory import MemoryManager

        manager = MemoryManager(self.client)
        return await manager.retrieve_by_scope(scope)

    def format_for_prompt(self, profile: Dict[str, Any]) -> str:
        """Format a profile dict into a system-prompt injectable string."""
        lines = ["<user_profile>"]
        for key, value in profile.items():
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        lines.append("</user_profile>")
        return "\n".join(lines)
