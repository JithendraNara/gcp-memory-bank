"""
Agent Platform Sessions management.

Sessions store chronological conversation history that feeds into
Memory Bank for long-term memory generation.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any, Dict, List, Optional

import structlog

from memory_bank.client import MemoryBankClient
from memory_bank.models import MemoryScope, SessionEvent

logger = structlog.get_logger(__name__)


class SessionManager:
    """Manages Agent Platform Sessions for conversation history."""

    def __init__(self, client: MemoryBankClient):
        self.client = client

    async def create(
        self,
        user_id: str,
        scope: Optional[MemoryScope] = None,
    ) -> str:
        """Create a new session. Returns session resource name."""
        loop = asyncio.get_event_loop()
        session = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.sessions.create(
                name=self.client.engine_name,
                user_id=user_id,
            ),
        )
        logger.info("session.created", session=session.response.name, user_id=user_id)
        return session.response.name

    async def append_event(
        self,
        session_name: str,
        event: SessionEvent,
    ) -> None:
        """Append a single event to a session."""
        content: Dict[str, Any] = {"role": event.role, "parts": [{"text": event.content}]}
        if event.tool_name:
            content["parts"][0]["function_call"] = {"name": event.tool_name}

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.sessions.events.append(
                name=session_name,
                author=event.role,
                invocation_id=event.invocation_id or "1",
                timestamp=event.timestamp or datetime.datetime.now(datetime.timezone.utc),
                config={"content": content},
            ),
        )
        logger.debug("session.event_appended", session=session_name, role=event.role)

    async def append_events(
        self,
        session_name: str,
        events: List[SessionEvent],
    ) -> None:
        """Batch append events to a session."""
        for ev in events:
            await self.append_event(session_name, ev)
        logger.info("session.events_appended", session=session_name, count=len(events))

    async def list_events(self, session_name: str) -> List[SessionEvent]:
        """Retrieve all events from a session."""
        loop = asyncio.get_event_loop()
        pager = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.sessions.events.list(
                name=session_name
            ),
        )

        events: List[SessionEvent] = []
        for page in pager:
            for raw in page.events if hasattr(page, "events") else [page]:
                text_parts = [
                    p.text for p in raw.config.content.parts if hasattr(p, "text")
                ]
                events.append(
                    SessionEvent(
                        role=raw.config.content.role,
                        content=" ".join(text_parts),
                        timestamp=raw.timestamp,
                        invocation_id=raw.invocation_id,
                    )
                )
        return events

    async def delete(self, session_name: str) -> None:
        """Delete a session and its events."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.sessions.delete(name=session_name),
        )
        logger.info("session.deleted", session=session_name)

    async def session_to_events(
        self,
        session_name: str,
    ) -> List[Dict[str, str]]:
        """Export session events as simple dicts for memory generation."""
        events = await self.list_events(session_name)
        return [{"role": e.role, "content": e.content} for e in events]
