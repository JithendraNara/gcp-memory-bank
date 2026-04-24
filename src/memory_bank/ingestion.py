"""
Event Ingestion (Preview).

Streaming pattern for continuous memory generation.
Decouples event ingestion from memory generation with automatic
triggering based on batching rules.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import structlog

from memory_bank.client import MemoryBankClient
from memory_bank.models import IngestEvent, MemoryScope

logger = structlog.get_logger(__name__)


class EventIngestionStream:
    """
    Manages a continuous ingestion stream.

    Events accumulate in a buffer. Memory generation triggers automatically
    when configured conditions (event count or idle time) are met.
    """

    def __init__(
        self,
        client: MemoryBankClient,
        scope: MemoryScope,
        stream_id: str = "default",
    ):
        self.client = client
        self.scope = scope
        self.stream_id = stream_id
        self._buffer: List[IngestEvent] = []

    async def ingest(
        self,
        events: List[IngestEvent],
        auto_trigger: bool = True,
    ) -> Optional[Any]:
        """
        Ingest events into the stream.

        Returns an LRO if auto_trigger=True and conditions are met,
        otherwise returns None.
        """
        self._buffer.extend(events)
        api_events = []
        for ev in events:
            api_ev: Dict[str, Any] = {
                "event_id": ev.event_id,
                "role": ev.role,
                "content": ev.content,
            }
            if ev.timestamp:
                api_ev["timestamp"] = ev.timestamp.isoformat()
            api_events.append(api_ev)

        loop = asyncio.get_event_loop()
        op = await loop.run_in_executor(
            None,
            lambda: self.client.raw_client.agent_engines.memories.ingest_events(
                name=self.client.engine_name,
                events=api_events,
                scope=self.scope.to_dict(),
                stream_id=self.stream_id,
            ),
        )

        if auto_trigger:
            result = await self.client._poll_lro(op)
            logger.info(
                "ingestion.stream_processed",
                stream=self.stream_id,
                events=len(events),
                scope=self.scope.to_dict(),
            )
            return result

        logger.debug(
            "ingestion.events_buffered",
            stream=self.stream_id,
            buffered=len(self._buffer),
        )
        return None

    async def flush(self) -> Any:
        """Force-trigger memory generation for buffered events."""
        if not self._buffer:
            logger.debug("ingestion.flush_empty", stream=self.stream_id)
            return None

        # Re-ingest all buffered events with explicit trigger
        result = await self.ingest(self._buffer, auto_trigger=True)
        self._buffer.clear()
        return result

    def clear_buffer(self) -> None:
        """Discard buffered events without processing."""
        count = len(self._buffer)
        self._buffer.clear()
        logger.warning("ingestion.buffer_cleared", stream=self.stream_id, count=count)
