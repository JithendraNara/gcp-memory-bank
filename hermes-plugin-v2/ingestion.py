"""Local event buffer + per-event CreateMemory fallback.

Used when GCP Sessions is disabled OR the vertex_session_source path fails.
This is the path v1's TEST_RESULTS.md proved actually works.

Sliding window keeps the buffer bounded so a long-running Hermes process
doesn't OOM. ``flush()`` clears on success — no double-ingestion.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from .observability import timed

logger = logging.getLogger(__name__)


class EventBuffer:
    """Bounded, thread-safe event buffer."""

    def __init__(self, *, max_events: int = 200) -> None:
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self._turn_count = 0
        self._lock = threading.Lock()

    def add_turn(self, user_text: str, assistant_text: str) -> int:
        events = [
            make_event(role="user", text=user_text),
            make_event(role="model", text=assistant_text),
        ]
        with self._lock:
            for ev in events:
                self._buffer.append(ev)
            self._turn_count += 1
            return self._turn_count

    def drain(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            self._turn_count = 0
        return items

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._buffer)

    @property
    def turn_count(self) -> int:
        with self._lock:
            return self._turn_count

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)


def make_event(*, role: str, text: str,
               event_id: Optional[str] = None) -> Dict[str, Any]:
    canonical = "user" if role == "user" else "model"
    return {
        "event_id": event_id or uuid.uuid4().hex,
        "content": {"role": canonical, "parts": [{"text": text or ""}]},
    }


def fallback_create_memories(
    *,
    client: Any,                # MemoryBankClient
    scope: Dict[str, str],
    events: List[Dict[str, Any]],
    revision_labels: Optional[Dict[str, str]] = None,
    label: str = "fallback",
) -> int:
    """Per-event ``CreateMemory`` — only path that always works in current SDK.

    Returns the count of memories actually created.
    """
    if not events:
        return 0
    with timed(f"fallback create_memories({label})") as ctx:
        ctx["events"] = len(events)
        n = client.create_memories_from_events(
            scope=scope,
            events=events,
            revision_labels=revision_labels,
        )
        ctx["created"] = n
        return n
