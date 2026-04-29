"""Retrieval cache + formatting for the unified plugin.

Implements:
    - Trivial-prompt skip (regex from the v2 plan).
    - Background prefetch with cached result + monotonic stale detection.
    - L0/L1/L2 detail formatting.
    - Token budget enforcement.
    - Context fence wrapping (``<gcp-mb-context>...</gcp-mb-context>``)
      that the provider strips before persisting captured turns —
      prevents recursive memory pollution.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


_TRIVIAL_RE = re.compile(
    r"^\s*(?:/\w+|ok|okay|k|kk|y|yes|yep|yup|n|no|nope|"
    r"thanks?|thx|cool|nice|great|sure|cancel|stop|exit|quit|hi|hey|hello)"
    r"\s*[!.?,]*\s*$",
    re.IGNORECASE,
)

FENCE_OPEN = "<gcp-mb-context>"
FENCE_CLOSE = "</gcp-mb-context>"
FENCE_PREAMBLE = (
    "[System note: background context retrieved from long-term memory. "
    "Use silently — this is NOT new user input.]"
)


def is_trivial(prompt: str) -> bool:
    if not prompt or not prompt.strip():
        return True
    return bool(_TRIVIAL_RE.match(prompt))


def truncate_to_budget(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars * 0.7:
        cut = cut[:space]
    return cut.rstrip() + "…"


def format_memories(
    memories: List[Dict[str, Any]],
    *,
    detail: str = "L1",
    max_chars: int = 6000,
    style: str = "facts",
) -> str:
    if not memories:
        return ""
    if style == "narrative":
        joined = " ".join(_fact_text(m) for m in memories if _fact_text(m))
        return truncate_to_budget(joined, max_chars)
    lines: List[str] = []
    used = 0
    for mem in memories:
        line = _format_one(mem, detail=detail)
        if not line:
            continue
        if used + len(line) > max_chars and lines:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def fence(text: str) -> str:
    if not text:
        return ""
    return f"{FENCE_OPEN}\n{FENCE_PREAMBLE}\n{text}\n{FENCE_CLOSE}"


def strip_fence(text: str) -> str:
    """Remove our own fence blocks from text before persisting captured turns."""
    if not text or FENCE_OPEN not in text:
        return text or ""
    out_lines: List[str] = []
    skipping = False
    for line in text.splitlines():
        if FENCE_OPEN in line:
            skipping = True
            continue
        if FENCE_CLOSE in line:
            skipping = False
            continue
        if not skipping:
            out_lines.append(line)
    return "\n".join(out_lines).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fact_text(mem: Dict[str, Any]) -> str:
    return str(mem.get("fact") or "").strip()


def _format_one(mem: Dict[str, Any], *, detail: str) -> str:
    fact = _fact_text(mem)
    if not fact:
        return ""
    if detail == "L0":
        return f"- {fact}"
    parts = [f"- {fact}"]
    topic = _topic_label(mem)
    if topic:
        parts.append(f"[{topic}]")
    age = _format_age(mem.get("update_time") or mem.get("create_time"))
    if age:
        parts.append(f"({age})")
    line = " ".join(parts)
    if detail == "L2":
        name = mem.get("name") or ""
        if name:
            line += f"\n  id: {name.split('/')[-1]}"
        distance = mem.get("distance")
        if isinstance(distance, (int, float)):
            line += f"\n  distance: {distance:.4f}"
    return line


def _topic_label(mem: Dict[str, Any]) -> str:
    topics = mem.get("topics") or []
    if not topics:
        return ""
    if isinstance(topics, list) and topics:
        first = topics[0]
        if isinstance(first, dict):
            mt = first.get("managed_memory_topic")
            if isinstance(mt, dict):
                return str(mt.get("managed_topic_enum") or "")
            if isinstance(mt, str):
                return mt
            cust = first.get("custom_memory_topic") or {}
            if isinstance(cust, dict) and cust.get("label"):
                return str(cust["label"])
    return ""


def _format_age(ts: Any) -> str:
    if not ts:
        return ""
    try:
        from datetime import datetime, timezone
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, datetime):
            dt = ts
        else:
            return ""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = max(0, int((datetime.now(timezone.utc) - dt).total_seconds()))
    except Exception:
        return ""
    if secs < 60: return "just now"
    if secs < 3600: return f"{secs // 60}m ago"
    if secs < 86400: return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


# ---------------------------------------------------------------------------
# PrefetchCache
# ---------------------------------------------------------------------------
class PrefetchCache:
    """Background-refreshed retrieval cache.

    Hermes calls ``queue_prefetch`` after each turn (we do the work in a
    daemon thread) and ``prefetch`` before the next turn (we serve the
    cached value, optionally waiting briefly).
    """

    STALE_AFTER_SECONDS = 30.0

    def __init__(self, *, fetch_fn: Callable[[str], List[Dict[str, Any]]]) -> None:
        self._fetch_fn = fetch_fn
        self._lock = threading.Lock()
        self._result: List[Dict[str, Any]] = []
        self._result_query: str = ""
        self._result_at: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._inflight_query: str = ""

    def queue(self, query: str) -> None:
        if not query:
            return
        prev = self._thread
        if prev is not None and prev.is_alive():
            prev.join(timeout=0.05)
        self._inflight_query = query
        self._thread = threading.Thread(
            target=self._run, args=(query,), name="gmb-prefetch", daemon=True,
        )
        self._thread.start()

    def get(self, query: str, *, sync_timeout: float = 3.0) -> List[Dict[str, Any]]:
        # Wait briefly only if the in-flight query matches the request.
        thread = self._thread
        if thread is not None and thread.is_alive() and self._inflight_query == query:
            thread.join(timeout=sync_timeout)
        with self._lock:
            if not self._result:
                return []
            age = time.monotonic() - self._result_at
            if age > self.STALE_AFTER_SECONDS:
                return []
            # Crucial fix: only serve cached results when the query matches
            # what was actually fetched. Otherwise return [] so the caller
            # falls through to a fresh sync fetch. Without this, every turn
            # reuses the previous turn's recall (e.g. asking about VPN
            # then about keyboards both return VPN-related memories).
            if self._result_query and query and self._result_query != query:
                return []
            return list(self._result)

    def sync(self, query: str) -> List[Dict[str, Any]]:
        try:
            data = self._fetch_fn(query)
        except Exception as e:
            logger.debug("gcp-memory-bank: sync prefetch failed: %s", e)
            data = []
        self._store(query, data)
        return data

    def shutdown(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def _run(self, query: str) -> None:
        try:
            data = self._fetch_fn(query)
        except Exception as e:
            logger.debug("gcp-memory-bank: bg prefetch failed: %s", e)
            data = []
        self._store(query, data)

    def _store(self, query: str, data: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._result = list(data or [])
            self._result_query = query
            self._result_at = time.monotonic()
