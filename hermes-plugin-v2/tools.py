"""Tool schemas + dict-dispatch handler for the unified plugin.

Tool names match v1's convention (``memory_*``) for backwards compatibility
with the model's learned habits and Hermes' built-in ``MEMORY.md`` ergonomics.

Eleven tools, four NEW behaviours over v1:
    - ``memory_search`` accepts ``topics`` filter + ``since`` filter.
    - ``memory_remember`` (alias of ``memory_store``) supports ``consolidate``.
    - ``memory_purge`` defaults to a scope-bound filter (no accidental cross-user purge).
    - ``memory_synthesize`` is REAL Gemini synthesis, not a string join.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
PROFILE_SCHEMA: Dict[str, Any] = {
    "name": "memory_profile",
    "description": (
        "Return all memories stored about the user in the active scope, as a "
        "single profile snapshot. Fast: one ListMemories call. Use when the "
        "user asks 'what do you remember about me?' or at conversation start."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
    },
}

SEARCH_SCHEMA: Dict[str, Any] = {
    "name": "memory_search",
    "description": (
        "Semantic search over GCP Memory Bank, scoped to the current user. "
        "Returns top_k facts ranked by Euclidean distance. Optional topics "
        "filter (managed enum or custom label) and 'since' (RFC3339) filter "
        "for recently-updated memories."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
            "topics": {"type": "array", "items": {"type": "string"}},
            "since": {"type": "string"},
        },
        "required": ["query"],
    },
}

STORE_SCHEMA: Dict[str, Any] = {
    "name": "memory_store",
    "description": (
        "Store a fact verbatim in long-term memory (CreateMemory — no LLM "
        "extraction, immediate availability). Use for explicit user "
        "preferences, decisions, or corrections. Set consolidate=true to "
        "instead route through Gemini extraction (slower, may dedupe)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fact": {"type": "string"},
            "topic": {"type": "string"},
            "metadata": {"type": "object"},
            "consolidate": {"type": "boolean"},
        },
        "required": ["fact"],
    },
}

GET_SCHEMA: Dict[str, Any] = {
    "name": "memory_get",
    "description": (
        "Fetch a single memory by full resource name (returned by "
        "memory_search or memory_profile). Includes scope, labels, "
        "create/update times, expiration."
    ),
    "parameters": {
        "type": "object",
        "properties": {"memory_name": {"type": "string"}},
        "required": ["memory_name"],
    },
}

DELETE_SCHEMA: Dict[str, Any] = {
    "name": "memory_delete",
    "description": (
        "Delete a single memory by resource name. Irreversible after the 48h "
        "revision-recovery window."
    ),
    "parameters": {
        "type": "object",
        "properties": {"memory_name": {"type": "string"}},
        "required": ["memory_name"],
    },
}

REVISIONS_SCHEMA: Dict[str, Any] = {
    "name": "memory_revisions",
    "description": (
        "List the revision history of a specific memory. Optional "
        "label_filter (e.g. 'labels.hermes_session=\"X\"')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_name": {"type": "string"},
            "label_filter": {"type": "string"},
        },
        "required": ["memory_name"],
    },
}

REVISION_GET_SCHEMA: Dict[str, Any] = {
    "name": "memory_revision_get",
    "description": "Fetch a single memory revision by its full resource name.",
    "parameters": {
        "type": "object",
        "properties": {"revision_name": {"type": "string"}},
        "required": ["revision_name"],
    },
}

ROLLBACK_SCHEMA: Dict[str, Any] = {
    "name": "memory_rollback",
    "description": (
        "Rollback a memory to a target revision id. Newer revisions are "
        "removed. Use to undo a bad consolidation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_name": {"type": "string"},
            "target_revision_id": {"type": "string"},
        },
        "required": ["memory_name", "target_revision_id"],
    },
}

PURGE_SCHEMA: Dict[str, Any] = {
    "name": "memory_purge",
    "description": (
        "Bulk-delete memories matching a filter. If no filter is given, "
        "defaults to deleting ALL memories under the current scope (still "
        "user-bound — never cross-user). Requires force=true to actually "
        "delete; otherwise returns dry-run count."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "filter": {"type": "string"},
            "force": {"type": "boolean"},
        },
    },
}

INGEST_SCHEMA: Dict[str, Any] = {
    "name": "memory_ingest",
    "description": (
        "Stream events into memory generation. NOTE: in current SDK versions "
        "the ingest_events API is unreliable (Preview); this tool falls back "
        "to per-event CreateMemory which is the only proven-working path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of {role, text} dicts.",
            },
        },
        "required": ["events"],
    },
}

SYNTHESIZE_SCHEMA: Dict[str, Any] = {
    "name": "memory_synthesize",
    "description": (
        "Compose a real narrative answer to a question, grounded in retrieved "
        "memories. Uses the configured synthesis_model (default "
        "gemini-2.5-flash). Returns 2-5 sentences. Use when the user asks an "
        "open question and you want continuity from past sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["query"],
    },
}


def all_schemas() -> List[Dict[str, Any]]:
    return [
        PROFILE_SCHEMA, SEARCH_SCHEMA, STORE_SCHEMA, GET_SCHEMA, DELETE_SCHEMA,
        REVISIONS_SCHEMA, REVISION_GET_SCHEMA, ROLLBACK_SCHEMA, PURGE_SCHEMA,
        INGEST_SCHEMA, SYNTHESIZE_SCHEMA,
    ]


def schemas_for_mode(recall_mode: str) -> List[Dict[str, Any]]:
    if recall_mode == "context":
        return []
    return all_schemas()


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------
def tool_ok(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps({"result": payload}, default=str)


def tool_error(message: str) -> str:
    try:
        from tools.registry import tool_error as _err  # type: ignore
        return _err(str(message))
    except Exception:
        return json.dumps({"error": str(message)})


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
class ToolDispatcher:
    """Dict-of-handlers (cleaner than v1's flat if/elif chain)."""

    def __init__(self, handlers: Dict[str, Callable[[Dict[str, Any]], str]]) -> None:
        self._handlers = dict(handlers)

    def handle(self, tool_name: str, args: Dict[str, Any]) -> str:
        fn = self._handlers.get(tool_name)
        if fn is None:
            return tool_error(f"Unknown tool: {tool_name}")
        try:
            return fn(args or {})
        except Exception as e:
            logger.debug("gcp-memory-bank: tool %s failed: %s", tool_name, e)
            return tool_error(f"{tool_name} failed: {e}")
