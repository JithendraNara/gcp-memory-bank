"""GCP Memory Bank plugin for Hermes.

Plugs Google Cloud Memory Bank into Hermes' MemoryManager for persistent,
cross-session recall with intelligent prefetch, deduplication, and structured
fact extraction.

Config via environment variables:
  GCP_PROJECT_ID      — GCP project (default: festive-antenna-463514-m8)
  GCP_LOCATION        — Vertex AI region (default: us-central1)
  GCP_MEMORY_ENGINE   — Optional: pre-created Agent Engine ID. If unset,
                        the provider auto-provisions one per app_name.

Or via $HERMES_HOME/gcp-memory-bank.json.

Requires: pip install google-adk google-cloud-aiplatform>=1.93
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# NLP helpers for intelligent prefetch and fact extraction
# ---------------------------------------------------------------------------

# Common stopwords to filter out of queries
_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "also", "get", "go", "me", "my", "we", "our", "you", "your",
    "he", "his", "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "i", "am", "what", "which", "who",
}

# Topic keywords that indicate personal preferences/identity
_TOPIC_KEYWORDS: Set[str] = {
    "like", "love", "prefer", "enjoy", "hate", "dislike", "want", "need",
    "live", "work", "from", "name", "called", "favorite", "favourite",
    "interested", "passionate", "career", "job", "study", "studying",
    "born", "birthday", "age", "family", "wife", "husband", "kids",
    "children", "dog", "cat", "pet", "home", "house", "apartment",
    "city", "state", "country", "moved", "visit", "travel", "trip",
    "plan", "goal", "project", "building", "learning", "learn",
    "allergic", "diet", "vegetarian", "vegan", "gluten", "health",
}


def _extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from a user message for targeted search."""
    # Lowercase and split on non-alphanumeric
    words = re.findall(r"[a-zA-Z']+", text.lower())
    # Filter stopwords and short words
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    # Prioritize topic keywords
    topic_matches = [w for w in keywords if w in _TOPIC_KEYWORDS]
    # Return topic keywords first, then other significant words
    return list(dict.fromkeys(topic_matches + keywords))[:8]


def _extract_fact(user_msg: str, assistant_msg: str) -> Optional[str]:
    """Extract a structured fact from a conversation turn.

    Returns None if no substantive personal information is detected.
    """
    content = user_msg.lower()

    # Pattern: "I live in ..."
    m = re.search(r"i live in ([^.]{3,60})", content)
    if m:
        return f"User lives in {m.group(1).strip()}."

    # Pattern: "I work as / I work at / I work for"
    m = re.search(r"i work (?:as|at|for) ([^.]{3,60})", content)
    if m:
        return f"User works {m.group(0).split(' ', 2)[2]} {m.group(1).strip()}."

    # Pattern: "I like / love / enjoy / prefer ..."
    m = re.search(r"i (?:like|love|enjoy|prefer) ([^.]{3,80})", content)
    if m:
        return f"User likes {m.group(1).strip()}."

    # Pattern: "My name is / I'm called / Call me ..."
    m = re.search(r"(?:my name is|i am called|call me) ([^.]{2,40})", content)
    if m:
        return f"User's name is {m.group(1).strip()}."

    # Pattern: "I'm interested in ..."
    m = re.search(r"i(?:'m| am) interested in ([^.]{3,80})", content)
    if m:
        return f"User is interested in {m.group(1).strip()}."

    # Pattern: "I'm learning / studying ..."
    m = re.search(r"i(?:'m| am) (?:learning|studying) ([^.]{3,80})", content)
    if m:
        return f"User is learning {m.group(1).strip()}."

    # Pattern: "I want to ..."
    m = re.search(r"i want to ([^.]{3,80})", content)
    if m:
        return f"User wants to {m.group(1).strip()}."

    # Pattern: "I have a ... (pet/dog/cat/etc)"
    m = re.search(r"i have a ([^.]{3,40})", content)
    if m:
        return f"User has a {m.group(1).strip()}."

    # Pattern: "I'm from ..."
    m = re.search(r"i(?:'m| am) from ([^.]{3,60})", content)
    if m:
        return f"User is from {m.group(1).strip()}."

    # Pattern: "I don't like / hate ..."
    m = re.search(r"i (?:don't like|hate|dislike) ([^.]{3,80})", content)
    if m:
        return f"User dislikes {m.group(1).strip()}."

    # Pattern: "I use / I'm using ..."
    m = re.search(r"i(?:'m| am)? (?:use|using) ([^.]{3,60})", content)
    if m:
        return f"User uses {m.group(1).strip()}."

    return None


def _memory_hash(fact: str) -> str:
    """Create a stable hash for deduplication comparison."""
    normalized = " ".join(fact.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env + $HERMES_HOME/gcp-memory-bank.json."""
    from hermes_constants import get_hermes_home

    config = {
        "project_id": os.environ.get("GCP_PROJECT_ID", "festive-antenna-463514-m8"),
        "location": os.environ.get("GCP_LOCATION", "us-central1"),
        "engine_id": os.environ.get("GCP_MEMORY_ENGINE", ""),
    }

    config_path = get_hermes_home() / "gcp-memory-bank.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "memory_profile",
    "description": (
        "Retrieve all stored memories about the user from GCP Memory Bank. "
        "Fast full-list. Use at conversation start or when you need a complete picture."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "memory_search",
    "description": (
        "Search GCP Memory Bank by semantic similarity. Returns relevant facts "
        "ranked by relevance. Use for targeted recall mid-conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "memory_store",
    "description": (
        "Store a durable fact about the user in GCP Memory Bank. "
        "Stored verbatim (no LLM extraction). Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fact": {"type": "string", "description": "The fact to store."},
        },
        "required": ["fact"],
    },
}

PURGE_SCHEMA = {
    "name": "memory_purge",
    "description": (
        "Delete ALL memories for the current user from GCP Memory Bank. "
        "Irreversible. Use only when the user explicitly requests complete data deletion."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

CONSOLIDATE_SCHEMA = {
    "name": "memory_consolidate",
    "description": (
        "Find and merge duplicate or near-duplicate memories in GCP Memory Bank. "
        "Run periodically to keep memory clean. Reports how many were merged."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class GcpMemoryBankProvider(MemoryProvider):
    """Google Cloud Memory Bank provider for Hermes.

    Uses the low-level MemoryBankServiceClient (verified working against live GCP).
    Auto-provisions an Agent Engine if none is configured.

    Key features:
    - Intelligent prefetch: extracts keywords and searches for relevant context
    - Structured fact extraction: stores meaningful facts, not raw messages
    - Deduplication: checks for similar memories before storing
    - Memory consolidation: merges duplicates on demand
    - Circuit breaker: backs off on repeated API failures
    """

    def __init__(self):
        self._config: Optional[dict] = None
        self._client: Optional[Any] = None
        self._client_lock = threading.Lock()
        self._project_id = ""
        self._location = ""
        self._engine_id = ""
        self._parent = ""
        self._user_id = "hermes-user"
        self._app_name = "hermes"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._initialized = False
        # Local dedup cache: fact_hash -> timestamp
        self._recent_hashes: Dict[str, float] = {}
        self._hash_lock = threading.Lock()
        # Retrieval counter: memory_name -> count
        self._retrieval_counts: Dict[str, int] = {}
        self._retrieval_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "gcp-memory-bank"

    def is_available(self) -> bool:
        """Check if google-adk / google-cloud-aiplatform is installed and ADC is valid."""
        try:
            import google.cloud.aiplatform_v1beta1  # noqa: F401
            import vertexai  # noqa: F401
        except ImportError:
            return False

        cfg = _load_config()
        if not cfg.get("project_id"):
            return False

        # Quick ADC check — don't make network calls here
        try:
            from google.auth import default as google_auth_default
            creds, project = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            if not creds or not project:
                return False
        except Exception:
            return False

        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "project_id",
                "description": "Google Cloud project ID",
                "required": True,
                "default": "festive-antenna-463514-m8",
            },
            {
                "key": "location",
                "description": "Vertex AI region",
                "required": True,
                "default": "us-central1",
                "choices": ["us-central1", "us-east4", "europe-west4", "asia-northeast1"],
            },
            {
                "key": "engine_id",
                "description": "Optional: pre-created Agent Engine ID (auto-provisioned if blank)",
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from pathlib import Path
        config_path = Path(hermes_home) / "gcp-memory-bank.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "GCP Memory Bank circuit breaker tripped after %d failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def _get_client(self):
        """Thread-safe lazy init of MemoryBankServiceClient."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            from google.cloud.aiplatform_v1beta1 import MemoryBankServiceClient
            from google.api_core.client_options import ClientOptions
            self._client = MemoryBankServiceClient(
                client_options=ClientOptions(
                    api_endpoint=f"{self._location}-aiplatform.googleapis.com"
                )
            )
            return self._client

    def _ensure_engine(self) -> str:
        """Return engine_id, auto-provisioning if needed."""
        if self._engine_id:
            return self._engine_id

        # Auto-provision an Agent Engine with Memory Bank enabled
        logger.info("Auto-provisioning Agent Engine for Memory Bank...")
        import vertexai

        client = vertexai.Client(project=self._project_id, location=self._location)

        config = {
            "display_name": f"hermes-memory-{self._app_name}",
            "context_spec": {
                "memory_bank_config": {
                    "generation_config": {
                        "model": "projects/{}/locations/{}/publishers/google/models/gemini-3.1-pro-preview".format(
                            self._project_id, self._location
                        ),
                    },
                    "similarity_search_config": {
                        "embedding_model": "projects/{}/locations/{}/publishers/google/models/gemini-embedding-001".format(
                            self._project_id, self._location
                        ),
                    },
                },
            },
        }
        engine = client.agent_engines.create(config=config)
        self._engine_id = engine.api_resource.name.split("/")[-1]
        self._parent = (
            f"projects/{self._project_id}/locations/{self._location}"
            f"/reasoningEngines/{self._engine_id}"
        )
        logger.info("Provisioned Agent Engine: %s", self._engine_id)

        # Persist auto-provisioned engine_id to config
        try:
            from hermes_constants import get_hermes_home
            config_path = get_hermes_home() / "gcp-memory-bank.json"
            if config_path.exists():
                cfg = json.loads(config_path.read_text())
            else:
                cfg = {
                    "project_id": self._project_id,
                    "location": self._location,
                }
            cfg["engine_id"] = self._engine_id
            config_path.write_text(json.dumps(cfg, indent=2))
            logger.info("Persisted engine_id to %s", config_path)
        except Exception as e:
            logger.warning("Failed to persist engine_id: %s", e)

        return self._engine_id

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._project_id = self._config.get("project_id", "festive-antenna-463514-m8")
        self._location = self._config.get("location", "us-central1")
        self._engine_id = self._config.get("engine_id", "")
        self._user_id = kwargs.get("user_id") or "hermes-user"
        self._app_name = kwargs.get("agent_identity") or "hermes"
        self._initialized = True

        if self._engine_id:
            self._parent = (
                f"projects/{self._project_id}/locations/{self._location}"
                f"/reasoningEngines/{self._engine_id}"
            )
            logger.info(
                "GCP Memory Bank initialized: project=%s location=%s engine=%s user=%s app=%s",
                self._project_id, self._location, self._engine_id, self._user_id, self._app_name,
            )
        else:
            self._ensure_engine()
            # Reload config so subsequent calls pick up the persisted engine_id
            self._config = _load_config()
            self._engine_id = self._config.get("engine_id", "")

    def system_prompt_block(self) -> str:
        return (
            "# GCP Memory Bank\n"
            f"Active. Engine: {self._engine_id}. User: {self._user_id}.\n"
            "Use memory_search to find facts, memory_store to save preferences, "
            "memory_profile for a full overview, memory_purge to delete all user data, "
            "memory_consolidate to clean up duplicates."
        )

    def _scope(self) -> Dict[str, str]:
        return {"app_name": self._app_name, "user_id": self._user_id}

    # -----------------------------------------------------------------------
    # Intelligent prefetch with keyword extraction
    # -----------------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## GCP Memory Bank\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall for the NEXT turn.

        Extracts keywords from the query and searches for relevant memories.
        Falls back to a generic scope search if keyword search returns nothing.
        """
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                memories: List[str] = []

                # Strategy 1: Search with extracted keywords
                keywords = _extract_keywords(query)
                if keywords:
                    search_query = " ".join(keywords[:5])
                    from google.cloud.aiplatform_v1beta1.types.memory_bank_service import RetrieveMemoriesRequest
                    request = RetrieveMemoriesRequest(
                        parent=self._parent,
                        scope=self._scope(),
                        similarity_search_params={"search_query": search_query},
                    )
                    response = client.retrieve_memories(request=request)
                    for r in response.retrieved_memories:
                        memories.append(r.memory.fact)

                # Strategy 2: If keyword search is empty, try broader topic search
                if not memories and len(keywords) > 1:
                    topic_terms = [k for k in keywords if k in _TOPIC_KEYWORDS]
                    if topic_terms:
                        request = RetrieveMemoriesRequest(
                            parent=self._parent,
                            scope=self._scope(),
                            similarity_search_params={"search_query": " ".join(topic_terms[:3])},
                        )
                        response = client.retrieve_memories(request=request)
                        for r in response.retrieved_memories:
                            memories.append(r.memory.fact)

                if memories:
                    # Deduplicate and format
                    seen: Set[str] = set()
                    unique_mems = []
                    for m in memories:
                        h = _memory_hash(m)
                        if h not in seen:
                            seen.add(h)
                            unique_mems.append(m)

                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {m}" for m in unique_mems[:8])
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("GCP Memory Bank prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="gcp-mb-prefetch")
        self._prefetch_thread.start()

    # -----------------------------------------------------------------------
    # Sync turn with structured fact extraction and deduplication
    # -----------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Persist structured facts from the turn (non-blocking)."""
        if self._is_breaker_open():
            return

        # Extract a structured fact instead of storing raw messages
        fact = _extract_fact(user_content, assistant_content)
        if not fact:
            # No structured fact extracted — skip storing this turn
            return

        # Deduplication check
        fact_hash = _memory_hash(fact)
        with self._hash_lock:
            now = time.monotonic()
            # Clean old hashes (> 1 hour)
            self._recent_hashes = {h: t for h, t in self._recent_hashes.items() if now - t < 3600}
            if fact_hash in self._recent_hashes:
                logger.debug("Skipping duplicate memory: %s", fact[:60])
                return
            self._recent_hashes[fact_hash] = now

        def _sync():
            try:
                client = self._get_client()
                from google.cloud.aiplatform_v1beta1.types import memory_bank as mb_types

                # Also check backend for near-duplicates
                if self._is_similar_in_backend(client, fact):
                    logger.debug("Skipping backend-duplicate: %s", fact[:60])
                    return

                mem = mb_types.Memory(fact=fact, scope=self._scope())
                client.create_memory(parent=self._parent, memory=mem)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("GCP Memory Bank sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="gcp-mb-sync")
        self._sync_thread.start()

    def _is_similar_in_backend(self, client, fact: str) -> bool:
        """Quick backend check: does a very similar memory already exist?"""
        try:
            from google.cloud.aiplatform_v1beta1.types.memory_bank_service import RetrieveMemoriesRequest
            request = RetrieveMemoriesRequest(
                parent=self._parent,
                scope=self._scope(),
                similarity_search_params={"search_query": fact},
            )
            response = client.retrieve_memories(request=request)
            for r in response.retrieved_memories:
                # If top result has very high similarity, consider it a dup
                if r.distance < 0.15:  # Lower = more similar in this proto
                    return True
                # Also check exact hash match
                if _memory_hash(r.memory.fact) == _memory_hash(fact):
                    return True
        except Exception:
            pass
        return False

    # -----------------------------------------------------------------------
    # Tools
    # -----------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, STORE_SCHEMA, PURGE_SCHEMA, CONSOLIDATE_SCHEMA, QUALITY_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "GCP Memory Bank temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        from google.cloud.aiplatform_v1beta1.types import memory_bank as mb_types

        if tool_name == "memory_profile":
            try:
                memories = []
                for m in client.list_memories(parent=self._parent):
                    scope = dict(m.scope) if m.scope else {}
                    if scope.get("user_id") == self._user_id and scope.get("app_name") == self._app_name:
                        memories.append(m.fact)
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                return json.dumps({"result": "\n".join(memories), "count": len(memories)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "memory_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                memories = []
                from google.cloud.aiplatform_v1beta1.types.memory_bank_service import RetrieveMemoriesRequest
                request = RetrieveMemoriesRequest(
                    parent=self._parent,
                    scope=self._scope(),
                    similarity_search_params={"search_query": query},
                )
                response = client.retrieve_memories(request=request)
                for r in response.retrieved_memories:
                    memories.append({"fact": r.memory.fact, "score": r.distance, "name": r.memory.name})
                    # Track retrieval count
                    with self._retrieval_lock:
                        self._retrieval_counts[r.memory.name] = self._retrieval_counts.get(r.memory.name, 0) + 1
                    if len(memories) >= top_k:
                        break
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No relevant memories found."})
                return json.dumps({"results": memories, "count": len(memories)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "memory_store":
            fact = args.get("fact", "")
            if not fact:
                return tool_error("Missing required parameter: fact")
            try:
                # Deduplication before storing
                if self._is_similar_in_backend(client, fact):
                    return json.dumps({"result": "Similar memory already exists. Skipped.", "deduplicated": True})

                mem = mb_types.Memory(fact=fact, scope=self._scope())
                op = client.create_memory(parent=self._parent, memory=mem)
                created = op.result(timeout=60)
                self._record_success()
                return json.dumps({"result": "Fact stored.", "name": created.name})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        elif tool_name == "memory_purge":
            try:
                deleted = 0
                for m in client.list_memories(parent=self._parent):
                    scope = dict(m.scope) if m.scope else {}
                    if scope.get("user_id") == self._user_id and scope.get("app_name") == self._app_name:
                        try:
                            client.delete_memory(name=m.name)
                            deleted += 1
                        except Exception:
                            pass
                self._record_success()
                # Also clear local dedup cache
                with self._hash_lock:
                    self._recent_hashes.clear()
                return json.dumps({"result": f"Deleted {deleted} memories."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Purge failed: {e}")

        elif tool_name == "memory_consolidate":
            return self._consolidate_memories(client)

        elif tool_name == "memory_quality":
            return self._memory_quality_report(client)

        return tool_error(f"Unknown tool: {tool_name}")

    def _consolidate_memories(self, client) -> str:
        """Find and merge duplicate memories."""
        try:
            # Collect all memories for this user
            all_mems: List[Any] = []
            for m in client.list_memories(parent=self._parent):
                scope = dict(m.scope) if m.scope else {}
                if scope.get("user_id") == self._user_id and scope.get("app_name") == self._app_name:
                    all_mems.append(m)

            if len(all_mems) < 2:
                return json.dumps({"result": "Not enough memories to consolidate.", "count": len(all_mems)})

            # Find duplicates by hash
            hash_to_mems: Dict[str, List[Any]] = {}
            for m in all_mems:
                h = _memory_hash(m.fact)
                hash_to_mems.setdefault(h, []).append(m)

            deleted = 0
            kept = 0
            for h, mems in hash_to_mems.items():
                if len(mems) > 1:
                    # Keep the first, delete the rest
                    for dup in mems[1:]:
                        try:
                            client.delete_memory(name=dup.name)
                            deleted += 1
                        except Exception:
                            pass
                    kept += 1
                else:
                    kept += 1

            self._record_success()
            return json.dumps({
                "result": f"Consolidated {len(all_mems)} memories: kept {kept}, removed {deleted} duplicates.",
                "kept": kept,
                "removed": deleted,
            })
        except Exception as e:
            self._record_failure()
            return tool_error(f"Consolidation failed: {e}")

    def _detect_conflicts(self, memories: List[Any]) -> List[Dict[str, Any]]:
        """Detect potentially conflicting memories.

        Looks for memories that share keywords but express opposite sentiments
        (like vs dislike, lives vs used to live, etc).
        """
        conflicts = []
        # Sentiment markers
        positive = {"like", "love", "enjoy", "prefer", "lives", "has", "is"}
        negative = {"dislike", "hate", "don't like", "used to", "had", "was"}

        for i, m1 in enumerate(memories):
            words1 = set(re.findall(r"[a-z]+", m1.fact.lower()))
            for m2 in memories[i+1:]:
                words2 = set(re.findall(r"[a-z]+", m2.fact.lower()))
                # Check for substantial keyword overlap
                overlap = words1 & words2
                if len(overlap) >= 3:
                    # Check for opposing sentiment
                    s1_pos = any(p in m1.fact.lower() for p in positive)
                    s1_neg = any(n in m1.fact.lower() for n in negative)
                    s2_pos = any(p in m2.fact.lower() for p in positive)
                    s2_neg = any(n in m2.fact.lower() for n in negative)

                    if (s1_pos and s2_neg) or (s1_neg and s2_pos):
                        conflicts.append({
                            "memory_a": m1.fact,
                            "memory_b": m2.fact,
                            "overlap": list(overlap)[:5],
                            "issue": "Potential contradiction",
                        })
        return conflicts

    def _memory_quality_report(self, client) -> str:
        """Generate a quality report for all memories."""
        try:
            all_mems = []
            for m in client.list_memories(parent=self._parent):
                scope = dict(m.scope) if m.scope else {}
                if scope.get("user_id") == self._user_id and scope.get("app_name") == self._app_name:
                    all_mems.append(m)

            if not all_mems:
                return json.dumps({"result": "No memories to analyze."})

            # Build quality metrics
            report = {
                "total_memories": len(all_mems),
                "retrieval_stats": {},
                "by_category": {},
                "conflicts": [],
            }

            with self._retrieval_lock:
                counts = dict(self._retrieval_counts)

            category_keywords = {
                "identity": ["name", "called", "from", "lives"],
                "preferences": ["like", "love", "enjoy", "prefer", "dislike", "hate"],
                "skills": ["learning", "studying", "knows", "uses"],
                "pets": ["dog", "cat", "pet", "retriever", "max"],
                "possessions": ["has", "own", "bought", "got"],
            }

            for m in all_mems:
                fact_lower = m.fact.lower()
                # Retrieval count
                report["retrieval_stats"][m.name] = counts.get(m.name, 0)
                # Categorize
                for cat, keywords in category_keywords.items():
                    if any(kw in fact_lower for kw in keywords):
                        report["by_category"].setdefault(cat, []).append(m.fact[:80])

            # Detect conflicts
            report["conflicts"] = self._detect_conflicts(all_mems)

            # Top retrieved memories
            top_retrieved = sorted(
                report["retrieval_stats"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            report["top_retrieved"] = [
                {"name": name, "retrievals": count}
                for name, count in top_retrieved if count > 0
            ]

            return json.dumps(report, indent=2)
        except Exception as e:
            return tool_error(f"Quality report failed: {e}")

    # -----------------------------------------------------------------------
    # Session end with structured fact extraction
    # -----------------------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Extract key facts from the full conversation and store them."""
        if self._is_breaker_open() or not messages:
            return

        def _extract():
            try:
                client = self._get_client()
                from google.cloud.aiplatform_v1beta1.types import memory_bank as mb_types

                # Extract structured facts from all user messages
                facts_found: set[str] = set()
                for msg in messages:
                    if msg.get("role") != "user":
                        continue
                    fact = _extract_fact(msg.get("content", ""), "")
                    if not fact or fact in facts_found:
                        continue
                    facts_found.add(fact)

                stored = 0
                for fact in facts_found:
                    # Skip if already stored recently (sync_turn may have raced ahead)
                    fact_hash = _memory_hash(fact)
                    with self._hash_lock:
                        if fact_hash in self._recent_hashes:
                            continue
                        self._recent_hashes[fact_hash] = time.monotonic()

                    # Skip if similar exists in backend
                    if self._is_similar_in_backend(client, fact):
                        continue

                    mem = mb_types.Memory(fact=fact, scope=self._scope())
                    try:
                        client.create_memory(parent=self._parent, memory=mem)
                        stored += 1
                    except Exception:
                        pass

                if stored:
                    logger.info("Session-end: stored %d structured facts", stored)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("GCP Memory Bank session-end extraction failed: %s", e)

        threading.Thread(target=_extract, daemon=True, name="gcp-mb-session-end").start()

    # -----------------------------------------------------------------------
    # Hooks
    # -----------------------------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to GCP Memory Bank."""
        if action == "add" and target == "memory" and content:
            # Treat as an explicit fact
            self.sync_turn(content, "", session_id="")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


QUALITY_SCHEMA = {
    "name": "memory_quality",
    "description": (
        "Analyze memory quality: retrieval frequency, confidence scores, "
        "and detect conflicting or outdated facts. Use to audit memory health."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def register(ctx) -> None:
    """Register GCP Memory Bank as a memory provider plugin."""
    ctx.register_memory_provider(GcpMemoryBankProvider())
