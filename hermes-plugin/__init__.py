"""GCP Memory Bank plugin for Hermes.

Uses Google Cloud Memory Bank via the official Agent Platform SDK.
Follows the patterns documented at:
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/setup
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/ingest-events
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/generate-memories
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/fetch-memories
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/revisions
  https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin

Config via environment variables:
  GCP_PROJECT_ID      — GCP project (default: )
  GCP_LOCATION        — Vertex AI region (default: us-central1)
  GCP_MEMORY_ENGINE   — Optional: pre-created Agent Engine ID. Auto-provisioned if unset.

Or via $HERMES_HOME/gcp-memory-bank.json.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    from hermes_constants import get_hermes_home
    config = {
        "project_id": os.environ.get("GCP_PROJECT_ID", ""),
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
        "ranked by relevance (Euclidean distance). Use for targeted recall mid-conversation. "
        "Optional filter string supports system field filtering (e.g. topics, create_time)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "filter": {"type": "string", "description": "Optional system field filter string (EBNF syntax)."},
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "memory_store",
    "description": (
        "Store a durable fact about the user in GCP Memory Bank. "
        "Stored verbatim (no LLM extraction). Use for explicit preferences, corrections, or decisions. "
        "Optional metadata can be attached for later filtering."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fact": {"type": "string", "description": "The fact to store."},
            "metadata": {
                "type": "object",
                "description": "Optional structured metadata as key-value pairs for filtering.",
            },
        },
        "required": ["fact"],
    },
}

REVISIONS_SCHEMA = {
    "name": "memory_revisions",
    "description": (
        "List the revision history of a specific memory. "
        "Shows how a memory evolved over time (created, updated, deleted). "
        "Supports optional filter by labels (e.g. labels.verified='true'). "
        "Requires the memory name returned by memory_search or memory_store."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_name": {"type": "string", "description": "Full memory resource name (e.g. projects/.../memories/...)"},
            "filter": {"type": "string", "description": "Optional EBNF filter string for label filtering (e.g. labels.verified='true')."},
        },
        "required": ["memory_name"],
    },
}

REVISION_GET_SCHEMA = {
    "name": "memory_revision_get",
    "description": (
        "Retrieve a single memory revision by its full resource name. "
        "Useful for time-travel queries or inspecting a specific version. "
        "Requires the revision name returned by memory_revisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "revision_name": {"type": "string", "description": "Full revision resource name (e.g. projects/.../memories/.../revisions/...)"},
        },
        "required": ["revision_name"],
    },
}

ROLLBACK_SCHEMA = {
    "name": "memory_rollback",
    "description": (
        "Rollback a memory to a previous revision. "
        "Irreversible for newer revisions (they are removed). Use for correcting "
        "incorrect consolidations or reverting to a known-good state. "
        "Requires the memory name and the target revision ID from memory_revisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_name": {"type": "string", "description": "Full memory resource name to rollback."},
            "target_revision_id": {"type": "string", "description": "The revision ID (last path segment) to restore to."},
        },
        "required": ["memory_name", "target_revision_id"],
    },
}

PURGE_SCHEMA = {
    "name": "memory_purge",
    "description": (
        "Delete memories from GCP Memory Bank matching an optional filter. "
        "If no filter is provided, deletes ALL memories for the current user. "
        "Irreversible. Use only when the user explicitly requests data deletion."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "description": "Optional EBNF filter string (e.g. 'scope.user_id=\"jithendra\"'). If omitted, scopes to current user/app automatically.",
            },
        },
        "required": [],
    },
}

GET_SCHEMA = {
    "name": "memory_get",
    "description": (
        "Retrieve a single memory by its full resource name. "
        "Returns fact, scope, labels, create/update times, and expiration. "
        "Use when you need details for a specific memory found via search or profile."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_name": {"type": "string", "description": "Full memory resource name (e.g. projects/.../memories/...)"},
        },
        "required": ["memory_name"],
    },
}

DELETE_SCHEMA = {
    "name": "memory_delete",
    "description": (
        "Delete a single memory by its full resource name. "
        "Irreversible. Use when a specific memory is outdated or incorrect."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_name": {"type": "string", "description": "Full memory resource name to delete."},
        },
        "required": ["memory_name"],
    },
}

INGEST_SCHEMA = {
    "name": "memory_ingest",
    "description": (
        "Stream events into GCP Memory Bank via ingest_events (background processing). "
        "More efficient than direct store for batching multiple events. "
        "Per Google docs: https://cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/ingest-events "
        "Events are processed asynchronously; memories are generated in the background. "
        "Use generation_trigger_config to control when the service generates memories from events."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "description": (
                    "List of event dicts. Each may contain 'event_id' (dedup key) "
                    "and 'text' (required). Events are wrapped in direct_contents_source "
                    "per the ingest_events API contract."
                ),
                "items": {"type": "object"},
            },
            "stream_id": {
                "type": "string",
                "description": "Optional session/stream identifier for grouping related events.",
            },
            "generation_trigger_config": {
                "type": "object",
                "description": (
                    "Optional config dict controlling when memories are generated from events. "
                    "Example: {'generation_rule': {'idle_duration': '60s'}}. "
                    "If omitted, uses idle_duration=60s default."
                ),
            },
        },
        "required": ["events"],
    },
}

SYNTHESIZE_SCHEMA = {
    "name": "memory_synthesize",
    "description": (
        "Synthesize a narrative summary from retrieved memories. "
        "Queries GCP Memory Bank for relevant facts and composes them into a coherent paragraph. "
        "Useful for reflect-like synthesis before making decisions or when you need context-aware insight."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Topic or question to synthesize memories about.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max memories to retrieve for synthesis (default: 10, max: 50).",
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class GcpMemoryBankProvider(MemoryProvider):
    """Google Cloud Memory Bank provider for Hermes.

    Follows official docs patterns:
    - Memory generation via generate_memories(direct_contents_source=...)
    - Memory retrieval via retrieve_memories(similarity_search_params=...)
    - Background generation with wait_for_completion=False
    - Native consolidation handled by Memory Bank (no custom dedup)
    """

    def __init__(self):
        self._config: Optional[dict] = None
        self._client: Optional[Any] = None
        self._vclient: Optional[Any] = None
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
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._initialized = False
        # Event buffer for generate_memories
        self._events: List[Dict[str, Any]] = []
        self._events_lock = threading.Lock()
        self._turn_count = 0
        # Mid-session generation thread tracking
        self._mid_gen_thread: Optional[threading.Thread] = None
        self._generate_every_n = 0
        self._auto_prefetch = True
        self._prefetch_mode = "facts"
        # GCP Session integration
        self._gcp_session_name: Optional[str] = None
        self._gcp_session_lock = threading.Lock()
        self._use_gcp_sessions: bool = True

    @property
    def name(self) -> str:
        return "gcp-memory-bank"

    def is_available(self) -> bool:
        try:
            import google.cloud.aiplatform_v1beta1  # noqa: F401
            import vertexai  # noqa: F401
        except ImportError:
            return False
        cfg = _load_config()
        if not cfg.get("project_id"):
            return False
        # Check for GCP credentials locally — do NOT make network calls
        # per Hermes contract. Accept service account JSON or gcloud ADC.
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.isfile(creds_path):
            return True
        # Default gcloud ADC path
        default_adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
        if os.path.isfile(default_adc):
            return True
        return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "project_id",
                "description": "Google Cloud project ID",
                "required": True,
                "default": "",
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
            {
                "key": "user_id",
                "description": "Canonical user identity for memory scoping (overrides platform user_id)",
                "required": False,
                "default": "",
            },
            {
                "key": "app_name",
                "description": "Application name for memory scoping (overrides agent profile)",
                "required": False,
                "default": "hermes",
            },
            {
                "key": "default_revision_labels",
                "description": "Optional labels applied to every revision created by on_memory_write (e.g. {'source': 'hermes', 'verified': 'true'})",
                "required": False,
                "default": {},
            },
            {
                "key": "create_ttl_days",
                "description": "TTL for manually created memories (granular_ttl_config.create_ttl)",
                "required": False,
                "default": 365,
            },
            {
                "key": "generate_created_ttl_days",
                "description": "TTL for AI-generated memories (granular_ttl_config.generate_created_ttl)",
                "required": False,
                "default": 365,
            },
            {
                "key": "generation_model",
                "description": "Gemini model for memory generation (e.g. gemini-2.5-flash, gemini-3.1-pro-preview)",
                "required": False,
                "default": "gemini-2.5-flash",
            },
            {
                "key": "embedding_model",
                "description": "Embedding model for memory retrieval (e.g. text-embedding-005, gemini-embedding-2)",
                "required": False,
                "default": "text-embedding-005",
            },
            {
                "key": "generate_every_n_turns",
                "description": "Generate memories every N turns mid-session (0 = disabled, only at session end).",
                "required": False,
                "default": 0,
            },
            {
                "key": "auto_prefetch",
                "description": "Automatically prefetch relevant memories at session start.",
                "required": False,
                "default": True,
            },
            {
                "key": "prefetch_mode",
                "description": "Prefetch formatting: facts (bullet list) or narrative (paragraph)",
                "required": False,
                "default": "facts",
                "choices": ["facts", "narrative"],
            },
            {
                "key": "use_gcp_sessions",
                "description": "Use GCP Agent Runtime Sessions as the primary event store for automatic memory generation via vertex_session_source.",
                "required": False,
                "default": True,
            },
            {
                "key": "gcp_session_ttl_seconds",
                "description": "TTL for GCP Sessions (minimum 86400 = 24 hours).",
                "required": False,
                "default": 86400,
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

    def _get_vertex_client(self):
        """Return a cached vertexai.Client.

        Creates once per provider instance to avoid repeated initialization overhead.
        """
        if self._vclient is not None:
            return self._vclient
        with self._client_lock:
            if self._vclient is not None:
                return self._vclient
            import vertexai
            self._vclient = vertexai.Client(project=self._project_id, location=self._location)
            return self._vclient

    def _get_client(self):
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

    def _ensure_gcp_session(self) -> Optional[str]:
        """Create or return an existing GCP Agent Runtime Session.

        Sessions are used as the official event store for vertex_session_source
        memory generation. Minimum TTL is 86400s (24h).
        """
        if not self._use_gcp_sessions:
            return None
        if self._gcp_session_name:
            return self._gcp_session_name
        with self._gcp_session_lock:
            if self._gcp_session_name:
                return self._gcp_session_name
            try:
                vclient = self._get_vertex_client()
                engine_name = (
                    f"projects/{self._project_id}/locations/{self._location}"
                    f"/reasoningEngines/{self._engine_id}"
                )
                ttl_secs = max(int(self._config.get("gcp_session_ttl_seconds", 86400)), 86400)
                op = vclient.agent_engines.sessions.create(
                    name=engine_name,
                    user_id=self._user_id,
                    config={
                        "display_name": f"hermes-{self._session_id[:16] if self._session_id else 'unknown'}",
                        "ttl": f"{ttl_secs}s",
                    },
                )
                self._gcp_session_name = op.response.name
                self._record_success()
                logger.info("Created GCP Session: %s", self._gcp_session_name)
                return self._gcp_session_name
            except Exception as e:
                self._record_failure()
                logger.warning("Failed to create GCP Session: %s", e)
                return None

    def _append_events_to_gcp_session(self, events: List[Dict[str, Any]], turn_marker: int = 0) -> None:
        """Synchronously append events to the active GCP Session.

        Call this from background threads only. Adds deterministic event_id
        for deduplication.
        """
        if not self._gcp_session_name:
            return
        import datetime as _dt
        try:
            vclient = self._get_vertex_client()
            for idx, ev in enumerate(events):
                role = ev.get("content", {}).get("role", "user")
                text = ev.get("content", {}).get("parts", [{}])[0].get("text", "")
                event_id = ev.get("event_id", f"{self._session_id or 'unknown'}-t{turn_marker}-e{idx}")
                vclient.agent_engines.sessions.events.append(
                    name=self._gcp_session_name,
                    author=role,
                    invocation_id=event_id,
                    timestamp=_dt.datetime.now(_dt.timezone.utc),
                    config={
                        "content": {
                            "role": role,
                            "parts": [{"text": text}]
                        }
                    },
                )
            logger.debug("Appended %d events to GCP Session %s", len(events), self._gcp_session_name)
            self._record_success()
        except Exception as e:
            self._record_failure()
            logger.warning("Failed to append events to GCP Session: %s", e)

    def _ingest_events(self, events: List[Dict[str, Any]], turn_count: int) -> None:
        """Fallback ingest via ingest_events (proven working path)."""
        try:
            vclient = self._get_vertex_client()
            engine_name = (
                f"projects/{self._project_id}/locations/{self._location}"
                f"/reasoningEngines/{self._engine_id}"
            )
            vclient.agent_engines.memories.ingest_events(
                name=engine_name,
                scope=self._retrieval_scope(),
                direct_contents_source={"events": events},
            )
            self._record_success()
            logger.info("Memory ingest triggered (%d events, %d turns).", len(events), turn_count)
        except Exception as e:
            self._record_failure()
            logger.warning("Memory ingest failed: %s", e)

    def _build_engine_config(self) -> Dict[str, Any]:
        """Return Memory Bank config per official docs."""
        return {
            "generation_config": {
                "model": f"projects/{self._project_id}/locations/global/publishers/google/models/{self._config.get('generation_model', 'gemini-2.5-flash')}",
            },
            "similarity_search_config": {
                "embedding_model": f"projects/{self._project_id}/locations/global/publishers/google/models/{self._config.get('embedding_model', 'gemini-embedding-001')}",
            },
            "ttl_config": {
                "granular_ttl_config": {
                    "create_ttl": f"{self._config.get('create_ttl_days', 365) * 24 * 60 * 60}s",
                    "generate_created_ttl": f"{self._config.get('generate_created_ttl_days', 365) * 24 * 60 * 60}s",
                },
            },
            "customization_configs": [
                {
                    "memory_topics": [
                        {"managed_memory_topic": {"managed_topic_enum": "USER_PERSONAL_INFO"}},
                        {"managed_memory_topic": {"managed_topic_enum": "USER_PREFERENCES"}},
                        {"managed_memory_topic": {"managed_topic_enum": "KEY_CONVERSATION_DETAILS"}},
                        {"managed_memory_topic": {"managed_topic_enum": "EXPLICIT_INSTRUCTIONS"}},
                    ],
                    "consolidation_config": {
                        "revisions_per_candidate_count": 1
                    },
                    "generate_memories_examples": [
                        {
                            "conversationSource": {
                                "events": [
                                    {
                                        "content": {
                                            "role": "model",
                                            "parts": [{"text": "Hey! What can I help you with today?"}]
                                        }
                                    },
                                    {
                                        "content": {
                                            "role": "user",
                                            "parts": [{"text": "I just moved to Fort Wayne, Indiana. Still getting used to the area."}]
                                        }
                                    }
                                ]
                            },
                            "generatedMemories": [
                                {"fact": "The user lives in Fort Wayne, Indiana."}
                            ]
                        },
                        {
                            "conversationSource": {
                                "events": [
                                    {
                                        "content": {
                                            "role": "model",
                                            "parts": [{"text": "Want me to use the latest Gemini model for this task?"}]
                                        }
                                    },
                                    {
                                        "content": {
                                            "role": "user",
                                            "parts": [{"text": "Yes, always use the newest available model. I don't care about cost."}]
                                        }
                                    }
                                ]
                            },
                            "generatedMemories": [
                                {"fact": "The user prefers using the newest available AI models regardless of cost."}
                            ]
                        },
                        {
                            "conversationSource": {
                                "events": [
                                    {
                                        "content": {
                                            "role": "model",
                                            "parts": [{"text": "Got it. Any other preferences I should know about?"}]
                                        }
                                    },
                                    {
                                        "content": {
                                            "role": "user",
                                            "parts": [{"text": "Remember to always follow official docs first. Don't make up your own rules."}]
                                        }
                                    }
                                ]
                            },
                            "generatedMemories": [
                                {"fact": "The user explicitly instructed the agent to always follow official documentation first and not invent its own rules."}
                            ]
                        },
                        {
                            "conversationSource": {
                                "events": [
                                    {
                                        "content": {
                                            "role": "model",
                                            "parts": [{"text": "Here's the weather forecast for Fort Wayne: sunny, 72°F."}]
                                        }
                                    },
                                    {
                                        "content": {
                                            "role": "user",
                                            "parts": [{"text": "Thanks, that's helpful."}]
                                        }
                                    }
                                ]
                            },
                            "generatedMemories": []
                        }
                    ],
                    "enable_third_person_memories": False,
                }
            ],
            "disable_memory_revisions": False,
        }

    def _ensure_engine(self) -> str:
        if self._engine_id:
            return self._engine_id
        logger.info("Auto-provisioning Agent Engine for Memory Bank...")
        import vertexai
        client = vertexai.Client(project=self._project_id, location=self._location)
        memory_bank_config = self._build_engine_config()
        config = {
            "display_name": f"hermes-memory-{self._app_name}",
            "context_spec": {
                "memory_bank_config": memory_bank_config,
            },
        }
        engine = client.agent_engines.create(config=config)
        self._engine_id = engine.api_resource.name.split("/")[-1]
        self._parent = (
            f"projects/{self._project_id}/locations/{self._location}"
            f"/reasoningEngines/{self._engine_id}"
        )
        logger.info("Provisioned Agent Engine: %s", self._engine_id)
        # Persist
        try:
            from hermes_constants import get_hermes_home
            config_path = get_hermes_home() / "gcp-memory-bank.json"
            if config_path.exists():
                cfg = json.loads(config_path.read_text())
            else:
                cfg = {"project_id": self._project_id, "location": self._location}
            cfg["engine_id"] = self._engine_id
            config_path.write_text(json.dumps(cfg, indent=2))
            logger.info("Persisted engine_id to %s", config_path)
        except Exception as e:
            logger.warning("Failed to persist engine_id: %s", e)
        return self._engine_id

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._project_id = self._config.get("project_id", "")
        self._location = self._config.get("location", "us-central1")
        self._engine_id = self._config.get("engine_id", "")
        self._user_id = self._config.get("user_id") or kwargs.get("user_id") or "hermes-user"
        self._app_name = self._config.get("app_name") or kwargs.get("agent_identity") or "hermes"
        self._agent_context = kwargs.get("agent_context", "primary")
        self._session_id = session_id
        self._initialized = True
        # Read new config options with safe defaults
        try:
            self._generate_every_n = int(self._config.get("generate_every_n_turns", 0))
        except (ValueError, TypeError):
            self._generate_every_n = 0
        self._auto_prefetch = bool(self._config.get("auto_prefetch", True))
        self._prefetch_mode = self._config.get("prefetch_mode", "facts")
        if self._prefetch_mode not in ("facts", "narrative"):
            self._prefetch_mode = "facts"
        self._use_gcp_sessions = bool(self._config.get("use_gcp_sessions", True))
        if self._engine_id:
            self._parent = (
                f"projects/{self._project_id}/locations/{self._location}"
                f"/reasoningEngines/{self._engine_id}"
            )
            logger.info(
                "GCP Memory Bank initialized: project=%s location=%s engine=%s user=%s app=%s context=%s generate_every_n=%s auto_prefetch=%s use_gcp_sessions=%s",
                self._project_id, self._location, self._engine_id, self._user_id, self._app_name, self._agent_context,
                self._generate_every_n, self._auto_prefetch, self._use_gcp_sessions,
            )
        else:
            self._ensure_engine()
            self._config = _load_config()
            # Preserve engine_id from _ensure_engine even if config file write failed
            if not self._engine_id:
                self._engine_id = self._config.get("engine_id", "")
            # Ensure parent is set if engine_id was recovered from config
            if self._engine_id and not self._parent:
                self._parent = (
                    f"projects/{self._project_id}/locations/{self._location}"
                    f"/reasoningEngines/{self._engine_id}"
                )
        # Kick off GCP Session creation in background so sync_turn isn't blocked
        if self._use_gcp_sessions and self._engine_id:
            threading.Thread(target=self._ensure_gcp_session, daemon=True, name="gcp-session-init").start()

    def system_prompt_block(self) -> str:
        session_info = f" Session: {self._gcp_session_name.split('/')[-1]}." if self._gcp_session_name else ""
        return (
            "# GCP Memory Bank\n"
            f"Active. Engine: {self._engine_id}. User: {self._user_id}.{session_info}\n"
            "Use memory_search to find facts, memory_store to save preferences, "
            "memory_get to fetch a specific memory, memory_delete to remove one, "
            "memory_profile for a full overview, memory_revisions to inspect history, "
            "memory_revision_get to retrieve a specific version, memory_rollback to revert, "
            "memory_ingest for batch/streaming events, memory_purge to delete matching memories (optionally filtered), "
            "memory_synthesize to compose a narrative summary from retrieved memories."
        )

    def _scope(self) -> Dict[str, str]:
        scope = {"app_name": self._app_name, "user_id": self._user_id}
        if self._session_id:
            scope["session_id"] = self._session_id
        return scope

    def _retrieval_scope(self) -> Dict[str, str]:
        """User-scoped retrieval (no session_id) so memories survive across sessions."""
        return {"app_name": self._app_name, "user_id": self._user_id}

    # -----------------------------------------------------------------------
    # Prefetch (retrieval only, per docs)
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
        if not self._auto_prefetch:
            return
        if self._is_breaker_open():
            return
        def _run():
            try:
                client = self._get_client()
                from google.cloud.aiplatform_v1beta1.types.memory_bank_service import RetrieveMemoriesRequest
                request = RetrieveMemoriesRequest(
                    parent=self._parent,
                    scope=self._retrieval_scope(),
                    similarity_search_params={"search_query": query, "top_k": 8},
                )
                response = client.retrieve_memories(request=request)
                memories = []
                for r in response.retrieved_memories:
                    memories.append(r.memory.fact)
                if memories:
                    if self._prefetch_mode == "narrative":
                        text = " ".join(f"{m}." if not m.endswith((".", "!", "?")) else m for m in memories)
                    else:
                        text = "\n".join(f"- {m}" for m in memories)
                    with self._prefetch_lock:
                        self._prefetch_result = text
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("GCP Memory Bank prefetch failed: %s", e)
        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="gcp-mb-prefetch")
        self._prefetch_thread.start()

    # -----------------------------------------------------------------------
    # Sync turn: buffer events for later generation (per docs)
    # -----------------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Buffer conversation events and mirror to GCP Session.

        Events are appended to the active GCP Session (fire-and-forget) for
        official vertex_session_source memory generation at session end.
        A local buffer is kept as fallback for ingest_events.
        """
        if self._agent_context != "primary":
            return
        user_event = {
            "content": {"role": "user", "parts": [{"text": user_content}]}
        }
        model_event = {
            "content": {"role": "model", "parts": [{"text": assistant_content}]}
        }
        events = [user_event, model_event]

        with self._events_lock:
            self._events.extend(events)
            self._turn_count += 1
            turn_count_snapshot = self._turn_count
            if len(self._events) > 200:
                self._events = self._events[-200:]
            should_generate_mid = self._generate_every_n > 0 and self._turn_count >= self._generate_every_n
            if should_generate_mid:
                events_copy = list(self._events)
                self._events.clear()
                self._turn_count = 0

        # Mirror to GCP Session (official event store)
        if self._use_gcp_sessions:
            if not self._gcp_session_name:
                self._ensure_gcp_session()
            if self._gcp_session_name:
                threading.Thread(
                    target=self._append_events_to_gcp_session,
                    args=(events, turn_count_snapshot),
                    daemon=True,
                    name=f"gcp-sess-append-{turn_count_snapshot}",
                ).start()

        # Mid-session fallback generation via ingest_events (proven working)
        if should_generate_mid:
            self._trigger_mid_session_ingest(events_copy, turn_count_snapshot)

    def _trigger_mid_session_ingest(self, events: List[Dict[str, Any]], turn_count: int) -> None:
        """Fire-and-forget mid-session memory ingest via ingest_events.

        Uses the proven ingest_events path rather than generate() which silently
        fails with direct_contents_source in current SDK versions.
        """
        if self._is_breaker_open():
            return
        def _ingest():
            try:
                self._ingest_events(events, turn_count)
                logger.info("Mid-session memory ingest completed (%d events, %d turns).", len(events), turn_count)
            except Exception as e:
                logger.warning("Mid-session memory ingest failed: %s", e)
        self._mid_gen_thread = threading.Thread(target=_ingest, daemon=True, name="gcp-mb-mid-ingest")
        self._mid_gen_thread.start()

    # -----------------------------------------------------------------------
    # Session end: generate memories via official API (per docs)
    # -----------------------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Generate memories from GCP Session via vertex_session_source.

        Primary path: append remaining buffered events to GCP Session, then call
        generate(vertex_session_source=...) for automatic memory extraction.

        Fallback path: if GCP Session is unavailable, use ingest_events with the
        local event buffer.
        """
        if self._agent_context != "primary":
            return
        if self._is_breaker_open():
            return
        with self._events_lock:
            events = list(self._events)
            self._events.clear()
            turn_count = self._turn_count
            self._turn_count = 0
        if not events and not self._gcp_session_name:
            return

        def _end_session():
            try:
                # 1. Flush remaining buffered events to GCP Session
                if self._gcp_session_name and events:
                    self._append_events_to_gcp_session(events, turn_count)

                # 2. Official path: generate from GCP Session
                if self._gcp_session_name:
                    vclient = self._get_vertex_client()
                    engine_name = (
                        f"projects/{self._project_id}/locations/{self._location}"
                        f"/reasoningEngines/{self._engine_id}"
                    )
                    vclient.agent_engines.memories.generate(
                        name=engine_name,
                        vertex_session_source={"session": self._gcp_session_name},
                        scope=self._retrieval_scope(),
                        config={"wait_for_completion": False},
                    )
                    self._record_success()
                    logger.info(
                        "Session-end memory generation via vertex_session_source triggered "
                        "(GCP Session: %s, %d events, %d turns).",
                        self._gcp_session_name, len(events), turn_count,
                    )
                else:
                    # Fallback: ingest_events with buffered events
                    if events:
                        self._ingest_events(events, turn_count)
            except Exception as e:
                self._record_failure()
                logger.warning("Session-end vertex_session_source failed: %s", e)
                if events:
                    self._ingest_events(events, turn_count)

        threading.Thread(target=_end_session, daemon=True, name="gcp-mb-session-end").start()

    # -----------------------------------------------------------------------
    # Pre-compress: flush buffered events before context truncation
    # -----------------------------------------------------------------------

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> None:
        """Generate memories from GCP Session before context compression discards them.

        Primary: vertex_session_source generation. Fallback: ingest_events.
        """
        if self._agent_context != "primary":
            return
        if self._is_breaker_open():
            return
        with self._events_lock:
            events = list(self._events)
            self._events.clear()
            turn_count = self._turn_count
            self._turn_count = 0
        if not events:
            return

        def _precompress():
            try:
                if self._gcp_session_name:
                    self._append_events_to_gcp_session(events, turn_count)
                    vclient = self._get_vertex_client()
                    engine_name = (
                        f"projects/{self._project_id}/locations/{self._location}"
                        f"/reasoningEngines/{self._engine_id}"
                    )
                    vclient.agent_engines.memories.generate(
                        name=engine_name,
                        vertex_session_source={"session": self._gcp_session_name},
                        scope=self._retrieval_scope(),
                        config={"wait_for_completion": False},
                    )
                    self._record_success()
                    logger.info(
                        "Pre-compress memory generation via vertex_session_source triggered "
                        "(%d events, %d turns).", len(events), turn_count,
                    )
                else:
                    self._ingest_events(events, turn_count)
            except Exception as e:
                self._record_failure()
                logger.warning("Pre-compress vertex_session_source failed: %s", e)
                self._ingest_events(events, turn_count)

        threading.Thread(target=_precompress, daemon=True, name="gcp-mb-precompress").start()

    def shutdown(self) -> None:
        """Clean shutdown — close the gRPC client connection."""
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.transport.close()
                except Exception:
                    pass
                self._client = None
            # High-level vertexai.Client does not expose .transport;
            # just drop the reference and let GC clean up.
            self._vclient = None

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to Memory Bank.

        When the agent edits MEMORY.md or USER.md, we also store the fact
        in Memory Bank so it survives profile changes.
        """
        if self._agent_context != "primary":
            return
        if self._is_breaker_open():
            return
        fact = f"[{action.upper()} {target}] {content}"

        def _write():
            try:
                vclient = self._get_vertex_client()
                engine_name = (
                    f"projects/{self._project_id}/locations/{self._location}"
                    f"/reasoningEngines/{self._engine_id}"
                )
                vclient.agent_engines.memories.create(
                    name=engine_name,
                    fact=fact,
                    scope=self._retrieval_scope(),
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("on_memory_write failed: %s", e)

        threading.Thread(target=_write, daemon=True, name="gcp-mb-write").start()

    # -----------------------------------------------------------------------
    # Tool schemas
    # -----------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            PROFILE_SCHEMA, SEARCH_SCHEMA, STORE_SCHEMA, GET_SCHEMA, DELETE_SCHEMA,
            REVISIONS_SCHEMA, REVISION_GET_SCHEMA, ROLLBACK_SCHEMA, INGEST_SCHEMA, PURGE_SCHEMA,
            SYNTHESIZE_SCHEMA,
        ]

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
                from google.cloud.aiplatform_v1beta1.types.memory_bank_service import RetrieveMemoriesRequest
                request = RetrieveMemoriesRequest(
                    parent=self._parent,
                    scope=self._retrieval_scope(),
                )
                response = client.retrieve_memories(request=request)
                memories = []
                for r in response.retrieved_memories:
                    memories.append(r.memory.fact)
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
            filter_str = args.get("filter", "")
            try:
                memories = []
                if filter_str:
                    # System field filtering requires high-level vertexai.Client
                    # per fetch-memories doc (low-level proto lacks filter field).
                    vclient = self._get_vertex_client()
                    engine_name = f"projects/{self._project_id}/locations/{self._location}/reasoningEngines/{self._engine_id}"
                    results = vclient.agent_engines.memories.retrieve(
                        name=engine_name,
                        scope=self._retrieval_scope(),
                        similarity_search_params={"search_query": query, "top_k": top_k},
                        config={"filter": filter_str},
                    )
                    for r in results:
                        memories.append({"fact": r.memory.fact, "distance": r.distance, "name": r.memory.name})
                else:
                    from google.cloud.aiplatform_v1beta1.types.memory_bank_service import RetrieveMemoriesRequest
                    request = RetrieveMemoriesRequest(
                        parent=self._parent,
                        scope=self._retrieval_scope(),
                        similarity_search_params={"search_query": query, "top_k": top_k},
                    )
                    response = client.retrieve_memories(request=request)
                    for r in response.retrieved_memories:
                        memories.append({"fact": r.memory.fact, "distance": r.distance, "name": r.memory.name})
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
            metadata = args.get("metadata") or {}
            try:
                vclient = self._get_vertex_client()
                engine_name = f"projects/{self._project_id}/locations/{self._location}/reasoningEngines/{self._engine_id}"
                typed_metadata = {}
                if metadata:
                    import vertexai
                    for k, v in metadata.items():
                        if isinstance(v, dict):
                            typed_metadata[k] = vertexai.types.MemoryMetadataValue(**v)
                        else:
                            typed_metadata[k] = vertexai.types.MemoryMetadataValue(string_value=str(v))
                op = vclient.agent_engines.memories.create(
                    name=engine_name,
                    fact=fact,
                    scope=self._retrieval_scope(),
                    config={"metadata": typed_metadata} if typed_metadata else None,
                )
                name = op.response.name if op.done else ""
                self._record_success()
                return json.dumps({"result": "Fact stored.", "name": name})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        elif tool_name == "memory_revisions":
            memory_name = args.get("memory_name", "")
            if not memory_name:
                return tool_error("Missing required parameter: memory_name")
            try:
                vclient = self._get_vertex_client()
                revisions = []
                list_kwargs = {"name": memory_name}
                flt = args.get("filter")
                if flt:
                    list_kwargs["config"] = {"filter": flt}
                for rev in vclient.agent_engines.memories.revisions.list(**list_kwargs):
                    entry = {
                        "name": rev.name,
                        "fact": rev.fact,
                        "create_time": rev.create_time.isoformat() if hasattr(rev, "create_time") and rev.create_time else None,
                    }
                    if hasattr(rev, "labels") and rev.labels:
                        entry["labels"] = dict(rev.labels)
                    revisions.append(entry)
                self._record_success()
                if not revisions:
                    return json.dumps({"result": "No revisions found for this memory."})
                return json.dumps({"revisions": revisions, "count": len(revisions)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to list revisions: {e}")

        elif tool_name == "memory_revision_get":
            revision_name = args.get("revision_name", "")
            if not revision_name:
                return tool_error("Missing required parameter: revision_name")
            try:
                vclient = self._get_vertex_client()
                rev = vclient.agent_engines.memories.revisions.get(name=revision_name)
                result = {
                    "name": rev.name,
                    "fact": rev.fact,
                    "create_time": rev.create_time.isoformat() if hasattr(rev, "create_time") and rev.create_time else None,
                }
                if hasattr(rev, "expire_time") and rev.expire_time:
                    result["expire_time"] = rev.expire_time.isoformat()
                if hasattr(rev, "labels") and rev.labels:
                    result["labels"] = dict(rev.labels)
                self._record_success()
                return json.dumps({"revision": result})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to get revision: {e}")

        elif tool_name == "memory_rollback":
            memory_name = args.get("memory_name", "")
            target_revision_id = args.get("target_revision_id", "")
            if not memory_name or not target_revision_id:
                return tool_error("Missing required parameters: memory_name and target_revision_id")
            try:
                vclient = self._get_vertex_client()
                vclient.agent_engines.memories.rollback(
                    name=memory_name,
                    target_revision_id=target_revision_id,
                )
                self._record_success()
                return json.dumps({
                    "result": "Rollback successful.",
                    "memory_name": memory_name,
                    "target_revision_id": target_revision_id,
                })
            except Exception as e:
                self._record_failure()
                return tool_error(f"Rollback failed: {e}")

        elif tool_name == "memory_get":
            memory_name = args.get("memory_name", "")
            if not memory_name:
                return tool_error("Missing required parameter: memory_name")
            try:
                vclient = self._get_vertex_client()
                mem = vclient.agent_engines.memories.get(name=memory_name)
                result = {
                    "name": mem.name,
                    "fact": mem.fact,
                    "scope": dict(mem.scope) if mem.scope else {},
                }
                if hasattr(mem, "create_time") and mem.create_time:
                    result["create_time"] = mem.create_time.isoformat()
                if hasattr(mem, "update_time") and mem.update_time:
                    result["update_time"] = mem.update_time.isoformat()
                if hasattr(mem, "expire_time") and mem.expire_time:
                    result["expire_time"] = mem.expire_time.isoformat()
                if hasattr(mem, "labels") and mem.labels:
                    result["labels"] = dict(mem.labels)
                self._record_success()
                return json.dumps({"memory": result})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to get memory: {e}")

        elif tool_name == "memory_delete":
            memory_name = args.get("memory_name", "")
            if not memory_name:
                return tool_error("Missing required parameter: memory_name")
            try:
                vclient = self._get_vertex_client()
                vclient.agent_engines.memories.delete(name=memory_name)
                self._record_success()
                return json.dumps({"result": "Memory deleted.", "name": memory_name})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to delete memory: {e}")

        elif tool_name == "memory_ingest":
            events = args.get("events", [])
            if not events:
                return tool_error("Missing required parameter: events")
            try:
                vclient = self._get_vertex_client()
                engine_name = (
                    f"projects/{self._project_id}/locations/{self._location}"
                    f"/reasoningEngines/{self._engine_id}"
                )
                # Format events per Google ingest_events API spec.
                # direct_contents_source.events[] is the canonical shape.
                formatted_events = []
                for ev in events:
                    if isinstance(ev, dict):
                        entry = {
                            "content": {
                                "role": ev.get("role", "user"),
                                "parts": [{"text": str(ev.get("text", ""))}]
                            }
                        }
                        if ev.get("event_id"):
                            entry["event_id"] = str(ev["event_id"])
                        formatted_events.append(entry)
                    else:
                        formatted_events.append({
                            "content": {
                                "role": "user",
                                "parts": [{"text": str(ev)}]
                            }
                        })

                ingest_kwargs = {
                    "name": engine_name,
                    "scope": self._retrieval_scope(),
                    "direct_contents_source": {"events": formatted_events},
                }

                # Optional stream_id for grouping related events
                stream_id = args.get("stream_id")
                if stream_id:
                    ingest_kwargs["stream_id"] = str(stream_id)

                # Optional generation trigger config (default: 60s idle)
                gtc = args.get("generation_trigger_config")
                if gtc:
                    ingest_kwargs["generation_trigger_config"] = gtc
                else:
                    ingest_kwargs["generation_trigger_config"] = {
                        "generation_rule": {"idle_duration": "60s"}
                    }

                op = vclient.agent_engines.memories.ingest_events(**ingest_kwargs)
                self._record_success()
                return json.dumps({
                    "result": "Events ingested (background processing).",
                    "event_count": len(formatted_events),
                    "operation": op.name if hasattr(op, "name") else "",
                    "scope": self._retrieval_scope(),
                })
            except Exception as e:
                self._record_failure()
                return tool_error(f"Ingest failed: {e}")

        elif tool_name == "memory_purge":
            try:
                vclient = self._get_vertex_client()
                engine_name = (
                    f"projects/{self._project_id}/locations/{self._location}"
                    f"/reasoningEngines/{self._engine_id}"
                )
                # Build filter from args or default to current scope
                user_filter = args.get("filter", "")
                if not user_filter:
                    user_filter = (
                        f'scope.user_id="{self._user_id}" AND '
                        f'scope.app_name="{self._app_name}"'
                    )
                operation = vclient.agent_engines.memories.purge(
                    name=engine_name,
                    filter=user_filter,
                    force=True,
                    config={"wait_for_completion": True},
                )
                purge_count = 0
                if hasattr(operation, "response") and operation.response:
                    purge_count = getattr(operation.response, "purge_count", 0)
                self._record_success()
                return json.dumps({
                    "result": f"Purged {purge_count} memories.",
                    "filter": user_filter,
                })
            except Exception as e:
                self._record_failure()
                return tool_error(f"Purge failed: {e}")

        elif tool_name == "memory_synthesize":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                vclient = self._get_vertex_client()
                engine_name = f"projects/{self._project_id}/locations/{self._location}/reasoningEngines/{self._engine_id}"
                results = vclient.agent_engines.memories.retrieve(
                    name=engine_name,
                    scope=self._retrieval_scope(),
                    similarity_search_params={"search_query": query, "top_k": top_k},
                )
                facts = []
                for r in results:
                    facts.append(r.memory.fact)
                self._record_success()
                if not facts:
                    return json.dumps({"result": "No relevant memories found to synthesize."})
                # Compose narrative synthesis (Hindsight-style reflect)
                narrative = " ".join(f"{f}." if not f.endswith((".", "!", "?")) else f for f in facts)
                return json.dumps({
                    "result": narrative,
                    "sources": facts,
                    "count": len(facts),
                })
            except Exception as e:
                self._record_failure()
                return tool_error(f"Synthesis failed: {e}")

        return tool_error(f"Unknown tool: {tool_name}")


def register(ctx) -> None:
    """Register GCP Memory Bank as a memory provider plugin."""
    ctx.register_memory_provider(GcpMemoryBankProvider())
