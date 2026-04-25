"""GCP Memory Bank plugin for Hermes.

Uses Google Cloud Memory Bank via the official Agent Platform SDK.
Follows the patterns documented at:
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/setup
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/generate-memories
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/fetch-memories

Config via environment variables:
  GCP_PROJECT_ID      — GCP project (default: festive-antenna-463514-m8)
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
        "ranked by relevance (Euclidean distance). Use for targeted recall mid-conversation."
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

    def _build_engine_config(self) -> Dict[str, Any]:
        """Return Memory Bank config per official docs."""
        return {
            "generation_config": {
                "model": f"projects/{self._project_id}/locations/{self._location}/publishers/google/models/gemini-3.1-pro-preview",
            },
            "similarity_search_config": {
                "embedding_model": f"projects/{self._project_id}/locations/{self._location}/publishers/google/models/text-embedding-005",
            },
            "ttl_config": {
                "default_ttl": f"{365 * 24 * 60 * 60}s"
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
            self._config = _load_config()
            self._engine_id = self._config.get("engine_id", "")

    def system_prompt_block(self) -> str:
        return (
            "# GCP Memory Bank\n"
            f"Active. Engine: {self._engine_id}. User: {self._user_id}.\n"
            "Use memory_search to find facts, memory_store to save preferences, "
            "memory_profile for a full overview, memory_purge to delete all user data."
        )

    def _scope(self) -> Dict[str, str]:
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
        if self._is_breaker_open():
            return
        def _run():
            try:
                client = self._get_client()
                from google.cloud.aiplatform_v1beta1.types.memory_bank_service import RetrieveMemoriesRequest
                request = RetrieveMemoriesRequest(
                    parent=self._parent,
                    scope=self._scope(),
                    similarity_search_params={"search_query": query, "top_k": 8},
                )
                response = client.retrieve_memories(request=request)
                memories = []
                for r in response.retrieved_memories:
                    memories.append(r.memory.fact)
                if memories:
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {m}" for m in memories)
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
        """Buffer conversation events. Memory generation happens at session end."""
        with self._events_lock:
            self._events.append({
                "content": {
                    "role": "model",
                    "parts": [{"text": assistant_content}]
                }
            })
            self._events.append({
                "content": {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            })
            self._turn_count += 1

    # -----------------------------------------------------------------------
    # Session end: generate memories via official API (per docs)
    # -----------------------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Generate memories from the full conversation using Memory Bank's native API.

        Per docs: https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/generate-memories
        Uses direct_contents_source with wait_for_completion=False for background processing.
        """
        if self._is_breaker_open():
            return
        with self._events_lock:
            events = list(self._events)
            self._events.clear()
            turn_count = self._turn_count
            self._turn_count = 0
        if not events:
            return

        def _generate():
            try:
                import vertexai
                client = vertexai.Client(project=self._project_id, location=self._location)
                engine_name = f"projects/{self._project_id}/locations/{self._location}/reasoningEngines/{self._engine_id}"
                client.agent_engines.memories.generate(
                    name=engine_name,
                    direct_contents_source={"events": events},
                    scope=self._scope(),
                    config={"wait_for_completion": False},
                )
                self._record_success()
                logger.info("Memory generation triggered in background (%d events, %d turns).", len(events), turn_count)
            except Exception as e:
                self._record_failure()
                logger.warning("Memory generation failed: %s", e)

        threading.Thread(target=_generate, daemon=True, name="gcp-mb-generate").start()

    # -----------------------------------------------------------------------
    # Tools
    # -----------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, STORE_SCHEMA, PURGE_SCHEMA]

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
            try:
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
                return json.dumps({"result": f"Deleted {deleted} memories."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Purge failed: {e}")

        return tool_error(f"Unknown tool: {tool_name}")
