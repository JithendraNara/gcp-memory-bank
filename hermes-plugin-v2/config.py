"""Configuration for the unified gcp-memory-bank Hermes provider.

Backwards compatible with v1 config keys (project_id, location, engine_id,
user_id, app_name, ...). Adds v2 knobs (recall_mode, recall_budget, fence,
trivial_skip, scope_template, ...) with safe defaults.

Layered resolution (highest priority first):
    1. Environment variables
    2. ``$HERMES_HOME/gcp-memory-bank.json``
    3. Built-in defaults
"""

from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "gcp-memory-bank.json"

# Memory Bank caps documented at the upstream docs.
SCOPE_KEY_LIMIT = 5
DIRECT_MEMORIES_LIMIT = 5
EVENT_BUFFER_LIMIT = 200            # sliding window per session
SESSION_TTL_MIN_SECONDS = 86400     # 24h is the documented minimum

DEFAULT_CONFIG: Dict[str, Any] = {
    # ---- Identity / scope ------------------------------------------------
    "project_id": "",
    "location": "us-central1",
    "engine_id": "",
    "user_id": "",            # If empty, resolved at init from kwargs.user_id (NEVER raw chat id).
    "app_name": "hermes",
    # Scope schema. v1 used hard-coded {app_name, user_id} — preserved as default.
    "scope_keys": ["app_name", "user_id"],
    "scope_template": {
        "app_name": "{app}",
        "user_id": "{user}",
    },
    "scope_includes_session": False,
    # ---- Models ----------------------------------------------------------
    "generation_model": "gemini-3.1-pro-preview",
    "embedding_model": "gemini-embedding-001",
    "synthesis_model": "gemini-2.5-flash",       # Used by memory_synthesize tool.
    # ---- TTL -------------------------------------------------------------
    "create_ttl_days": 365,
    "generate_created_ttl_days": 365,
    "revision_ttl_days": 365,
    # ---- Recall (NEW v2) -------------------------------------------------
    "recall_mode": "hybrid",          # context | tools | hybrid
    "recall_budget": "mid",           # low=3 / mid=8 / high=15
    "recall_detail": "L1",            # L0 fact / L1 +topic+age / L2 +revisions
    "recall_max_input_chars": 4000,
    "context_token_budget": 1500,
    "auto_prefetch": True,
    "prefetch_mode": "facts",         # facts | narrative
    "trivial_skip": True,
    # ---- Ingestion -------------------------------------------------------
    "use_gcp_sessions": True,
    "gcp_session_ttl_seconds": 86400,
    "generate_every_n_turns": 3,      # CHANGED from v1 default 0 — now mid-session generation by default.
    "skip_empty_session_end": True,   # NEW v2 — don't burn round-trips on 0-event sessions.
    "session_reuse": True,            # NEW v2 — reuse one GCP session across the Hermes process.
    "cross_process_session_reuse": True,  # NEW v2 — reattach to last session across Hermes restarts / agent-cache evictions.
    # ---- Reliability -----------------------------------------------------
    "circuit_breaker": {"threshold": 5, "cooldown_seconds": 120},
    "lro_poll_max_seconds": 60,       # cap how long we'll wait on a foreground LRO.
    "background_observability": True, # log completion+latency for fire-and-forget threads.
    # ---- Topics + customization -----------------------------------------
    "custom_topics_enabled": True,
    "custom_topics": None,            # None means use topics.DEFAULT_CUSTOM_TOPICS
    "allowed_topics": None,
    "few_shot_examples_enabled": True,
    "consolidation_revisions_per_candidate": 5,
    "enable_third_person_memories": False,
    "disable_memory_revisions": False,
    # ---- Write mirror behaviour -----------------------------------------
    "mirror_memory_md_writes": True,        # on_memory_write → CreateMemory
    "mirror_drop_action_prefix": True,      # NEW v2: drop "[ADD user] " prefix that polluted v1 facts.
    "default_revision_labels": {},
    # ---- Hard write gating ----------------------------------------------
    "primary_only": True,                   # Skip writes outside agent_context=primary.
    "skip_contexts": ["cron", "flush", "subagent"],  # NEW v2 explicit list.
}


_TEMPLATE_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_SEGMENT_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._\-]+")
# Reject obvious raw identifiers that should never appear in scope.
_NUMERIC_ONLY_RE = re.compile(r"^\d{6,}$")  # Telegram chat ids are 9-10 digits.


def _sanitize_segment(value: Any) -> str:
    if value is None:
        return "default"
    cleaned = _SEGMENT_SANITIZE_RE.sub("-", str(value))
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-._") or "default"
    return cleaned[:128]


def _render_template(template: str, ctx: Dict[str, str]) -> str:
    def repl(match: "re.Match[str]") -> str:
        key = match.group(1)
        return _sanitize_segment(ctx.get(key, key))
    return _TEMPLATE_RE.sub(repl, template)


@dataclass
class GmbConfig:
    """Resolved configuration for a provider instance."""

    raw: Dict[str, Any] = field(default_factory=lambda: deepcopy(DEFAULT_CONFIG))

    # --- Convenience accessors --------------------------------------------
    @property
    def project(self) -> str: return str(self.raw.get("project_id") or "")

    @property
    def location(self) -> str: return str(self.raw.get("location") or "us-central1")

    @property
    def engine_id(self) -> str: return str(self.raw.get("engine_id") or "")

    @property
    def app_name(self) -> str: return str(self.raw.get("app_name") or "hermes")

    @property
    def scope_keys(self) -> List[str]:
        keys = list(self.raw.get("scope_keys") or [])
        if not keys:
            keys = ["app_name", "user_id"]
        if len(keys) > SCOPE_KEY_LIMIT:
            logger.warning("gcp-memory-bank: scope_keys=%d truncated to %d", len(keys), SCOPE_KEY_LIMIT)
            keys = keys[:SCOPE_KEY_LIMIT]
        return keys

    @property
    def recall_mode(self) -> str:
        m = str(self.raw.get("recall_mode", "hybrid")).lower()
        return m if m in {"context", "tools", "hybrid"} else "hybrid"

    @property
    def recall_top_k(self) -> int:
        return {"low": 3, "mid": 8, "high": 15}.get(
            str(self.raw.get("recall_budget", "mid")).lower(), 8
        )

    @property
    def context_token_budget(self) -> int:
        return int(self.raw.get("context_token_budget", 1500))

    @property
    def primary_only(self) -> bool:
        return bool(self.raw.get("primary_only", True))

    @property
    def skip_contexts(self) -> set:
        return set(self.raw.get("skip_contexts") or [])

    def engine_name(self) -> str:
        if not self.engine_id or not self.project:
            return ""
        return (
            f"projects/{self.project}/locations/{self.location}"
            f"/reasoningEngines/{self.engine_id}"
        )

    # --- Scope resolution -------------------------------------------------
    def resolve_user_id(self, kwargs_user: str = "") -> Tuple[str, Optional[str]]:
        """Return (user_id, drift_warning_msg).

        Hard guardrails learned from the runtime audit:
        - NEVER accept a numeric-only id (Telegram chat ids leaked into v1).
        - NEVER use the placeholder ``hermes-user`` if a real id is available.
        - Prefer config.user_id > kwargs.user_id > "hermes-user".
        """
        configured = str(self.raw.get("user_id") or "").strip()
        from_kwargs = str(kwargs_user or "").strip()
        warn: Optional[str] = None
        candidate = configured or from_kwargs or "hermes-user"
        if _NUMERIC_ONLY_RE.match(candidate):
            warn = (
                f"Refused to use raw numeric id {candidate!r} as user_id "
                "(looks like a Telegram chat id). Falling back to 'hermes-user'. "
                "Set user_id in gcp-memory-bank.json to override."
            )
            candidate = configured if configured and not _NUMERIC_ONLY_RE.match(configured) else "hermes-user"
        return candidate, warn

    def resolve_scope(self, *, user_id: str, agent_identity: str = "",
                      session_id: str = "", platform: str = "",
                      workspace: str = "") -> Dict[str, str]:
        ctx = {
            "user": user_id or "hermes-user",
            "app": self.app_name or "hermes",
            "profile": agent_identity or "hermes",
            "session": session_id or "default",
            "platform": platform or "cli",
            "workspace": workspace or "default",
        }
        template = dict(self.raw.get("scope_template") or {})
        scope: Dict[str, str] = {}
        for key in self.scope_keys:
            tmpl = template.get(key, "{" + key + "}")
            scope[key] = _render_template(tmpl, ctx)
        if self.raw.get("scope_includes_session") and "session_id" not in scope:
            scope["session_id"] = _sanitize_segment(session_id or "default")
        if "*" in scope.values():
            raise ValueError("gcp-memory-bank: '*' is not allowed in Memory Bank scope values.")
        if len(scope) > SCOPE_KEY_LIMIT:
            raise ValueError(f"gcp-memory-bank: scope has {len(scope)} keys; max {SCOPE_KEY_LIMIT}.")
        return scope


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------
def _config_path(hermes_home: str) -> Path:
    return Path(hermes_home) / CONFIG_FILENAME


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    for k, v in overlay.items():
        if v in ("", None):
            continue
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


_LEGACY_KEY_MAP = {
    # v1 → v2 (we already accept v1 names; this is for forward migration helpers).
}


def load_config(hermes_home: str) -> GmbConfig:
    raw = deepcopy(DEFAULT_CONFIG)
    path = _config_path(hermes_home)
    if path.exists():
        try:
            on_disk = json.loads(path.read_text())
            if isinstance(on_disk, dict):
                _deep_merge(raw, on_disk)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("gcp-memory-bank: bad config at %s: %s", path, e)

    # Env overrides.
    for env_key, cfg_key in (
        ("GCP_PROJECT_ID", "project_id"),
        ("GOOGLE_CLOUD_PROJECT", "project_id"),
        ("GCP_LOCATION", "location"),
        ("GOOGLE_CLOUD_LOCATION", "location"),
        ("GCP_MEMORY_ENGINE", "engine_id"),
        ("GOOGLE_CLOUD_AGENT_ENGINE_ID", "engine_id"),
    ):
        v = os.environ.get(env_key)
        if v:
            raw[cfg_key] = v

    # Allow short engine id + project + location.
    eid = str(raw.get("engine_id") or "")
    if eid and eid.startswith("projects/"):
        # Already a full resource name — extract id back out.
        try:
            raw["engine_id"] = eid.rsplit("/", 1)[-1]
        except Exception:
            pass
    return GmbConfig(raw=raw)


def save_config_file(values: Dict[str, Any], hermes_home: str) -> None:
    path = _config_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, json.JSONDecodeError):
            existing = {}
    cleaned = {k: v for k, v in (values or {}).items() if v not in ("", None)}
    _deep_merge(existing, cleaned)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))
