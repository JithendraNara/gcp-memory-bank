"""Vertex AI client wrapper for the unified gcp-memory-bank plugin.

Combines:
    - Lazy SDK import (no failure at module load).
    - Configurable circuit breaker (Mem0 idiom).
    - Tenacity-based retry on transient errors.
    - Bounded LRO polling.
    - Dual-client routing: low-level proto for plain RetrieveMemories
      (lowest latency), high-level vertexai.Client for everything that
      needs filter/config/sources.
    - Sane error normalisation for tool returns.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


class CircuitBreakerOpen(RuntimeError):
    """Raised when the breaker is open and short-circuits a call."""


class _CircuitBreaker:
    def __init__(self, threshold: int = 5, cooldown_seconds: int = 120) -> None:
        self.threshold = max(1, int(threshold))
        self.cooldown = max(1, int(cooldown_seconds))
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._failures < self.threshold:
                return True
            if (time.monotonic() - self._opened_at) >= self.cooldown:
                self._failures = self.threshold - 1  # half-open
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures == self.threshold:
                self._opened_at = time.monotonic()
                logger.warning(
                    "gcp-memory-bank: circuit breaker OPEN for %ds after %d failures.",
                    self.cooldown, self.threshold,
                )

    @property
    def state(self) -> str:
        with self._lock:
            if self._failures < self.threshold:
                return "closed"
            if (time.monotonic() - self._opened_at) >= self.cooldown:
                return "half-open"
            return "open"


def _retry(fn: Callable, *args: Any, attempts: int = 3, **kwargs: Any) -> Any:
    """Tiny retry helper. We don't hard-depend on tenacity for the unit tests."""
    try:
        from tenacity import (  # type: ignore
            retry, stop_after_attempt, wait_exponential,
            retry_if_exception_type,
        )
    except ImportError:
        # Fallback: plain best-effort 3-try retry.
        last: Optional[BaseException] = None
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last = e
                if i + 1 < attempts:
                    time.sleep(min(2 ** i, 30))
        if last is not None:
            raise last
        return None

    # Real tenacity path with a meaningful filter.
    try:
        from google.api_core import exceptions as gx  # type: ignore
        retryable_exceptions = (
            gx.ResourceExhausted, gx.ServiceUnavailable, gx.DeadlineExceeded,
            ConnectionError,
        )
    except Exception:
        retryable_exceptions = (ConnectionError,)

    def _is_throttle(e: BaseException) -> bool:
        """Detect Gemini code-8 throttle errors that surface as RuntimeError.
        Observed in production at 2026-04-29 00:17 — gemini-3.1-pro-preview
        threw {"code": 8, "message": "... throttled. Please try again."}
        wrapped in RuntimeError. The standard ResourceExhausted catch misses
        these because the SDK rewraps them."""
        if isinstance(e, retryable_exceptions):
            return True
        msg = str(e)
        if "throttle" in msg.lower():
            return True
        if "'code': 8" in msg or "code: 8" in msg:
            return True
        if "RESOURCE_EXHAUSTED" in msg:
            return True
        return False

    from tenacity import retry_if_exception  # type: ignore

    decorator = retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_throttle),
        reraise=True,
    )
    return decorator(fn)(*args, **kwargs)


class MemoryBankClient:
    """All Memory Bank + Sessions traffic flows through this class."""

    def __init__(
        self,
        *,
        project: str,
        location: str,
        engine_id: str,
        breaker_threshold: int = 5,
        breaker_cooldown: int = 120,
        lro_poll_max_seconds: int = 60,
    ) -> None:
        self.project = project
        self.location = location
        self.engine_id = engine_id
        self._breaker = _CircuitBreaker(breaker_threshold, breaker_cooldown)
        self._vclient: Optional[Any] = None
        self._proto_client: Optional[Any] = None
        self._lock = threading.Lock()
        self._lro_poll_max = max(1, int(lro_poll_max_seconds))

    @property
    def engine_name(self) -> str:
        return (
            f"projects/{self.project}/locations/{self.location}"
            f"/reasoningEngines/{self.engine_id}"
        )

    @property
    def breaker_state(self) -> str:
        return self._breaker.state

    # ------------------------------------------------------------------
    # SDK init
    # ------------------------------------------------------------------
    def _ensure_vclient(self) -> Any:
        if self._vclient is not None:
            return self._vclient
        with self._lock:
            if self._vclient is not None:
                return self._vclient
            try:
                import vertexai  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "gcp-memory-bank: vertexai not installed. "
                    "Run: pip install 'google-cloud-aiplatform>=1.148.0'"
                ) from e
            self._vclient = vertexai.Client(project=self.project, location=self.location)
            return self._vclient

    def _ensure_proto_client(self) -> Any:
        """Low-level gRPC client for plain RetrieveMemories (lowest latency)."""
        if self._proto_client is not None:
            return self._proto_client
        with self._lock:
            if self._proto_client is not None:
                return self._proto_client
            try:
                from google.api_core.client_options import ClientOptions  # type: ignore
                from google.cloud.aiplatform_v1beta1 import MemoryBankServiceClient  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "gcp-memory-bank: google.cloud.aiplatform_v1beta1 not importable."
                ) from e
            self._proto_client = MemoryBankServiceClient(
                client_options=ClientOptions(
                    api_endpoint=f"{self.location}-aiplatform.googleapis.com",
                )
            )
            return self._proto_client

    # ------------------------------------------------------------------
    # Call wrapper
    # ------------------------------------------------------------------
    def _call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if not self._breaker.allow():
            raise CircuitBreakerOpen("gcp-memory-bank breaker open")
        try:
            result = _retry(fn, *args, **kwargs)
        except Exception:
            self._breaker.record_failure()
            raise
        self._breaker.record_success()
        return result

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------
    def get_engine(self) -> Any:
        c = self._ensure_vclient()
        return self._call(c.agent_engines.get, name=self.engine_name)

    def update_engine_config(self, memory_bank_config: Dict[str, Any]) -> Any:
        c = self._ensure_vclient()
        return self._call(
            c.agent_engines.update,
            name=self.engine_name,
            config={"context_spec": {"memory_bank_config": memory_bank_config}},
        )

    def create_engine(self, memory_bank_config: Optional[Dict[str, Any]] = None,
                      display_name: str = "hermes-memory-bank") -> Any:
        c = self._ensure_vclient()
        body: Dict[str, Any] = {"display_name": display_name}
        if memory_bank_config:
            body["context_spec"] = {"memory_bank_config": memory_bank_config}
        return self._call(c.agent_engines.create, config=body)

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------
    def create_memory(
        self,
        *,
        scope: Dict[str, str],
        fact: str,
        metadata: Optional[Dict[str, Any]] = None,
        revision_labels: Optional[Dict[str, str]] = None,  # accepted, ignored
    ) -> Any:
        """Create a memory.

        NOTE: ``revision_labels`` is accepted for API parity with the
        ``generate_memories`` path but is silently dropped — the Vertex AI
        SDK's ``AgentEngineMemoryConfig`` does NOT permit ``revision_labels``
        on Create (Pydantic ``extra='forbid'`` rejects it). Labels are only
        valid on ``GenerateMemories.config``.
        """
        c = self._ensure_vclient()
        body: Dict[str, Any] = {
            "name": self.engine_name,
            "scope": scope,
            "fact": fact,
        }
        if metadata:
            body["config"] = {"metadata": _box_metadata(metadata)}
        return self._call(c.agent_engines.memories.create, **body)

    def retrieve(
        self,
        *,
        scope: Dict[str, str],
        query: Optional[str] = None,
        top_k: int = 8,
        filter_expr: Optional[str] = None,
        no_retry: bool = False,
    ) -> List[Dict[str, Any]]:
        # Fast path for plain similarity search → low-level proto client.
        if query and not filter_expr:
            try:
                pc = self._ensure_proto_client()
                from google.cloud.aiplatform_v1beta1.types.memory_bank_service import (  # type: ignore
                    RetrieveMemoriesRequest,
                )
                req = RetrieveMemoriesRequest(
                    parent=self.engine_name,
                    scope=scope,
                    similarity_search_params={"search_query": query, "top_k": int(top_k)},
                )
                if no_retry:
                    # Hot-path call (e.g. prefetch). Skip retries entirely so a
                    # transient error returns fast instead of blocking the
                    # turn for up to ~14s of exponential backoff.
                    if not self._breaker.allow():
                        return []
                    try:
                        resp = pc.retrieve_memories(request=req, timeout=4.0)
                        self._breaker.record_success()
                    except Exception:
                        self._breaker.record_failure()
                        raise
                else:
                    resp = self._call(pc.retrieve_memories, request=req)
                return [
                    {
                        "name": getattr(r.memory, "name", ""),
                        "fact": getattr(r.memory, "fact", ""),
                        "scope": dict(getattr(r.memory, "scope", {}) or {}),
                        "distance": getattr(r, "distance", None),
                    }
                    for r in getattr(resp, "retrieved_memories", []) or []
                ]
            except Exception as e:
                logger.debug("gcp-memory-bank: proto retrieve failed (%s); falling back.", e)

        # Filtered / fallback path → high-level client.
        c = self._ensure_vclient()
        params: Dict[str, Any] = {"name": self.engine_name, "scope": scope}
        ssp: Dict[str, Any] = {}
        if query:
            ssp["search_query"] = query
            ssp["top_k"] = int(top_k)
        if ssp:
            params["similarity_search_params"] = ssp
        config: Dict[str, Any] = {}
        if filter_expr:
            config["filter"] = filter_expr
        if config:
            params["config"] = config
        try:
            resp = self._call(c.agent_engines.memories.retrieve, **params)
        except Exception as e:
            logger.debug("gcp-memory-bank: retrieve failed: %s", e)
            return []
        return _normalize_memories(resp)

    def list_memories(
        self,
        *,
        filter_expr: Optional[str] = None,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        c = self._ensure_vclient()
        config: Dict[str, Any] = {"page_size": page_size}
        if filter_expr:
            config["filter"] = filter_expr
        try:
            pager = self._call(c.agent_engines.memories.list, name=self.engine_name, config=config)
        except Exception as e:
            logger.debug("gcp-memory-bank: list failed: %s", e)
            return []
        out: List[Dict[str, Any]] = []
        try:
            for memory in pager:
                out.extend(_normalize_memories(memory))
        except TypeError:
            out.extend(_normalize_memories(pager))
        return out

    def get_memory(self, name: str) -> Optional[Dict[str, Any]]:
        c = self._ensure_vclient()
        try:
            resp = self._call(c.agent_engines.memories.get, name=name)
            return _to_dict(resp)
        except Exception as e:
            logger.debug("gcp-memory-bank: get failed: %s", e)
            return None

    def delete_memory(self, name: str) -> Any:
        c = self._ensure_vclient()
        return self._call(c.agent_engines.memories.delete, name=name)

    def purge(self, *, filter_expr: str, force: bool = False,
              wait: bool = True) -> Any:
        c = self._ensure_vclient()
        return self._call(
            c.agent_engines.memories.purge,
            name=self.engine_name,
            filter=filter_expr,
            config={"force": bool(force), "wait_for_completion": bool(wait)},
        )

    # ------------------------------------------------------------------
    # Generation — only vertex_session_source is reliable per
    # exploration/GCP_SESSIONS_INTEGRATION.md. Other paths kept as
    # fallbacks but flagged as Preview-broken.
    # ------------------------------------------------------------------
    def generate_from_session(
        self,
        *,
        scope: Dict[str, str],
        session_name: str,
        revision_labels: Optional[Dict[str, str]] = None,
        wait: bool = True,
    ) -> Any:
        c = self._ensure_vclient()
        body: Dict[str, Any] = {
            "name": self.engine_name,
            "scope": scope,
            "vertex_session_source": {"session": session_name},
        }
        if revision_labels:
            body["config"] = {"revision_labels": revision_labels}
        op = self._call(c.agent_engines.memories.generate, **body)
        return self._optionally_wait(op, wait)

    def create_memories_from_events(
        self,
        *,
        scope: Dict[str, str],
        events: List[Dict[str, Any]],
        revision_labels: Optional[Dict[str, str]] = None,
        min_text_chars: int = 10,
    ) -> int:
        """Per-event ``CreateMemory`` fallback. Used when GCP Sessions is
        disabled or the session-source generation fails. This is the path
        that v1's TEST_RESULTS.md proved to actually work."""
        created = 0
        for ev in events:
            text = _extract_text(ev).strip()
            if len(text) < min_text_chars:
                continue
            try:
                self.create_memory(
                    scope=scope,
                    fact=text[:1024],
                    revision_labels=revision_labels,
                )
                created += 1
            except Exception as e:
                logger.debug("gcp-memory-bank: per-event create failed: %s", e)
                continue
        return created

    # ------------------------------------------------------------------
    # Revisions
    # ------------------------------------------------------------------
    def list_revisions(self, memory_name: str, *,
                       label_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        c = self._ensure_vclient()
        config: Dict[str, Any] = {}
        if label_filter:
            config["filter"] = label_filter
        try:
            pager = self._call(
                c.agent_engines.memories.revisions.list,
                name=memory_name,
                config=config or None,
            )
        except Exception as e:
            logger.debug("gcp-memory-bank: list_revisions failed: %s", e)
            return []
        out: List[Dict[str, Any]] = []
        try:
            for rev in pager:
                out.extend(_normalize_revisions(rev))
        except TypeError:
            out.extend(_normalize_revisions(pager))
        return out

    def get_revision(self, name: str) -> Optional[Dict[str, Any]]:
        c = self._ensure_vclient()
        try:
            return _to_dict(self._call(c.agent_engines.memories.revisions.get, name=name))
        except Exception as e:
            logger.debug("gcp-memory-bank: get_revision failed: %s", e)
            return None

    def rollback(self, memory_name: str, target_revision_id: str) -> Any:
        c = self._ensure_vclient()
        return self._call(
            c.agent_engines.memories.rollback,
            name=memory_name,
            target_revision_id=target_revision_id,
        )

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def create_session(self, *, user_id: str, display_name: str = "",
                       ttl_seconds: int = 86400) -> str:
        c = self._ensure_vclient()
        config: Dict[str, Any] = {}
        if display_name:
            config["display_name"] = display_name
        if ttl_seconds:
            config["ttl"] = f"{max(86400, int(ttl_seconds))}s"
        op = self._call(
            c.agent_engines.sessions.create,
            name=self.engine_name,
            user_id=user_id,
            config=config or None,
        )
        # SDK returns AgentEngineSessionOperation — name is on .response.
        try:
            return getattr(op.response, "name", "") or ""
        except Exception:
            return ""

    def append_event(self, *, session_name: str, author: str,
                     invocation_id: str, timestamp: Any,
                     content: Dict[str, Any]) -> None:
        c = self._ensure_vclient()
        self._call(
            c.agent_engines.sessions.events.append,
            name=session_name,
            author=author,
            invocation_id=invocation_id,
            timestamp=timestamp,
            config={"content": content},
        )

    def list_sessions(self, *, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        c = self._ensure_vclient()
        params: Dict[str, Any] = {"name": self.engine_name}
        if user_id:
            params["user_id"] = user_id
        try:
            pager = self._call(c.agent_engines.sessions.list, **params)
        except Exception as e:
            logger.debug("gcp-memory-bank: list_sessions failed: %s", e)
            return []
        return [_to_dict(s) for s in (pager if hasattr(pager, "__iter__") else [pager])]

    def delete_session(self, session_name: str) -> None:
        c = self._ensure_vclient()
        self._call(c.agent_engines.sessions.delete, name=session_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _optionally_wait(self, op: Any, wait: bool) -> Any:
        """Bounded wait for an LRO. We don't trust ``wait_for_completion=False``
        because v1 found it broken; instead we either block briefly or return
        immediately and let the caller choose."""
        if not wait:
            return op
        deadline = time.monotonic() + self._lro_poll_max
        while time.monotonic() < deadline:
            done = getattr(op, "done", None)
            if callable(done):
                try:
                    if done():
                        return op
                except Exception:
                    pass
            elif done is True:
                return op
            time.sleep(1.0)
            # Try refreshing the operation if the SDK exposes it.
            refresh = getattr(op, "result", None)
            if callable(refresh):
                try:
                    refresh(timeout=5)
                    return op
                except Exception:
                    continue
        logger.info("gcp-memory-bank: LRO did not complete within %ds; returning handle.",
                    self._lro_poll_max)
        return op


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    # Pydantic v2 (current Vertex AI SDK) — model_dump returns the canonical dict.
    for attr in ("model_dump", "to_dict", "dict"):
        if hasattr(obj, attr):
            try:
                fn = getattr(obj, attr)
                v = fn() if callable(fn) else fn
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    # Fallback: introspect public scalar attributes only (avoid descriptor calls).
    result: Dict[str, Any] = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if callable(v):
            continue
        result[k] = v
    return result


def _normalize_memories(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        items: Iterable[Any] = obj
    elif isinstance(obj, dict) and "memories" in obj:
        items = obj.get("memories") or []
    elif hasattr(obj, "memories"):
        items = getattr(obj, "memories") or []
    elif hasattr(obj, "retrieved_memories"):
        items = getattr(obj, "retrieved_memories") or []
    else:
        items = [obj]
    out: List[Dict[str, Any]] = []
    for item in items:
        d = _to_dict(item)
        if "memory" in d and isinstance(d["memory"], dict):
            inner = dict(d["memory"])
            if "distance" in d:
                inner["distance"] = d["distance"]
            out.append(inner)
        else:
            out.append(d)
    return out


def _normalize_revisions(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict) and "memory_revisions" in obj:
        return [_to_dict(x) for x in obj.get("memory_revisions") or []]
    if hasattr(obj, "memory_revisions"):
        return [_to_dict(x) for x in getattr(obj, "memory_revisions") or []]
    return [_to_dict(obj)]


def _box_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Box Python types into Memory Bank's MemoryMetadataValue shape."""
    from datetime import datetime
    out: Dict[str, Any] = {}
    for k, v in (metadata or {}).items():
        if isinstance(v, bool):
            out[k] = {"bool_value": v}
        elif isinstance(v, (int, float)):
            out[k] = {"double_value": float(v)}
        elif isinstance(v, datetime):
            out[k] = {"timestamp_value": v.isoformat()}
        elif isinstance(v, str):
            out[k] = {"string_value": v}
        else:
            out[k] = {"string_value": str(v)}
    return out


def _extract_text(event: Dict[str, Any]) -> str:
    if not event:
        return ""
    content = event.get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        return ""
    text = parts[0].get("text") if isinstance(parts[0], dict) else None
    return str(text or "")
