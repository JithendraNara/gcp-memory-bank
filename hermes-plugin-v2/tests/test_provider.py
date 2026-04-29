"""Unit tests for gcp-memory-bank v2.

Runs without google-cloud-aiplatform installed by patching MemoryBankClient
with an in-memory fake. Covers all hooks, all 11 tools, and the v2 audit
fixes (drift detection, scope migration safety, fence sanitize, trivial
skip, primary_only gating, mid-session generation, real synthesize).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest


# Bootstrap: copy the plugin to a tempdir under an underscore name so it's
# importable as a package.
import os as _os
PLUGIN_DIR = Path(_os.environ.get(
    "GMB_PLUGIN_DIR",
    "/Users/jithendranara/projects/gcp-memory-bank/hermes-plugin-v2",
)).resolve()

_pkg_root = Path(tempfile.mkdtemp(prefix="gmb-v2-test-"))
_pkg_dir = _pkg_root / "gcp_memory_bank_v2"
shutil.copytree(PLUGIN_DIR, _pkg_dir, ignore=shutil.ignore_patterns("__pycache__", "tests", ".pytest_cache"))
sys.path.insert(0, str(_pkg_root))

import gcp_memory_bank_v2 as gmb  # noqa: E402


# ---------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------
class FakeClient:
    def __init__(self, **_: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.memories: List[Dict[str, Any]] = []
        self.revisions: Dict[str, List[Dict[str, Any]]] = {}
        self.sessions: List[str] = []
        self.appended_events: List[Dict[str, Any]] = []
        self.engine_updated: Optional[Dict[str, Any]] = None
        self._next_id = 0
        self.engine_name = "projects/p/locations/us/reasoningEngines/test"
        self.breaker_state = "closed"

    def _record(self, op: str, **kw: Any) -> None:
        self.calls.append({"op": op, **kw})

    def get_engine(self) -> Dict[str, Any]:
        self._record("get_engine")
        return {"name": self.engine_name}

    def update_engine_config(self, body: Dict[str, Any]) -> Dict[str, Any]:
        self._record("update_engine_config", body=body)
        self.engine_updated = body
        return {"name": "op1"}

    def create_engine(self, memory_bank_config=None, display_name="x"):
        self._record("create_engine")
        return {"name": "projects/p/locations/us/reasoningEngines/new"}

    def create_memory(self, *, scope, fact, metadata=None, revision_labels=None):
        self._next_id += 1
        name = f"{self.engine_name}/memories/{self._next_id}"
        mem = {"name": name, "scope": dict(scope), "fact": fact}
        self.memories.append(mem)
        self.revisions[name] = [{"name": f"{name}/revisions/r1", "fact": fact}]
        self._record("create_memory", scope=scope, fact=fact, labels=revision_labels)
        return mem

    def retrieve(self, *, scope, query=None, top_k=8, filter_expr=None):
        self._record("retrieve", scope=scope, query=query, filter=filter_expr)
        return [m for m in self.memories if m["scope"] == scope][: top_k or 8]

    def list_memories(self, *, filter_expr=None, page_size=100):
        self._record("list_memories", filter=filter_expr)
        return list(self.memories)

    def get_memory(self, name: str):
        self._record("get_memory", name=name)
        for m in self.memories:
            if m["name"] == name:
                return m
        return None

    def delete_memory(self, name: str):
        self.memories = [m for m in self.memories if m["name"] != name]
        self._record("delete_memory", name=name)

    def purge(self, *, filter_expr, force=False, wait=False):
        self._record("purge", filter=filter_expr, force=force)
        if force:
            n = len(self.memories)
            self.memories.clear()
            return {"purge_count": n}
        return {"purge_count": 0}

    def generate_from_session(self, *, scope, session_name, revision_labels=None, wait=True):
        self._record("generate_from_session", scope=scope, session=session_name)

        class _Op:
            done = True

            class response:
                generated_memories = [type("G", (), {"action": "CREATED"})()]
        return _Op()

    def create_memories_from_events(self, *, scope, events, revision_labels=None, min_text_chars=10):
        n = 0
        for ev in events:
            text = (((ev.get("content") or {}).get("parts") or [{}])[0].get("text") or "")
            if len(text.strip()) >= min_text_chars:
                self.create_memory(scope=scope, fact=text)
                n += 1
        self._record("create_memories_from_events", events=len(events), created=n)
        return n

    # Revisions
    def list_revisions(self, name, *, label_filter=None):
        self._record("list_revisions", name=name)
        return list(self.revisions.get(name, []))

    def get_revision(self, name):
        self._record("get_revision", name=name)
        for revs in self.revisions.values():
            for r in revs:
                if r["name"] == name:
                    return r
        return None

    def rollback(self, memory, target):
        self._record("rollback", memory=memory, target=target)

    # Sessions
    def create_session(self, *, user_id, display_name="", ttl_seconds=86400):
        name = f"{self.engine_name}/sessions/sess-{len(self.sessions)+1}"
        self.sessions.append(name)
        self._record("create_session", user_id=user_id)
        return name

    def append_event(self, *, session_name, author, invocation_id, timestamp, content):
        self.appended_events.append({"session": session_name, "author": author})
        self._record("append_event", session=session_name, author=author)

    def list_sessions(self, *, user_id=None):
        self._record("list_sessions", user_id=user_id)
        return [{"name": s} for s in self.sessions]

    def delete_session(self, name):
        self.sessions = [s for s in self.sessions if s != name]
        self._record("delete_session", name=name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "festive-antenna-463514-m8")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "4938048007586185216")
    return tmp_path


@pytest.fixture
def provider(hermes_home: Path):
    fake = FakeClient()
    with patch.object(gmb, "MemoryBankClient", return_value=fake):
        p = gmb.GcpMemoryBankProvider()
        p.initialize(
            session_id="20260428_session",
            hermes_home=str(hermes_home),
            agent_identity="hermes",
            user_id="jithendra",
            platform="cli",
            agent_context="primary",
        )
        yield p, fake
        p.shutdown()


# ---------------------------------------------------------------------------
# Identity & availability
# ---------------------------------------------------------------------------
class TestProviderIdentity:
    def test_name(self):
        assert gmb.GcpMemoryBankProvider().name == "gcp-memory-bank"

    def test_no_sdk_means_unavailable(self, monkeypatch):
        monkeypatch.setattr(gmb, "_SDK_PRESENT", False)
        assert gmb.GcpMemoryBankProvider().is_available() is False


# ---------------------------------------------------------------------------
# v2 audit fixes
# ---------------------------------------------------------------------------
class TestUserIdGuardrails:
    def test_telegram_chat_id_rejected(self, hermes_home):
        from gcp_memory_bank_v2.config import GmbConfig
        cfg = GmbConfig()
        uid, warn = cfg.resolve_user_id("8405386815")  # raw Telegram chat id
        assert uid == "hermes-user"
        assert warn and "Telegram chat id" in warn

    def test_real_username_passes(self, hermes_home):
        from gcp_memory_bank_v2.config import GmbConfig
        cfg = GmbConfig()
        uid, warn = cfg.resolve_user_id("jithendra")
        assert uid == "jithendra"
        assert warn is None

    def test_config_override_wins(self, hermes_home):
        from gcp_memory_bank_v2.config import GmbConfig
        cfg = GmbConfig()
        cfg.raw["user_id"] = "alice"
        uid, _ = cfg.resolve_user_id("8405386815")
        assert uid == "alice"


class TestScopeDriftDetection:
    def test_drift_warning(self, caplog):
        from gcp_memory_bank_v2.observability import ScopeDriftDetector
        d = ScopeDriftDetector()
        d.record(user_id="alice", app_name="hermes", engine_id="e1")
        with caplog.at_level("WARNING"):
            d.record(user_id="bob", app_name="hermes", engine_id="e1")
        assert any("DRIFT" in r.message for r in caplog.records)


class TestScopeSafetyAndFence:
    def test_default_scope(self, provider):
        p, _ = provider
        assert p._scope == {"app_name": "hermes", "user_id": "jithendra"}

    def test_star_rejected(self):
        from gcp_memory_bank_v2.config import GmbConfig
        cfg = GmbConfig()
        cfg.raw["scope_template"] = {"user_id": "*"}
        cfg.raw["scope_keys"] = ["user_id"]
        with pytest.raises(ValueError):
            cfg.resolve_scope(user_id="*")

    def test_fence_strip(self):
        from gcp_memory_bank_v2.retrieval import strip_fence
        text = "<gcp-mb-context>\n[note]\n- fact\n</gcp-mb-context>\nReal input."
        assert strip_fence(text) == "Real input."


class TestTrivialSkip:
    def test_obvious_trivials_blocked(self, provider):
        p, fake = provider
        p.queue_prefetch("ok")
        p.queue_prefetch("thanks")
        p.queue_prefetch("/help")
        time.sleep(0.1)
        assert not any(c["op"] == "retrieve" for c in fake.calls)


class TestAgentContextGating:
    def test_subagent_context_blocks_writes(self, hermes_home):
        fake = FakeClient()
        with patch.object(gmb, "MemoryBankClient", return_value=fake):
            p = gmb.GcpMemoryBankProvider()
            p.initialize(session_id="s", hermes_home=str(hermes_home),
                         agent_identity="h", user_id="alice",
                         agent_context="subagent")
            p.sync_turn("hi", "hey")
            time.sleep(0.2)
            p.shutdown()
        assert not any(c["op"] in {"create_memory", "create_session", "append_event"} for c in fake.calls)

    def test_cron_context_blocks_writes(self, hermes_home):
        fake = FakeClient()
        with patch.object(gmb, "MemoryBankClient", return_value=fake):
            p = gmb.GcpMemoryBankProvider()
            p.initialize(session_id="s", hermes_home=str(hermes_home),
                         agent_identity="h", user_id="alice",
                         agent_context="cron")
            p.on_memory_write("add", "USER.md", "x")
            time.sleep(0.2)
            p.shutdown()
        assert not any(c["op"] == "create_memory" for c in fake.calls)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
class TestTools:
    def test_search(self, provider):
        p, fake = provider
        fake.create_memory(scope=p._scope, fact="Likes Postgres")
        out = json.loads(p.handle_tool_call("memory_search", {"query": "db"}))
        assert out["result"]["count"] == 1

    def test_search_with_topics(self, provider):
        p, fake = provider
        out = json.loads(p.handle_tool_call("memory_search", {
            "query": "x", "topics": ["USER_PREFERENCES"], "since": "2026-01-01T00:00:00Z",
        }))
        assert "result" in out
        last = [c for c in fake.calls if c["op"] == "retrieve"][-1]
        assert "USER_PREFERENCES" in last["filter"]
        assert "update_time" in last["filter"]

    def test_store(self, provider):
        p, fake = provider
        out = json.loads(p.handle_tool_call("memory_store", {"fact": "Allergic to nuts"}))
        assert out["result"]["status"] == "stored"

    def test_get(self, provider):
        p, fake = provider
        m = fake.create_memory(scope=p._scope, fact="x")
        out = json.loads(p.handle_tool_call("memory_get", {"memory_name": m["name"]}))
        assert out["result"]["memory"]["fact"] == "x"

    def test_delete(self, provider):
        p, fake = provider
        m = fake.create_memory(scope=p._scope, fact="x")
        out = json.loads(p.handle_tool_call("memory_delete", {"memory_name": m["name"]}))
        assert out["result"]["status"] == "deleted"

    def test_revisions_workflow(self, provider):
        p, fake = provider
        m = fake.create_memory(scope=p._scope, fact="hello")
        listed = json.loads(p.handle_tool_call("memory_revisions", {"memory_name": m["name"]}))
        assert listed["result"]["count"] == 1
        rolled = json.loads(p.handle_tool_call("memory_rollback", {
            "memory_name": m["name"], "target_revision_id": "r1",
        }))
        assert rolled["result"]["status"] == "rolled_back"

    def test_purge_dry_run_default(self, provider):
        p, fake = provider
        fake.create_memory(scope=p._scope, fact="x")
        out = json.loads(p.handle_tool_call("memory_purge", {}))
        assert out["result"]["status"] == "dry-run"
        # Filter must be scope-bound — never cross-user.
        assert "user_id" in out["result"]["filter"]

    def test_purge_force(self, provider):
        p, fake = provider
        fake.create_memory(scope=p._scope, fact="x")
        out = json.loads(p.handle_tool_call("memory_purge", {"force": True}))
        assert out["result"]["status"] == "purged"

    def test_profile(self, provider):
        p, fake = provider
        fake.create_memory(scope=p._scope, fact="A")
        fake.create_memory(scope=p._scope, fact="B")
        out = json.loads(p.handle_tool_call("memory_profile", {}))
        assert out["result"]["count"] == 2

    def test_ingest_uses_fallback(self, provider):
        p, fake = provider
        out = json.loads(p.handle_tool_call("memory_ingest", {
            "events": [{"role": "user", "text": "Cherry MX Blue switches feel great."},
                       {"role": "user", "text": "ok"}],
        }))
        # Only the long event passes the min_text_chars filter.
        assert out["result"]["created"] == 1

    def test_synthesize_real_or_fallback(self, provider):
        p, fake = provider
        fake.create_memory(scope=p._scope, fact="Lives in Bangalore")
        fake.create_memory(scope=p._scope, fact="Likes Postgres")
        out = json.loads(p.handle_tool_call("memory_synthesize", {"query": "where do I live?"}))
        # Either real LLM (unavailable in tests) or join fallback — both produce text.
        assert out["result"]["narrative"]
        assert out["result"]["sources"] == 2


# ---------------------------------------------------------------------------
# sync_turn / lifecycle
# ---------------------------------------------------------------------------
class TestSyncTurnAndSessions:
    def test_sync_turn_non_blocking(self, provider):
        p, _ = provider
        t0 = time.monotonic()
        for i in range(5):
            p.sync_turn(f"hi {i}", f"hello {i}")
        assert time.monotonic() - t0 < 0.5  # < 100ms per call

    def test_sessions_created_and_events_appended(self, provider):
        p, fake = provider
        p.sync_turn("Hello world", "Hi back")
        time.sleep(0.4)
        assert any(c["op"] == "create_session" for c in fake.calls)
        assert any(c["op"] == "append_event" for c in fake.calls)

    def test_mid_session_generation_after_n_turns(self, provider):
        p, fake = provider
        # Default generate_every_n_turns is 3.
        p.sync_turn("A long technical message", "An informative reply")
        p.sync_turn("Another important fact", "Acknowledged")
        p.sync_turn("Final detail to remember", "Got it")
        time.sleep(0.5)
        # The fallback CreateMemory path should have been hit.
        assert any(c["op"] == "create_memories_from_events" for c in fake.calls), [c["op"] for c in fake.calls]


class TestLifecycleHooks:
    def test_session_end_skips_empty(self, provider, caplog):
        p, fake = provider
        with caplog.at_level("INFO"):
            p.on_session_end([])
        time.sleep(0.2)
        assert not any(c["op"] == "generate_from_session" for c in fake.calls)
        assert any("session-end skipped" in r.message for r in caplog.records)

    def test_session_end_uses_vertex_session_source(self, provider):
        p, fake = provider
        p.sync_turn("Some context", "Some reply")
        time.sleep(0.3)
        p.on_session_end([])
        time.sleep(0.5)
        assert any(c["op"] == "generate_from_session" for c in fake.calls)

    def test_pre_compress_uses_session_then_fallback(self, provider):
        p, fake = provider
        p.sync_turn("First turn", "Reply")
        time.sleep(0.3)
        p.on_pre_compress([{"role": "user", "content": "compressed message"}])
        time.sleep(0.5)
        ops = [c["op"] for c in fake.calls]
        assert "generate_from_session" in ops or "create_memories_from_events" in ops

    def test_on_memory_write_no_action_prefix(self, provider):
        p, fake = provider
        p.on_memory_write("add", "USER.md", "User likes oat milk.")
        time.sleep(0.3)
        # v2 fix: the fact must NOT be prefixed with [ADD USER.md].
        creates = [c for c in fake.calls if c["op"] == "create_memory"]
        assert creates
        assert not creates[-1]["fact"].startswith("[")


# ---------------------------------------------------------------------------
# Synthesis fallback
# ---------------------------------------------------------------------------
class TestSynthesizeFallback:
    def test_join_fallback_when_genai_missing(self):
        from gcp_memory_bank_v2.synthesize import synthesize_memories
        # No google.genai installed in tests; must fall through to join.
        narrative = synthesize_memories(
            project="p", location="us-central1", model="gemini-2.5-flash",
            query="x", memories=[{"fact": "A"}, {"fact": "B"}],
        )
        assert "A" in narrative and "B" in narrative


# ---------------------------------------------------------------------------
# Topics build
# ---------------------------------------------------------------------------
class TestTopicsBuild:
    def test_real_few_shots_emitted(self):
        from gcp_memory_bank_v2.topics import (
            DEFAULT_FEW_SHOT_EXAMPLES,
            build_memory_bank_config,
        )
        cfg = build_memory_bank_config(project_id="p")
        cust = cfg["customization_configs"][0]
        # Verify the SDK-correct nested topic shape.
        first_managed = next(t for t in cust["memory_topics"] if "managed_memory_topic" in t)
        assert "managed_topic_enum" in first_managed["managed_memory_topic"]
        # Few-shots present and non-empty.
        assert len(cust["generate_memories_examples"]) >= len(DEFAULT_FEW_SHOT_EXAMPLES)

    def test_granular_ttl(self):
        from gcp_memory_bank_v2.topics import build_memory_bank_config
        cfg = build_memory_bank_config(project_id="p", create_ttl_days=90,
                                       generate_created_ttl_days=180)
        ttls = cfg["ttl_config"]["granular_ttl_config"]
        assert ttls["create_ttl"] == f"{90 * 86400}s"
        assert ttls["generate_created_ttl"] == f"{180 * 86400}s"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
class TestSystemPrompt:
    def test_contains_scope_and_tools(self, provider):
        p, _ = provider
        block = p.system_prompt_block()
        assert "GCP Memory Bank" in block
        assert "user_id=jithendra" in block
        assert "memory_search" in block
        assert "<gcp-mb-context>" in block


# ---------------------------------------------------------------------------
# Recall mode gating
# ---------------------------------------------------------------------------
class TestRecallMode:
    def test_context_mode_no_tools(self, hermes_home):
        (hermes_home / "gcp-memory-bank.json").write_text(json.dumps({"recall_mode": "context"}))
        fake = FakeClient()
        with patch.object(gmb, "MemoryBankClient", return_value=fake):
            p = gmb.GcpMemoryBankProvider()
            p.initialize(session_id="s", hermes_home=str(hermes_home),
                         agent_identity="h", user_id="alice")
            assert p.get_tool_schemas() == []
            p.shutdown()

    def test_tools_mode_no_auto_prefetch(self, hermes_home):
        (hermes_home / "gcp-memory-bank.json").write_text(json.dumps({"recall_mode": "tools"}))
        fake = FakeClient()
        with patch.object(gmb, "MemoryBankClient", return_value=fake):
            p = gmb.GcpMemoryBankProvider()
            p.initialize(session_id="s", hermes_home=str(hermes_home),
                         agent_identity="h", user_id="alice")
            assert p.prefetch("anything") == ""
            schemas = p.get_tool_schemas()
            assert any(s["name"] == "memory_search" for s in schemas)
            p.shutdown()


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
class TestCircuitBreaker:
    def test_opens_then_recovers(self):
        from gcp_memory_bank_v2.client import _CircuitBreaker
        b = _CircuitBreaker(threshold=2, cooldown_seconds=1)
        b.record_failure(); b.record_failure()
        assert b.state == "open" and not b.allow()
        time.sleep(1.05)
        assert b.allow()
        b.record_success()
        assert b.state == "closed"


# ---------------------------------------------------------------------------
# Schemas exposed
# ---------------------------------------------------------------------------
class TestSchemas:
    def test_eleven_tools(self):
        from gcp_memory_bank_v2 import all_schemas
        names = [s["name"] for s in all_schemas()]
        assert len(names) == 11
        assert "memory_synthesize" in names
        assert all(n.startswith("memory_") for n in names)


# ---------------------------------------------------------------------------
# v2.1 fixes — pollution filter, debounce, throttle retry detection
# ---------------------------------------------------------------------------
class TestPollutionFilter:
    def test_pollution_patterns(self):
        from gcp_memory_bank_v2.retrieval import is_pollution
        # Real captures from the live engine on 2026-04-29.
        assert is_pollution("Review the conversation above and consider whether a skill should be saved or updated.")
        assert is_pollution("**Nothing to save.**\n\nThe task class: ...")
        assert is_pollution("Health check memory created at 2026-04-29T06:03:04Z. Test ID: health-check-140b6734")
        assert is_pollution("[IMPORTANT: Background process proc_ce5ce3398892 completed (exit code 1).")
        # Real-content false positives
        assert not is_pollution("My favorite VPN protocol is WireGuard.")
        assert not is_pollution("User uses Postgres with pgvector for embedding storage.")

    def test_pollution_blocks_sync_turn(self, provider):
        p, fake = provider
        # Real polluting message that was captured on 2026-04-29.
        p.sync_turn("Review the conversation above and consider whether a skill should be saved.",
                    "Some assistant reply.")
        time.sleep(0.3)
        assert not any(c["op"] in {"create_session", "append_event", "create_memory"} for c in fake.calls)


class TestSessionEndDebounce:
    def test_back_to_back_calls_debounced(self, provider):
        p, fake = provider
        p.sync_turn("Real user message about postgres.", "Real reply.")
        time.sleep(0.3)
        # First call dispatches generate.
        p.on_session_end([])
        time.sleep(0.4)
        first_count = sum(1 for c in fake.calls if c["op"] == "generate_from_session")
        # Second call within cooldown should NOT dispatch.
        p.on_session_end([])
        time.sleep(0.3)
        second_count = sum(1 for c in fake.calls if c["op"] == "generate_from_session")
        assert second_count == first_count, "expected debounce to drop second call"


class TestThrottleRetry:
    def test_code_8_throttle_detected(self):
        from gcp_memory_bank_v2.client import _retry
        attempts = {"n": 0}
        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError(
                    "Failed to generate memory: {'code': 8, 'message': 'throttled. Please try again.'}"
                )
            return "ok"
        # Force the no-tenacity path so the test is portable.
        import sys
        if "tenacity" in sys.modules:
            # If tenacity is available, the real path should also retry on
            # the throttle detector. Either way, after 2 attempts we see "ok".
            pass
        result = _retry(flaky, attempts=3)
        assert result == "ok"
        assert attempts["n"] == 2
