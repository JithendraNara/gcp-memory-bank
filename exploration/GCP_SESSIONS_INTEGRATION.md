# GCP Sessions × Hermes Integration

**Date:** 2026-04-27
**Status:** ✅ Implemented in plugin
**Plugin:** `~/.hermes/plugins/gcp-memory-bank/__init__.py`

---

## What We Built

The GCP Memory Bank plugin now uses **GCP Agent Runtime Sessions** as the official event store for memory generation, with `ingest_events()` as a fallback.

### Architecture

```
User → Hermes → sync_turn()
              ├─> [LOCAL] Buffer events in _events (fallback)
              ├─> [BG THREAD] GCP Session.events.append() (official store)
              └─> [BG THREAD] ingest_events() (mid-session fallback)

Session End → on_session_end()
              ├─> Flush remaining events to GCP Session
              ├─> generate(vertex_session_source={"session": ...}) ← OFFICIAL PATH
              └─> Fallback: ingest_events(_events) if GCP Session fails
```

---

## Key Findings

### 1. `generate()` Works — But Only With `vertex_session_source`

| API Path | Result |
|----------|--------|
| `generate(direct_contents_source={"events": [...]})` | ❌ Silently fails (no memories created) |
| `generate(vertex_session_source={"session": ...})` | ✅ Creates/updates memories automatically |
| `ingest_events(direct_contents_source={"events": [...]})` | ✅ Works (background processing) |

**The official Google architecture is:**
```python
client.agent_engines.sessions.create(name=engine, user_id=UID, config={...})
client.agent_engines.sessions.events.append(name=session, author=..., invocation_id=..., timestamp=..., config={...})
client.agent_engines.memories.generate(
    name=engine,
    vertex_session_source={"session": session_name},
    scope={"user_id": UID},
    config={"wait_for_completion": False},
)
```

### 2. SDK API Quirks (Verified Live)

**Session creation:**
```python
# CORRECT (current SDK)
op = client.agent_engines.sessions.create(
    name=engine_name,        # NOT "parent"
    user_id="jithendra",     # Required
    config={
        "display_name": "...",
        "ttl": "86400s",     # Minimum 24h
    },
)
session_name = op.response.name  # NOT op.result()
```

**Event append:**
```python
# CORRECT (current SDK)
client.agent_engines.sessions.events.append(
    name=session_name,
    author="user",           # Required
    invocation_id="1",       # Required (used for dedup)
    timestamp=datetime.now(timezone.utc),  # Required, must have tzinfo
    config={
        "content": {
            "role": "user",
            "parts": [{"text": "hello"}]
        }
    },
)
```

**Memory generation from session:**
```python
# CORRECT (current SDK)
client.agent_engines.memories.generate(
    name=engine_name,
    vertex_session_source={"session": session_name},  # NOT {"name": ...}
    scope={"app_name": "hermes", "user_id": "jithendra"},
    config={"wait_for_completion": False},
)
```

### 3. Performance

| Operation | Latency |
|-----------|---------|
| Session creation | ~1s |
| Event append | ~100-200ms |
| Memory generation (wait=True) | ~15-20s |
| Memory generation (wait=False) | ~50ms (fire-and-forget) |
| Memory search | ~500ms |

### 4. Memory Consolidation

GCP Memory Bank **consolidates** similar memories. If you say "I like Python" multiple times, it may UPDATE an existing memory instead of creating a new one. This is by design.

To force a new memory, use highly unique content. Example test that proved CREATION:
```
User: "My secret codename for this project is Purple Unicorn 42."
→ Action: CREATED
→ Memory: "My secret codename for my project is Purple Unicorn 42."
```

---

## Plugin Implementation Details

### New Config Options

| Key | Default | Description |
|-----|---------|-------------|
| `use_gcp_sessions` | `True` | Enable GCP Session integration |
| `gcp_session_ttl_seconds` | `86400` | Session TTL (min 86400 = 24h) |

### New Methods

```python
_ensure_gcp_session()          # Creates session lazily in background
_append_events_to_gcp_session(events, turn_marker)  # Sync append (call from bg thread)
_ingest_events(events, turn_count)                  # Fallback ingest (proven working)
```

### Modified Hooks

| Hook | Behavior |
|------|----------|
| `initialize()` | Kicks off `_ensure_gcp_session()` in background thread |
| `sync_turn()` | Buffers locally + fires `events.append()` to GCP Session (bg thread) |
| `on_session_end()` | Flushes to GCP Session → `generate(vertex_session_source=...)` → fallback to `ingest_events()` |
| `on_pre_compress()` | Same pattern as `on_session_end()` |
| `on_memory_write()` | Now always uses `memories.create()` (broken `generate()` path removed) |

---

## Verified End-to-End Flow

```
1. Plugin initializes → creates GCP Session (bg thread)
2. sync_turn(user, assistant) →
   a. Buffer events locally
   b. Fire events.append() to GCP Session
3. on_session_end() →
   a. Flush remaining events to GCP Session
   b. Call generate(vertex_session_source={"session": ...})
   c. Background processing extracts memories (~15-20s)
4. Next session prefetch() → retrieves generated memories
```

**Live test result:** Memory "My secret codename for my project is Purple Unicorn 42." created from session events and retrieved via `memory_search`.

---

## Files

| File | Purpose |
|------|---------|
| `~/.hermes/plugins/gcp-memory-bank/__init__.py` | Installed plugin (live) |
| `~/projects/gcp-memory-bank/hermes-plugin/__init__.py` | Repo copy |
| `~/projects/gcp-memory-bank/hermes-plugin/plugin.yaml` | Plugin manifest |

---

## Related Docs

- https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/sessions
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/sessions/manage-with-api
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/generate-memories
- https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin
