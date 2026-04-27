# GCP Memory Bank Plugin for Hermes

Google Cloud Memory Bank integration for Hermes Agent. Provides persistent,
cross-session memory with LLM-powered extraction, automatic consolidation,
and semantic search via the Gemini Enterprise Agent Platform.

## Features

- **Automatic memory extraction** — Conversation facts extracted by Gemini
- **Semantic search** — Retrieve memories by meaning, not keyword
- **Multi-user isolation** — Scoped by `app_name` + `user_id`
- **Native consolidation** — Duplicate detection handled by Memory Bank
- **4 managed topics** — Personal info, preferences, conversation details, explicit instructions
- **4 few-shot examples** — Teaches Memory Bank your extraction patterns
- **Circuit breaker** — Pauses API calls after 5 consecutive failures
- **Revision history** — Track how memories evolved over time

## Setup

```bash
hermes memory setup
```

Or manually create `~/.hermes/gcp-memory-bank.json`:

```json
{
  "project_id": "your-gcp-project-id",
  "location": "us-central1",
  "engine_id": ""
}
```

Leave `engine_id` blank to auto-provision an Agent Engine.

Then activate:

```yaml
# ~/.hermes/config.yaml
memory:
  provider: gcp-memory-bank
```

## Tools

| Tool | Purpose |
|------|---------|
| `memory_search` | Semantic search by query |
| `memory_store` | Store an explicit fact |
| `memory_profile` | List all stored memories |
| `memory_get` | Fetch a specific memory by name |
| `memory_delete` | Delete a specific memory |
| `memory_revisions` | List revision history for a memory |
| `memory_revision_get` | Retrieve a specific revision |
| `memory_rollback` | Rollback a memory to a previous revision |
| `memory_ingest` | Stream batch events for background processing |
| `memory_purge` | Delete all user memories (optionally filtered) |
| `memory_synthesize` | Compose a narrative summary from memories |

## Architecture

- Events buffered during conversation via `sync_turn()`
- `ingest_events()` triggered at session end (background, non-blocking)
- `retrieve_memories()` for prefetch before each turn
- `on_pre_compress()` flushes events before context truncation
- Memories queryable ~60s after session ends (background processing)

## Known Issues & Fixes

### `generate()` silently fails — fixed 2026-04-27

**Problem:** `on_session_end()` and `on_pre_compress()` originally used `vertexai.Client.agent_engines.memories.generate()` to create memories from buffered conversation events. This API path **silently fails** — the call returns without error, but no memories are generated.

**Fix:** Both hooks now use `ingest_events()` (the same proven path as the `memory_ingest` tool), which correctly processes events in the background and generates memories asynchronously.

**Verification:** After the fix, a 3-turn conversation about "Cherry MX keyboards" was processed by `on_session_end()`. Within 20 seconds, searching for "Cherry MX" returned: *"Owns a mechanical keyboard with Cherry MX Blue switches used for coding and gaming..."*

## Docs

- https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/setup
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/ingest-events
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/fetch-memories
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/revisions
