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
| `memory_purge` | Delete all user memories |

## Architecture

- Events buffered during conversation
- `generate_memories()` triggered at session end (background)
- `retrieve_memories()` for prefetch before each turn
- Memories queryable ~60s after session ends

## Docs

- https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/setup
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/generate-memories
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/memory-bank/fetch-memories
