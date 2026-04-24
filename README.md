# gcp-memory-bank

Production-grade Python SDK for Google Gemini Enterprise Agent Platform **Memory Bank**.

Provides async-first, type-safe access to managed long-term memory for AI agents with scoped isolation, structured profiles, and comprehensive retrieval strategies.

## What is Memory Bank?

Memory Bank is Google's managed long-term memory service for AI agents (part of the Gemini Enterprise Agent Platform). It:

- **Extracts** meaningful facts from conversations automatically
- **Consolidates** new info with existing memories (update/merge/delete)
- **Retrieves** via similarity search scoped to specific identities
- **Persists** across sessions, runtimes, and frameworks

## Features

| Feature | Status |
|---------|--------|
| Async-first client | ✅ GA |
| Memory CRUD (create, read, update, delete, purge) | ✅ GA |
| Session management | ✅ GA |
| Similarity search | ✅ GA |
| Scope-based isolation | ✅ GA |
| Structured profiles | 🔶 Preview |
| Event ingestion (streaming) | 🔶 Preview |
| Memory revisions / audit trail | ✅ GA |
| IAM Conditions | ✅ GA |
| Multi-strategy retrieval | ✅ SDK-layer |
| Bridge pattern for external agents | ✅ SDK-layer |

## Installation

```bash
pip install "google-cloud-aiplatform[agent_engines]"
pip install -e .
```

## Quick Start

```python
import asyncio
from memory_bank import MemoryBankClient, MemoryBankConfig
from memory_bank.memory import MemoryManager
from memory_bank.utils import build_scope

async def main():
    async with MemoryBankClient(
        project="your-gcp-project-id",
        location="us-central1"
    ) as client:
        # Create instance
        engine = await client.create_instance(MemoryBankConfig())
        
        manager = MemoryManager(client)
        scope = build_scope(user_id="jithendra")
        
        # Generate memories from conversation
        events = [
            {"role": "user", "content": "I prefer Python for backend work."},
            {"role": "user", "content": "Use us-central1 for all GCP deployments."},
        ]
        generated = await manager.generate_from_events(events, scope=scope)
        
        # Search memories
        results = await manager.search("deployment region", scope=scope)
        for r in results:
            print(f"- {r.memory.fact}")

asyncio.run(main())
```

## Architecture

```
gcp-memory-bank/
├── src/memory_bank/
│   ├── __init__.py          # Public API exports
│   ├── client.py            # Core async client + instance lifecycle
│   ├── models.py            # Pydantic models (Memory, Scope, Revision, ...)
│   ├── config.py            # Instance configuration + pre-built presets
│   ├── memory.py            # Memory CRUD + generation + search
│   ├── sessions.py          # Agent Platform Session management
│   ├── retrieval.py         # Multi-strategy retrieval (similarity, scope, hybrid)
│   ├── profiles.py          # Structured profiles (Preview)
│   ├── ingestion.py         # Event streaming (Preview)
│   ├── revisions.py         # Audit trail + rollback inspection
│   ├── iam.py               # IAM Conditions for scope-based security
│   ├── bridge.py            # Bridge pattern for Hermes/OpenClaw integration
│   └── utils.py             # Formatters, scope builders, Jinja templates
├── examples/
│   ├── basic_usage.py
│   ├── hermes_bridge_demo.py
│   └── openclaw_bridge_demo.py
├── tests/
└── pyproject.toml
```

## Modules

### `client.MemoryBankClient`

Async-first wrapper around the vertexai SDK:

- `create_instance(config)` — Create Agent Platform instance with Memory Bank
- `update_instance(config)` — Update topics, TTL, schemas
- `delete_instance()` — Cascade delete all sessions/memories
- `list_instances()` — Discover existing instances
- Automatic retry on provisioning delays
- Context manager support (`async with`)

### `memory.MemoryManager`

High-level memory operations:

| Method | What it does |
|--------|-------------|
| `create(fact, scope)` | Direct write (no consolidation) |
| `generate_from_session(session_name, scope)` | Extract from session history |
| `generate_from_events(events, scope)` | Extract from raw events |
| `generate_from_facts(facts, scope)` | Agent-controlled writes |
| `get(memory_name)` | Fetch single memory |
| `list_all()` | Paginated list |
| `retrieve_by_scope(scope)` | All memories for exact scope |
| `search(query, scope, top_k)` | Embedding similarity search |
| `search_to_prompt(query, scope)` | Search + format for prompt injection |
| `delete(memory_name)` | Remove single memory |
| `purge(filter_string)` | Bulk delete by criteria |
| `purge_scope(scope)` | Delete all memories for scope |

### `retrieval`

Pluggable retrieval strategies:

- `SimilaritySearchStrategy` — Embedding-based (default)
- `ScopeRetrievalStrategy` — Exact scope match
- `HybridRetrievalStrategy` — Multi-strategy fusion + deduplication
- `MultiScopeRetrieval` — Retrieve across nested scope levels

### `bridge`

Abstract `BaseMemoryBridge` for integrating Memory Bank into external agents:

- `HermesBridgeExample` — Reference implementation for Hermes (NOT wired)
- `OpenClawBridgeExample` — Reference implementation for OpenClaw (NOT wired)

Implement `on_session_start`, `on_session_end`, `on_agent_fact`, and `recall` to wire into any agent framework.

### `config`

Pre-built configurations:

- `HERMES_MEMORY_CONFIG` — 7 topics, 90-day TTL, 5 revision depth
- `OPENCLAW_MEMORY_CONFIG` — 4 topics, 30-day TTL, 3 revision depth

### `profiles.ProfileManager`

Structured schemas for low-latency retrieval (Preview):

```python
profile = await profile_manager.get_profile("user-profile", scope)
# Returns: {"technical_stack": "ADK, Python", "preferred_language": "Python", ...}
```

### `ingestion.EventIngestionStream`

Streaming pattern (Preview):

```python
stream = EventIngestionStream(client, scope, stream_id="realtime")
await stream.ingest(events)  # Auto-triggers generation when threshold met
await stream.flush()         # Force trigger
```

### `revisions.RevisionManager`

Audit trail:

```python
revisions = await revision_manager.list_revisions(memory_name)
await revision_manager.diff_revisions(memory_name, rev_index_a=-2, rev_index_b=-1)
audit = await revision_manager.audit_memory(memory_name)
```

## Configuration

### Environment Variables

| Variable | Required | Default |
|----------|----------|---------|
| `GOOGLE_CLOUD_PROJECT` | Yes | — |
| `GOOGLE_CLOUD_LOCATION` | No | `us-central1` |
| `HERMES_MEMORY_ENGINE` | No | Auto-discover |

### Required GCP Setup

```bash
# Enable API
gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT

# Required roles
# - roles/aiplatform.user
# - roles/serviceusage.serviceUsageAdmin (to enable APIs)
```

## Memory Scope Isolation

Memories are strictly isolated by **scope** — a dict of up to 5 key-value pairs:

```python
# Broad
{"user_id": "jithendra"}

# Narrow
{"user_id": "jithendra", "project": "openclaw"}

# Agent-specific
{"user_id": "jithendra", "agent": "hermes"}
```

**Rule:** Only memories with the **exact same scope** are consolidated or retrieved together.

## Cost

Memory Bank burns your GCP credit:

| Operation | Estimated Cost |
|-----------|---------------|
| Generate memories (per session) | ~$0.02–0.05 |
| Retrieve memories | ~$0.001 |
| Storage | Negligible |
| **Monthly** (100 sessions) | **~$5–10** |

## Tests

```bash
pytest tests/ -v
```

## License

Apache 2.0
