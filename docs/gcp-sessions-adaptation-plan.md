# GCP Sessions Adaptation Plan

Date: 2026-05-16

## Decision

Extend the existing `gcp-memory-bank` Hermes provider. Do not build a separate `gcp-sessions` memory provider.

Rationale:

- Hermes supports one active external memory provider at a time.
- Google Sessions are the short/medium-term event source for Memory Bank extraction, not the durable memory store.
- Hermes already owns canonical session state in `state.db`.
- A separate general plugin would duplicate lifecycle handling and introduce coordination risk with the Memory Bank provider.

## Target Boundary

```text
Hermes state.db
  Canonical sessions, resume, search, lineage, platform source tags

gcp-memory-bank/sessions.py
  Remote chronological mirror of Hermes turns

GCP Memory Bank
  Durable extracted memories and structured profiles
```

GCP Sessions should mirror Hermes conversations. They should not replace Hermes session storage, and they should not force durable Memory Bank scopes to include `session_id` by default.

## Current Fit

The provider already maps the main Hermes hooks:

- `sync_turn()` appends user/model events asynchronously.
- `on_session_end()` generates memories from `vertex_session_source`.
- `on_pre_compress()` drains pending local events before Hermes compresses context.
- `shutdown()` closes resources without forcing expensive extraction.

The provider also already keeps config and local session mirror files under `$HERMES_HOME`, which satisfies Hermes profile isolation.

## Needed Changes

### 1. Key GCP sessions by Hermes conversation identity

Current cross-process reuse is broad. It should become conversation-aware:

```text
profile + platform + canonical_user_id + hermes_session_id_or_lineage_root
```

Use Hermes lineage/root session if available. Otherwise use the current `session_id`.

### 2. Verify persisted remote sessions before reuse

Before reattaching to a persisted GCP session:

- call `get_session()` or equivalent SDK method
- verify the session exists and still belongs to the expected user
- clear the local persisted pointer if missing or expired

### 3. Retry once on stale append

If `sessions.events.append()` fails because the remote session is stale or deleted:

- clear local persisted state
- create a fresh GCP session
- retry the append once

Do not retry indefinitely.

### 4. Add session inspection CLI

Add non-destructive commands:

```bash
hermes gcp-memory-bank sessions local
hermes gcp-memory-bank sessions verify
hermes gcp-memory-bank sessions describe SESSION
hermes gcp-memory-bank sessions events SESSION
```

Keep destructive commands explicit:

```bash
hermes gcp-memory-bank sessions delete SESSION
hermes gcp-memory-bank sessions clean --force
```

### 5. Keep Memory Bank scope stable

Default durable memory scope should remain:

```json
{"app_name": "hermes", "user_id": "demo-user"}
```

Add `profile`, `workspace`, or `session_id` only when the operator explicitly chooses that partitioning strategy. Exact-scope retrieval means scope changes strand old memories unless migrated.

### 6. Add config flags

Proposed additions:

```json
{
  "gcp_session_key_strategy": "hermes_session",
  "gcp_session_delete_on_end": false,
  "gcp_session_reuse_stale_retry": true
}
```

Possible future value:

```json
{"gcp_session_key_strategy": "lineage_root"}
```

Use it only after verifying Hermes passes enough lineage data into memory-provider initialization.

## Verification Plan

Unit tests:

- key strategy builds unique file paths
- profile-local persistence stays under `$HERMES_HOME`
- stale persisted session is rejected
- append failure retries once with a new session
- `sync_turn()` remains non-blocking
- CLI `sessions local/verify/describe/events` parse and call the expected client methods

Hermes integration tests:

- `MemoryManager.initialize_all()`
- `sync_all()`
- `on_pre_compress()`
- `on_session_end()`
- `shutdown_all()`

Live smoke test with private credentials:

- create a temporary GCP session
- append one user event and one model event
- list/read events
- generate Memory Bank memories from the session source, or run in dry mode if avoiding extraction cost
- delete the temporary session
- confirm no live resource names or credentials are committed
