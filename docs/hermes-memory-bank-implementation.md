# Hermes + GCP Memory Bank Implementation

Date: 2026-05-16

This document describes the long-term memory stack in this repository. It combines Google's managed Memory Bank and Sessions APIs with Hermes Agent's memory-provider lifecycle, while keeping local credentials and deployment identifiers out of the repository.

## What This Provides

- Managed long-term memory for agents using Google Gemini Enterprise Agent Platform Memory Bank.
- A Hermes `MemoryProvider` plugin that follows the provider ABC, setup schema, hooks, profile isolation, and threading contract.
- GCP Sessions as a remote chronological mirror for Hermes turns, used as the source for Memory Bank extraction.
- Durable cross-session recall with exact-scope isolation.
- Structured profile retrieval for stable user, project, and workflow facts.
- Operational safeguards: circuit breaker, bounded retry, background operation logging, pollution filtering, scope-drift detection, and conservative fallback paths.
- A profile-aware admin CLI modeled after Hermes' bundled Honcho memory plugin.
- A dependency-light Deep Research Agent scaffold that can use Memory Bank through adapter protocols.

## Architecture

```text
Hermes state.db
  Canonical local session store: resume, lineage, platform source, messages

Hermes MemoryProvider hooks
  initialize()       -> load profile-local config
  sync_turn()        -> append user/model events asynchronously
  prefetch()         -> retrieve scoped memories for prompt context
  on_pre_compress()  -> flush pending raw turns before compression
  on_session_end()   -> generate durable memories from GCP session source
  shutdown()         -> close transports without expensive generation

GCP Sessions
  Remote chronological event mirror for a Hermes conversation

GCP Memory Bank
  Durable cross-session memories, revisions, structured profiles, retrieval
```

The important boundary: Hermes remains the source of truth for sessions. GCP Sessions are not a replacement for Hermes `state.db`; they are the remote event source that Memory Bank can use to extract durable memory.

## Hermes Provider Surface

The active provider lives in `hermes-plugin-v2/`:

- `__init__.py` wires the Hermes hooks.
- `config.py` handles defaults, scope templates, profile-local config loading, and user-id guardrails.
- `client.py` wraps Memory Bank and Sessions SDK calls.
- `sessions.py` owns GCP Session lifecycle and event appends.
- `retrieval.py` handles prompt-safe recall formatting.
- `ingestion.py` keeps the proven per-event `CreateMemory` fallback.
- `tools.py` exposes memory tools.
- `cli.py` exposes profile-aware admin commands.

The plugin is intentionally one provider. Splitting Sessions into a second memory provider would conflict with Hermes' single external memory-provider model and would blur the boundary between Hermes local sessions and Google session mirrors.

## Admin CLI

When `gcp-memory-bank` is the active Hermes memory provider:

```bash
hermes gcp-memory-bank status
hermes gcp-memory-bank status --all
hermes gcp-memory-bank config path
hermes gcp-memory-bank config show --effective
hermes gcp-memory-bank --target-profile default config show
hermes gcp-memory-bank --target-profile research config set user_id demo-user
hermes gcp-memory-bank --target-profile research config unset scope_template.workspace
hermes gcp-memory-bank sessions list
hermes gcp-memory-bank doctor
```

`--target-profile` lets operators inspect or update another profile's `gcp-memory-bank.json` without switching the active Hermes profile.

## Security Posture

The repository is designed to be public-safe:

- No API keys, ADC files, service accounts, `.env`, or local `gcp-memory-bank.json` files are required in git.
- `.gitignore` excludes common secret and local-config filenames.
- Documentation uses placeholders such as `YOUR_PROJECT_ID`, `YOUR_ENGINE_ID`, and `demo-user`.
- Real validation can happen locally through ADC and profile-local config.
- Commands that delete memories, purge scopes, or delete engines should remain opt-in and guarded.

Memory scopes are not secrets. They are routing boundaries. Use stable non-sensitive identifiers in examples and keep live user/profile names in local config.

## Verification Summary

Local deterministic coverage:

```bash
GMB_PLUGIN_DIR=$PWD/hermes-plugin-v2 python3 -m pytest hermes-plugin-v2/tests -q
PYTHONPATH=src python3 -m pytest tests -q
```

Hermes plugin verification:

```bash
hermes memory status
hermes doctor
hermes gcp-memory-bank status
hermes gcp-memory-bank config show --effective
```

Live verification should use real ADC credentials and a private engine, but the repository should only record:

- test class or command name
- pass/fail status
- SDK version
- high-level operation result
- redacted resource names

Do not commit access tokens, service-account JSON, live config files, raw chat IDs, or private transcript content.

## Implementation Notes

This implementation is designed as an operational agent-memory system rather than a toy vector-store example:

- It respects the host agent's lifecycle.
- It keeps profile and workspace isolation explicit.
- It handles real SDK quirks instead of assuming every documented endpoint works equally.
- It separates local session truth from remote memory extraction.
- It includes tests for concurrency, hooks, tools, scope behavior, CLI config writes, and MemoryManager integration.
- It provides a path for higher-level research agents to use memory without coupling the core planner to one provider.
