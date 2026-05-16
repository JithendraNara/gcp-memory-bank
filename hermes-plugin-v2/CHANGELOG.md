# Changelog

## v2.2.0 — 2026-05-16 — docs-aligned Memory Bank defaults

### Changed
- Default Memory Bank generation model changed to Google's current documented default: `gemini-2.5-flash`.
- Default Memory Bank embedding model changed to Google's current documented default: `text-embedding-005`.
- Active local config updated to the same defaults.

### Documentation
- README now documents Hermes-profile-scoped config/state, exact-scope retrieval, and Hindsight-style optional profile/workspace partitioning through `scope_keys` + `scope_template`.
- `gcp-memory-bank-ops` skill updated to the current Google default model pair.

---

## v2.1.0 — 2026-05-15 — Memory Profiles spike

### Added
- `memory_profiles` tool — structured schema-shaped snapshot via `retrieve_profiles(...)`. The private production engine was configured with the `hermes-profile` schema fields: `name`, `location`, `communication_style`, `technical_stack`, `active_projects`, `operational_preferences`. Schema ID: `hermes-profile`.
- `GcpMemoryBankProvider._tool_profiles()` handler + `STRUCTURED_PROFILES_SCHEMA` tool definition in `tools.py`.
- `MemoryBankClient.retrieve_profiles(scope)` wrapper in `client.py`.
- `DEFAULT_STRUCTURED_SCHEMA` in `topics.py` (and integrated into `build_memory_bank_config`).
- `retrieve_profiles` fake in test suite + `test_structured_profiles` test case.
- Tool count updated from 11 → 12 in tests.

### Fixed
- `memory_purge` SDK signature: `force` moved from inside `config` dict to top-level kwarg. Root cause of `extra_forbidden` errors in gateway.error.log before v2 plugin upgrade.
- `ingest_events` LRO blocking: confirmed non-cancellable, blocks engine deletion. Plugin retains proven per-event `CreateMemory` fallback — IngestEvents not used for production ingestion.

### Changed
- Production engine generation model was moved off the earlier preview model during the private deployment; public examples now use Google's documented defaults.
- `agent/memory_manager.py` patched: post-init tool schema re-index in `initialize_all` so gcp-memory-bank's lazily-populated `get_tool_schemas()` are discoverable after `initialize()` is called.

### Documentation
- `README.md` updated: model change, `memory_profiles` tool, purge SDK quirk, IngestEvents warning.
- `gcp-memory-bank-ops` skill updated: purge quirk, profile schema, tool count.
- This CHANGELOG created.

---

## v2.0.0 — 2026-05-05 — v2 rewrite

Full rewrite with recall modes, circuit breaker, GCP session mirror, real Gemini synthesis, context fence, scope drift detection, agent_context gating.
