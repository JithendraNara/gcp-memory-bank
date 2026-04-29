# gcp-memory-bank — Hermes Plugin v2 (unified)

The unified rewrite that combines the battle-tested v1 SDK reality with all the missing UX, observability, and safety features. **Backwards compatible** with engine `4938048007586185216` and the existing 200+ memories under `{app_name=hermes, user_id=jithendra}`.

## What changed vs v1

| Area | v1 | v2 |
|---|---|---|
| **Tools** | 11 (incl. fake `memory_synthesize`) | 11 (real Gemini synthesis) |
| **Tool dispatch** | Flat 300-line if/elif | Dict-of-handlers |
| **`memory_synthesize`** | `" ".join(facts)` | Real Gemini call (with join fallback) |
| **`on_memory_write`** | Wrote `[ADD USER.md] ...` polluted facts | Drops the prefix; clean fact text |
| **`generate_every_n_turns`** | `0` (mid-session generation OFF) | `3` by default |
| **Session reuse** | New session every 18s observed | Reuses one session per process; atexit cleanup |
| **Empty session-end** | Burns a round-trip on (0 events, 0 turns) | Skipped by default |
| **`user_id`** | Accepted raw chat ids (Telegram `8405386815` leaked) | Refuses numeric-only ids; logs migration warning |
| **Scope drift** | Silent — 3 user_ids and 3 engines accumulated | `ScopeDriftDetector` warns on every change |
| **Recall modes** | Always-on prefetch | `recall_mode ∈ {context, tools, hybrid}` |
| **Recall budget** | Hardcoded `top_k=8` | `low/mid/high` → 3/8/15 |
| **Recall detail** | Flat fact list | L0/L1/L2 with topic + age |
| **Trivial skip** | None | Regex-skip on "ok"/"thanks"/"/help"/etc. |
| **Context fence** | Plain markdown header | `<gcp-mb-context>...</gcp-mb-context>` + sanitize-before-capture |
| **Background ops** | 23 dispatched, 0 completion logs | `timed()` wraps every fire-and-forget |
| **Circuit breaker** | Hardcoded 5/120 | Configurable; tenacity retry on transient errors |
| **`agent_context` gate** | Only `primary` checked | Strict skip set: `{cron, flush, subagent}` |
| **Topic schema** | Already correct (`{managed_topic_enum: ...}`) | Same |
| **Few-shot examples** | 4 inline (Fort Wayne, etc.) | 5 (added a TECHNICAL_DECISIONS positive) |
| **CLI** | 12 commands | 14 (adds `audit`, `scope-migrate`, `instance update-config`) |

## Module layout

```
hermes-plugin-v2/
├── plugin.yaml
├── __init__.py             # GcpMemoryBankProvider — orchestrator (~700 lines)
├── config.py               # GmbConfig, scope template, user_id guardrails
├── client.py               # Dual-client (proto + vertexai), breaker, tenacity, LRO polling
├── topics.py               # 4 managed + 3 custom topics + 5 verified few-shots
├── sessions.py             # Reuse + atexit + skip empty
├── ingestion.py            # Sliding-window buffer + per-event CreateMemory fallback
├── retrieval.py            # PrefetchCache, L0/L1/L2 format, fence + sanitize, trivial skip
├── synthesize.py           # REAL Gemini synthesis with join fallback
├── tools.py                # 11 schemas + dict-dispatch
├── cli.py                  # 14 subcommands incl. audit + scope-migrate
├── observability.py        # timed() + ScopeDriftDetector
└── tests/
    └── test_provider.py    # 38 tests, no real GCP calls
```

## Installation — swap from v1

```bash
# 1. Back up v1
mv ~/.hermes/plugins/gcp-memory-bank ~/.hermes/plugins/gcp-memory-bank.v1.bak

# 2. Install v2 (symlink keeps the projects/ checkout authoritative)
ln -s /Users/jithendranara/projects/gcp-memory-bank/hermes-plugin-v2 \
      ~/.hermes/plugins/gcp-memory-bank

# 3. Verify
hermes gcp-memory-bank doctor

# 4. Audit existing memories for drift
hermes gcp-memory-bank audit

# 5. (If needed) consolidate the `8405386815` and `hermes-user` shards
hermes gcp-memory-bank scope-migrate --from-user hermes-user --to-user jithendra
hermes gcp-memory-bank scope-migrate --from-user 8405386815 --to-user jithendra --force

# 6. Restart Hermes
```

## Live audit findings this addresses

From the runtime log scan (4 days, 91 inits):

- ✅ **3 different `user_id` values** — `hermes-user`, `8405386815` (Telegram chat id), `jithendra`. v2's `resolve_user_id` rejects numeric-only ids; `scope-migrate` CLI re-keys old memories.
- ✅ **18 sessions created, 11 ended, 7 leaked**. v2 `session_reuse=true` keeps one session per process; `atexit` flushes; `skip_empty_session_end=true` skips the 2/11 wasted round-trips.
- ✅ **Tools never invoked** in 4 days. v2 `system_prompt_block` rewords the tool guidance to "Use memory_search FIRST when the user references past context."
- ✅ **`memory_synthesize` was fake**. v2 calls Gemini for real (with join fallback if google-genai isn't installed).
- ✅ **No background-op completion logs**. v2 `timed()` wraps every daemon thread with start / done / fail + ms.
- ✅ **`generate_every_n_turns: 0`** kept memories stuck at session-end-only. v2 default is `3`.
- ✅ **`[ADD user]` prefix poisoning** in mirrored writes. v2 drops the prefix.
- ✅ **No scope-drift warnings**. v2 `ScopeDriftDetector` logs once per (user, app, engine) tuple, screams on change.
- ✅ **Hindsight + gcp-memory-bank both configured**. `doctor` now flags this.

## Configuration reference

Wizard-prompted (minimal):

| Key | Default | Notes |
|---|---|---|
| `project_id` | — | Required |
| `location` | `us-central1` | |
| `engine_id` | — | Auto-provisioned if blank via `instance create` |
| `user_id` | — | If empty, resolved from kwargs but **rejects numeric-only** |
| `app_name` | `hermes` | |

Everything else lives in `~/.hermes/gcp-memory-bank.json`:

| Key | Default |
|---|---|
| `scope_keys` | `["app_name", "user_id"]` |
| `scope_template` | `{"app_name":"{app}", "user_id":"{user}"}` |
| `recall_mode` | `hybrid` |
| `recall_budget` | `mid` (top_k=8) |
| `recall_detail` | `L1` |
| `trivial_skip` | `true` |
| `auto_prefetch` | `true` |
| `prefetch_mode` | `facts` |
| `use_gcp_sessions` | `true` |
| `gcp_session_ttl_seconds` | `86400` |
| `session_reuse` | `true` |
| `skip_empty_session_end` | `true` |
| `generate_every_n_turns` | `3` |
| `generation_model` | `gemini-3.1-pro-preview` |
| `embedding_model` | `gemini-embedding-001` |
| `synthesis_model` | `gemini-2.5-flash` |
| `create_ttl_days` / `generate_created_ttl_days` / `revision_ttl_days` | `365` |
| `circuit_breaker.threshold` / `cooldown_seconds` | `5` / `120` |
| `lro_poll_max_seconds` | `60` |
| `mirror_memory_md_writes` | `true` |
| `mirror_drop_action_prefix` | `true` |
| `default_revision_labels` | `{}` |
| `primary_only` | `true` |
| `skip_contexts` | `["cron", "flush", "subagent"]` |
| `consolidation_revisions_per_candidate` | `5` |
| `enable_third_person_memories` | `false` |
| `disable_memory_revisions` | `false` |

## CLI

```
hermes gcp-memory-bank status
hermes gcp-memory-bank doctor                # ADC, SDK, engine reachability, dual-provider warning
hermes gcp-memory-bank audit                 # NEW: scope drift report + leaked sessions
hermes gcp-memory-bank scope [--set k=tmpl ...]
hermes gcp-memory-bank scope-migrate --from-user X --to-user Y [--force]   # NEW: re-key memories
hermes gcp-memory-bank instance describe / create / update-config          # update-config NEW
hermes gcp-memory-bank topics list
hermes gcp-memory-bank revisions list MEMORY_ID [--label k=v]
hermes gcp-memory-bank revisions get MEMORY_ID REVISION_ID
hermes gcp-memory-bank rollback MEMORY_ID REVISION_ID
hermes gcp-memory-bank purge --filter EXPR [--force]
hermes gcp-memory-bank sessions list / delete
hermes gcp-memory-bank iam check
```

## Tools surface

| Name | Notes |
|---|---|
| `memory_search` | Adds `topics` + `since` filter on top of v1 |
| `memory_store` | Verbatim write |
| `memory_profile` | Scope-bound list |
| `memory_get` / `memory_delete` | Unchanged from v1 |
| `memory_revisions` / `memory_revision_get` / `memory_rollback` | Unchanged |
| `memory_purge` | Filter is ALWAYS scope-bound when omitted (never cross-user) |
| `memory_ingest` | Routes through proven CreateMemory fallback (since `ingest_events` SDK is broken) |
| `memory_synthesize` | **Real Gemini synthesis** with join fallback |

## Tests

```bash
GMB_PLUGIN_DIR=/Users/jithendranara/projects/gcp-memory-bank/hermes-plugin-v2 \
  python3 -m pytest /Users/jithendranara/projects/gcp-memory-bank/hermes-plugin-v2/tests/ -v
```

38 tests: identity, availability, user_id guardrails, scope drift, fence + sanitize, trivial skip, agent_context gating, all 11 tools, sync_turn non-blocking, mid-session generation, session-end (incl. empty-skip), pre-compress, on_memory_write (no prefix), real synthesize fallback, topic build (correct nested schema), system prompt, recall_mode gating, circuit breaker.

## SDK quirk reference (preserved from v1's TEST_RESULTS.md)

| Call | Status |
|---|---|
| `memories.generate(vertex_session_source=...)` | ✅ Works (sync, 15-35s) |
| `memories.create(...)` | ✅ Works (immediate) |
| `memories.generate(direct_contents_source=...)` | ❌ Silently fails — DO NOT USE |
| `memories.generate(config={"wait_for_completion": False})` | ❌ Returns `done=None`, never processes |
| `memories.ingest_events(...)` | ❌ Returns `done=None`, never processes |
| `sessions.create(name=engine)` | ✅ Works |
| `sessions.events.append(...)` | ✅ Works (200-500ms) |

v2 honours these. The only working extraction path is `vertex_session_source`; everything else falls back to per-event `CreateMemory`.

## License

MIT.
