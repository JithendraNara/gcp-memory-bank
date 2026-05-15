# gcp-memory-bank — Hermes Plugin

**Plugin:** `~/.hermes/plugins/gcp-memory-bank/__init__.py`
**Engine:** `4938048007586185216` (`hermes-memory-global-3-1-test`)
**SDK:** `google-cloud-aiplatform==1.149.0` (Hermes venv at `/Users/jithendranara/.hermes/hermes-agent/venv`)

## Installed Package Versions

```
google-cloud-aiplatform==1.149.0   # Memory Bank API client + proto definitions
google-genai==1.73.1               # Gemini API (not used directly by this plugin)
mistralai==2.4.1                   # Third-party; pins otel-semantic-conventions<0.61
opentelemetry-api==1.39.1          # Resolved: satisfies mistralai + gcp exporters
opentelemetry-sdk==1.39.1           # Resolved: satisfies gcp exporters
opentelemetry-semantic-conventions==0.60b1  # Resolved: satisfies mistralai
opentelemetry-instrumentation==0.60b1       # + fastapi/asgi/aiohttp-client @ 0.60b1
opentelemetry-exporter-prometheus==0.60b1
opentelemetry-exporter-gcp-logging==1.12.0a0
opentelemetry-resourcedetector-gcp==1.12.0a0
opentelemetry-exporter-gcp-trace==1.12.0
```

`pip check` passes cleanly — all constraints satisfied.

## SDK Import Path

The proto classes live here (the `vertexai.types` import from forum posts does NOT exist in 1.149.0):

```python
from google.cloud.aiplatform_v1beta1.types.reasoning_engine import ReasoningEngineContextSpec
MemoryBankConfig = ReasoningEngineContextSpec.MemoryBankConfig
```

## Engine Config (Production)

Retrieved from `GET /reasoningEngines/4938048007586185216`:
- Generation model: `MiniMax-M2.7` (active, May 2026) | Previously `gemini-3.1-pro-preview`
- Embedding model: `gemini-embedding-001` (global path)
- TTL: 365 days
- Custom topics: `TECHNICAL_DECISIONS`, `PROJECT_CONTEXT`, `CORRECTED_MISTAKES`

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

```bash
hermes memory status   # shows the active memory provider, including gcp-memory-bank when selected
hermes doctor          # Hermes CLI health check; includes active memory-provider diagnostics
```

Plugin maintenance commands live in the plugin's own docs/CLI entry points, not as Hermes top-level subcommands. The supported runtime surface for Hermes users is the memory provider itself: `memory_search`, `memory_store`, `memory_profile`, `memory_get`, `memory_delete`, `memory_revisions`, `memory_revision_get`, `memory_rollback`, `memory_purge`, `memory_ingest`, and `memory_synthesize`.

## Tools surface

| Name | Notes |
|---|---|
| `memory_search` | Adds `topics` + `since` filter on top of v1 |
| `memory_store` | Verbatim write |
| `memory_profile` / `memory_profiles` | Profile: scope-bound list. **Profiles: structured schema-shaped snapshot** via `retrieve_profiles(...)` (12 tools total) |
| `memory_get` / `memory_delete` | Unchanged from v1 |
| `memory_revisions` / `memory_revision_get` / `memory_rollback` | Unchanged |
| `memory_purge` | Filter is ALWAYS scope-bound when omitted (never cross-user). **SDK quirk:** `force` must be a top-level kwarg, NOT inside `config` — `config={"force": True}` raises `PurgeAgentEngineMemoriesConfig extra_forbidden` |
| `memory_ingest` | Routes through proven CreateMemory fallback. ⚠️ `ingest_events(...)` SDK returns a non-cancellable LRO that blocks engine deletion — do not use for production ingestion |
| `memory_synthesize` | **Real Gemini synthesis** with join fallback |

## Tests

```bash
GMB_PLUGIN_DIR=/Users/jithendranara/projects/gcp-memory-bank/hermes-plugin-v2 \
  python3 -m pytest /Users/jithendranara/projects/gcp-memory-bank/hermes-plugin-v2/tests/ -v
```

47 tests: identity, availability, user_id guardrails, scope drift, fence + sanitize, trivial skip, agent_context gating, all 11 tools, sync_turn non-blocking, mid-session generation, session-end (incl. empty-skip), pre-compress, on_memory_write (no prefix), real synthesize fallback, topic build (correct nested schema), system prompt, recall_mode gating, circuit breaker, session-list filtering, transport close cleanup.

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

## SDK version + customization features

**Verified May 5, 2026:** Hermes venv has `google-cloud-aiplatform==1.149.0` installed. Memory Bank TTL / generation-model / embedding-model config classes are available, but **not** through the forum-style `vertexai.types` import path. Use the nested proto classes instead:

```python
from google.cloud.aiplatform_v1beta1.types.reasoning_engine import ReasoningEngineContextSpec
from google.protobuf.duration_pb2 import Duration

MemoryBankConfig = ReasoningEngineContextSpec.MemoryBankConfig
TtlConfig = MemoryBankConfig.TtlConfig
GenerationConfig = MemoryBankConfig.GenerationConfig
SimilaritySearchConfig = MemoryBankConfig.SimilaritySearchConfig

memory_bank_config = MemoryBankConfig(
    ttl_config=TtlConfig(default_ttl=Duration(seconds=2_592_000)),  # 30 days
    generation_config=GenerationConfig(
        model="projects/festive-antenna-463514-m8/locations/us-central1/publishers/google/models/gemini-2.5-flash"
    ),
    similarity_search_config=SimilaritySearchConfig(
        embedding_model="projects/festive-antenna-463514-m8/locations/us-central1/publishers/google/models/text-multilingual-embedding-002"
    ),
)
```

Validated locally with `/Users/jithendranara/.hermes/hermes-agent/venv/bin/python3`: `MemoryBankConfig(...)` constructs successfully. Custom topic / few-shot config classes were **not** found in the 1.149.0 Python package under the documented names, so do not wire those into production until the actual SDK symbols are verified.

## REST API endpoint reference

The GCP Memory Bank engine supports a subset of REST API endpoints. Direct REST access is useful for health checks and debugging, but the SDK (`MemoryBankClient`) is the reliable path for all read/write operations.

**Base URL:** `https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/us-central1`

**Supported REST endpoints:**

| Endpoint | Method | Auth | Status |
|---|---|---|---|
| `reasoningEngines` (ListEngines) | GET | ADC token | ✅ Works |
| `reasoningEngines/{id}` (GetEngine) | GET | ADC token | ✅ Works |
| `reasoningEngines/{id}/memories` (ListMemories) | GET | ADC token | ✅ Works |
| `sessions` (ListSessions) | GET | ADC token | ✅ Works |
| `reasoning:searchMemory` | POST | ADC token | ❌ **404 — wrong endpoint path** |
| `retrieveMemories` (REST) | POST | ADC token | ❌ **404 — use SDK proto path instead** |

**Key insight:** The `reasoning:searchMemory` and bare `retrieveMemories` REST endpoints return 404. The SDK's search works via the **proto/gRPC client** (`MemoryBankServiceClient`), which uses a different transport path. For any health checks that involve search/retrieve, use the SDK:

```python
from gcp_memory_bank.client import MemoryBankClient
client = MemoryBankClient(project='...', location='us-central1', engine_id='...')
results = client.retrieve(scope={...}, query='...', top_k=3)
```

Or use the plugin tools: `memory_search` (via Hermes tools) which call the SDK correctly.

**Correct REST health check (read-only):**
```bash
TOKEN=$(gcloud auth print-access-token)
ENGINE_ID="4938048007586185216"
PROJECT_ID="festive-antenna-463514-m8"

# Check engine exists
curl -s "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/us-central1/reasoningEngines/${ENGINE_ID}" \
  -H "Authorization: Bearer ${TOKEN}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('Engine:', d.get('displayName',''), '✓')"

curl -s "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/${PROJECT_ID}/locations/us-central1/reasoningEngines/${ENGINE_ID}/memories?page_size=3" \
  -H "Authorization: Bearer ${TOKEN}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); [print(' -', m.get('fact','')[:80]) for m in d.get('memories',[])]"

```

**DO NOT use** `reasoning:searchMemory` for REST health checks — it returns 404. The SDK `retrieve()` method works correctly and is the authoritative search path.

## License

MIT.
