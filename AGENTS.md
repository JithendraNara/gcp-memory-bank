READ ~/AGENTS.MD BEFORE ANYTHING (skip if missing).

## Project: GCP Memory Bank

**What it does:** Production-grade Python SDK for Google Gemini Enterprise Agent Platform Memory Bank — async-first, type-safe long-term memory for AI agents with scoped isolation and structured retrieval.

**Core package:** `src/`
**CLI:** `PYTHONPATH=src python3 -m deep_research_agent.cli`
**Tests:** `PYTHONPATH=src /Users/jithendranara/.hermes/hermes-agent/venv/bin/python3 -m pytest tests -q`
**Demo:** `examples/deep_research_agent_demo.py`
**Design doc:** `docs/deep-research-agent-v0.md`

**Verify install:**
```bash
PYTHONPATH=src /Users/jithendranara/.hermes/hermes-agent/venv/bin/python3 -m pytest tests -q
```

**GCP Memory Bank facts:**
- Engine: `4938048007586185216`
- Project: `festive-antenna-463514-m8`
- Region: `us-central1`
- SDK: `google-cloud-aiplatform` 1.149.0 (proto path)
- REST: ListMemories works; search/retrieveMemories 404 → use SDK

**Key ops (from `gcp-memory-bank-ops` skill):**
- Test: `python3 -c "from google.cloud import aiplatform; print(aiplatform.__version__)"`
- Auth: `gcloud auth application-default-login --project=festive-antenna-463514-m8`
- Memory search: use `memory_search` tool (MCP) not REST search endpoint
- Ingest events: use per-event `memory_store` (not batch ingest_events which is unreliable in current SDK)