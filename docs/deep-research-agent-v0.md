# Deep Research Agent v0 Scaffold

**Date (ET):** 2026-05-05

## Goal

Create a small, testable research-agent core that can later plug into GCP Memory Bank, Gemini Search grounding, Gemini Deep Research, MiniMax search, or browser-based fetchers without coupling the orchestration logic to one API.

## What shipped

- `src/deep_research_agent/` — dependency-light Python package.
- `src/deep_research_agent/adapters.py` — callable SearchProvider/MemoryStore adapters plus live-search result normalization helpers.
- `src/deep_research_agent/file_adapters.py` — JSON-fixture SearchProvider and JSONL FileMemoryStore for offline/CI runs.
- `src/deep_research_agent/report.py` — Markdown/JSON report renderer with citation-preserving output.
- `src/deep_research_agent/cli.py` — runnable v0 CLI that writes `.md` and `.json` reports.
- `src/deep_research_agent/promotion.py` — deterministic long-term-memory promotion policy and MemoryStore decorator.
- `tests/test_deep_research_agent.py`, `tests/test_deep_research_agent_cli.py`, and `tests/test_deep_research_agent_runtime_adapters.py` — offline deterministic tests.
- `examples/deep_research_agent_demo.py` — runnable offline demo.
- `examples/search_fixture.json` — CLI/demo search fixture.

## Architecture

The scaffold separates eight responsibilities:

1. **Planning** — `planner.py` turns a question into a compact fan-out plan: `scope`, `current`, `tradeoffs`.
2. **Search** — `SearchProvider` protocol accepts any backend that returns `SearchHit` objects.
3. **Memory** — `MemoryStore` protocol supports recall and report persistence; v0 ships `NullMemoryStore`, `InMemoryStore`, and `ProfileAugmentedMemoryStore` for structured profile context.
4. **Structured profiles** — `normalize_profile_result()` flattens GCP Memory Bank `memory_profiles` / `retrieve_profiles(...)` output into durable profile facts that get prepended to recall context.
5. **Evidence normalization** — `DeepResearchAgent.collect_evidence()` converts search hits into cited evidence and deduplicates URLs.
6. **Synthesis** — `Synthesizer` protocol supports provider-backed synthesis later; v0 ships deterministic `ExtractiveSynthesizer` for CI.
7. **Runtime adapters** — `CallableSearchProvider` and `CallableMemoryStore` can wrap Hermes/GCP/web-search tool callables without coupling core code to injected tools.
8. **Report persistence** — the CLI writes Markdown and JSON reports under `reports/deep-research/` while optionally appending durable report facts to a JSONL memory file.
9. **Memory promotion** — `MemoryPromotionPolicy` and `PromotingMemoryStore` prevent weak, uncited, or failure-report output from entering long-term memory.

## Why this shape

- Provider-specific adapters can fail independently without breaking the core planner.
- Offline CI can prove orchestration without network/API keys.
- GCP Memory Bank can be introduced through the `MemoryStore` protocol.
- Gemini Deep Research or Vertex/Gemini grounded generation can be introduced through either `SearchProvider` or `Synthesizer` depending on the final product shape.

## CLI smoke test

```bash
PYTHONPATH=src python3 -m deep_research_agent.cli \
  --question 'How should Deep Research Agent v0 connect memory, search, and reports?' \
  --search-fixture examples/search_fixture.json \
  --memory-file /tmp/deep_research_agent_memory.jsonl \
  --output-dir reports/deep-research \
  --slug smoke-cli
# writes reports/deep-research/smoke-cli.md and reports/deep-research/smoke-cli.json
```

## Verification

Local targeted command:

```bash
PYTHONPATH=src python3 -m pytest tests/test_deep_research_agent.py tests/test_deep_research_agent_cli.py tests/test_deep_research_agent_runtime_adapters.py -q
# 20 passed in 0.02s
```

Local full-suite command, using the Hermes venv because system Python lacks project dependencies:

```bash
PYTHONPATH=src /Users/jithendranara/.hermes/hermes-agent/venv/bin/python3 -m pytest tests -q
# 40 passed, 1 warning in 1.45s
```

HP/HermesBox targeted command:

```bash
/Users/jithendranara/.hermes/tools/hermes_box.py run --repo /Users/jithendranara/projects/gcp-memory-bank -- scripts/run_deep_research_agent_tests.sh
# 17 passed, 1 warning in 0.05s
```

HP/HermesBox full-suite command after creating `.venv` and installing project test/runtime deps on HP:

```bash
/Users/jithendranara/.hermes/tools/hermes_box.py run --repo /Users/jithendranara/projects/gcp-memory-bank -- scripts/run_all_tests.sh
# 37 passed, 22 warnings in 3.18s
```

Demo command:

```bash
python3 examples/deep_research_agent_demo.py
```

## Runtime adapter smoke test

`tests/test_deep_research_agent_runtime_adapters.py` verifies these live-wiring boundaries without real network calls:

- `CallableSearchProvider` normalizes MiniMax-style `organic` results with `link`, `snippet`, and `date` fields.
- `normalize_profile_result` flattens GCP Memory Bank `memory_profiles` / `retrieve_profiles(...)` payloads into deterministic profile facts.
- `ProfileAugmentedMemoryStore` prepends structured profile facts to normal recall context without changing report persistence.
- `CitationPreservingSynthesizer` wraps a provider callable and appends missing `[Source: URL]` citations if the provider omits evidence URLs.
- `MemoryPromotionPolicy` blocks weak/uncited/failure reports before persistence.
- `PromotingMemoryStore` decorates any MemoryStore, including future GCP Memory Bank stores, with that policy.

## Next adapter tasks

1. Add a tiny Hermes runtime bridge that instantiates `CallableSearchProvider(search_fn=mcp_minimax_web_search_or_other_tool)`, `ProfileAugmentedMemoryStore(profile_facts_fn=lambda: normalize_profile_result(memory_profiles()))`, and `PromotingMemoryStore(inner=CallableMemoryStore(...))` inside agent sessions.
2. Add a provider-backed synthesizer callable using the active model path, wrapped by `CitationPreservingSynthesizer`.
3. Add a direct `GcpMemoryBankStore` using the existing Hermes plugin/client surface when running outside the real Hermes tool-injection environment.
4. Add one guarded live E2E command that writes a report only after search, synthesis, citation preservation, profile-context injection, promotion-policy, and ops logging all pass.
