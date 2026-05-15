#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x .venv/bin/python ]; then
  PYTHON_BIN=".venv/bin/python"
fi
"$PYTHON_BIN" -m pytest tests/test_deep_research_agent.py tests/test_deep_research_agent_cli.py tests/test_deep_research_agent_runtime_adapters.py -q
