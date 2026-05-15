#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x .venv/bin/python ]; then
  PYTHON_BIN=".venv/bin/python"
fi
"$PYTHON_BIN" -m pytest tests -q
