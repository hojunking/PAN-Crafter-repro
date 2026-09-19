#!/usr/bin/env bash
# Explicit activation only; no source checkout, no existing trainer termination.
set -euo pipefail
QG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QG_PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"
[ -x "$QG_PY" ] || QG_PY=python
export PYTHONDONTWRITEBYTECODE=1
cd "$QG_ROOT"
exec "$QG_PY" tools/qg40_runner.py start "$@"
