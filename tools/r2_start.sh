#!/usr/bin/env bash
# Explicit R2 priority handoff; keep the current campaign, source and Git HEAD.
set -euo pipefail
R2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R2_PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"
[ -x "$R2_PY" ] || R2_PY=python
export PYTHONDONTWRITEBYTECODE=1
cd "$R2_ROOT"
exec "$R2_PY" tools/r2_runner.py start "$@"
