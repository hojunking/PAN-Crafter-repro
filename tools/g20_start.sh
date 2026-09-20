#!/usr/bin/env bash
# Explicit activation only. No implicit t0, checkout, or trainer termination.
set -euo pipefail
G20_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G20_PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"
[ -x "$G20_PY" ] || G20_PY=python
export PYTHONDONTWRITEBYTECODE=1
cd "$G20_ROOT"
exec "$G20_PY" tools/g20_runner.py start "$@"
