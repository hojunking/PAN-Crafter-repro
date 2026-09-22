#!/usr/bin/env bash
set -euo pipefail
GFP40_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GFP40_PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
cd "$GFP40_ROOT"
exec "$GFP40_PY" tools/gfp40_runner.py start "$@"
