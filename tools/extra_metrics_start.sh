#!/usr/bin/env bash
# Run from any directory; retain current Git HEAD and all training files.
set -euo pipefail
EXTRA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTRA_PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"
[ -x "$EXTRA_PY" ] || EXTRA_PY=python
export PYTHONDONTWRITEBYTECODE=1
exec "$EXTRA_PY" "$EXTRA_ROOT/tools/extra_metrics.py" start "$@"
