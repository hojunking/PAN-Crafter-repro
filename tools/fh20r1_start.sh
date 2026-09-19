#!/usr/bin/env bash
# Explicit start only: CORE completion + >=20 qualified hours, no hard deadline.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"
[ -x "$PY" ] || PY=python
exec "$PY" tools/fh20r1_runner.py start "$@"
