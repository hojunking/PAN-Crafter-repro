#!/usr/bin/env bash
# One explicit action; safe drain, local preflight/calibration, immutable 12h.
# Pulling this file or generating configs never starts an experiment.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"
[ -x "$PY" ] || PY=python
exec "$PY" tools/fh12_runner.py start "$@"
