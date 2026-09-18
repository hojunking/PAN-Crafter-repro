#!/usr/bin/env bash
# One command: infer server, generate/rebase config, archive legacy hold,
# start the finite runner with automatic evaluation/upload. First start +20h.
#   bash tools/mix20h_switch.sh
# Preview only: --dry-run. Shared clock remains optional: --start-at <UTC T0>.
# No kill, fresh retry, automatic time extension, or cron installation.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"
[ -x "$PY" ] || PY=python
exec "$PY" tools/mix20h_runner.py start "$@"
