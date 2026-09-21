#!/usr/bin/env bash
# One command per server, same actual --t0. No implicit clock or legacy kill.
set -euo pipefail
L100_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
L100_PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
cd "$L100_ROOT"
exec "$L100_PY" tools/l100_runner.py start "$@"
