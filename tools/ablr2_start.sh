#!/usr/bin/env bash
# --lease-hours 72 is explicit initial/renewed operator authorization, not auto-renewal.
set -euo pipefail
ABLR2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLR2_PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
cd "$ABLR2_ROOT"
exec "$ABLR2_PY" tools/ablr2_runner.py start "$@"
