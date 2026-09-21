#!/usr/bin/env bash
# PYTHON chooses the HOST interpreter used to verify/freeze the committed release.
set -euo pipefail
ABLR2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLR2_PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
cd "$ABLR2_ROOT"
exec "$ABLR2_PY" tools/ablr2_docker_start.py "$@"
