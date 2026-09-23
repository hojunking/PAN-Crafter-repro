#!/usr/bin/env bash
# Host interpreter verifies/freezes the release; --until-operator-stop is explicit.
set -euo pipefail
ABLR2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLR2_PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
cd "$ABLR2_ROOT"
exec "$ABLR2_PY" tools/ablr2_docker_start.py "$@"
