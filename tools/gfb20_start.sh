#!/usr/bin/env bash
# Same explicit t0 on s3/s4/s5. No implicit restart of the twenty-hour window.
set -euo pipefail
GFB20_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GFB20_PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
cd "$GFB20_ROOT"
exec "$GFB20_PY" tools/gfb20_runner.py start "$@"
