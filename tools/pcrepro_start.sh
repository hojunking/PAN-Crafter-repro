#!/usr/bin/env bash
set -euo pipefail
PCREPRO_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PCREPRO_PYTHON="${PANCRAFTER_PYTHON:-python}"
exec "$PCREPRO_PYTHON" "$PCREPRO_REPO/tools/pcrepro_runner.py" start "$@"
