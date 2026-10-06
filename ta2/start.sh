#!/usr/bin/env bash
# Portable explicit launch. Pulling code alone never starts/stops a process.
set -euo pipefail
ta2_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec "${TA2_HOST_PYTHON:-python3}" "$ta2_repo/ta2/launch.py" "$@"
