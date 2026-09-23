#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PANCRAFTER_PYTHON:-python}" "$script_dir/g23sens_docker_start.py" "$@"
