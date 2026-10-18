#!/usr/bin/env bash
# Explicit invocation only. Existing experiments are never interrupted.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s(1|3|4)$ ]]; then
  echo 'Usage: bash scripts/run_panda30r.sh s1|s3|s4 (WV3, max30h)' >&2
  exit 2
fi
panda30r_server="$1"
panda30r_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$panda30r_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "${PAN_PYTHON:-python}" -m panda30r run --server "$panda30r_server"
