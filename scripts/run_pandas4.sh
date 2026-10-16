#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 0 ]]; then
  echo 'Usage: bash scripts/run_pandas4.sh (s4 WV3, max40h)' >&2
  exit 2
fi
pandas4_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pandas4_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "${PAN_PYTHON:-python}" -m pandas4 run
