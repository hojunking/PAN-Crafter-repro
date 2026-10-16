#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s(1|3)$ ]]; then
  echo 'Usage: bash scripts/run_donor24.sh s1|s3 (both WV3)' >&2
  exit 2
fi
donor24_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$donor24_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "${PAN_PYTHON:-python}" -m donor24 run --server "$1"
