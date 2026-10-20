#!/usr/bin/env bash
# Explicit invocation; no other campaign is stopped or replaced.
set -euo pipefail
if [[ $# -ne 1 || "$1" != s1 ]]; then
  echo 'Usage: bash scripts/run_panda_band30.sh s1 (WV3, fresh Stage1 + warp cases, 30h)' >&2
  exit 2
fi
band30_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$band30_root"
if [[ ! -f gspread/server.txt || "$(tr -d '[:space:]' < gspread/server.txt)" != s1 ]]; then
  echo 'Local server marker must be s1; nothing started.' >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "${PAN_PYTHON:-python}" -m panda_band30 run --server s1
