#!/usr/bin/env bash
# Explicit invocation; no other campaign is stopped or replaced.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/run_panda_next30.sh s1|s3 (WV3, mature donor + Lanczos4/sampling/PAN-difference Stage2, unlimited queue)' >&2
  exit 2
fi
next30_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$next30_root"
if [[ ! -f gspread/server.txt || "$(tr -d '[:space:]' < gspread/server.txt)" != "$1" ]]; then
  echo 'Local server marker must match requested s1/s3; nothing started.' >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec "${PAN_PYTHON:-python}" -m panda_next30 run --server "$1"
