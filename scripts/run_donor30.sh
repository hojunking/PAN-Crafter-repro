#!/usr/bin/env bash
# Training controller only; the Docker launcher starts the host publisher.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s(1|3)$ ]]; then
  echo 'Usage: bash scripts/run_donor30.sh s1|s3' >&2
  exit 2
fi
donor_server="$1"
donor_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$donor_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MKL_THREADING_LAYER=GNU
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec "${PAN_PYTHON:-python}" -m donor30 run --server "$donor_server"
