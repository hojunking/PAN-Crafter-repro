#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^s[123]$ ]]; then
  echo 'Usage: bash scripts/run_edgeres48.sh s1|s2|s3 [extra run arguments]' >&2
  exit 2
fi
server="$1"
shift
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "${PAN_PYTHON:-python}" -m edgeres48 run --server "$server" "$@"
