#!/usr/bin/env bash
# Training only. No Google publisher, process termination, or checkpoint reuse.
set -euo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^s(1|3)$ ]]; then
  echo 'Usage: bash scripts/run_directshift48.sh s1|s3 [extra run arguments]' >&2
  exit 2
fi
direct_server="$1"
shift
direct_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$direct_root"
direct_args=("$@")
direct_device=cuda
for ((i=0; i<${#direct_args[@]}; i++)); do
  case "${direct_args[i]}" in
    --device)
      if ((i+1 >= ${#direct_args[@]})); then
        echo 'Missing --device value.' >&2
        exit 2
      fi
      direct_device="${direct_args[i+1]}"
      i=$((i+1))
      ;;
    --device=*) direct_device="${direct_args[i]#--device=}";;
  esac
done
if [[ "$direct_device" == cuda* ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 'Cannot verify GPU ownership: nvidia-smi is unavailable. Training was not started.' >&2
    exit 1
  fi
  if ! direct_gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"; then
    echo 'Cannot verify GPU ownership: nvidia-smi query failed. Training was not started.' >&2
    exit 1
  fi
  if [[ -n "${direct_gpu_processes//[[:space:]]/}" ]]; then
    echo 'A GPU compute workload is still active. No process was stopped.' >&2
    echo "$direct_gpu_processes" >&2
    echo 'Finish or explicitly stop the intended old experiment, then run this command again.' >&2
    exit 1
  fi
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MKL_THREADING_LAYER=GNU
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec "${PAN_PYTHON:-python}" -m directshift48 run --server "$direct_server" "${direct_args[@]}"
