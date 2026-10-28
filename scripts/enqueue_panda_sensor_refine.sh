#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/enqueue_panda_sensor_refine.sh s1|s3' >&2
  exit 2
fi
refine_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$refine_root"
refine_python="${PAN_HOST_PYTHON:-}"
if [[ -z "$refine_python" ]]; then
  refine_base="$(command -v python || command -v python3)"
  refine_prefix="$("$refine_base" -c 'import sys; print(sys.base_prefix)')"
  for refine_candidate in "$refine_base" "$refine_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$refine_candidate" ]] && "$refine_candidate" -c 'import torch,google.oauth2.service_account,requests' >/dev/null 2>&1; then
      refine_python="$refine_candidate"
      break
    fi
  done
fi
if [[ -z "$refine_python" ]]; then
  echo 'Set PAN_HOST_PYTHON to the existing training/uploader Python.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 PAN_HOST_PYTHON="$refine_python"
exec "$refine_python" -m panda_sensor_refine.queue --server "$1" --enqueue
