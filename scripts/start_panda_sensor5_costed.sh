#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/start_panda_sensor5_costed.sh s1|s3' >&2
  exit 2
fi
sensor5_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$sensor5_root"
sensor5_python="${PAN_HOST_PYTHON:-}"
if [[ -z "$sensor5_python" ]]; then
  sensor5_base="$(command -v python || command -v python3)"
  sensor5_prefix="$("$sensor5_base" -c 'import sys; print(sys.base_prefix)')"
  for sensor5_candidate in "$sensor5_base" "$sensor5_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$sensor5_candidate" ]] && "$sensor5_candidate" -c 'import torch,google.oauth2.service_account,requests' >/dev/null 2>&1; then
      sensor5_python="$sensor5_candidate"
      break
    fi
  done
fi
if [[ -z "$sensor5_python" ]]; then
  echo 'Set PAN_HOST_PYTHON to the existing training/uploader Python.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 PAN_HOST_PYTHON="$sensor5_python"
exec "$sensor5_python" -m panda_sensor5.launch --server "$1"
