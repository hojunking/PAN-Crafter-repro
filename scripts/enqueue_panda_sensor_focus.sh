#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/enqueue_panda_sensor_focus.sh s1|s3' >&2
  exit 2
fi
focus_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$focus_root"
focus_python="${PAN_HOST_PYTHON:-}"
if [[ -z "$focus_python" ]]; then
  focus_base="$(command -v python || command -v python3)"
  focus_prefix="$("$focus_base" -c 'import sys; print(sys.base_prefix)')"
  for focus_candidate in "$focus_base" "$focus_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$focus_candidate" ]] && "$focus_candidate" -c 'import torch,google.oauth2.service_account,requests' >/dev/null 2>&1; then
      focus_python="$focus_candidate"
      break
    fi
  done
fi
if [[ -z "$focus_python" ]]; then
  echo 'Set PAN_HOST_PYTHON to the existing training/uploader Python.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 PAN_HOST_PYTHON="$focus_python"
exec "$focus_python" -m panda_sensor_focus.queue --server "$1" --enqueue
