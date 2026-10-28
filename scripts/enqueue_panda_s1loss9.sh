#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/enqueue_panda_s1loss9.sh s1|s3' >&2
  exit 2
fi
s1loss9_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$s1loss9_root"
s1loss9_python="${PAN_HOST_PYTHON:-}"
if [[ -z "$s1loss9_python" ]]; then
  s1loss9_base="$(command -v python || command -v python3)"
  s1loss9_prefix="$("$s1loss9_base" -c 'import sys; print(sys.base_prefix)')"
  for s1loss9_candidate in "$s1loss9_base" "$s1loss9_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$s1loss9_candidate" ]] && "$s1loss9_candidate" -c 'import torch,google.oauth2.service_account,requests' >/dev/null 2>&1; then
      s1loss9_python="$s1loss9_candidate"
      break
    fi
  done
fi
if [[ -z "$s1loss9_python" ]]; then
  echo 'Set PAN_HOST_PYTHON to the existing training/uploader Python.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 PAN_HOST_PYTHON="$s1loss9_python"
exec "$s1loss9_python" -m panda_s1loss9.queue --server "$1" --enqueue
