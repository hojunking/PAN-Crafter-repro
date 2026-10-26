#!/usr/bin/env bash
# One command: measured cost sidecars, then unchanged training and publication.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/start_panda_hard36_costed.sh s1|s3' >&2
  exit 2
fi
panda_costed_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$panda_costed_root"
panda_follow_host_python="${PAN_HOST_PYTHON:-}"
if [[ -z "$panda_follow_host_python" ]]; then
  panda_follow_base_python="$(command -v python || command -v python3)"
  panda_follow_base_prefix="$("$panda_follow_base_python" -c 'import sys; print(sys.base_prefix)')"
  for panda_follow_candidate in "$panda_follow_base_python" "$panda_follow_base_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$panda_follow_candidate" ]] && "$panda_follow_candidate" -c 'import torch, google.oauth2.service_account, requests' >/dev/null 2>&1; then
      panda_follow_host_python="$panda_follow_candidate"
      break
    fi
  done
fi
if [[ -z "$panda_follow_host_python" ]]; then
  echo 'Set PAN_HOST_PYTHON to the existing training/uploader Python.' >&2
  exit 1
fi
export PAN_HOST_PYTHON="$panda_follow_host_python" PYTHONDONTWRITEBYTECODE=1
"$PAN_HOST_PYTHON" -m panda_hard36.transition --server "$1" --prepare-only --require-complete --check-container
mkdir -p "work_dir/panda_hard36_launchers/$1"
exec 9>"work_dir/panda_hard36_launchers/$1/.start.lock"
flock -n 9 || { echo 'Another Hard36 startup is active; nothing started.' >&2; exit 1; }
bash scripts/prepare_panda_hard36_costs.sh "$1"
bash scripts/run_panda_hard36_docker.sh "$1"
