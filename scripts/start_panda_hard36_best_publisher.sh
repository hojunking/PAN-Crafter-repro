#!/usr/bin/env bash
# Switch Sheet watcher only; no GPU use and no trainer stop/restart.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ! "$1" =~ ^s[13]$ || ( $# -eq 2 && "$2" != '--restart' ) ]]; then
  echo 'Usage: bash scripts/start_panda_hard36_best_publisher.sh s1|s3 [--restart]' >&2
  exit 2
fi
panda_best_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$panda_best_root"
panda_best_python="${PAN_HOST_PYTHON:-}"
if [[ -z "$panda_best_python" ]]; then
  panda_best_base="$(command -v python || command -v python3)"
  panda_best_prefix="$("$panda_best_base" -c 'import sys; print(sys.base_prefix)')"
  for panda_best_candidate in "$panda_best_base" "$panda_best_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$panda_best_candidate" ]] && "$panda_best_candidate" -c 'import google.auth.transport.requests, google.oauth2.service_account, requests' >/dev/null 2>&1; then
      panda_best_python="$panda_best_candidate"
      break
    fi
  done
fi
if [[ -z "$panda_best_python" ]]; then
  echo 'Set PAN_HOST_PYTHON to the existing uploader Python.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
panda_best_args=(--server "$1")
if [[ $# -eq 2 ]]; then
  panda_best_args+=(--restart)
fi
exec "$panda_best_python" -m panda_hard36.best_publication.restart "${panda_best_args[@]}"
