#!/usr/bin/env bash
# Register a detached local successor without touching or stopping ZERO30.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/start_panda_scale30_after_zero30.sh s1|s3' >&2
  exit 2
fi
panda_wait_server="$1"
panda_wait_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$panda_wait_root"
if [[ ! -f gspread/server.txt || "$(tr -d '[:space:]' < gspread/server.txt)" != "$panda_wait_server" ]]; then
  echo 'Local server marker mismatch; no successor registered.' >&2
  exit 1
fi
panda_wait_python="${PAN_HOST_PYTHON:-}"
if [[ -z "$panda_wait_python" ]]; then
  panda_wait_base="$(command -v python || command -v python3)"
  panda_wait_prefix="$("$panda_wait_base" -c 'import sys; print(sys.base_prefix)')"
  for panda_wait_candidate in "$panda_wait_base" "$panda_wait_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$panda_wait_candidate" ]] && "$panda_wait_candidate" -c 'import torch, google.oauth2.service_account, google.auth.transport.requests, requests' >/dev/null 2>&1; then
      panda_wait_python="$panda_wait_candidate"
      break
    fi
  done
fi
if [[ -z "$panda_wait_python" ]]; then
  echo 'Set PAN_HOST_PYTHON to the existing environment with torch, google-auth and requests.' >&2
  exit 1
fi
command -v setsid >/dev/null
panda_wait_directory="$panda_wait_root/work_dir/panda_scale30_launchers/$panda_wait_server"
mkdir -p "$panda_wait_directory"
exec 8>"$panda_wait_directory/.register.lock"
flock -n 8 || { echo 'Another successor registration is active.' >&2; exit 1; }
if [[ -f "$panda_wait_directory/pid" ]]; then
  panda_wait_previous_pid="$(cat "$panda_wait_directory/pid")"
  if [[ "$panda_wait_previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$panda_wait_previous_pid" 2>/dev/null; then
    if [[ -r "/proc/$panda_wait_previous_pid/cmdline" ]] && tr '\0' ' ' < "/proc/$panda_wait_previous_pid/cmdline" | grep -Fq 'panda_scale30.wait_previous'; then
      echo "Scale30 successor watcher already active: PID $panda_wait_previous_pid"
      exit 0
    fi
  fi
fi
PAN_HOST_PYTHON="$panda_wait_python" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  setsid "$panda_wait_python" -m panda_scale30.wait_previous --server "$panda_wait_server" \
  >> "$panda_wait_directory/watcher.log" 2>&1 < /dev/null 8>&- &
panda_wait_pid=$!
echo "$panda_wait_pid" > "$panda_wait_directory/pid"
sleep 2
if ! kill -0 "$panda_wait_pid" 2>/dev/null; then
  echo "Successor watcher failed; inspect $panda_wait_directory/watcher.log" >&2
  exit 1
fi
echo "Registered Scale30 successor PID $panda_wait_pid. Receipt: $panda_wait_directory/receipt.json"
echo 'The 30h budget begins with cost profiling after ZERO30 succeeds and the GPU is idle.'
