#!/usr/bin/env bash
# Wait for the current schedule queue to finish normally; never stop or skip it.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/start_panda_sampling30_transition.sh s1|s3' >&2
  exit 2
fi
panda_transition_server="$1"
panda_transition_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$panda_transition_root"
if [[ ! -f gspread/server.txt || "$(tr -d '[:space:]' < gspread/server.txt)" != "$panda_transition_server" ]]; then
  echo 'Local server marker mismatch; nothing registered.' >&2
  exit 1
fi
panda_transition_python="${PAN_HOST_PYTHON:-}"
if [[ -z "$panda_transition_python" ]]; then
  panda_transition_base="$(command -v python || command -v python3)"
  panda_transition_prefix="$("$panda_transition_base" -c 'import sys; print(sys.base_prefix)')"
  for panda_transition_candidate in "$panda_transition_base" "$panda_transition_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$panda_transition_candidate" ]] && "$panda_transition_candidate" -c 'import torch, google.oauth2.service_account, requests' >/dev/null 2>&1; then
      panda_transition_python="$panda_transition_candidate"
      break
    fi
  done
fi
if [[ -z "$panda_transition_python" ]]; then
  echo 'Set PAN_HOST_PYTHON to the existing training/uploader Python.' >&2
  exit 1
fi
command -v setsid >/dev/null
panda_transition_directory="$panda_transition_root/work_dir/panda_sampling30_launchers/$panda_transition_server"
mkdir -p "$panda_transition_directory"
exec 8>"$panda_transition_directory/.register.lock"
flock -n 8 || { echo 'Another sampling successor registration is active.' >&2; exit 1; }
if [[ -f "$panda_transition_directory/pid" ]]; then
  panda_transition_previous="$(cat "$panda_transition_directory/pid")"
  if [[ "$panda_transition_previous" =~ ^[0-9]+$ ]] && kill -0 "$panda_transition_previous" 2>/dev/null; then
    if [[ -r "/proc/$panda_transition_previous/cmdline" ]] && tr '\0' ' ' < "/proc/$panda_transition_previous/cmdline" | grep -Fq 'panda_sampling30.transition'; then
      echo "Sampling successor watcher already active: PID $panda_transition_previous"
      exit 0
    fi
  fi
fi
PAN_HOST_PYTHON="$panda_transition_python" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  setsid "$panda_transition_python" -m panda_sampling30.transition \
  --server "$panda_transition_server" \
  >> "$panda_transition_directory/watcher.log" 2>&1 < /dev/null 8>&- &
panda_transition_pid=$!
echo "$panda_transition_pid" > "$panda_transition_directory/pid"
sleep 2
if ! kill -0 "$panda_transition_pid" 2>/dev/null; then
  echo "Transition failed; inspect $panda_transition_directory/watcher.log" >&2
  exit 1
fi
echo "Registered sampling successor PID $panda_transition_pid. Receipt: $panda_transition_directory/transition_receipt.json"
echo 'Current schedule queue finishes normally. Completed local S020C30 R1/R2 controls are required. New sampling queue has no wall deadline.'
