#!/usr/bin/env bash
# Restart this sampling campaign's Sheet watcher only; preserve all trainers.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/restart_panda_sampling30_publisher.sh s1|s3' >&2
  exit 2
fi
sampling_publish_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$sampling_publish_root/scripts/start_panda_sampling30_best_publisher.sh" "$1" --restart
