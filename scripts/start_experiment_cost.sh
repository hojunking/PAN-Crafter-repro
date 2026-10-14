#!/usr/bin/env bash
# Local-server cost measurement + replacement qualified-view publisher.
# Never stops/restarts training; profiles briefly under a guarded SIGSTOP lease.
set -euo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^s(1|3)$ ]]; then
  echo 'Usage: bash scripts/start_experiment_cost.sh s1|s3 [--container NAME] [--credentials FILE]' >&2
  exit 2
fi
cost_server="$1"
shift
cost_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$cost_root"
exec "${PAN_PYTHON:-python}" -m experiment_cost start --server "$cost_server" "$@"
