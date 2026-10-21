#!/usr/bin/env bash
# One command: measured cost sidecars, then unchanged training and publication.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/start_panda_zero30_costed.sh s1|s3' >&2
  exit 2
fi
panda_costed_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$panda_costed_root"
bash scripts/prepare_panda_zero30_costs.sh "$1"
bash scripts/run_panda_zero30_docker.sh "$1"

