#!/usr/bin/env bash
# Single-action M20 start, identical on s1..s5. --dry-run changes nothing.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO/tools/mix20h_switch.sh" "$@"
