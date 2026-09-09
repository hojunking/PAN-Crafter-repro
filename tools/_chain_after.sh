#!/usr/bin/env bash
# 지금 도는 체인이 끝나면(체인·학습 프로세스 없음) 다음 큐를 기동한다. 서버에 체인은 하나씩만 뜬다(_run_cases.sh flock).
#
#   setsid nohup ./tools/_chain_after.sh config/queues/pa_s1.txt 30 pa-s1 > work_dir/chain_after.log 2>&1 < /dev/null &
#
# 자기 자신을 [_]run_cases 로 잡지 않도록 패턴을 bracket 으로 쓴다 (CLAUDE.md 함정).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
QUEUE="$1"; HOURS="${2:-24}"; LABEL="${3:-after}"
echo "[after] $(date -Iseconds) 대기 시작 — 다음 큐 $QUEUE (${HOURS}h, $LABEL)"
while ps -eo args | grep -q '[_]run_cases\.sh' || ps -eo args | grep -qE '^python .*main\.py --config'; do sleep 120; done
echo "[after] $(date -Iseconds) 이전 체인 종료 확인 — 기동"
./tools/campaign_start.sh --queue "$QUEUE" --hours "$HOURS" --label "$LABEL"
./tools/_watchdog.sh --install
