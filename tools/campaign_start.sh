#!/usr/bin/env bash
# 새 캠페인 부트스트랩 — 이전 캠페인의 잔존 상태가 새 체인을 조용히 무산시키는
# 함정 3가지를 일괄 처리하고 체인을 기동한다.
#
#   함정 1: cases_deadline.txt 가 지난 마감이면 전 case 가 '마감 경과' 스킵 후 즉시 DONE
#   함정 2: cases_chain.log 에 이전 DONE 이 남아 있으면 감시자가 죽은 체인을 영영 재기동 안 함
#   함정 3: 큐 미지정 시 기본 ORDER(과거 캠페인)가 돌아 완료 확인만 하고 DONE
#
# 사용:
#   ./tools/campaign_start.sh --queue <큐파일> [--hours N] [--label 이름]
#     --queue : 실행명 목록 파일 (한 줄 하나, # 주석). work_dir/cases_queue.txt 로 복사
#     --hours : 마감까지 시간 (기본 24)
#     --label : 회전된 이전 로그의 접미사 (기본 타임스탬프)
#
# 이전 실패 원장(cases_failed.txt)은 자동으로 지우지 않는다 — 남길지 여부는
# 사람이 정한다 (같은 case 를 재도전하려면 해당 줄을 지울 것).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

QUEUE=""; HOURS=24; LABEL="$(date +%Y%m%d-%H%M)"
while [ $# -gt 0 ]; do
    case "$1" in
        --queue) QUEUE="$2"; shift 2 ;;
        --hours) HOURS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        *) echo "알 수 없는 인자: $1" >&2; exit 1 ;;
    esac
done
[ -n "$QUEUE" ] || { echo "--queue <큐파일> 이 필요하다 (함정 3 방지)" >&2; exit 1; }
[ -f "$QUEUE" ] || { echo "큐 파일 없음: $QUEUE" >&2; exit 1; }

# 기존 체인이 살아 있으면 건드리지 않는다 — 죽이는 판단은 사람이 한다
if pgrep -f "bash .*tools/_run_cases.sh" > /dev/null; then
    echo "체인이 이미 실행 중이다 — 종료 후 다시 실행할 것" >&2; exit 1
fi

# 큐의 모든 config 존재를 먼저 확인 (부분 무산 방지)
MISS=0
while read -r TAG; do
    case "$TAG" in ''|'#'*) continue ;; esac
    [ -f "config/${TAG}.yaml" ] || [ -f "config/pancrafter_${TAG}.yaml" ] || {
        echo "config 없음: $TAG" >&2; MISS=1; }
done < "$QUEUE"
[ $MISS -eq 0 ] || { echo "누락 config 를 먼저 만들 것 (git pull 필요?)" >&2; exit 1; }

cp "$QUEUE" work_dir/cases_queue.txt
date -Iseconds -d "+${HOURS} hours" > work_dir/cases_deadline.txt
if [ -f work_dir/cases_chain.log ]; then
    mv work_dir/cases_chain.log "work_dir/cases_chain_${LABEL}.log"
    echo "이전 로그 -> work_dir/cases_chain_${LABEL}.log"
fi
[ -f work_dir/cases_failed.txt ] && \
    echo "주의: 실패 원장이 남아 있다 ($(wc -l < work_dir/cases_failed.txt)건) — 재도전할 case 는 줄을 지울 것"

setsid nohup ./tools/_run_cases.sh > work_dir/cases_chain.log 2>&1 < /dev/null &
sleep 3
if ps -eo pid,ppid,args | grep -v grep | grep -q "bash ./tools/_run_cases.sh"; then
    echo "체인 기동 완료 — 마감 $(cat work_dir/cases_deadline.txt), 큐 $(grep -cvE '^[[:space:]]*(#|$)' work_dir/cases_queue.txt)건"
    ps -eo pid,ppid,args | grep -v grep | grep "bash ./tools/_run_cases.sh"
else
    echo "기동 실패 — work_dir/cases_chain.log 확인" >&2; exit 1
fi
