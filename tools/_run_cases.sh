#!/usr/bin/env bash
# 24h 아키텍처 탐색 — 장애 대비 v2. 계획: research_log/2026-08-28_architecture-search-24h-plan.md
#   setsid nohup ./tools/_run_cases.sh > work_dir/cases_chain.log 2>&1 < /dev/null &
#   (새 캠페인은 반드시 기존 로그를 mv 로 치우고 새 로그로 시작 — 감시자가
#    이전 캠페인의 "[cases] DONE" 을 보면 재기동하지 않는다)
#
# 큐 결정
#   - work_dir/cases_queue.txt 가 있으면 그 목록(한 줄 한 실행명, # 주석 허용)을 쓴다.
#     s2 는 이 파일로 자기 큐를 정한다 — git 추적 파일을 건드리지 않는다.
#   - 없으면 아래 ORDER(s1 기본 9건)를 쓴다.
#   - 본 큐가 끝나면 tools/campaign_gate.py 가 결과 ERGAS 로 조건부 case
#     (R5·A3·L2)를 판정해 추가 실행한다.
#
# 장애 대응
#   - 학습 전 smoke(tools/smoke_cases.py): build·forward·backward·FR 형상 + 실배치
#     OOM 검사. 실패하면 2h 슬롯을 태우지 않고 즉시 FAILED 처리
#   - 완료 판정: reduced_best_hqnr.mat 와 full_best_hqnr.mat **둘 다** 존재
#   - 체크포인트(epoch-*/checkpoint-*, mtime 최신)가 있으면 **--resume 을 먼저** 시도.
#     resume 실패(비 NaN)면 처음부터 1회 재시도. exit 3(NaN)은 재시도하지 않는다
#   - 최종 실패는 work_dir/cases_failed.txt 에 기록되고 재기동 시 건너뛴다(무한 재시도 방지).
#     수동 재도전: 그 줄을 지우고 로그를 교대한 뒤 재기동
#   - work_dir/cases_deadline.txt(ISO 시각)를 지나면 새 case 를 시작하지 않는다
#   - 체인 자체가 죽는 경우는 tools/_watchdog.sh(cron) 가 재기동한다
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
# 수동 재기동과 cron 감시자가 겹쳐도 체인은 하나만 뜬다
exec 8>"$REPO/work_dir/.cases_chain.lock"; flock -n 8 || { echo "[cases] 이미 실행 중 — 종료"; exit 0; }
# 주의: fd 8 이 자식(run.sh -> python)에 상속되면, 체인 셸이 죽어도 자식이 잠금을
# 쥐고 있어 재기동이 막힌다. 실제로 겪었다. 자식 호출마다 8>&- 로 닫는다.
QUEUE_FILE="$REPO/work_dir/cases_queue.txt"
DEADLINE_FILE="$REPO/work_dir/cases_deadline.txt"
LEDGER="$REPO/work_dir/cases_failed.txt"
if [ -f "$QUEUE_FILE" ]; then
    mapfile -t ORDER < <(grep -vE '^[[:space:]]*(#|$)' "$QUEUE_FILE")
    echo "[cases] 큐 파일 사용: $QUEUE_FILE (${#ORDER[@]}건)"
else
    # s1 기본 큐. 분기 정보가 큰 것 먼저(9ch 유지 -> stage 제거 -> width -> 비대칭 -> 새 구조)
    ORDER=(N3_9_d124_noattn R1_w128_d024_noattn R3_w112_d124_noattn R4_w96_d124_noattn
           A1_asym_114_10 L1_11_lr_fuse_w64 L1_9_lr_fuse_w64 A2_asym_014_10 R6_w96_d024_noattn)
fi
CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/knuvi/miniconda3)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate pancrafter
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
running(){ ps -eo args | grep -v grep | grep -qE "^python .*main\.py --config"; }
while running; do echo "[cases] 대기 $(date -Iseconds)"; sleep 300; done
echo "[cases] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개"

latest_ckpt(){ ls -dt "work_dir/$1"/epoch-* "work_dir/$1"/checkpoint-* 2>/dev/null | head -1; }
complete(){ [ -f "work_dir/$1/results/reduced_best_hqnr.mat" ] && [ -f "work_dir/$1/results/full_best_hqnr.mat" ]; }
ledgered(){ [ -f "$LEDGER" ] && grep -q "^$1 " "$LEDGER"; }
past_deadline(){
    [ -f "$DEADLINE_FILE" ] || return 1
    local dl; dl=$(date -d "$(cat "$DEADLINE_FILE")" +%s 2>/dev/null) || return 1
    [ "$(date +%s)" -gt "$dl" ]
}

FAILED=""; DEADLINE_SKIPPED=""; MISSING_CFG=""
run_case(){  # $1=TAG $2=순번표시
    local TAG=$1 i=$2 rc CK
    if ledgered "$TAG"; then
        echo "[cases] ($i) $TAG 이전 실패 기록(cases_failed.txt) — 건너뜀"
        FAILED="${FAILED:-}$TAG "; return
    fi
    if complete "$TAG"; then
        echo "[cases] ($i) $TAG 완료됨 — 업로드만 확인"
        ./tools/_upload.sh "$TAG" 8>&-; return
    fi
    if past_deadline; then
        echo "[cases] ($i) $TAG 마감($(cat "$DEADLINE_FILE")) 경과 — 시작하지 않음"
        DEADLINE_SKIPPED="${DEADLINE_SKIPPED:-}$TAG "; return
    fi
    echo "[cases] ($i) === $TAG 시작 $(date -Iseconds) ==="
    # 새 아키텍처의 config·build·OOM 오류는 학습 전에 분 단위로 걸러낸다.
    # rc=2 는 config 부재(예: s2 가 pull 전) — 일시 사유라 원장에 남기지 않는다.
    set +e; python tools/smoke_cases.py "$TAG" 8>&-; smoke_rc=$?; set -e
    if [ $smoke_rc -eq 2 ]; then
        echo "[cases] ($i) $TAG config 없음 — 이번 패스 건너뜀 (원장 미기록, pull 후 재기동 시 재시도)"
        MISSING_CFG="${MISSING_CFG:-}$TAG "; return
    elif [ $smoke_rc -ne 0 ]; then
        echo "[cases] ($i) FAILED $TAG (smoke 불통과 — 학습 시작 안 함) $(date -Iseconds)"
        echo "$TAG rc=smoke $(date -Iseconds)" >> "$LEDGER"
        FAILED="${FAILED:-}$TAG "; return
    fi
    # 체크포인트가 있으면(중단된 실행) 이어 돌리는 게 먼저다 — 처음부터가 아니라
    CK=$(latest_ckpt "$TAG")
    if [ -n "$CK" ]; then
        echo "[cases] ($i) $TAG 체크포인트 발견 — $CK 에서 재개"
        set +e; ./tools/run.sh "$TAG" --resume "$CK" 8>&-; rc=$?; set -e
        if [ $rc -ne 0 ] && [ $rc -ne 3 ] && [ $rc -ne 4 ]; then
            echo "[cases] ($i) $TAG 재개 실패(rc=$rc) — 처음부터 재시도"
            set +e; ./tools/run.sh "$TAG" 8>&-; rc=$?; set -e
        fi
    else
        set +e; ./tools/run.sh "$TAG" 8>&-; rc=$?; set -e
        if [ $rc -ne 0 ] && [ $rc -ne 3 ] && [ $rc -ne 4 ]; then
            CK=$(latest_ckpt "$TAG")
            if [ -n "$CK" ]; then
                echo "[cases] ($i) $TAG 실패(rc=$rc) — $CK 에서 재개 재시도"
                set +e; ./tools/run.sh "$TAG" --resume "$CK" 8>&-; rc=$?; set -e
            else
                echo "[cases] ($i) $TAG 실패(rc=$rc) — 체크포인트 없음, 처음부터 재시도"
                set +e; ./tools/run.sh "$TAG" 8>&-; rc=$?; set -e
            fi
        fi
    fi
    if [ $rc -eq 3 ]; then
        echo "[cases] ($i) $TAG NaN 중단 — 재시도하지 않는다"
    elif [ $rc -eq 4 ]; then
        echo "[cases] ($i) $TAG 사전 gate 불통과(예: ShiftNet pretrain·resume provenance) — 학습 시작 안 함, 재시도하지 않는다"
    fi
    if [ $rc -eq 0 ] && complete "$TAG"; then
        echo "[cases] ($i) $TAG 완료 $(date -Iseconds)"
        python tools/eval_dlpan.py "work_dir/$TAG/results/reduced_best_hqnr.mat" --preset wv3 8>&- 2>/dev/null | grep "$TAG" || true
        ./tools/_upload.sh "$TAG" 8>&-
    else
        echo "[cases] ($i) FAILED $TAG (rc=$rc) $(date -Iseconds)"
        echo "$TAG rc=$rc $(date -Iseconds)" >> "$LEDGER"
        FAILED="${FAILED:-}$TAG "
    fi
}

i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1)); run_case "$TAG" "$i/${#ORDER[@]}"
done

# 본 큐 종료 후: 결과 기반 조건부 case. 게이트는 **다중 패스**로 평가한다 —
# 2단 체인(예: SW4 결과가 SW6 을 열고, SW6 결과가 SW8 을 여는 구조)은 단일
# 패스로는 영영 닫힌 채 끝나기 때문이다. 패스마다 새로 열린 것이 없으면 종료.
if ! past_deadline; then
    PREV_EXTRA=""
    for pass in 1 2 3 4; do
        # 게이트: stdout 은 실행할 tag 목록, 판정 사유는 stderr(체인 로그)로 남는다
        mapfile -t EXTRA < <(python tools/campaign_gate.py 8>&- | grep -E '^[A-Za-z0-9_]+$' || true)
        if [ ${#EXTRA[@]} -eq 0 ]; then
            echo "[cases] 조건부 게이트(패스 $pass): 추가 실행 없음"; break
        fi
        if [ "${EXTRA[*]}" = "$PREV_EXTRA" ]; then
            echo "[cases] 조건부 게이트(패스 $pass): 진전 없음(${EXTRA[*]}) — 종료"; break
        fi
        PREV_EXTRA="${EXTRA[*]}"
        echo "[cases] 조건부 게이트(패스 $pass) 통과: ${EXTRA[*]}"
        j=0
        for TAG in "${EXTRA[@]}"; do
            j=$((j+1)); run_case "$TAG" "gate$pass $j/${#EXTRA[@]}"
        done
        past_deadline && { echo "[cases] 게이트 루프 마감 도달 — 종료"; break; }
    done
fi

# DONE 은 감시자의 종료 신호다. 실패가 있어도 재기동 루프를 막기 위해 DONE 은 찍되
# 실패·마감스킵 목록을 함께 남긴다 — 사람이 로그만 보고 정상 완료로 오인하지 않게.
# (실패는 이미 재시도를 소진했고 ledger 에 남아, 재기동해도 다시 태우지 않는다)
echo "[cases] DONE $(date -Iseconds)${FAILED:+  !! 실패: $FAILED}${DEADLINE_SKIPPED:+  !! 마감스킵: $DEADLINE_SKIPPED}${MISSING_CFG:+  !! config없음(미실행): $MISSING_CFG}"
