#!/usr/bin/env bash
# 논문 격차(ERGAS 2.040 vs 재현 2.16) 원인 진단 2종.
# 대조군은 이미 끝난 work_dir/sweep_W128D2222A5 (동일 Teacher config·25K) 이고,
# 아래 둘은 거기서 손잡이 하나씩만 다르다.
#   d1_nocrop  : 증강 crop(=자른 뒤 원크기 복원 scale jitter) 제거
#   d2_lmsbase : 잔차 기준선 bicubic(ms,x4) -> dataset lms
#
#   setsid nohup ./tools/_run_diag.sh > work_dir/diag_chain.log 2>&1 < /dev/null &
#
# 2축 체인이 GPU 를 쓰고 있으면 끝날 때까지 기다렸다가 시작한다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ORDER=(d1_nocrop d2_lmsbase)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter

# --- 선행 체인 대기 ---
while pgrep -f "bash .*_run_twoaxis.sh" > /dev/null; do
    echo "[diag] 2축 체인 진행 중 — 대기 $(date -Iseconds)"; sleep 300
done
echo "[diag] 선행 체인 종료 확인, 시작 $(date -Iseconds)  총 ${#ORDER[@]}개  예상 4.7h"

collect(){ python tools/collect_exp.py "d[12]_*" results_log/2026-08-24_WIP_diag.md \
           > work_dir/diag_collect.log 2>&1 || echo "[diag] collect 실패"; }
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    [ -f "work_dir/${TAG}/results/reduced_best_val.mat" ] && { echo "[diag] ($i/${#ORDER[@]}) $TAG 완료됨 — 건너뜀"; continue; }
    echo "[diag] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "$TAG"; then echo "[diag] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
    else echo "[diag] ($i/${#ORDER[@]}) FAILED $TAG $(date -Iseconds)"; fi
    collect; echo "[diag] ($i/${#ORDER[@]}) results_log 갱신 $(date -Iseconds)"
done
collect
echo "[diag] DIAG_DONE $(date -Iseconds)"
