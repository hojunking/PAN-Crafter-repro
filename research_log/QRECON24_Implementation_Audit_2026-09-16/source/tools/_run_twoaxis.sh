#!/usr/bin/env bash
# 두 축 결합 6종 + 시드 반복 2종.
#   setsid nohup ./tools/_run_twoaxis.sh > work_dir/twoaxis_chain.log 2>&1 < /dev/null &
# 완료 판정은 results/reduced_best_val.mat. 재시작하면 이어서 진행.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
# 정보 가치 순: 폭만 줄인 기준점(+9.4%)과 직접 대조되는 것 먼저
ORDER=(y_W96D1121_A1 y_W96D1121_A2 y_W128D1121_A2 y_W96D2222_A2
       y_W64D1121_A1 y_W64D1121_A2 z_seed1234 z_seed7777)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter
collect(){ python tools/collect_exp.py "[xyz]*" results_log/2026-08-24_WIP_twoaxis.md \
           > work_dir/twoaxis_collect.log 2>&1 || echo "[2ax] collect 실패"; }
echo "[2ax] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개  예상 7.3h"
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    [ -f "work_dir/${TAG}/results/reduced_best_val.mat" ] && { echo "[2ax] ($i/${#ORDER[@]}) $TAG 완료됨 — 건너뜀"; continue; }
    echo "[2ax] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "$TAG"; then echo "[2ax] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
    else echo "[2ax] ($i/${#ORDER[@]}) FAILED $TAG $(date -Iseconds)"; fi
    collect; echo "[2ax] ($i/${#ORDER[@]}) results_log 갱신 $(date -Iseconds)"
done
collect
echo "[2ax] TWOAXIS_DONE $(date -Iseconds)"
