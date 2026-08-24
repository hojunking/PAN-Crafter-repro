#!/usr/bin/env bash
# 우선순위 체인: 논문 충실 재구성본을 먼저 돌리고, 그 뒤 기존 진단 2종을 이어간다.
#   setsid nohup ./tools/_run_priority.sh > work_dir/priority_chain.log 2>&1 < /dev/null &
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ORDER=(paper_wv3 d1_nocrop d2_lmsbase)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter

while pgrep -f "bash .*_run_twoaxis.sh" > /dev/null; do
    echo "[pri] 2축 체인 진행 중 — 대기 $(date -Iseconds)"; sleep 300
done
echo "[pri] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개"

collect(){ python tools/collect_exp.py "[yzd]*" results_log/2026-08-24_WIP_running.md \
           > work_dir/priority_collect.log 2>&1 || echo "[pri] collect 실패"; }
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    [ -f "work_dir/${TAG}/results/reduced_best_val.mat" ] && { echo "[pri] ($i/${#ORDER[@]}) $TAG 완료됨 — 건너뜀"; continue; }
    echo "[pri] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "$TAG"; then echo "[pri] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
    else echo "[pri] ($i/${#ORDER[@]}) FAILED $TAG $(date -Iseconds)"; fi
    collect
done
collect
echo "[pri] PRIORITY_DONE $(date -Iseconds)"
