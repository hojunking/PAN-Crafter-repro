#!/usr/bin/env bash
# 논문 충실 재구성본 실험 체인.
#   setsid nohup ./tools/_run_priority.sh > work_dir/priority_chain.log 2>&1 < /dev/null &
#
# Phase 1 (5.7h)  paper_wv3           50K. 논문 2.040 대조 — 재구성 가설의 검증
# Phase 2 (7.8h)  p25_s{2025,1234,7777} 25K x3. 시드 오차율 + 이후 탐색의 대조군
# Phase 3 (13h)   p25_*               25K 구조 탐색 5종. 정보가치 순
#
# 완료 판정은 results/reduced_best_val.mat. 재시작하면 이어서 진행한다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ORDER=(paper_wv3
       p25_s2025 p25_s1234 p25_s7777
       p25_d124 p25_w96 p25_a2 p25_d322 p25_d222
       d1_nocrop d2_lmsbase)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter

while pgrep -f "bash .*_run_twoaxis.sh" > /dev/null; do
    echo "[pri] 2축 체인 진행 중 — 대기 $(date -Iseconds)"; sleep 300
done
echo "[pri] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개  예상 27h"

collect(){ python tools/collect_exp.py "p*" results_log/2026-08-25_WIP_paper.md \
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
