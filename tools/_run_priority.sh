#!/usr/bin/env bash
# 논문 충실 재구성본 실험.
#   setsid nohup ./tools/_run_priority.sh > work_dir/priority_chain.log 2>&1 < /dev/null &
#
# Phase 1 (5.7h)   paper_wv3 50K — 논문 2.040 대조. **여기서 성능이 확인돼야 이후가 의미 있다**
# Phase 2 (7.8h)   기준선 25K x3 seed — 시드 오차율 확정
# Phase 3 (39h)    탐색 5종 x3 seed — 구성별로 3벌을 몰아서 끝낸다(부분 결과도 해석 가능하도록)
#
# 완료 판정은 results/reduced_best_val.mat. 재시작하면 이어서 진행한다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ORDER=(paper_wv3
       p25_s2025 p25_s1234 p25_s7777
       p25_d124 p25_d124_s1234 p25_d124_s7777
       p25_w96  p25_w96_s1234  p25_w96_s7777
       p25_a2   p25_a2_s1234   p25_a2_s7777
       p25_d322 p25_d322_s1234 p25_d322_s7777
       p25_d222 p25_d222_s1234 p25_d222_s7777
       d1_nocrop d2_lmsbase)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter

while pgrep -f "bash .*_run_twoaxis.sh" > /dev/null; do
    echo "[pri] 2축 체인 진행 중 — 대기 $(date -Iseconds)"; sleep 300
done
echo "[pri] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개  예상 53h"

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
    # Phase 1 관문: 재구성본 50K 성능을 로그에 바로 남긴다
    if [ "$TAG" = "paper_wv3" ] && [ -f "work_dir/paper_wv3/results/reduced_best_val.mat" ]; then
        echo "[pri] --- Phase 1 결과 (논문 2.040 대조) ---"
        python tools/eval_dlpan.py work_dir/paper_wv3/results/reduced_best_val.mat \
               --preset wv3 2>/dev/null | grep paper_wv3 || true
    fi
done
collect
echo "[pri] PRIORITY_DONE $(date -Iseconds)"
