#!/usr/bin/env bash
# 서브모듈 제거 진단 실험 8종. 기준(A3)은 work_dir/sweep_W128D2222A3 를 재사용한다.
#
#   setsid nohup ./tools/_run_ablation.sh > work_dir/ablation_chain.log 2>&1 < /dev/null &
#
# 완료 판정은 results/reduced_best_val.mat 존재. 재시작하면 남은 것부터 이어간다. 약 12.3시간.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# 정보 가치 순: B그룹(cross-modality 검증) 먼저, 그다음 위치(A), 마지막 mode(C)
ORDER=(B2panbr B1kpan B3hf A2enc A3dec A1btl C1gate C2resmod)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter
collect(){ python tools/collect_ablation.py > work_dir/ablation_collect.log 2>&1 || echo "[chain] collect 실패(무시)"; }
echo "[chain] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개  예상 12.3h"
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    if [ -f "work_dir/abl_${TAG}/results/reduced_best_val.mat" ]; then
        echo "[chain] ($i/${#ORDER[@]}) $TAG 이미 완료 — 건너뜀"; continue; fi
    echo "[chain] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "abl_${TAG}"; then echo "[chain] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
    else echo "[chain] ($i/${#ORDER[@]}) FAILED $TAG — 계속 $(date -Iseconds)"; fi
    collect; echo "[chain] ($i/${#ORDER[@]}) results_log 갱신 $(date -Iseconds)"
done
collect
echo "[chain] ABLATION_DONE $(date -Iseconds)"
