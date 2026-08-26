#!/usr/bin/env bash
# 경량화 Tier A + M1. 명세: research_log/lightweight_case_specs_v1.md
#   setsid nohup ./tools/_run_cases.sh > work_dir/cases_chain.log 2>&1 < /dev/null &
# 완료 판정: results/reduced_best_hqnr.mat (select_on=hqnr 의 산출물)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ORDER=(c1_nopan c4_noattn c3b_btl c3e_enc c2_encbtl m1_single)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
running(){ ps -eo args | grep -v grep | grep -qE "^python .*main\.py --config"; }
while running; do echo "[cases] 대기 $(date -Iseconds)"; sleep 300; done
echo "[cases] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개  예상 30h"
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    [ -f "work_dir/${TAG}/results/reduced_best_hqnr.mat" ] && { echo "[cases] ($i/${#ORDER[@]}) $TAG 완료됨"; continue; }
    echo "[cases] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "$TAG"; then
        echo "[cases] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
        python tools/eval_dlpan.py "work_dir/$TAG/results/reduced_best_hqnr.mat" --preset wv3 2>/dev/null | grep "$TAG" || true
        ./tools/_upload.sh "$TAG"
    else
        echo "[cases] ($i/${#ORDER[@]}) FAILED $TAG $(date -Iseconds)"
    fi
done
echo "[cases] DONE $(date -Iseconds)"
