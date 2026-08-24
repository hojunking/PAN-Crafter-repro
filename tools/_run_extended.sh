#!/usr/bin/env bash
# 확장 실험 8종: ① 비싼 원자 조합, ② A5 기준 재확인, ③ 최대 감축 50K
#   setsid nohup ./tools/_run_extended.sh > work_dir/extended_chain.log 2>&1 < /dev/null &
# 완료 판정은 results/reduced_best_val.mat. 재시작하면 이어서 진행. 약 15.1시간.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
# 정보 가치 순: A5 기준 검증(결론의 조건부성 제거) → 새 영역 → 50K
ORDER=(x2_A5_gate_resmod x2_A5_panbr x1_panbr_enc4 x3_maxcut_50k
       x1_panbr_enc4_dec4 x1_enc4_dec4 x2_A5_enc4 x1_panbr_dec4)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter
collect(){ python tools/collect_exp.py "x*" > work_dir/extended_collect.log 2>&1 \
           || echo "[ext] collect 실패 — extended_collect.log 참고"; }
echo "[ext] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개  예상 15.1h"
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    [ -f "work_dir/${TAG}/results/reduced_best_val.mat" ] && { echo "[ext] ($i/${#ORDER[@]}) $TAG 완료됨 — 건너뜀"; continue; }
    echo "[ext] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "$TAG"; then echo "[ext] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
    else echo "[ext] ($i/${#ORDER[@]}) FAILED $TAG $(date -Iseconds)"; fi
    collect; echo "[ext] ($i/${#ORDER[@]}) results_log 갱신 $(date -Iseconds)"
done
collect
echo "[ext] EXTENDED_DONE $(date -Iseconds)"
