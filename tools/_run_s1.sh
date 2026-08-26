#!/usr/bin/env bash
# s1 Teacher screening — LayerNorm + nocrop, 단일 시드 50K.
#   setsid nohup ./tools/_run_s1.sh > work_dir/s1_chain.log 2>&1 < /dev/null &
#
#   s1_A0  LN · nocrop · 9ch  · depth(2,2,4)   7.1707M   crop 효과 + 로컬 기준선
#   s1_A1  A0 + 11ch 입력                      7.1730M   입력 구성 효과
#   s1_A2  A1 + depth(4,4,4)                   9.5425M   용량 효과
#
# best 체크포인트는 검증셋 ERGAS 로 고른다 (배포 코드의 ERGAS 기준을 그대로 두고
# 데이터만 테스트셋 -> 검증셋으로 바꾼 것. KNOWN_ISSUES E-1).
#
# crop 효과는 paper_ln(crop=True, 50K, 같은 설정)과 대조해서 읽는다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ORDER=(s1_A0 s1_A1 s1_A2)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter

running(){ ps -eo args | grep -v grep | grep -qE "^python .*main\.py --config"; }
while running; do echo "[s1] 다른 학습 진행 중 — 대기 $(date -Iseconds)"; sleep 300; done
echo "[s1] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개  예상 17h"

i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    [ -f "work_dir/${TAG}/results/reduced_best_val.mat" ] && { echo "[s1] ($i/${#ORDER[@]}) $TAG 완료됨 — 건너뜀"; continue; }
    echo "[s1] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "$TAG"; then
        echo "[s1] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
        python tools/eval_dlpan.py "work_dir/$TAG/results/reduced_best_val.mat" --preset wv3 2>/dev/null | grep "$TAG" || true
        ./tools/_upload.sh "$TAG"
    else
        echo "[s1] ($i/${#ORDER[@]}) FAILED $TAG $(date -Iseconds)"
    fi
done
echo "[s1] DONE $(date -Iseconds)"
