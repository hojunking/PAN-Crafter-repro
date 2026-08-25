#!/usr/bin/env bash
# s1 큐: 축소한 p25 탐색 -> Teacher architecture screening
#   setsid nohup ./tools/_run_s1.sh > work_dir/s1_chain.log 2>&1 < /dev/null &
#
# [1] p25 축소 (정보가치 상위 2종 x 3시드, 15.5h)
#     d124 는 이미 돌고 있어 그 시드 1벌로 끝낸다. a2/d222 는 취소했다 —
#     a2 는 AttnBlock 축소, d222 는 bottleneck 축소로 FLOPs 기여가 1.5% 뿐이라
#     정보가치가 낮다.
#
# [2] Teacher screening (계획서 research_log/s1_teacher_architecture_screening_plan.md)
#     A0 T7 LN 9ch nocrop        7.1707M   s1 로컬 기준선
#     A1 A0 + 11ch 입력          7.1730M   입력 구성 효과
#     A2 A1 + depth(4,4,4)       9.5425M   용량 효과 (같은 구조 안에서)
#     셋 다 50K. 계획서의 25K 중간 checkpoint 방식 대신 50K 완주로 간다 —
#     25K 시점 값은 metrics.csv 에 남으므로 별도 run 이 필요 없다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ORDER=(p25_w96 p25_w96_s1234 p25_w96_s7777
       p25_d322 p25_d322_s1234 p25_d322_s7777
       s1_A0 s1_A1 s1_A2)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter

# PyTorch 는 프로세스명을 pt_main_thread 로 바꾼다. comm 으로 걸러선 안 된다.
# args 로 보되, 이 스크립트 자신과 다른 셸 명령은 제외한다.
running(){ ps -eo args | grep -v grep | grep -qE "^python .*main\.py --config"; }
while running; do echo "[s1] 다른 학습 진행 중 — 대기 $(date -Iseconds)"; sleep 300; done
echo "[s1] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개  예상 31h"

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
