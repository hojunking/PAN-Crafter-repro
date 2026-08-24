#!/usr/bin/env bash
# Student 아키텍처 스윕 — 12개 구성을 순차 학습하고 매번 결과를 results_log 에 반영한다.
#
#   setsid nohup ./tools/_run_student_sweep.sh > work_dir/sweep_chain.log 2>&1 < /dev/null &
#
# 완료 판정은 results/reduced_best_val.mat 존재로 한다. run.sh 의 trap 은 실패 시에도
# finished_at.txt 를 남기므로 그것으로는 판정할 수 없다.
# 중단 후 같은 명령으로 재시작하면 남은 것부터 이어서 진행된다. 총 약 12~13시간.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# 정보 가치가 큰 순서. 중간에 끊겨도 쓸모 있는 부분집합이 남는다.
ORDER=(W128D2222A5 W96D1121A3 W96D1121A5 W96D2222A3 W128D2222A3 W96D2222A5
       W96D1121A1 W64D1121A3 W64D1121A5 W64D1121A1 W32D1121A5 W32D1121A3)

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pancrafter

collect () { python tools/collect_sweep.py > work_dir/sweep_collect.log 2>&1 || echo "[chain] collect 실패(무시)"; }

echo "[chain] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개  예상 12~13h"
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    if [ -f "work_dir/sweep_${TAG}/results/reduced_best_val.mat" ]; then
        echo "[chain] ($i/${#ORDER[@]}) $TAG 이미 완료 — 건너뜀"; continue
    fi
    echo "[chain] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "sweep_${TAG}"; then
        echo "[chain] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
    else
        echo "[chain] ($i/${#ORDER[@]}) FAILED $TAG — 다음 구성으로 계속 $(date -Iseconds)"
    fi
    collect
    echo "[chain] ($i/${#ORDER[@]}) results_log 갱신 $(date -Iseconds)"
done

collect
echo "[chain] SWEEP_DONE $(date -Iseconds)"
