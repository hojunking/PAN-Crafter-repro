#!/usr/bin/env bash
# 170hx 서버: 논문 충실 재구성본 기준선 2벌.
#   setsid nohup ./tools/_run_170hx.sh > work_dir/chain_170hx.log 2>&1 < /dev/null &
#
# 학습 수치는 서버 간 섞을 수 없으므로(같은 설정 4벌 폭 0.79%) 이 서버에서
# 재구성본의 50K / 25K 를 각각 한 벌씩 확보한다. 이후 25K 로 student 를 찾을 때
#   - p25_s2025 가 대조군
#   - paper_wv3 가 "25K 가 50K 대비 얼마나 덜 수렴했는가" 의 보정 기준
# 이 된다. 둘 다 이 서버 GPU 라 내부 비교가 성립한다.
#
# 완료 판정은 results/reduced_best_val.mat 존재로 한다
# (run.sh 의 trap 은 실패해도 finished_at.txt 를 쓴다).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ORDER=(paper_wv3 p25_s2025)
echo "[170hx] 시작 $(date -Iseconds)  ${#ORDER[@]}벌"
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    if [ -f "work_dir/${TAG}/results/reduced_best_val.mat" ]; then
        echo "[170hx] ($i/${#ORDER[@]}) $TAG 완료됨 — 건너뜀"; continue
    fi
    echo "[170hx] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "$TAG"; then
        echo "[170hx] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
        python tools/eval_dlpan.py "work_dir/${TAG}/results/reduced_best_val.mat" \
               --preset wv3 --baseline 2>/dev/null | tail -6 || true
    else
        echo "[170hx] ($i/${#ORDER[@]}) FAILED $TAG $(date -Iseconds)"
    fi
done
echo "[170hx] CHAIN_DONE $(date -Iseconds)"
