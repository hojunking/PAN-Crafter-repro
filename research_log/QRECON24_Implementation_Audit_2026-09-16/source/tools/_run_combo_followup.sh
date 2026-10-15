#!/usr/bin/env bash
# 단일 제거 실험이 끝나면 자동으로 조합(중복) 실험을 이어서 돌린다.
#
#   setsid nohup ./tools/_run_combo_followup.sh > work_dir/combo_chain.log 2>&1 < /dev/null &
#
# 1) work_dir/ablation_chain.log 에 ABLATION_DONE 이 나올 때까지 대기
# 2) tools/plan_combos.py 로 결과 기반 조합 config 자동 생성 (열화가 작은 조합부터)
# 3) 마감시각까지 순차 실행. 남은 시간이 부족하면 시작하지 않고 종료.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
DEADLINE=$(date -d "2026-08-21 16:00:00" +%s)
RESERVE=6000          # 구성당 약 1.5h + 여유. 이보다 적게 남으면 새로 시작하지 않는다

source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter

echo "[combo] 대기 시작 $(date -Iseconds) — 마감 $(date -d @$DEADLINE -Iseconds)"
while ! grep -q "ABLATION_DONE" work_dir/ablation_chain.log 2>/dev/null; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "[combo] 대기 중 마감 도달. 종료."; exit 0; fi
    sleep 120
done
echo "[combo] 단일 실험 완료 감지 $(date -Iseconds)"

python tools/collect_ablation.py > work_dir/ablation_collect.log 2>&1 || true
if ! python tools/plan_combos.py > work_dir/combo_plan.log 2>&1; then
    echo "[combo] 조합 계획 실패 — combo_plan.log 참고. 종료."; cat work_dir/combo_plan.log; exit 1
fi
cat work_dir/combo_plan.log

mapfile -t ORDER < work_dir/combo_order.txt
echo "[combo] 조합 ${#ORDER[@]}개 계획됨 $(date -Iseconds)"
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1)); [ -z "$TAG" ] && continue
    LEFT=$(( DEADLINE - $(date +%s) ))
    if [ "$LEFT" -lt "$RESERVE" ]; then
        echo "[combo] 남은 시간 $((LEFT/60))분 < 필요 $((RESERVE/60))분 — 여기서 중단 ($((i-1))/${#ORDER[@]} 완료)"
        break
    fi
    if [ -f "work_dir/combo_${TAG}/results/reduced_best_val.mat" ]; then
        echo "[combo] ($i/${#ORDER[@]}) $TAG 이미 완료 — 건너뜀"; continue; fi
    echo "[combo] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds), 남은 $((LEFT/3600))h$(( (LEFT%3600)/60 ))m ==="
    if ./tools/run.sh "combo_${TAG}"; then echo "[combo] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
    else echo "[combo] ($i/${#ORDER[@]}) FAILED $TAG — 계속 $(date -Iseconds)"; fi
    python tools/collect_combo.py > work_dir/combo_collect.log 2>&1 || echo "[combo] collect 실패(무시)"
done
python tools/collect_combo.py > work_dir/combo_collect.log 2>&1 || true
echo "[combo] COMBO_DONE $(date -Iseconds)"
