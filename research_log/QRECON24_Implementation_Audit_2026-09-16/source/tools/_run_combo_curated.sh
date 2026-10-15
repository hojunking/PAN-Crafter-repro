#!/usr/bin/env bash
# 8/21 재배치: 단일 결과를 본 뒤 직접 고른 조합만 순차 실행한다 (재계획 없음).
# gate+resmod 동시 제거는 붕괴(+119%)가 확인돼 제외했다. 쌍은 이미 6개 확보돼 삼중/사중에 집중.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
DEADLINE=$(date -d "2026-08-21 16:00:00" +%s); RESERVE=6000
ORDER=(hf_dec4_btl_resmod hf_gate_dec4 dec4_btl_resmod)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter
echo "[curated] 시작 $(date -Iseconds), 마감 $(date -d @$DEADLINE -Iseconds), ${#ORDER[@]}개"
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1)); LEFT=$(( DEADLINE - $(date +%s) ))
    if [ "$LEFT" -lt "$RESERVE" ]; then
        echo "[curated] 남은 $((LEFT/60))분 < 필요 $((RESERVE/60))분 — 중단 ($((i-1))/${#ORDER[@]})"; break; fi
    [ -f "work_dir/combo_${TAG}/results/reduced_best_val.mat" ] && { echo "[curated] $TAG 완료됨 — 건너뜀"; continue; }
    echo "[curated] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds), 남은 $((LEFT/3600))h$(((LEFT%3600)/60))m ==="
    if ./tools/run.sh "combo_${TAG}"; then echo "[curated] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
    else echo "[curated] ($i/${#ORDER[@]}) FAILED $TAG $(date -Iseconds)"; fi
    python tools/collect_combo.py > work_dir/combo_collect.log 2>&1 || echo "[curated] collect 실패"
done
python tools/collect_combo.py > work_dir/combo_collect.log 2>&1 || true
echo "[curated] CURATED_DONE $(date -Iseconds)"
