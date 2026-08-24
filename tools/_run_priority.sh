#!/usr/bin/env bash
# 논문 재구성본 — 재현 충실도 우선.
#   setsid nohup ./tools/_run_priority.sh > work_dir/priority_chain2.log 2>&1 < /dev/null &
#
# 목표는 논문 ERGAS 2.040 에 얼마나 근접하는가다. 프루닝 탐색(구 Phase 2·3)은 중단했다.
# 재구성이 논문 수치를 내지 못하면 경량화 논의 자체가 기준선을 잃기 때문이다.
#
#   paper_wv3      50K  GroupNorm32 (배포본과 같은 정규화)        <- 실행 중
#   paper_ln       50K  LayerNorm  (논문 Eq (5) 서술)             A
#   paper_ln_mlp1  50K  LN + mlp 1.0 + ResBlock 13개              C  (params 적중의 다른 갈래)
#
# depth 배분(B)과 학습 파이프라인 진단(d1/d2)은 위 결과를 보고 정한다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ORDER=(paper_wv3 paper_ln paper_ln_mlp1)
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter

# 이미 돌고 있는 학습이 있으면 끝날 때까지 기다린다
while pgrep -f "main.py --config .*paper_wv3" > /dev/null; do
    echo "[pri] paper_wv3 학습 중 — 대기 $(date -Iseconds)"; sleep 300
done
echo "[pri] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개"

report(){  # 논문 2.040 대조를 로그에 바로 남긴다
    local t="$1"
    [ -f "work_dir/$t/results/reduced_best_val.mat" ] || return 0
    echo "[pri] --- $t 결과 (논문 ERGAS 2.040 대조) ---"
    python tools/eval_dlpan.py "work_dir/$t/results/reduced_best_val.mat" --preset wv3 2>/dev/null \
      | grep -E "$t|PSNR" || true
}
collect(){ python tools/collect_exp.py "paper*" results_log/2026-08-25_WIP_paper.md \
           > work_dir/priority_collect.log 2>&1 || echo "[pri] collect 실패"; }
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    if [ -f "work_dir/${TAG}/results/reduced_best_val.mat" ]; then
        echo "[pri] ($i/${#ORDER[@]}) $TAG 완료됨 — 건너뜀"; report "$TAG"; continue
    fi
    echo "[pri] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "$TAG"; then echo "[pri] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
    else echo "[pri] ($i/${#ORDER[@]}) FAILED $TAG $(date -Iseconds)"; fi
    report "$TAG"; collect
done
collect
echo "[pri] DONE $(date -Iseconds)"
