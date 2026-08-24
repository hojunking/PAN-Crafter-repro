#!/usr/bin/env bash
# 170hx: 관문(paper_wv3 50K) 결과에 따라 갈리는 두 분기.
#
#   ./tools/_run_170hx_branch.sh ladder     # 관문 통과 (ERGAS <= 2.13): 감축 사다리
#   ./tools/_run_170hx_branch.sh fallback   # 관문 실패 (ERGAS >= 2.13): 재구성본 미확정 선택지
#
# 진행 중인 체인(_run_170hx.sh)이 끝나기를 기다렸다 시작한다.
# 완료 판정은 results/reduced_best_val.mat 존재 — run.sh 의 trap 은 실패해도
# finished_at.txt 를 쓰기 때문이다.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

case "${1:-}" in
  ladder)   ORDER=(p25_L1_w96d124a2 p25_L2_w96d124a1 p25_L3_w64d124a2 p25_L4_w64d112a1) ;;
  fallback) ORDER=(p25_drop02 p25_nocrop p25_in11 p25_d134) ;;
  *) echo "usage: $0 {ladder|fallback}" >&2; exit 1 ;;
esac

while pgrep -f "bash \./tools/_run_170hx\.sh" > /dev/null; do
    echo "[br] 선행 체인 진행 중 — 대기 $(date -Iseconds)"; sleep 300
done
echo "[br] $1 시작 $(date -Iseconds)  ${#ORDER[@]}벌"
i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    if [ -f "work_dir/${TAG}/results/reduced_best_val.mat" ]; then
        echo "[br] ($i/${#ORDER[@]}) $TAG 완료됨 — 건너뜀"; continue
    fi
    echo "[br] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    if ./tools/run.sh "$TAG"; then
        echo "[br] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
        python tools/eval_dlpan.py "work_dir/${TAG}/results/reduced_best_val.mat" \
               --preset wv3 2>/dev/null | tail -4 || true
    else
        echo "[br] ($i/${#ORDER[@]}) FAILED $TAG $(date -Iseconds)"
    fi
done
echo "[br] BRANCH_DONE $1 $(date -Iseconds)"
