#!/usr/bin/env bash
# 진행 중인 p25_in11 종료를 기다렸다 paper_nocrop(50K) 을 돌린다.
# p25_d134(depth 배분)는 뺐다 — nocrop 이 -3.42% 를 낸 뒤로 우선순위가 내려갔고,
# 유사 실험(p25_d322)은 원 서버가 맡는다.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
echo "[n50] p25_in11 종료 대기 $(date -Iseconds)"
until [ -f work_dir/p25_in11/results/reduced_best_val.mat ]; do sleep 60; done
echo "[n50] === paper_nocrop 50K 시작 $(date -Iseconds) ==="
if ./tools/run.sh paper_nocrop; then
    echo "[n50] 완료 $(date -Iseconds)"
    python tools/eval_dlpan.py work_dir/paper_nocrop/results/reduced_best_val.mat \
           --preset wv3 2>/dev/null | tail -4 || true
else
    echo "[n50] FAILED $(date -Iseconds)"
fi
echo "[n50] DONE $(date -Iseconds)"
