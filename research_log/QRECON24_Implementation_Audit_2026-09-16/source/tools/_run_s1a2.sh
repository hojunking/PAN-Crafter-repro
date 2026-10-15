#!/usr/bin/env bash
# s1_A2 재시도. OOM 이 단편화(reserved 1.07GB 미사용)로 보여 expandable_segments 로 먼저 시도.
# 그래도 OOM 이면 수동 개입 필요 (batch 축소는 학습 수학이 바뀌므로 자동으로 하지 않는다).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate pancrafter
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
running(){ ps -eo args | grep -v grep | grep -qE "^python .*main\.py --config"; }
while running; do echo "[a2] 대기 $(date -Iseconds)"; sleep 300; done
echo "[a2] s1_A2 재시도 (expandable_segments) $(date -Iseconds)"
if ./tools/run.sh s1_A2; then
    echo "[a2] 완료 $(date -Iseconds)"
    python tools/eval_dlpan.py work_dir/s1_A2/results/reduced_best_val.mat --preset wv3 2>/dev/null | grep s1_A2 || true
    ./tools/_upload.sh s1_A2
else
    echo "[a2] FAILED — batch 축소가 필요하다. 자동으로 하지 않는다 (학습 수학 변경)"
fi
