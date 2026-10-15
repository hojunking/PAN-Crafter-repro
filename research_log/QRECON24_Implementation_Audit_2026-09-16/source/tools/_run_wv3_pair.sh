#!/usr/bin/env bash
# WV3 재현 체인 — baseline(배포본 그대로) 다음 fixed(KNOWN_ISSUES A-1/A-2 적용).
# GPU 1장이므로 순차 실행한다. 약 5시간 x 2.
#
#   setsid nohup ./tools/_run_wv3_pair.sh > work_dir/wv3_pair_chain.log 2>&1 < /dev/null &
#
# baseline 이 실패하면 fixed 는 돌리지 않는다. 같은 원인으로 5시간을 더 버릴 이유가 없다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "[chain] 시작 $(date -Iseconds)"

echo "[chain] === 1/2 baseline (wv3) ==="
if ./tools/run.sh wv3; then
    echo "[chain] baseline 완료 $(date -Iseconds)"
else
    echo "[chain] BASELINE_FAILED exit=$? $(date -Iseconds)"
    echo "[chain] fixed 는 건너뛴다."
    exit 1
fi

echo "[chain] === 2/2 fixed (wv3_fixed) ==="
if ./tools/run.sh wv3_fixed; then
    echo "[chain] fixed 완료 $(date -Iseconds)"
else
    echo "[chain] FIXED_FAILED exit=$? $(date -Iseconds)"
    exit 1
fi

echo "[chain] === 결과 정리 ==="
# run.sh 안의 conda activate 는 그 서브셸에서만 유효하다. 여기서 다시 활성화하지 않으면
# base python(matplotlib 없음)으로 실행돼 ModuleNotFoundError 가 난다.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pancrafter
python tools/make_report_figures.py || true

echo "[chain] CHAIN_DONE $(date -Iseconds)"
