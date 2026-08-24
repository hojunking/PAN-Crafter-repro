#!/usr/bin/env bash
# WV3 재현 2차 — 검증셋으로 best 를 고르는 방식(select_on=val)으로 baseline/fixed 를 다시 학습.
# 1차(work_dir/wv3_{baseline,fixed})는 테스트셋으로 골랐다. 둘 다 보존한다.
#
#   setsid nohup ./tools/_run_wv3_valsel_pair.sh > work_dir/wv3_valsel_chain.log 2>&1 < /dev/null &
#
# 실행당 약 5h11m (학습 4h55m + val 평가 49회 x 20초).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "[chain] 시작 $(date -Iseconds)"
for V in baseline fixed; do
    echo "[chain] === wv3_${V}_valsel ==="
    if ./tools/run.sh "wv3_${V}_valsel"; then
        echo "[chain] wv3_${V}_valsel 완료 $(date -Iseconds)"
    else
        echo "[chain] FAILED wv3_${V}_valsel exit=$? $(date -Iseconds)"
        exit 1
    fi
done

echo "[chain] CHAIN_DONE $(date -Iseconds)"
