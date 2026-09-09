#!/usr/bin/env bash
# A1–A3 (PAN 앞단 전역 정합) — 서버마다 이 한 줄로 점검하고 자기 block 큐를 기동한다.
#
#   ./tools/pa_prepare.sh                # gspread/server.txt 의 서버(s1/s2/s3)에 맞는 config/queues/pa_<server>.txt 기동 (30h 마감)
#   ./tools/pa_prepare.sh --no-start     # 점검만
#   ./tools/pa_prepare.sh --after        # 지금 도는 체인이 끝나면 기동 (tools/_chain_after.sh)
#
# 하는 일: 환경·데이터·논문 세트 확인 → verify_metrics → tools/pa_unit_tests.py (§9 gate) → 큐 config smoke → 기동.
# 명세 research_log/PAN_A1_A3_Global_PAN_Alignment_W96_D124_2026-09-09_v2.md · 구현 노트 research_log/2026-09-09_pa-a1-a3-implementation.md
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
START=1; AFTER=0; HOURS=30
while [ $# -gt 0 ]; do case "$1" in --no-start) START=0;; --after) AFTER=1;; --hours) HOURS="$2"; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift; done
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null || PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python
[ -f gspread/server.txt ] || { echo "!! gspread/server.txt 없음 (s1/s2/s3)"; exit 1; }
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; QUEUE="config/queues/pa_${SERVER}.txt"
[ -f "$QUEUE" ] || { echo "!! $QUEUE 없음 — server.txt 는 s1/s2/s3 여야 한다"; exit 1; }
CASES=$(grep -v '^#' "$QUEUE" | grep -v '^$' | tr '\n' ' ')
echo "[pa] server=$SERVER queue=$QUEUE: $CASES"
echo "[pa] 1/5 데이터·논문 세트"
for f in train_wv3.h5 train_wv3_pan.h5 valid_wv3.h5 valid_wv3_pan.h5 reduced_examples_h5/test_wv3_multiExm1.h5; do
  [ -f "data/PanCollection/WV3/$f" ] || { echo "!! data/PanCollection/WV3/$f 없음 — SETUP.md §4"; exit 1; }
done
if [ -f data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20_pan.h5 ]; then echo "  논문 세트 있음"; else PYTHON="$PY" ./tools/build_paperset_all.sh wv3; fi
echo "[pa] 2/5 지표 이식 검사"; "$PY" tools/verify_metrics.py
echo "[pa] 3/5 A1–A3 gate (tools/pa_unit_tests.py)"; "$PY" tools/pa_unit_tests.py
echo "[pa] 4/5 smoke"
# shellcheck disable=SC2086
"$PY" tools/smoke_cases.py $CASES
echo "[pa] 5/5 기동"
if [ "$START" = 0 ]; then echo "  준비 완료 — 기동: ./tools/campaign_start.sh --queue $QUEUE --hours $HOURS --label pa-$SERVER"; exit 0; fi
if ps -eo args | grep -q '[_]run_cases\.sh'; then
  if [ "$AFTER" = 1 ]; then
    mkdir -p work_dir; setsid nohup ./tools/_chain_after.sh "$QUEUE" "$HOURS" "pa-$SERVER" > work_dir/chain_after.log 2>&1 < /dev/null &
    echo "  다른 체인이 돌고 있다 — 끝나면 자동 기동 (work_dir/chain_after.log)"; exit 0
  fi
  echo "!! 다른 체인이 돌고 있다 — --after 로 예약하거나 [cases] DONE 뒤에 다시 실행"; exit 1
fi
./tools/campaign_start.sh --queue "$QUEUE" --hours "$HOURS" --label "pa-$SERVER"
./tools/_watchdog.sh --install && echo "  감시자 cron 등록"
