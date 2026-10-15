#!/usr/bin/env bash
# 새 baseline(W96·D124 · MS+PAN 9ch · 단일 HRMS task) WV3 3-seed — s3 에서 이 한 줄로 점검하고 기동한다.
#
#   ./tools/base_w96_prepare.sh                 # 점검 + 체인 기동 (24h 마감)
#   ./tools/base_w96_prepare.sh --no-start      # 점검만
#   ./tools/base_w96_prepare.sh --hours 12
#
# 하는 일:
#   1. gspread/server.txt · PANCRAFTER_DLPAN · WV3 학습/검증/RR 데이터 확인
#   2. 논문 FR 세트 h5 (data/PanCollection/WV3/full_examples_mat20/) — 없으면 tools/build_paperset_all.sh wv3
#   3. 지표 이식 검사 (tools/verify_metrics.py) · config 3벌 smoke (tools/smoke_cases.py)
#   4. 다른 체인이 돌고 있으면 기동하지 않는다 (체인은 하나씩)
#   5. 체인 기동: config/queues/base_w96_d124_mspan_wv3_3seed.txt → run 마다 tools/_upload.sh 가 논문 세트 평가 → 시트 WV3-<server>
# 기준 문서: research_log/PAN_research_baseline_W96_D124_2026-09-09.md · 실행 준비: research_log/2026-09-09_w96-d124-mspan-wv3-3seed-launch.md
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
START=1; HOURS=24
while [ $# -gt 0 ]; do case "$1" in --no-start) START=0;; --hours) HOURS="$2"; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift; done
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null || PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python
QUEUE=config/queues/base_w96_d124_mspan_wv3_3seed.txt
CASES=$(grep -v '^#' "$QUEUE" | grep -v '^$' | tr '\n' ' ')

echo "[base96] 1/5 환경"
[ -f gspread/server.txt ] || { echo "!! gspread/server.txt 없음 — 서버 식별자(s1/s2/s3) 를 넣어야 시트에 올라간다 (CLAUDE.md)"; exit 1; }
echo "  server=$(cat gspread/server.txt)  DLPan=$PANCRAFTER_DLPAN"
[ -f "$PANCRAFTER_DLPAN/wald_utilities.py" ] || [ -d "$PANCRAFTER_DLPAN" ] || { echo "!! PANCRAFTER_DLPAN 경로가 없다: $PANCRAFTER_DLPAN"; exit 1; }
for f in train_wv3.h5 train_wv3_pan.h5 valid_wv3.h5 valid_wv3_pan.h5 reduced_examples_h5/test_wv3_multiExm1.h5 reduced_examples_h5/test_wv3_multiExm1_pan.h5; do
  [ -f "data/PanCollection/WV3/$f" ] || { echo "!! data/PanCollection/WV3/$f 없음 — SETUP.md §4"; exit 1; }
done
echo "[base96] 2/5 논문 FR 세트 (WV3 .mat 20장 → full_examples_mat20)"
if [ -f data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5 ] && [ -f data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20_pan.h5 ]; then
  echo "  있음"
else
  PYTHON="$PY" ./tools/build_paperset_all.sh wv3
fi
echo "[base96] 3/5 지표 이식 검사 + smoke"; "$PY" tools/verify_metrics.py
# shellcheck disable=SC2086
"$PY" tools/smoke_cases.py $CASES
echo "[base96] 4/5 체인 상태"
if ps -eo pid,args | grep '[_]run_cases' >/dev/null; then
  echo "  다른 체인이 돌고 있다: $(ps -eo pid,args | grep '[_]run_cases' | head -1)"
  [ "$START" = 1 ] && { echo "!! [cases] DONE 뒤에 다시 실행한다 (체인은 하나씩)"; exit 1; }
else
  echo "  돌고 있는 체인 없음"
fi
if [ "$START" = 1 ]; then
  echo "[base96] 5/5 체인 기동 (${HOURS}h)"; ./tools/campaign_start.sh --queue "$QUEUE" --hours "$HOURS" --label base-w96-mspan
  ./tools/_watchdog.sh --install && echo "  감시자 cron 등록 (PANCRAFTER-WATCHDOG, 15분)"
  echo "  진행: tail -n +1 -f work_dir/cases_chain.log | grep --line-buffered '\[cases\]\|핵심'"
else
  echo "[base96] 준비 완료 — 기동: ./tools/campaign_start.sh --queue $QUEUE --hours $HOURS --label base-w96-mspan"
fi
