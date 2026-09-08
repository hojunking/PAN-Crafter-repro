#!/usr/bin/env bash
# 2026-09-07 지표 v2 로 서버(s2/s3)의 모든 run 을 다시 재고 시트에 올린다 — 한 번만 돌리면 된다.
#
#   ./tools/metric_v2_prepare.sh                 # 전부
#   ./tools/metric_v2_prepare.sh --no-upload     # 시트는 올리지 않고 fr_mat20.json 까지만
#   ./tools/metric_v2_prepare.sh --shards 4      # 논문 세트 평가를 4 프로세스로 (GPU 가 비어 있을 때)
#
# 하는 일 (results_log/2026-09-07_metric-comparability-audit.md · KNOWN_ISSUES D-7 / F-2):
#   1. PanCollection **.mat 형식** WV3 FR 테스트셋 20장(= 논문 세트) 을 받아 입력 h5 로 만든다
#      (Google Drive 폴더 16pGIqvwWfyQVvkk3s1xrwLpavqQd0Bv7, gdown). 이미 있으면 건너뛴다.
#   2. python tools/verify_metrics.py — 지표 이식 검사 (SCC/SSIM/PSNR/D_λ/D_s/HQNR 기대값과 상대오차 0)
#   3. python tools/eval_fr_paperset.py --all — run 마다 results/fr_mat20.json (eval_version 2026-09-07.2)
#   4. python gspread/gspread_upload.py --all --replace — 새 탭 <DS>-<server> (옛 탭은 <DS>-<server>_v1 로 남아 있다)
#   5. python gspread/apply_layout.py --sheet WV3-<server> --ref WV3-<server>_v1 — 옛 탭과 같은 순서·구분행,
#      옛 탭에 없던 run 은 WV3-<server>-extra 로
# 전제: PANCRAFTER_DLPAN (DLPan-Toolbox 경로), gspread/server.txt, gspread/account.json, conda env pancrafter.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
UPLOAD=1; SHARDS=1
while [ $# -gt 0 ]; do case "$1" in
  --no-upload) UPLOAD=0;; --shards) SHARDS="$2"; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift; done
# (tools/setup_paths.sh 는 대화형 sourcing 용이라 여기서는 부르지 않는다 — set -e 아래에서 스크립트를 끝내 버린다)
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
[ -d "$PANCRAFTER_DLPAN" ] || { echo "!! PANCRAFTER_DLPAN=$PANCRAFTER_DLPAN 없음 — DLPan-Toolbox 를 clone 하고 경로를 지정할 것"; exit 1; }
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null || PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python
SERVER="$(cat gspread/server.txt 2>/dev/null || true)"
[ -n "$SERVER" ] || { echo "!! gspread/server.txt 가 없다 (예: echo s2 > gspread/server.txt)"; exit 1; }
echo "[prepare] server=$SERVER  python=$PY  DLPan=$PANCRAFTER_DLPAN"

echo "[prepare] 1/5 논문 FR 세트 (센서별 .mat 20장 → h5)"
PYTHON="$PY" ./tools/build_paperset_all.sh          # 데이터가 있는 센서만; 이미 있으면 건너뜀
[ -f data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5 ] || { echo "!! WV3 논문 세트 h5 가 없다"; exit 1; }
echo "[prepare] 2/5 지표 이식 검사"; "$PY" tools/verify_metrics.py
echo "[prepare] 3/5 논문 세트 평가 (shards=$SHARDS)"
if [ "$SHARDS" -gt 1 ]; then
  for i in $(seq 0 $((SHARDS-1))); do "$PY" tools/eval_fr_paperset.py --all --shard "$i/$SHARDS" > "work_dir/fr_mat20_shard$i.log" 2>&1 & done; wait
  cat work_dir/fr_mat20_shard*.log | grep -c HQNR | xargs -I{} echo "  {} run 완료"
  "$PY" tools/eval_fr_paperset.py --all          # OOM 등으로 빠진 것 순차 보충
else
  "$PY" tools/eval_fr_paperset.py --all
fi
if [ "$UPLOAD" = 1 ]; then
  echo "[prepare] 4/5 시트 전체 업로드 (탭 WV3-$SERVER)"; "$PY" gspread/gspread_upload.py --all --replace
  echo "[prepare] 5/5 옛 탭 순서로 재배치"; "$PY" gspread/apply_layout.py --sheet "WV3-$SERVER" --ref "WV3-${SERVER}_v1" || \
    echo "  (참조 탭 WV3-${SERVER}_v1 이 없으면 python gspread/refile_sheet.py --sheet WV3-$SERVER 로 범주 정렬만 해도 된다)"
fi
echo "[prepare] 완료"
