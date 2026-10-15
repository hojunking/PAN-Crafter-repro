#!/usr/bin/env bash
# SMEC12 분석 runner — A00 → D10 → I20 → I23 → I24 → I25 → X40 → REPORT (있는 자산만; 준비 학습이 끝날 때마다 다시 돌리면 새 모델 분이 덧붙는다). detached:
#   setsid nohup ./tools/smec12_run.sh >> work_dir/_smec12_<server>_campaign/run.log 2>&1 < /dev/null &
# 실패하면 그 stage 에서 멈춘다. 이미 done 인 stage 라도 D10/I20/I23/I24/I25 는 증분이라 --force 없이도 다시 돈다(SMEC12_PHASE 로 ledger 이름 구분).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"; SERVER="$(tr -d '[:space:]' < gspread/server.txt 2>/dev/null || echo s1)"; CAMP="work_dir/_smec12_${SERVER}_campaign"; mkdir -p "$CAMP"
exec 8>"$CAMP/.runner.lock"; flock -n 8 || { echo "[smec12] 이미 실행 중"; exit 0; }
PHASE="${SMEC12_PHASE:-$(date +%m%d%H%M)}"; echo "[smec12] 시작 $(date -Iseconds) phase $PHASE"
for st in a00 d10 i20 i23 i24 i25 x40 report; do
  echo "[smec12] === $st $(date -Iseconds) ==="; SMEC12_PHASE="$PHASE" "$PY" tools/smec12.py "$st" 8>&-; rc=$?
  if [ $rc -ne 0 ]; then echo "[smec12] !! $st 실패 (rc=$rc) — 중단 $(date -Iseconds)"; echo "STOPPED $st rc=$rc $(date -Iseconds)" >> "$CAMP/run_status.txt"; exit $rc; fi
done
echo "[smec12] DONE $(date -Iseconds)"; echo "DONE $(date -Iseconds)" >> "$CAMP/run_status.txt"
