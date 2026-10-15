#!/usr/bin/env bash
# DCR12 runner (s1) — 계획 research_log/PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md. detached:
#   setsid nohup ./tools/dcr12_run.sh >> work_dir/_dcr12_s1_campaign/run.log 2>&1 < /dev/null &
# phase pre : D00 → D01 → D02 → D03 (T0 + 이미 있는 B0/B1 checkpoint)
# phase train: 큐 config/queues/dcr12_s1.txt 의 run 이 전부 끝날 때까지 대기 (체인이 없으면 campaign_start 로 기동; 실패는 cases_failed.txt 로 판정)
# phase post: D01 → D02 → D03 (새 checkpoint 분 추가) → D04 → RESULTS → REPORT. 이미 끝난 stage 는 ledger(done)+산출물 검사로 건너뛴다.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"; SERVER="$(tr -d '[:space:]' < gspread/server.txt 2>/dev/null || echo s1)"; CAMP="work_dir/_dcr12_${SERVER}_campaign"; QUEUE=config/queues/dcr12_s1.txt; mkdir -p "$CAMP"
exec 8>"$CAMP/.runner.lock"; flock -n 8 || { echo "[dcr12] 이미 실행 중"; exit 0; }
[ -f "$CAMP/T0.txt" ] || date -Iseconds > "$CAMP/T0.txt"; echo "[dcr12] T0 $(cat "$CAMP/T0.txt") · 시작 $(date -Iseconds)"
done_stage() { [ -f "$CAMP/time_ledger.jsonl" ] && grep -q "\"stage\": \"$1\", \"status\": \"done\"" "$CAMP/time_ledger.jsonl" && "$PY" tools/dcr12.py check --of "$2" > /dev/null 2>&1; }
stage() {  # $1 ledger name, $2 cli stage, $3 phase env
  if done_stage "$1" "$2"; then echo "[dcr12] $1 이미 완료 — 건너뜀"; return 0; fi
  echo "[dcr12] === $1 $(date -Iseconds) ==="; DCR12_PHASE="$3" "$PY" tools/dcr12.py "$2" 8>&-; rc=$?
  if [ $rc -ne 0 ]; then echo "[dcr12] !! $1 실패 (rc=$rc) — 중단 $(date -Iseconds)"; echo "STOPPED $1 rc=$rc $(date -Iseconds)" >> "$CAMP/run_status.txt"; exit $rc; fi
}
stage D00 d00 ""; stage D01 d01 ""; stage D02 d02 ""; stage D03 d03 ""
# ---- train phase
mapfile -t RUNS < <(grep -vE '^[[:space:]]*(#|$)' "$QUEUE")
complete() { [ -f "work_dir/$1/results/reduced_best_hqnr.mat" ] && [ -f "work_dir/$1/results/full_best_hqnr.mat" ]; }
failed() { [ -f work_dir/cases_failed.txt ] && grep -q "^$1 " work_dir/cases_failed.txt; }
all_done() { for r in "${RUNS[@]}"; do complete "$r" || return 1; done; return 0; }
if ! all_done; then
  if ! ps -eo args | grep -q '[_]run_cases\.sh'; then echo "[dcr12] 학습 체인 없음 — 기동 $(date -Iseconds)"; ./tools/campaign_start.sh --queue "$QUEUE" --hours 20 --label "dcr12-$SERVER-$(date +%m%d-%H%M)" 8>&- || { echo "[dcr12] !! 기동 실패"; exit 1; }; ./tools/_watchdog.sh --install 8>&- > /dev/null 2>&1 || true; fi
  echo "[dcr12] 학습 대기: ${RUNS[*]}"
  while ! all_done; do
    for r in "${RUNS[@]}"; do if failed "$r"; then echo "[dcr12] !! $r 실패(cases_failed.txt) — INVALID, 중단 $(date -Iseconds)"; echo "STOPPED train $r failed $(date -Iseconds)" >> "$CAMP/run_status.txt"; exit 3; fi; done
    if ! ps -eo args | grep -q '[_]run_cases\.sh' && grep -q '\[cases\] DONE' work_dir/cases_chain.log 2>/dev/null && ! all_done; then echo "[dcr12] !! 체인이 끝났는데 미완 run 있음 — 중단 $(date -Iseconds)"; echo "STOPPED train incomplete $(date -Iseconds)" >> "$CAMP/run_status.txt"; exit 3; fi
    sleep 300
  done
fi
echo "[dcr12] 학습 완료 $(date -Iseconds)"
# ---- post phase
stage D01-post d01 post; stage D02-post d02 post; stage D03-post d03 post; stage D04 d04 post; stage RESULTS results post; stage REPORT report post
echo "[dcr12] DONE $(date -Iseconds)"; echo "DONE $(date -Iseconds)" >> "$CAMP/run_status.txt"
