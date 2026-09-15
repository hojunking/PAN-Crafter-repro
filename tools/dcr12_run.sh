#!/usr/bin/env bash
# DCR12 runner (서버 공용) — 계획 research_log/PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md, 노트 research_log/2026-09-15_dcr12-implementation.md. detached:
#   setsid nohup ./tools/dcr12_run.sh >> work_dir/_dcr12_<server>_campaign/run.log 2>&1 < /dev/null &
# 큐는 config/queues/dcr12_<server>.txt (gspread/server.txt) — 없으면 시작하지 않는다. 다른 호스트에서 학습한 pair 는 먼저 tools/dcr12_bundle.py install (tools/dcr12_prepare.sh 가 한다).
# phase pre : D00 → D01 → D02 → D03 (T0 + 이미 있는 **완료** B0/B1 checkpoint)
# phase train: 큐의 run 이 전부 끝날 때까지 대기. 다른 큐의 chain 이 돌고 있으면 끝날 때까지 기다렸다가(5 분 간격) gate 'pakd50' 를 비우고(백업 work_dir/_dcr12_gates_backup.txt) campaign_start 로 우리 큐를 기동한다.
#              실패(cases_failed.txt; 마감 DEFERRED 는 rc=4) 면 INVALID 로 멈춘다. chain 이 미완 run 을 남기고 끝나면(마감스킵·config 없음) 최대 3 회 다시 기동한다.
# phase post: D01 → D02 → D03 (새 checkpoint 분 추가) → D04 → RESULTS → REPORT. 이미 끝난 stage 는 ledger(done)+산출물 검사로 건너뛴다.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"; SERVER="$(tr -d '[:space:]' < gspread/server.txt 2>/dev/null || echo s1)"; CAMP="work_dir/_dcr12_${SERVER}_campaign"; QUEUE="config/queues/dcr12_${SERVER}.txt"; mkdir -p "$CAMP"
[ -f "$QUEUE" ] || { echo "[dcr12] !! 큐 없음: $QUEUE — 서버 $SERVER 용 큐를 먼저 만든다 (노트 §5)"; exit 1; }
exec 8>"$CAMP/.runner.lock"; flock -n 8 || { echo "[dcr12] 이미 실행 중"; exit 0; }
[ -f "$CAMP/T0.txt" ] || date -Iseconds > "$CAMP/T0.txt"; echo "[dcr12] T0 $(cat "$CAMP/T0.txt") · 시작 $(date -Iseconds) · 서버 $SERVER · 큐 $QUEUE"
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
chain_alive() { ps -eo args | grep -q '[_]run_cases\.sh'; }
ours_queued() { [ -f work_dir/cases_queue.txt ] || return 1; local r; for r in "${RUNS[@]}"; do grep -qx "$r" work_dir/cases_queue.txt || return 1; done; return 0; }
LAUNCHES=0
launch_ours() {
  if [ "$LAUNCHES" -ge 3 ]; then echo "[dcr12] !! chain 을 3 회 기동했는데 미완 run 이 남는다 — 중단 $(date -Iseconds) (work_dir/cases_chain*.log 확인)"; echo "STOPPED train relaunch-limit $(date -Iseconds)" >> "$CAMP/run_status.txt"; exit 3; fi
  if [ -s work_dir/campaign_gates_enabled.txt ]; then cp work_dir/campaign_gates_enabled.txt work_dir/_dcr12_gates_backup.txt; : > work_dir/campaign_gates_enabled.txt; echo "[dcr12] gate 비움 (백업 work_dir/_dcr12_gates_backup.txt: $(tr '\n' ' ' < work_dir/_dcr12_gates_backup.txt))"; fi
  echo "[dcr12] 학습 chain 기동 $(date -Iseconds): ${RUNS[*]}"; LAUNCHES=$((LAUNCHES+1))
  ./tools/campaign_start.sh --queue "$QUEUE" --hours 20 --label "dcr12-$SERVER-$(date +%m%d-%H%M)" 8>&- || { echo "[dcr12] !! 기동 실패"; echo "STOPPED launch $(date -Iseconds)" >> "$CAMP/run_status.txt"; exit 1; }
  ./tools/_watchdog.sh --install 8>&- > /dev/null 2>&1 || true
}
if ! all_done; then
  echo "[dcr12] 학습 대기: ${RUNS[*]}"; WAITED_FOREIGN=0
  while ! all_done; do
    for r in "${RUNS[@]}"; do if failed "$r"; then echo "[dcr12] !! $r 실패(cases_failed.txt; 마감 DEFERRED 는 rc=4 — config kdv.budget.training_deadline) — INVALID, 중단 $(date -Iseconds). 재도전: 그 줄을 지우고 runner 재기동"; echo "STOPPED train $r failed $(date -Iseconds)" >> "$CAMP/run_status.txt"; exit 3; fi; done
    if chain_alive; then
      if ! ours_queued && [ "$WAITED_FOREIGN" = 0 ]; then echo "[dcr12] 다른 큐의 chain 실행 중 — 끝나면 우리 큐로 기동 $(date -Iseconds)"; WAITED_FOREIGN=1; fi
    else
      launch_ours
    fi
    sleep 300
  done
fi
echo "[dcr12] 학습 완료 $(date -Iseconds)"
# ---- post phase
stage D01-post d01 post; stage D02-post d02 post; stage D03-post d03 post; stage D04 d04 post; stage RESULTS results post; stage REPORT report post
echo "[dcr12] DONE $(date -Iseconds)"; echo "DONE $(date -Iseconds)" >> "$CAMP/run_status.txt"
