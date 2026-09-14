#!/usr/bin/env bash
# EQREC4-S1-v1 runner — G00 → D10 → D20 → D30 → D40 → D50 → K10 → K20 → report 를 순서대로 (계획 §17). detached:
#   setsid nohup ./tools/eqrec4_run.sh > work_dir/_eqrec4_s1_campaign/run.log 2>&1 < /dev/null &
# 각 stage 는 tools/eqrec4.py <stage>; 실패하면 그 stage 에서 멈추고 time_ledger.jsonl 에 error 를 남긴다 (다음 stage 를 강행하지 않는다 — §17.3).
# 이미 끝난 stage 는 다시 돌리지 않는다 (ledger 의 done). 재기동: 같은 명령.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"; CAMP=work_dir/_eqrec4_s1_campaign; mkdir -p "$CAMP"
exec 8>"$CAMP/.runner.lock"; flock -n 8 || { echo "[eqrec4] 이미 실행 중"; exit 0; }
[ -f "$CAMP/T0.txt" ] || date -Iseconds > "$CAMP/T0.txt"; echo "[eqrec4] T0 $(cat "$CAMP/T0.txt") · 시작 $(date -Iseconds)"
done_stage() { [ -f "$CAMP/time_ledger.jsonl" ] && grep -q "\"stage\": \"$1\", \"status\": \"done\"" "$CAMP/time_ledger.jsonl"; }
for st in g00:G00 d10:D10 d20:D20 d30:D30 d40:D40 d50:D50 k10:K10 k20:K20 report:REPORT; do
  cmd=${st%%:*}; name=${st##*:}
  if done_stage "$name"; then echo "[eqrec4] $name 이미 완료 — 건너뜀"; continue; fi
  echo "[eqrec4] === $name $(date -Iseconds) (T0 로부터 $(( ($(date +%s) - $(date -d "$(cat "$CAMP/T0.txt")" +%s)) / 60 )) min) ==="
  "$PY" tools/eqrec4.py "$cmd" 8>&-; rc=$?
  if [ $rc -ne 0 ]; then echo "[eqrec4] !! $name 실패 (rc=$rc) — 중단 $(date -Iseconds)"; echo "[eqrec4] STOPPED $name rc=$rc $(date -Iseconds)" >> "$CAMP/run_status.txt"; exit $rc; fi
done
echo "[eqrec4] DONE $(date -Iseconds)"; echo "[eqrec4] DONE $(date -Iseconds)" >> "$CAMP/run_status.txt"
