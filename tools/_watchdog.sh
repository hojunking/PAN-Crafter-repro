#!/usr/bin/env bash
# 체인 감시자. cron 이 15분마다 부른다. 체인이 죽어 있고 DONE 이 아니면 재기동한다.
# 머신 재부팅 후에도 @reboot cron 으로 되살아난다. flock 으로 중복 기동을 막는다.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "${1:-}" = "--install" ]; then     # cron 등록 (s2 등 새 서버용, 멱등)
  ( crontab -l 2>/dev/null | grep -v "PANCRAFTER-WATCHDOG"
    echo "*/15 * * * * $REPO/tools/_watchdog.sh # PANCRAFTER-WATCHDOG"
    echo "@reboot sleep 120 && $REPO/tools/_watchdog.sh # PANCRAFTER-WATCHDOG"
  ) | crontab -
  echo "cron 등록 완료"; exit 0
fi
LOG="$REPO/work_dir/cases_chain.log"
LOCK="$REPO/work_dir/.watchdog.lock"
exec 9>"$LOCK"; flock -n 9 || exit 0
# Explicitly activated FH12 owns future admissions, including after its window
# closes. Pulling code/configs alone does not create this local pointer/window.
FH12_SERVER=""
if [ -r "$REPO/work_dir/_fh12/local_server.txt" ]; then
  IFS= read -r FH12_SERVER < "$REPO/work_dir/_fh12/local_server.txt" || true
fi
case "$FH12_SERVER" in s1|s2|s3|s4|s5)
  if [ -f "$REPO/work_dir/_fh12/$FH12_SERVER/window.json" ]; then
    cd "$REPO"
    PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"
    [ -x "$PY" ] || PY=python
    setsid nohup "$PY" tools/fh12_runner.py run --server "$FH12_SERVER" >> "$REPO/work_dir/_fh12/$FH12_SERVER/runner.log" 2>&1 < /dev/null 9>&- &
    exit 0
  fi;;
esac
[ ! -e "$REPO/work_dir/_fh12/local_server.txt" ] || { echo "FH12 registration incomplete/invalid; preserving legacy admission hold" >> "$LOG"; exit 0; }
if [ -f "$REPO/work_dir/_qrc24_mix20h/plan_manifest.json" ]; then
  cd "$REPO"
  PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"
  [ -x "$PY" ] || PY=python
  # Manifest persists after deadline/DONE: never revive historical queues.
  setsid nohup "$PY" tools/mix20h_runner.py run --wait >> "$REPO/work_dir/_qrc24_mix20h/runner.log" 2>&1 < /dev/null 9>&- &
  exit 0
fi
pgrep -f "bash .*tools/_run_cases.sh" > /dev/null && exit 0          # 체인 살아있음
[ -f work_dir/_eval_phase/hold.json ] && { echo "$(date -Iseconds) 평가 phase hold — 재기동하지 않는다 (tools/eval_phase.py)" >> "$LOG"; exit 0; }   # 계획 2026-09-18 §2
grep -q "\[cases\] DONE" "$LOG" 2>/dev/null && exit 0                # 이미 끝남
echo "[watchdog] $(date -Iseconds) 체인 재기동" >> "$REPO/work_dir/watchdog.log"
cd "$REPO"
export PATH="/home/knuvi/miniconda3/bin:$PATH"   # cron 의 최소 PATH 대비
setsid nohup ./tools/_run_cases.sh >> "$LOG" 2>&1 < /dev/null &
