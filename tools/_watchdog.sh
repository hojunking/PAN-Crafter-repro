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
pgrep -f "bash .*tools/_run_cases.sh" > /dev/null && exit 0          # 체인 살아있음
grep -q "\[cases\] DONE" "$LOG" 2>/dev/null && exit 0                # 이미 끝남
echo "[watchdog] $(date -Iseconds) 체인 재기동" >> "$REPO/work_dir/watchdog.log"
cd "$REPO"
for _mc in "$HOME/miniconda3/bin" /home/knuvi/miniconda3/bin; do   # cron 최소 PATH 대비 (서버 무관)
  [ -d "$_mc" ] && export PATH="$_mc:$PATH"
done
setsid nohup ./tools/_run_cases.sh >> "$LOG" 2>&1 < /dev/null &
