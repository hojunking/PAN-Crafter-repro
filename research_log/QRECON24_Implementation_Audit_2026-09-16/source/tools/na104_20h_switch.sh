#!/usr/bin/env bash
# NA104 20H 우선순위로 전환 (s2·s3 운영자 실행) — research_log/01_S2_20H_PRIORITY.md · 02_S3_20H_PRIORITY.md §2.
#   ./tools/na104_20h_switch.sh            # ① 시계 시작(지금) + 감시자 cron 해제 ② 체인 runner 만 종료(현재 학습 run 은 원 설정으로 끝까지; 그 잔여 시간은 예산에 계상) ③ 완료 대기 → export/업로드
#                                          # ④ smoke(P1 4 + CF01) + unit test — **실패하면 기동하지 않는다** ⑤ gate token na104_20h ⑥ 남은 예산 시간을 마감으로 20h 큐 기동 ⑦ 감시자 재등록
#   ./tools/na104_20h_switch.sh --no-wait  # 현재 학습 run 이 없을 때
#   ./tools/na104_20h_switch.sh --reenter  # 체인이 DONE 으로 끝난 뒤 조건이 열렸을 때(예: s2 가 s3 token 을 받은 뒤): 예산 확인 → 같은 큐로 재진입(완료분은 건너뛰고 gate 가 CF01/X02 를 연다)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; case "$SERVER" in s2|s3) ;; *) echo "!! s2·s3 전용 (현재 $SERVER)"; exit 1;; esac
QUEUE="config/queues/na104_20h_${SERVER}.txt"; CAMP=work_dir/_na104_20h; mkdir -p "$CAMP"; MODE=switch; [ "${1:-}" = "--no-wait" ] && MODE=nowait; [ "${1:-}" = "--reenter" ] && MODE=reenter
T_PREP0=$(date +%s)
if [ "$MODE" != reenter ]; then
  CH=$(ps -eo pid,args | awk '/bash \.\/tools\/_run_cases\.sh$/{print $1}'); TR=$(ps -eo pid,args | awk '/python -u main\.py --config .*NA104_/{print $1}' | head -1)
  CUR=""; [ -n "$TR" ] && CUR=$(ps -o args= -p "$TR" | sed 's#.*/config/##; s#\.yaml.*##')
  echo "[20h] ① 시계 시작 $(date -Iseconds) · 감시자 cron 일시 해제 · 체인 runner [$CH] · 현재 학습 run [$CUR]"; crontab -l 2>/dev/null | grep -v PANCRAFTER-WATCHDOG | crontab - || true
  $PY - "$CAMP" "$SERVER" "$CUR" <<'PYEOF'
import json, os, sys, time
camp, srv, cur = sys.argv[1:4]; lp = os.path.join(camp, "ledger.json"); d = json.load(open(lp)) if os.path.exists(lp) else {}
if not d.get("start"):
    d.update(server=srv, start=time.strftime("%Y-%m-%dT%H:%M:%S"), cap_hours=20.0, est_run_hours=(1.5 if srv == "s2" else 0.9), pre_switch_run=(cur or None), post_run_overhead_hours=0.1, overhead_hours=0.0,
             note="20h 시계 = switch 시각. 전환 전 run 의 잔여 시간·실패/중단 시도·준비·export overhead 를 모두 계상 (계획 §2·§6; 리뷰 P1-2)")
    json.dump(d, open(lp, "w"), indent=1); print("   ledger start", d["start"], "pre_switch_run", cur or None)
else:
    print("   ledger 이미 시작", d["start"], "(재실행 — 시계는 그대로)")
PYEOF
  if [ -n "$CH" ]; then kill "$CH" && echo "[20h] ② 체인 runner 종료 (학습 프로세스는 그대로 — 원 설정으로 끝까지)"; sleep 2; fi
  if [ -n "$CUR" ] && [ "$MODE" = switch ]; then
    echo "[20h] ③ $CUR 완료 대기 ($(date +%H:%M))"; while ps -p "$TR" > /dev/null 2>&1; do sleep 60; done; echo "[20h]    학습 종료 $(date +%H:%M)"
    if [ -f "work_dir/$CUR/results/reduced_best_hqnr.mat" ]; then ./tools/_upload.sh "$CUR" 8>&- || echo "!! 업로드 실패 — 나중에 ./tools/_upload.sh $CUR"; else echo "!! $CUR 산출물 없음 (실패?) — 확인 필요"; fi
  fi
else
  echo "[20h] --reenter: DONE 뒤 재진입 ($(date -Iseconds))"; crontab -l 2>/dev/null | grep -v PANCRAFTER-WATCHDOG | crontab - || true
  if ps -eo args | grep -q '[_]run_cases\.sh'; then echo "!! 체인이 아직 돌고 있다 — 재진입 불필요"; exit 1; fi
  [ -f "$CAMP/ledger.json" ] || { echo "!! ledger 없음 — 먼저 switch 로 시작한 캠페인이어야 한다"; exit 1; }
fi
echo "[20h] ④ 사전 검사 — smoke(P1 4 + CF01) + unit test (실패하면 기동하지 않는다, 리뷰 P1-1)"
CF=$(ls config/NA104_CF01_*_S777_v2.yaml | xargs -n1 basename | sed 's/\.yaml//'); P1=$(grep -v '^#' "$QUEUE" | grep -v '^$' | tr '\n' ' ')
[ -f work_dir/NA104_Q36_W104_D122_WV3_N0_GCH_S1234_v1/calibration_resolved.json ] || echo "   (주의) Q36 S1234 v1 의 calibration_resolved.json 이 없다 — CF01 smoke 가 실패한다 (조건부 분기라 P1 기동은 막지 않는다)"
# shellcheck disable=SC2086
"$PY" tools/smoke_cases.py $P1 > "$CAMP/smoke_p1.log" 2>&1; RC_P1=$?; grep -v Warning "$CAMP/smoke_p1.log" | tail -5
"$PY" tools/smoke_cases.py $CF > "$CAMP/smoke_cf01.log" 2>&1; RC_CF=$?; grep -v Warning "$CAMP/smoke_cf01.log" | tail -2
"$PY" tools/na104_20h_unit_tests.py > "$CAMP/unit_tests.log" 2>&1; RC_UT=$?; tail -1 "$CAMP/unit_tests.log"
[ $RC_P1 -eq 0 ] || { echo "!! P1 smoke 실패 (rc=$RC_P1) — 기동하지 않는다: $CAMP/smoke_p1.log"; exit 1; }
[ $RC_UT -eq 0 ] || { echo "!! unit test 실패 (rc=$RC_UT) — 기동하지 않는다: $CAMP/unit_tests.log"; exit 1; }
[ $RC_CF -eq 0 ] || { echo "   CF01 smoke 실패 (rc=$RC_CF) — CF01 분기는 gate 가 열어도 trainer gate(CALIBRATION_SOURCE_INVALID) 에서 멈춘다. P1 은 기동한다: $CAMP/smoke_cf01.log"; }
$PY - "$CAMP" "$T_PREP0" <<'PYEOF'
import json, os, sys, time
camp, t0 = sys.argv[1], float(sys.argv[2]); lp = os.path.join(camp, "ledger.json"); d = json.load(open(lp)); d["overhead_hours"] = round(float(d.get("overhead_hours", 0.0)) + (time.time() - t0) / 3600.0, 4)
json.dump(d, open(lp, "w"), indent=1); print("   준비·smoke overhead 계상:", d["overhead_hours"], "h")
PYEOF
REM=$($PY -c "
import sys; sys.path.insert(0, '.'); from tools.na104_20h import budget; b = budget(); print(max(0.5, round(b['remaining_hours'], 2)))")
echo "[20h] 남은 예산 $REM h (시계 $(cat $CAMP/ledger.json | $PY -c 'import json,sys; print(json.load(sys.stdin)["start"])'))"
if [ "$MODE" = reenter ]; then
  OPEN=$($PY tools/na104_20h.py decide --server "$SERVER" | $PY -c "import json,sys; print(len(json.load(sys.stdin)['open']))")
  [ "$OPEN" -gt 0 ] || { echo "!! 재진입: 열릴 분기가 없다 (조건 미충족/예산 부족) — python tools/na104_20h.py decide 참조"; exit 2; }
  echo "[20h] 재진입: 열릴 분기 $OPEN 개 — 같은 큐로 체인 재기동(완료분 건너뜀 → gate)"
fi
echo "na104_20h" > work_dir/campaign_gates_enabled.txt; echo "[20h] ⑤ 조건부 gate token = na104_20h"
echo "[20h] ⑥ 큐 기동 (마감 = 남은 예산 $REM h)"; ./tools/campaign_start.sh --queue "$QUEUE" --hours "$REM" --label "na104-20h-$SERVER-$(date +%m%d%H%M)" || { echo "!! 기동 실패"; exit 1; }
./tools/_watchdog.sh --install && echo "[20h] ⑦ 감시자 cron 재등록"
"$PY" tools/na104_20h.py report --server "$SERVER" | tail -9
