#!/usr/bin/env bash
# NA104 20H 우선순위로 전환 (s2·s3 운영자 실행) — research_log/01_S2_20H_PRIORITY.md · 02_S3_20H_PRIORITY.md §2.
#   ./tools/na104_20h_switch.sh            # ① 감시자 cron 일시 해제 ② 체인 runner 만 종료(현재 학습 run 은 원 설정으로 끝까지) ③ 완료 대기 → 그 run 의 export/업로드 ④ 예산 시계 시작
#                                          # ⑤ smoke(P1 4 + CF01 1) ⑥ gate token na104_20h ⑦ 20h 큐 기동 ⑧ 감시자 재등록
#   ./tools/na104_20h_switch.sh --no-wait  # 현재 학습 run 이 없을 때 (또는 이미 끝났을 때)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; case "$SERVER" in s2|s3) ;; *) echo "!! s2·s3 전용 (현재 $SERVER)"; exit 1;; esac
QUEUE="config/queues/na104_20h_${SERVER}.txt"; CAMP=work_dir/_na104_20h; mkdir -p "$CAMP"; WAIT=1; [ "${1:-}" = "--no-wait" ] && WAIT=0
echo "[20h] ① 감시자 cron 일시 해제"; crontab -l 2>/dev/null | grep -v PANCRAFTER-WATCHDOG | crontab - || true
CH=$(ps -eo pid,args | awk '/bash \.\/tools\/_run_cases\.sh$/{print $1}'); TR=$(ps -eo pid,args | awk '/python -u main\.py --config .*NA104_/{print $1}' | head -1)
CUR=""; [ -n "$TR" ] && CUR=$(ps -o args= -p "$TR" | sed 's#.*/config/##; s#\.yaml.*##')
echo "[20h] 체인 runner [$CH] · 현재 학습 run [$CUR]"
if [ -n "$CH" ]; then kill "$CH" && echo "[20h] ② 체인 runner 종료 (학습 프로세스는 그대로 — 원 설정으로 끝까지 간다)"; sleep 2; fi
if [ -n "$CUR" ] && [ "$WAIT" = 1 ]; then
  echo "[20h] ③ $CUR 완료 대기 ($(date +%H:%M))"; while ps -p "$TR" > /dev/null 2>&1; do sleep 60; done; echo "[20h]    학습 종료 $(date +%H:%M)"
  if [ -f "work_dir/$CUR/results/reduced_best_hqnr.mat" ]; then ./tools/_upload.sh "$CUR" 8>&- || echo "!! 업로드 실패 — 나중에 ./tools/_upload.sh $CUR"; else echo "!! $CUR 산출물 없음 (실패?) — 확인 필요"; fi
fi
$PY - "$CAMP" "$SERVER" "$CUR" <<'PYEOF'
import json, os, sys, time
camp, srv, cur = sys.argv[1:4]; lp = os.path.join(camp, "ledger.json"); d = json.load(open(lp)) if os.path.exists(lp) else {}
if not d.get("start"):
    d.update(server=srv, start=time.strftime("%Y-%m-%dT%H:%M:%S"), cap_hours=20.0, est_run_hours=(1.5 if srv == "s2" else 0.9), finished_before_switch=cur,
             note="20h 시계 시작 (switch). 현재 run 은 원 설정으로 마무리했고 그 잔여 시간은 whitelist 밖(계획 §2). 예산 판정은 tools/na104_20h.py budget()")
    json.dump(d, open(lp, "w"), indent=1); print("[20h] ④ 예산 시계 시작", d["start"])
else:
    print("[20h] ④ 예산 시계 이미 시작됨", d["start"])
PYEOF
echo "[20h] ⑤ smoke — P1 4 + CF01 pilot (Teacher strict load·loss 구성; CF01 은 Q36 S1234 v1 의 calibration_resolved.json 이 있어야 λ_C 를 읽는다)"
CF=$(ls config/NA104_CF01_*_S777_v2.yaml | xargs -n1 basename | sed 's/\.yaml//'); P1=$(grep -v '^#' "$QUEUE" | grep -v '^$' | tr '\n' ' ')
[ -f work_dir/NA104_Q36_W104_D122_WV3_N0_GCH_S1234_v1/calibration_resolved.json ] || echo "!! Q36 S1234 v1 의 calibration_resolved.json 이 없다 — CF01 은 gate(CALIBRATION_SOURCE_MISSING) 에서 멈춘다"
# shellcheck disable=SC2086
"$PY" tools/smoke_cases.py $P1 $CF 2>&1 | grep -v Warning | tail -7
"$PY" tools/na104_20h_unit_tests.py | tail -1
echo "na104_20h" > work_dir/campaign_gates_enabled.txt; echo "[20h] ⑥ 조건부 gate token = na104_20h (CF01/X02 는 §4·§5 조건으로만 열린다)"
echo "[20h] ⑦ 큐 기동"; ./tools/campaign_start.sh --queue "$QUEUE" --hours 30 --label na104-20h-$SERVER
./tools/_watchdog.sh --install && echo "[20h] ⑧ 감시자 cron 재등록"
"$PY" tools/na104_20h.py report --server "$SERVER" | tail -8
