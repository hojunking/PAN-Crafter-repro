#!/usr/bin/env bash
# PAKD50 — 돌고 있는 체인의 큐를 gate 편성(J0 → λE → JQ → F0 → FQ → JR → FR → XJ)으로 바꾼다. **현재 학습 run 은 건드리지 않는다** (감사 F02).
#   ./tools/pakd50_requeue.sh
#   ① 감시자 cron 일시 해제 ② 체인 runner(bash _run_cases.sh) 만 종료 — 학습 프로세스(run.sh → python)는 그대로
#   ③ 예산 ledger 를 50h/4h 규약으로, 50h 시계에 공통 마감 기록 ④ 새 runner 기동(큐 = J0 만; runner 는 현재 학습이 끝날 때까지 스스로 기다린다)
#   ⑤ 체인 마감 = 공통 training_deadline ⑥ 감시자 cron 재등록
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; QUEUE="config/queues/pakd50_${SERVER}_stage1.txt"; LED=work_dir/_pakd50/ledger.json
fail() { echo "!! $1"; exit 1; }
[ -f "$LED" ] || fail "$LED 없음 — 이 서버는 prepare 를 아직 안 했다 (requeue 대상 아님)"
[ -f "$QUEUE" ] || fail "$QUEUE 없음"
grep -q '^PAKD50_J0_' "$QUEUE" || fail "$QUEUE 가 새 형식(J0 만)이 아니다 — gen_pakd50_configs.py --server $SERVER 로 다시 만들 것"
"$PY" - <<'PYEOF' || fail "ledger/시계 갱신 실패"
import json, os, sys; sys.path.insert(0, "."); from tools._ledger_update import locked; from tools import gen_pakd50_configs as G
c = G.campaign_clock(); assert c.get("training_deadline"), "assets/pakd50/campaign_clock.json 없음"
p = G.LEDGER
with locked(p):
    d = json.load(open(p)); d["total_gpu_hours"] = 50.0; json.dump(d, open(p, "w"), indent=1)         # 50h 벽시계 = GPU-h, reserve 4 는 config 가 (학습 ≤ 46h)
l = json.load(open("work_dir/_pakd50/ledger.json")); l["local_prepared_at"] = l.get("local_prepared_at") or l["start"]
l.update(start=c["start"], training_deadline=c["training_deadline"], final_deadline=c["final_deadline"], clock_source=G.CLOCK_PATH); json.dump(l, open("work_dir/_pakd50/ledger.json", "w"), indent=1)
print(f"   ledger total 50h · 공통 시계 {c['start']} → 학습 마감 {c['training_deadline']}")
PYEOF
DL=$("$PY" -c "import json; print(json.load(open('$LED'))['training_deadline'])")
REM=$("$PY" -c "import time; t=time.mktime(time.strptime('$DL'[:19],'%Y-%m-%dT%H:%M:%S')); print('%.2f' % max(0.0, (t-time.time())/3600))")
[ "$("$PY" -c "print(1 if float('$REM') > 0.5 else 0)")" = 1 ] || fail "학습 마감($DL) 이 지났다"
HOURS=$("$PY" -c "import math; print(max(1, math.ceil(float('$REM'))))")
echo "[requeue] ① 감시자 cron 일시 해제"; crontab -l 2>/dev/null | grep -v PANCRAFTER-WATCHDOG | crontab - || true
CH=$(ps -eo pid,args | awk '/bash \.\/tools\/_run_cases\.sh$/{print $1}'); TR=$(ps -eo pid,args | awk '/python .*main\.py --config .*PAKD50_/{print $1}' | head -1)
echo "[requeue] ② 체인 runner [${CH:-없음}] 종료 · 현재 학습 [${TR:-없음}] 은 그대로"
if [ -n "$CH" ]; then kill $CH; for _ in 1 2 3 4 5 6 7 8 9 10; do ps -eo args | grep -q '[_]run_cases\.sh' || break; sleep 1; done; fi
ps -eo args | grep -q '[_]run_cases\.sh' && fail "runner 가 아직 살아 있다"
echo "[requeue] ④ 새 runner (큐 $QUEUE; 남은 ${REM}h)"; ./tools/campaign_start.sh --queue "$QUEUE" --hours "$HOURS" --label "pakd50-$SERVER-requeue-$(date +%m%d-%H%M)" || fail "기동 실패"
echo "$DL" > work_dir/cases_deadline.txt; echo "[requeue] ⑤ 체인 마감 = $DL"
./tools/_watchdog.sh --install && echo "[requeue] ⑥ 감시자 cron 재등록"
tail -3 work_dir/cases_chain.log
