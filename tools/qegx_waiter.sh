#!/usr/bin/env bash
# QEGX 대기자 (s3·s4) — QEGX 필수 run(work_dir/_qegx/mandatory_runs.txt) 이 모두 끝날 때까지 5 분마다:
#   (s3) J0 S2026 v2 의 exact50K last 가 생겼는데 c_E3(work_dir/_qegx/qec3_cE.json) 가 없으면 → tools/qedge9_cue.py pilot --branch qegx (한 번; 실패는 로그에 남기고 다음 tick 재시도)
#   chain 이 죽었고(DONE 포함) 미완 run 이 남았으면 → cue 자산(θq / QEC3 의 c_E3) 이 준비된 run 이 하나라도 있을 때 chain 을 다시 연다(campaign_start 큐 config/queues/qegx_<srv>.txt; 마감 파일 제거 = 상한 없음),
#   아직 준비 안 됐으면 status WAITING_FOR_CUE 를 남기고 기다린다. 전부 끝나면 DONE. 기존 runner 의 gate 4 pass 제한·"진전 없음" 종료를 이 대기자가 보완한다 (runner 자체는 손대지 않는다).
#   setsid nohup ./tools/qegx_waiter.sh >> work_dir/_qegx/waiter.log 2>&1 < /dev/null &      (flock 으로 하나만)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"; export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$("$PY" -c "import sys; sys.path.insert(0, '.'); from tools.gen_pakd50_configs import server_id; print(server_id(open('gspread/server.txt').read()))")"; CAMP=work_dir/_qegx; QUEUE="config/queues/qegx_${SERVER}.txt"; mkdir -p "$CAMP"
PR3=PAKD50_J0_W104_D121_WV3_T0_S2026_FRESH50_v2
exec 8>"$CAMP/.waiter.lock"; flock -n 8 || { echo "[qegx-waiter] 이미 실행 중"; exit 0; }
status() { "$PY" - "$SERVER" <<'PYEOF'
import json, os, subprocess, sys, time; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal
srv = sys.argv[1]; p = os.path.join(G.ROOT, G.QEGX_MANDATORY_FILE); runs = [l.strip() for l in open(p) if l.strip() and not l.startswith("#")] if os.path.exists(p) else []
pending = [r for r in runs if not terminal(r)]; ready = [r for r in pending if G.cue_ready(r)]; waiting = [r for r in pending if not G.cue_ready(r)]
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout; alive = any("tools/_run_cases.sh" in l and "bash" in l for l in ps.splitlines()); training = any("main.py" in l and "--config" in l for l in ps.splitlines())
st = "DONE" if not pending else ("RUNNING" if (alive or training) else ("READY_TO_RESTART" if ready else "WAITING_FOR_CUE"))
json.dump(dict(server=srv, status=st, pending=pending, ready=ready, waiting_for_cue=waiting, chain_alive=alive, training=training, checked=time.strftime("%Y-%m-%dT%H:%M:%S")), open(os.path.join(G.ROOT, "work_dir", "_qegx", "status.json"), "w"), indent=1)
print(st)
PYEOF
}
while true; do
  if [ "$SERVER" = s3 ] && [ ! -f "$CAMP/qec3_cE.json" ] && [ -f "work_dir/$PR3/last_meta.json" ] && [ -f "work_dir/$PR3/last/model.safetensors" ] && [ -f "work_dir/$PR3/results/full_best_hqnr.mat" ]; then
    echo "[qegx-waiter] $(date -Iseconds) pilot $PR3 완료 → c_E3"; "$PY" tools/qedge9_cue.py pilot --branch qegx --run "$PR3" --tag last 2>&1 | grep -v Warning | tail -2; echo "[qegx-waiter] pilot rc=${PIPESTATUS[0]}"
  fi
  ST=$(status | tail -1)
  case "$ST" in
    DONE) echo "[qegx-waiter] $(date -Iseconds) QEGX 필수 run 전부 종료 — 대기자 종료"; exit 0;;
    RUNNING) :;;
    WAITING_FOR_CUE) echo "[qegx-waiter] $(date -Iseconds) chain 없음 · 미완 run 은 cue 대기(θq/c_E3) — 자산이 생기면 자동 재기동";;
    READY_TO_RESTART)
      echo "[qegx-waiter] $(date -Iseconds) chain 없음 · 준비된 미완 run 있음 → chain 재기동 ($QUEUE; 마감 파일 제거 = 상한 없음)"
      ./tools/campaign_start.sh --queue "$QUEUE" --hours 24 --label "qegx-$SERVER-restart-$(date +%m%d-%H%M)" >> "$CAMP/waiter.log" 2>&1 && rm -f work_dir/cases_deadline.txt && ./tools/_watchdog.sh --install > /dev/null 2>&1;;
  esac
  sleep 300
done
