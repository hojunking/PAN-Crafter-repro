#!/usr/bin/env bash
# QRECON24 대기자 (s1–s5) — QRECON24 필수 run(work_dir/_qrecon24/mandatory_runs.txt) 이 모두 끝날 때까지 5 분마다:
#   chain 이 죽었고(DONE 포함; s1 처럼 gate 가 꺼진 서버는 이전 큐 DONE 뒤) 미완 run 이 남았으면 → 준비된 run 이 있을 때 chain 을 다시 연다(campaign_start 큐 config/queues/qrecon24_<srv>.txt; 마감 파일 제거),
#   전부 끝나면 DONE (그때 §8.3 24h 확인은 사람이 tools/qrecon24_switch.sh --extend). 기존 runner 의 gate 4 pass 제한·"진전 없음" 종료를 보완한다 (runner 자체는 손대지 않는다).
#   setsid nohup ./tools/qrecon24_waiter.sh >> work_dir/_qrecon24/waiter.log 2>&1 < /dev/null &      (flock 으로 하나만)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-/home/knuvi/miniconda3/envs/pancrafter/bin/python}"; export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$("$PY" -c "import sys; sys.path.insert(0, '.'); from tools.gen_pakd50_configs import server_id; print(server_id(open('gspread/server.txt').read()))")"; CAMP=work_dir/_qrecon24; mkdir -p "$CAMP"
queue_file() { if [ -f "$CAMP/queue_active.txt" ]; then echo "$CAMP/queue_active.txt"; else echo "config/queues/qrecon24_${SERVER}.txt"; fi; }     # §8.3 확장 뒤에는 활성 큐(기본 + 확장) 로 재기동 (감사 F03)
exec 8>"$CAMP/.waiter.lock"; flock -n 8 || { echo "[qrecon24-waiter] 이미 실행 중"; exit 0; }
status() { "$PY" - "$SERVER" <<'PYEOF'
import json, os, subprocess, sys, time; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal
srv = sys.argv[1]; p = os.path.join(G.ROOT, G.QRC24_MANDATORY_FILE); runs = [l.strip() for l in open(p) if l.strip() and not l.startswith("#")] if os.path.exists(p) else []
pending = [r for r in runs if not terminal(r)]; ready = [r for r in pending if G.cue_ready(r)]; waiting = [r for r in pending if not G.cue_ready(r)]
hold = G.eval_hold()          # 계획 2026-09-18 §2: 평가 phase 중에는 chain 을 재기동하지 않는다 (진행 중 학습은 그대로 끝난다)
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout; alive = any("tools/_run_cases.sh" in l and "bash" in l for l in ps.splitlines()); training = any("main.py" in l and "--config" in l for l in ps.splitlines())
st = "DONE" if not pending else ("RUNNING" if (alive or training) else ("HOLD_EVAL" if hold else ("READY_TO_RESTART" if ready else "WAITING_FOR_CUE")))
json.dump(dict(server=srv, status=st, pending=pending, ready=ready, waiting_for_cue=waiting, chain_alive=alive, training=training, checked=time.strftime("%Y-%m-%dT%H:%M:%S")), open(os.path.join(G.ROOT, "work_dir", "_qrecon24", "status.json"), "w"), indent=1)
print(st)
PYEOF
}
while true; do
  ST=$(status | tail -1)
  case "$ST" in
    DONE) echo "[qrecon24-waiter] $(date -Iseconds) QRECON24 필수 run 전부 종료 — 대기자 종료 (24h 확인·확장은 tools/qrecon24_switch.sh --extend)"; exit 0;;
    RUNNING) :;;
    HOLD_EVAL) echo "[qrecon24-waiter] $(date -Iseconds) 평가 phase hold — chain 을 재기동하지 않는다 (tools/eval_phase.py status)";;
    WAITING_FOR_CUE) echo "[qrecon24-waiter] $(date -Iseconds) chain 없음 · 미완 run 은 cue 대기 — 자산이 생기면 자동 재기동";;
    READY_TO_RESTART)
      echo "[qrecon24-waiter] $(date -Iseconds) chain 없음 · 준비된 미완 run 있음 → chain 재기동 ($(queue_file); 마감 파일 제거 = 상한 없음)"
      QUEUE="$(queue_file)"; ./tools/campaign_start.sh --queue "$QUEUE" --hours 24 --label "qrecon24-$SERVER-restart-$(date +%m%d-%H%M)" >> "$CAMP/waiter.log" 2>&1 && rm -f work_dir/cases_deadline.txt && ./tools/_watchdog.sh --install > /dev/null 2>&1;;
  esac
  sleep 300
done
