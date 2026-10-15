#!/usr/bin/env bash
# QEDGE9 s1 (2026-09-15 17:20 사용자 결정: seed 1234 묶음을 s4 대신 s1 이) — 계획 research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md §6, 노트 research_log/2026-09-15_qedge9-implementation.md §9
#   ./tools/qedge9_prepare_s1.sh [--dry-run]
#   ① gate(K01–K33) ② cue verify(s1 T0) ③ 큐 config/queues/qedge9_s1.txt 의 5 config(v2) == 생성기 · 예약 표 ④ (apply) s1 로컬 mandatory/reservations · QEDGE9 mandatory
#   ⑤ (apply) waiter 둘을 detached 로: launch_when_idle(현재 SMEC12 chain 이 DONE 되고 학습 프로세스가 없으면 campaign_start → 마감 파일 제거 = soft 정책) · pilot_when_ready(J0 v2 exact50K 가 생기면 c_E 산출; QEC 는 큐 마지막)
#   gate 'pakd50' 는 s1 에서 꺼져 있다(work_dir/campaign_gates_enabled.txt 비어 있음) — 큐 순서 그대로 돈다. 기존 SMEC12 chain·runner 는 건드리지 않는다.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; DRY=0
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; *) echo "!! 알 수 없는 인자 $1"; exit 1;; esac; shift; done
fail() { echo "!! $1"; exit 1; }
[ "$SERVER" = s1 ] || fail "이 스크립트는 s1 용 (이 서버: $SERVER; s5 는 tools/qedge9_switch.sh)"
QUEUE=config/queues/qedge9_s1.txt; CAMP=work_dir/_qedge9; mkdir -p "$CAMP" work_dir/_qedge9_budget; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
[ -s work_dir/campaign_gates_enabled.txt ] && fail "work_dir/campaign_gates_enabled.txt 가 비어 있지 않다 — s1 의 QEDGE9 는 큐 순서로만 돈다(옛 gate 가 다른 run 을 재주입하지 않게 비운다)"
echo "[qedge9-s1] ① unit gate (K01–K33)"
"$PY" tools/pakd50_unit_tests.py > "$CAMP/gate_s1.log" 2>&1 || { grep -v Warning "$CAMP/gate_s1.log" | grep FAIL | head -5; fail "unit gate 실패 — $CAMP/gate_s1.log"; }; tail -1 "$CAMP/gate_s1.log"
echo "[qedge9-s1] ② cue 자산 verify"; "$PY" tools/qedge9_cue.py verify --n 96 2>&1 | grep -v Warning | tail -1 || fail "cue verify 실패"
echo "[qedge9-s1] ③ 큐 config == 생성기 (v2) · 예약 표"
mapfile -t RUNS < <(grep -vE '^[[:space:]]*(#|$)' "$QUEUE")
"$PY" tools/gen_pakd50_configs.py --server s1 --cases "$(IFS=,; echo "${RUNS[*]}")" --out-dir "$TMP/cfg" 2>&1 | grep -v Warning | tail -1 | cut -c1-100; [ "${PIPESTATUS[0]}" -eq 0 ] || fail "config 생성 실패"
"$PY" - "$TMP/cfg" "${RUNS[@]}" <<'PYEOF' || fail "config 검사 실패"
import filecmp, os, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal
tmp = sys.argv[1]; bad = []
for r in sys.argv[2:]:
    f = os.path.join("config", r + ".yaml"); t = os.path.join(tmp, r + ".yaml"); st = "완료/실패" if terminal(r) else "대기"
    if not os.path.exists(f): bad.append(f"{r}: config 없음"); continue
    same = filecmp.cmp(f, t, shallow=False); print(f"   {r:<52} {G.branch_for('s1', r)} {st} — config {'== 생성기' if same else '!= 생성기'}"); same or bad.append(f"{r}: config != 생성기")
    assert r.endswith("_v2") and G.branch_for("s1", r) == "QEDGE9", r
if bad: print("\n".join("!! " + b for b in bad)); sys.exit(1)
PYEOF
"$PY" tools/gen_pakd50_configs.py --plan --server s1 2>&1 | grep -v Warning
if [ "$DRY" = 1 ]; then echo "[qedge9-s1] --dry-run — 운영 파일·waiter 를 만들지 않았다"; exit 0; fi
echo "[qedge9-s1] ④ s1 로컬 예약·mandatory"
"$PY" - <<'PYEOF' || fail "로컬 파일 기록 실패"
import os, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
runs = G.write_mandatory_file("s1"); res = G.write_reservation_file("s1", G.priority_for("s1"), G.measured_hours_all("s1"))
p = os.path.join(G.ROOT, G.QEDGE9_MANDATORY_FILE); os.makedirs(os.path.dirname(p), exist_ok=True)
open(p, "w").write("# s1 QEDGE9 run (seed 1234, v2; 자체 ledger work_dir/_qedge9_budget; soft 9h 경고만) — tools/qedge9_prepare_s1.sh\n" + "\n".join(G.priority_for("s1")) + "\n")
print(f"   {G.MANDATORY_FILE}: {len(runs)} · {G.QEDGE9_MANDATORY_FILE}: {len(G.priority_for('s1'))} · {G.RESERVATION_FILE}: {len(res['runs'])}")
PYEOF
echo "[qedge9-s1] ⑤ waiter 기동 (detached)"
cat > "$CAMP/launch_when_idle_s1.sh" <<EOS
#!/usr/bin/env bash
cd "$REPO"; export PANCRAFTER_DLPAN="$PANCRAFTER_DLPAN"
while ps -eo args | grep -q '[_]run_cases\.sh' || ps -eo args | grep -q '[m]ain\.py --config'; do sleep 300; done
./tools/campaign_start.sh --queue "$QUEUE" --hours 24 --label "qedge9-s1-\$(date +%m%d-%H%M)" && rm -f work_dir/cases_deadline.txt && ./tools/_watchdog.sh --install
echo "[qedge9-s1] launched \$(date -Iseconds) (마감 파일 없음 = soft 정책)" >> "$CAMP/launch_s1.log"
EOS
cat > "$CAMP/pilot_when_ready_s1.sh" <<EOS
#!/usr/bin/env bash
cd "$REPO"; export PANCRAFTER_DLPAN="$PANCRAFTER_DLPAN"; PR=PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v2
while ! { [ -f "work_dir/\$PR/last_meta.json" ] && [ -f "work_dir/\$PR/last/model.safetensors" ] && [ -f "work_dir/\$PR/results/full_best_hqnr.mat" ]; }; do sleep 300; done
sleep 60; echo "[qedge9-s1] \$(date -Iseconds) pilot \$PR 완료 → c_E" >> "$CAMP/launch_s1.log"; "$PY" tools/qedge9_cue.py pilot --run "\$PR" --tag last >> "$CAMP/launch_s1.log" 2>&1; echo "[qedge9-s1] pilot rc=\$?" >> "$CAMP/launch_s1.log"
EOS
chmod +x "$CAMP/launch_when_idle_s1.sh" "$CAMP/pilot_when_ready_s1.sh"
setsid nohup "$CAMP/launch_when_idle_s1.sh" >> "$CAMP/launch_s1.log" 2>&1 < /dev/null & echo "   launch waiter pid $!"
setsid nohup "$CAMP/pilot_when_ready_s1.sh" >> "$CAMP/launch_s1.log" 2>&1 < /dev/null & echo "   pilot waiter pid $!"
echo "[qedge9-s1] 준비 완료 — SMEC12 chain 이 끝나면 자동으로 J0→JQ→QE50→QES→QEC (v2) 를 돈다. 진행: tail -f work_dir/cases_chain.log · $CAMP/launch_s1.log"
