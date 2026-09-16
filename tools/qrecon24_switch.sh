#!/usr/bin/env bash
# QRECON24 전환 (s1–s5; 계획 research_log/PAN_QRECON24_S1_S5_FixedMethod_Tuning_Plan_2026-09-16.md §9–§10, 노트 research_log/2026-09-16_qrecon24-implementation.md) — pull 뒤 한 번.
#   ./tools/qrecon24_switch.sh --dry-run   # 검사만: gate(K01–K48) · cue verify(이 서버 T0) · 명시 큐 config == 생성기 · 예약 표 — 운영 파일은 **쓰지 않는다**
#   ./tools/qrecon24_switch.sh             # 적용: 위 검사 + 서버 로컬 mandatory(PAKD50 + QRECON24)·reservations · 옛 extra_priority 보존 분리 · chain 마감 파일 제거(상한 없음)
#                                          #      · chain 이 살아 있으면 **그대로 둔다**(현재 run 은 그 runner 가 마지막 update 까지; gate 가 켜진 서버는 다음 pass 부터 새 순서, s1 은 chain DONE 뒤 대기자가 큐로 재기동)
#                                          #      · chain 이 없으면 큐 config/queues/qrecon24_<srv>.txt 로 campaign_start · 대기자 tools/qrecon24_waiter.sh
#   ./tools/qrecon24_switch.sh --extend    # §8.3: 기본 큐를 다 마쳤는데 신규 완료 Train(h) 합 < 24h 이면 지정 3 run(추가 seed) 을 extra_priority 에 추가 (+config 생성)
# 원칙(§9.1·§10.1): Sheet 만 보고 프로세스를 kill 하지 않는다 · 완료 ID 덮어쓰기 없음 · 실행 중 코드에 pull 하지 않는다(현재 run 은 그 코드로 끝난다) · 실패/NaN 자동 반복 없음
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$("$PY" -c "import sys; sys.path.insert(0, '.'); from tools.gen_pakd50_configs import server_id; print(server_id(open('gspread/server.txt').read()))")"; DRY=0; EXTEND=0
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --extend) EXTEND=1;; *) echo "!! 알 수 없는 인자 $1"; exit 1;; esac; shift; done
fail() { echo "!! $1"; exit 1; }
case "$SERVER" in s1|s2|s3|s4|s5) ;; *) fail "QRECON24 switch 는 s1–s5 용 (이 서버: $SERVER)";; esac
[ -f work_dir/_pakd50/ledger.json ] || fail "work_dir/_pakd50/ledger.json 없음 — 이 서버는 pakd50_prepare 를 아직 안 했다"
CAMP=work_dir/_qrecon24; QUEUE="config/queues/qrecon24_${SERVER}.txt"; [ -f "$QUEUE" ] || fail "$QUEUE 없음 (git pull)"
mkdir -p "$CAMP" work_dir/_qrecon24_budget; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
GATE_ON=0; grep -qx 'pakd50' work_dir/campaign_gates_enabled.txt 2>/dev/null && GATE_ON=1
if [ "$EXTEND" = 1 ]; then
  [ "$DRY" = 1 ] && fail "--extend 는 --dry-run 과 같이 쓰지 않는다"
  "$PY" - "$SERVER" <<'PYEOF' || fail "--extend 실패"
import json, os, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal, complete
srv = sys.argv[1]; base = G.priority_for(srv); pend = [r for r in base if not terminal(r)]
if pend: sys.exit(f"!! 기본 큐 미완 {len(pend)} run — 확장은 기본 큐를 다 마친 뒤 (§8.3)")
m = G.measured_hours_all(srv); led = G._load_json(G.QRC24_LEDGER); hrs = [float(e.get("hours_total") or e.get("hours") or 0) for rid, e in (led.get("entries") or {}).items() if e.get("kind") == "run" and str(e.get("status", "")).startswith("FINISHED") and "QRC24" in rid and "#" not in rid]
tot = sum(hrs); print(f"   신규 완료 Train(h) 합 {tot:.2f} h ({len(hrs)} run; ledger {G.QRC24_LEDGER})")
if tot >= 24.0: sys.exit("   ≥ 24h — 확장 불필요 (§8.3)")
sd0, profs = G.QRC24_EXTRA[srv]; seeds = [sd0] + list(G.QRC24_RESERVE_SEEDS); items = None
for sd in seeds:
    cand = [G.qrc24_run_name(srv, p, sd) for p in profs]
    if not any(complete(r) for r in cand): items = cand; break
if items is None: sys.exit("!! 지정 seed 와 reserve seed 전부 이미 결과가 있다 — 사람이 결정")
G.generate(srv, items, os.path.join(G.ROOT, "config"), projected=None); xp = os.path.join(G.ROOT, G.EXTRA_PRIORITY_FILE); cur = G.extra_priority()
with open(xp, "a") as fh: fh.write("# QRECON24 §8.3 확장 (신규 완료 Train(h) 합 < 24h)\n" + "\n".join(r for r in items if r not in cur) + "\n")
print("   추가:", items, "→", G.EXTRA_PRIORITY_FILE, "(config 생성; gate/대기자가 편성)")
PYEOF
  exit 0
fi
echo "[qrecon24] ① $SERVER — unit gate (K01–K48; 임시 fixture — 운영 파일 기록 없음)"
"$PY" tools/pakd50_unit_tests.py > "$CAMP/gate_$SERVER.log" 2>&1 || { grep -v Warning "$CAMP/gate_$SERVER.log" | grep "FAIL" | head -5; fail "unit gate 실패 — $CAMP/gate_$SERVER.log"; }; tail -1 "$CAMP/gate_$SERVER.log"
echo "[qrecon24] ② cue 자산 verify (raw q 의 출처: 이 서버 T0 aligner 로 96 base × 4 state 재계산)"
"$PY" tools/qedge9_cue.py verify --n 96 2>&1 | grep -v Warning | tail -1 || fail "cue verify 실패 — work_dir/_qedge9/verify_$SERVER.json"
echo "[qrecon24] ③ 명시 큐 config 검사 (생성기 == config/; 완료·실행 중 run 은 생략; 서버 토큰 == 이 서버)"
mapfile -t ITEMS < <(grep -vE '^[[:space:]]*(#|$)' "$QUEUE")
"$PY" tools/gen_pakd50_configs.py --server "$SERVER" --cases "$(IFS=,; echo "${ITEMS[*]}")" --out-dir "$TMP/cfg" 2>&1 | grep -v Warning | tail -1 | cut -c1-120
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "config 생성 실패"
"$PY" - "$SERVER" "$TMP/cfg" "$DRY" "${ITEMS[@]}" <<'PYEOF' || fail "config 검사 실패"
import filecmp, json, os, subprocess, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal, complete
srv, tmp, dry = sys.argv[1], sys.argv[2], sys.argv[3] == "1"; items = sys.argv[4:]; ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout; bad = []
assert items == G.priority_for(srv), "큐 파일 != 생성기 명시 순서 (git pull)"
for run in items + G.extra_priority():
    f = os.path.join("config", run + ".yaml"); t = os.path.join(tmp, run + ".yaml"); br = G.branch_for(srv, run) or "PAKD50"
    if br == "QRECON24" and G.qrc24_parse(G.case_of(run))[0] != srv: bad.append(f"{run}: 서버 토큰이 {srv} 가 아니다"); continue
    st = "완료/실패" if terminal(run) else ("실행 중" if any("main.py" in l and "--config" in l and run in l for l in ps.splitlines()) else ("대기" if G.cue_ready(run) else "대기(cue 미준비)"))
    if st.startswith("완료") or st.startswith("실행 중"):
        print(f"   {run:<62} {br:<9} {st}"); continue
    if not os.path.exists(f): bad.append(f"{run}: config 없음 (git pull)"); continue
    if not os.path.exists(t): print(f"   {run:<62} {br:<9} {st} — (extra; 생성기 비교 생략)"); continue
    same = filecmp.cmp(f, t, shallow=False); print(f"   {run:<62} {br:<9} {st} — config {'== 생성기' if same else '!= 생성기'}"); same or bad.append(f"{run}: config != 생성기")
alive = any("tools/_run_cases.sh" in l for l in ps.splitlines()); cur = [l for l in ps.splitlines() if "main.py" in l and "--config" in l]
print("   chain:", "살아 있음" if alive else "없음", "· 학습 중:", (cur[0][:120] if cur else "없음"), "— 진행 중 run 은 마지막 update 까지 둔다 (§9.1)")
if bad: print("\n".join("!! " + b for b in bad)); sys.exit(1)
PYEOF
echo "[qrecon24] ④ 예약 표 (§8.2 1.20×R_s + 10/60; 실측은 PAKD50+QEDGE9+QEGX+EDGEBAL+QRECON24 ledger 통합)"; "$PY" tools/gen_pakd50_configs.py --plan --server "$SERVER" 2>&1 | grep -v Warning
if [ "$DRY" = 1 ]; then echo "[qrecon24] --dry-run — 운영 파일·chain·cron 을 바꾸지 않았다 (gate 'pakd50' $( [ "$GATE_ON" = 1 ] && echo 켜짐 || echo 꺼짐 ))"; exit 0; fi
echo "[qrecon24] ⑤ 서버 로컬 기본 묶음·예약 파일 · 옛 extra_priority 보존 분리"
"$PY" - "$SERVER" <<'PYEOF' || fail "로컬 파일 기록 실패"
import os, sys, time; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
srv = sys.argv[1]; xp = os.path.join(G.ROOT, G.EXTRA_PRIORITY_FILE); old = [x for x in G.extra_priority() if G.branch_for(srv, x) != "QRECON24"]
if old:
    keep = xp.replace("extra_priority.txt", f"extra_priority.pre_qrecon24_{time.strftime('%m%d-%H%M')}.txt"); os.replace(xp, keep)
    open(xp, "w").write(f"# QRECON24 전환({time.strftime('%Y-%m-%dT%H:%M')}): 이전 추가 편성 {len(old)} 항목은 {os.path.basename(keep)} 에 보존 — QRECON24 의 추가(§8.3 --extend; run 이름) 만 여기에\n" + "\n".join(x for x in G.extra_priority() if G.branch_for(srv, x) == "QRECON24") + "\n")
    print(f"   extra_priority {len(old)} 항목(비 QRECON24) → {os.path.basename(keep)} (보존; 편성에서 제외 — 이전 QEDGE9/QEGX/EDGEBAL 미완 항목은 superseded, 필요하면 run 이름으로 다시 적는다)")
runs = G.write_mandatory_file(srv); m = G.measured_hours_all(srv); res = G.write_reservation_file(srv, G.priority_for(srv) + G.extra_priority(), m)
q = [it for it in G.priority_for(srv) + G.extra_priority() if G.branch_for(srv, it) == "QRECON24"]; p = os.path.join(G.ROOT, G.QRC24_MANDATORY_FILE); os.makedirs(os.path.dirname(p), exist_ok=True)
open(p, "w").write(f"# {srv} QRECON24 run (자체 ledger {G.QRC24_LEDGER}; 상한 없음, required 경고만; 대기자 tools/qrecon24_waiter.sh 의 pending 목록) — tools/qrecon24_switch.sh\n" + "\n".join(q) + "\n")
print(f"   {G.MANDATORY_FILE}: {len(runs)} run · {G.QRC24_MANDATORY_FILE}: {len(q)} run · {G.RESERVATION_FILE}: {len(res['runs'])} run (실측 {m or '없음'})")
PYEOF
echo "[qrecon24] ⑥ chain: 상한 없음 — 마감 파일 제거; runner 는 그대로(현재 run·후처리 유지), 없으면 기동; 대기자 기동"
if [ -f work_dir/cases_deadline.txt ]; then cp work_dir/cases_deadline.txt "$CAMP/cases_deadline.before_qrecon24.txt"; rm -f work_dir/cases_deadline.txt; echo "   cases_deadline.txt 제거 (사본 $CAMP/cases_deadline.before_qrecon24.txt)"; fi
if ps -eo args | grep -q '[_]run_cases\.sh'; then
  if [ "$GATE_ON" = 1 ]; then echo "   chain 살아 있음 — 손대지 않는다. 현재 run 은 그 runner 가 끝내고 다음 gate pass 부터 QRECON24 순서"; else echo "   chain 살아 있음(gate 꺼짐: $SERVER) — 현재 큐를 끝내면 DONE; 대기자가 QRECON24 큐로 재기동한다"; fi
else
  ./tools/campaign_start.sh --queue "$QUEUE" --hours 24 --label "qrecon24-$SERVER-$(date +%m%d-%H%M)" || fail "기동 실패"; rm -f work_dir/cases_deadline.txt; ./tools/_watchdog.sh --install > /dev/null && echo "   chain 기동($QUEUE) · 마감 파일 제거 · 감시자 cron 등록"
fi
setsid nohup ./tools/qrecon24_waiter.sh >> "$CAMP/waiter.log" 2>&1 < /dev/null & sleep 1; echo "   대기자 pid $! ($CAMP/waiter.log · 상태 $CAMP/status.json)"
echo "[qrecon24] 완료 — $SERVER 명시 큐 $(grep -cvE '^[[:space:]]*(#|$)' "$QUEUE") run (마감 admission 제외). 확인: tail -f work_dir/cases_chain.log · $CAMP/status.json · 결과 selector: python tools/qrecon24_select.py <run>"
