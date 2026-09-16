#!/usr/bin/env bash
# EDGEBAL 전환 (s2·s5; 계획 research_log/PAN_EDGEBAL_S2_S5_Experiment_Plan_2026-09-16.md §10, 노트 research_log/2026-09-16_edgebal-implementation.md) — pull 뒤 한 번.
#   ./tools/edgebal_switch.sh --dry-run   # 검사만: gate(K01–K43) · cue verify(이 서버 T0) · 명시 순서 config == 생성기 · 재사용 대조(§5/§6 J0/JQ/QE50) 완결 검증 · 예약 표 — 운영 파일은 **쓰지 않는다**
#   ./tools/edgebal_switch.sh             # 적용: 위 검사 + 서버 로컬 mandatory(PAKD50 + EDGEBAL)·reservations · 옛 extra_priority 보존 분리 · chain 마감 파일 제거(상한 없음)
#                                         #      · chain 이 살아 있으면 **그대로 둔다**(현재 run — s5 의 QE50 S777 등 — 은 그 runner 가 끝낸다; 다음 gate pass 부터 새 순서) · 없으면(DONE) 큐 config/queues/edgebal_<srv>.txt 로 campaign_start
#                                         #      · 대기자 tools/edgebal_waiter.sh (DONE-with-pending 재기동)
# 원칙(QEDGE9 감사 F01–F08 계승): runner 를 죽이지 않고 run 경계 인계 · Sheet 만 보고 kill 하지 않는다 · dry-run 무기록 · 완료 ID 덮어쓰기 없음 · 실측 통합(PAKD50+QEDGE9+QEGX+EDGEBAL ledger)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$("$PY" -c "import sys; sys.path.insert(0, '.'); from tools.gen_pakd50_configs import server_id; print(server_id(open('gspread/server.txt').read()))")"; DRY=0
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; *) echo "!! 알 수 없는 인자 $1"; exit 1;; esac; shift; done
fail() { echo "!! $1"; exit 1; }
case "$SERVER" in s2|s5) ;; *) fail "EDGEBAL switch 는 s2·s5 용 (이 서버: $SERVER)";; esac
[ -f work_dir/_pakd50/ledger.json ] || fail "work_dir/_pakd50/ledger.json 없음 — 이 서버는 pakd50_prepare 를 아직 안 했다"
grep -qx 'pakd50' work_dir/campaign_gates_enabled.txt 2>/dev/null || fail "work_dir/campaign_gates_enabled.txt 에 pakd50 이 없다 (편성 gate)"
CAMP=work_dir/_edgebal; QUEUE="config/queues/edgebal_${SERVER}.txt"; [ -f "$QUEUE" ] || fail "$QUEUE 없음 (git pull)"
mkdir -p "$CAMP" work_dir/_edgebal_budget; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
echo "[edgebal] ① $SERVER — unit gate (K01–K43; 임시 fixture — 운영 파일 기록 없음)"
"$PY" tools/pakd50_unit_tests.py > "$CAMP/gate_$SERVER.log" 2>&1 || { grep -v Warning "$CAMP/gate_$SERVER.log" | grep "FAIL" | head -5; fail "unit gate 실패 — $CAMP/gate_$SERVER.log"; }; tail -1 "$CAMP/gate_$SERVER.log"
echo "[edgebal] ② cue 자산 verify (내부 일관성 · 이 서버 T0 aligner 로 96 base × 4 state 재계산; EB_QF* 의 g 표)"
"$PY" tools/qedge9_cue.py verify --n 96 2>&1 | grep -v Warning | tail -1 || fail "cue verify 실패 — work_dir/_qedge9/verify_$SERVER.json"
echo "[edgebal] ③ 명시 순서 config 검사 (생성기 == config/; 완료·실행 중 run 은 생략) + 재사용 대조(§5/§6: 이 서버 seed 의 J0/JQ/QE50) 완결 검증 (§7.3; 보고 — 불통과면 사람이 refresh 결정)"
mapfile -t ITEMS < <("$PY" -c "import sys; sys.path.insert(0, '.'); from tools.gen_pakd50_configs import priority_for, extra_priority; print('\n'.join(priority_for('$SERVER') + extra_priority()))")
"$PY" tools/gen_pakd50_configs.py --server "$SERVER" --cases "$(IFS=,; echo "${ITEMS[*]}")" --out-dir "$TMP/cfg" 2>&1 | grep -v Warning | tail -1 | cut -c1-120
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "config 생성 실패"
"$PY" - "$SERVER" "$TMP/cfg" "$DRY" <<'PYEOF' || fail "config/control 검사 실패"
import filecmp, json, os, subprocess, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal, complete
srv, tmp, dry = sys.argv[1], sys.argv[2], sys.argv[3] == "1"; seed = G.SERVER_SEED[srv]; ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout; bad = []; ver = {}
for c in G.priority_for(srv) + G.extra_priority():
    run = G.to_tag(c, seed); f = os.path.join("config", run + ".yaml"); t = os.path.join(tmp, run + ".yaml"); br = G.branch_for(srv, c) or "PAKD50"
    st = "완료/실패" if terminal(run) else ("실행 중" if any("main.py" in l and "--config" in l and run in l for l in ps.splitlines()) else ("대기" if G.cue_ready(c) else "대기(cue 미준비)"))
    if complete(run):
        v = G.verified_complete(run); ver[run] = v; st += " · 검증 " + ("OK" if v["ok"] else "!! 불통과 " + str([k for k, x in v["checks"].items() if x is False]))
    if br == "EDGEBAL" and not G.is_tag(c):
        bad.append(f"{c}: EDGEBAL 항목은 run 이름(seed·version 포함) 이어야 한다 (s2 v1 · s5 v2)")
    if st.startswith("완료") or st.startswith("실행 중"):
        print(f"   {c:<50} {br:<8} {st}"); continue
    if not os.path.exists(f):
        bad.append(f"{c}: config/{run}.yaml 없음 (git pull)"); continue
    same = filecmp.cmp(f, t, shallow=False); print(f"   {c:<50} {br:<8} {st} — config {'== 생성기' if same else '!= 생성기'}"); same or bad.append(f"{c}: config/{run}.yaml 가 생성기 출력과 다르다")
print("   재사용 대조(§7.3 전제: 완주·exact50K·후보 격자·Teacher T0·train sha·init hash):")
for sd, ctl in (G.EDGEBAL_CONTROLS.get(srv) or {}).items():
    for role, run in ctl.items():
        if complete(run):
            v = G.verified_complete(run); ver[run] = v; print(f"   S{sd} {role:<5} {run:<52} 완료 · 검증 {'OK' if v['ok'] else '!! 불통과 ' + str([k for k, x in v['checks'].items() if x is False])}")
        else:
            print(f"   S{sd} {role:<5} {run:<52} {'실행 중' if any('main.py' in l and run in l for l in ps.splitlines()) else '미완/없음'} — 이 seed 의 대조는 같은 큐의 EB_N0/EB_R3E100 또는 완료 뒤 사용 (Sheet 만 보고 kill 하지 않는다)")
if not dry:
    json.dump(ver, open(os.path.join(G.ROOT, "work_dir", "_edgebal", "control_verification.json"), "w"), indent=1, ensure_ascii=False)
if bad:
    print("\n".join("!! " + b for b in bad)); sys.exit(1)
PYEOF
echo "[edgebal] ④ 예약 표 (실측은 PAKD50 + QEDGE9 + QEGX + EDGEBAL ledger 통합; 계획 §7.2 s2 31.59 h / s5 26.57 h)"; "$PY" tools/gen_pakd50_configs.py --plan --server "$SERVER" 2>&1 | grep -v Warning
if [ "$DRY" = 1 ]; then
  "$PY" - "$SERVER" "$TMP" <<'PYEOF'
import sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
srv, tmp = sys.argv[1], sys.argv[2]; res = G.write_reservation_file(srv, G.priority_for(srv) + G.extra_priority(), G.measured_hours_all(srv), path=tmp + "/reservations.json")
print(f"   (dry-run) reservations 미리보기 {len(res['runs'])} run → 임시 경로 (운영 파일 미기록) · extra_priority 현재 {G.extra_priority() or '없음'}")
PYEOF
  echo "[edgebal] --dry-run — 운영 파일·chain·cron 을 바꾸지 않았다"; exit 0
fi
echo "[edgebal] ⑤ 서버 로컬 기본 묶음·예약 파일 · 옛 extra_priority 보존 분리"
"$PY" - "$SERVER" <<'PYEOF' || fail "로컬 파일 기록 실패"
import os, sys, time; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
srv = sys.argv[1]; xp = os.path.join(G.ROOT, G.EXTRA_PRIORITY_FILE); cur = G.extra_priority(); old = [x for x in cur if G.branch_for(srv, x) != "EDGEBAL"]; keep = [x for x in cur if G.branch_for(srv, x) == "EDGEBAL"]   # 감사 F04: 옮기기 전에 메모리에 읽는다
if old:
    keep = xp.replace("extra_priority.txt", f"extra_priority.pre_edgebal_{time.strftime('%m%d-%H%M')}.txt"); os.replace(xp, keep)
    open(xp, "w").write(f"# EDGEBAL 전환({time.strftime('%Y-%m-%dT%H:%M')}): 이전 추가 편성 {len(old)} 항목은 {os.path.basename(keep)} 에 보존 — EDGEBAL 의 추가 편성(§9 조건부 확장; run 이름) 만 여기에\n" + "\n".join(keep) + "\n")
    print(f"   extra_priority {len(old)} 항목(비 EDGEBAL) → {os.path.basename(keep)} (보존; 편성에서 제외)")
runs = G.write_mandatory_file(srv); m = G.measured_hours_all(srv); res = G.write_reservation_file(srv, G.priority_for(srv) + G.extra_priority(), m)
eb = [G.to_tag(it, G.SERVER_SEED[srv]) for it in G.priority_for(srv) + G.extra_priority() if G.branch_for(srv, it) == "EDGEBAL"]; p = os.path.join(G.ROOT, G.EDGEBAL_MANDATORY_FILE); os.makedirs(os.path.dirname(p), exist_ok=True)
open(p, "w").write(f"# {srv} EDGEBAL run (자체 ledger {G.EDGEBAL_LEDGER}; 상한 없음, required 경고만; 대기자 tools/edgebal_waiter.sh 의 pending 목록) — tools/edgebal_switch.sh\n" + "\n".join(eb) + "\n")
print(f"   {G.MANDATORY_FILE}: {len(runs)} run · {G.EDGEBAL_MANDATORY_FILE}: {len(eb)} run · {G.RESERVATION_FILE}: {len(res['runs'])} run (실측 {m or '없음'})")
PYEOF
echo "[edgebal] ⑥ chain: 상한 없음 — 마감 파일 제거; runner 는 그대로(현재 run·후처리 유지), 없으면 기동; 대기자 기동"
if [ -f work_dir/cases_deadline.txt ]; then cp work_dir/cases_deadline.txt "$CAMP/cases_deadline.before_edgebal.txt"; rm -f work_dir/cases_deadline.txt; echo "   cases_deadline.txt 제거 (사본 $CAMP/cases_deadline.before_edgebal.txt) — 새 run 시작에 절대 마감 없음"; fi
if ps -eo args | grep -q '[_]run_cases\.sh'; then
  echo "   chain 살아 있음 — 손대지 않는다. 현재 run 은 그 runner 가 끝내고(평가·업로드 포함) 다음 gate pass 부터 새 순서 (gate 는 매 pass tools/campaign_gate.py 를 새로 부른다)"
else
  ./tools/campaign_start.sh --queue "$QUEUE" --hours 24 --label "edgebal-$SERVER-$(date +%m%d-%H%M)" || fail "기동 실패"; rm -f work_dir/cases_deadline.txt; ./tools/_watchdog.sh --install > /dev/null && echo "   chain 기동($QUEUE) · 마감 파일 제거 · 감시자 cron 등록"
fi
setsid nohup ./tools/edgebal_waiter.sh >> "$CAMP/waiter.log" 2>&1 < /dev/null & sleep 1; echo "   대기자 pid $! ($CAMP/waiter.log · 상태 $CAMP/status.json)"
echo "[edgebal] 완료 — $SERVER 명시 순서 (EDGEBAL run 은 마감 admission 제외; EB_QF* 는 cue 자산이 있어야 열린다). 확인: tail -f work_dir/cases_chain.log · $CAMP/status.json"
