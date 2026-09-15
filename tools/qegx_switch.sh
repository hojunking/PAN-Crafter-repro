#!/usr/bin/env bash
# QEGX 전환 (s3·s4; 계획 research_log/PAN_QEGX_S3_S4_W104D121_Experiment_Plan_2026-09-15.md §11.4·§12, 노트 research_log/2026-09-15_qegx-implementation.md) — pull 뒤 한 번.
#   ./tools/qegx_switch.sh --dry-run          # 검사만: gate(K01–K38) · cue verify(이 서버 T0) · 명시 순서 config == 생성기 · (s4) 기존 W104 control 완결 검증 · 예약 표 — 운영 파일은 **쓰지 않는다**
#   ./tools/qegx_switch.sh                    # 적용: 위 검사 + 서버 로컬 mandatory(PAKD50 + QEGX)·reservations · 옛 extra_priority 보존 분리 · chain 마감 파일 제거(상한 없음)
#                                             #      · chain 이 살아 있으면 **그대로 둔다**(현재 run·후처리·업로드는 그 runner 가 끝낸다; 다음 gate pass 부터 새 순서) · 없으면(DONE) 큐 config/queues/qegx_<srv>.txt 로 campaign_start
#                                             #      · 대기자 tools/qegx_waiter.sh (DONE-with-pending 재기동 · (s3) J0 S2026 v2 exact50K 가 생기면 QEC3 의 c_E3 pilot 자동)
#   ./tools/qegx_switch.sh --pilot            # (s3) c_E3 만 (없을 때) — 대기자가 다음 tick 에 QEC3 를 연다
#   ./tools/qegx_switch.sh --refresh-controls # (s4) 재사용 전제 불통과 시: J0/JQ/XJ@W104 S1234 **v3** 를 QEGX branch 로 extra_priority 에 추가 (+config 생성) — 계획 §6 '새 이름으로 다시'
# 감사 원칙(QEDGE9 F01–F08 계승): runner 를 죽이지 않고 run 경계 인계 · dry-run 무기록 · pilot identity 고정 · 완료 검증 보고 · 실측 통합(PAKD50+QEDGE9+QEGX ledger) · extra 보존 분리
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$("$PY" -c "import sys; sys.path.insert(0, '.'); from tools.gen_pakd50_configs import server_id; print(server_id(open('gspread/server.txt').read()))")"; DRY=0; PILOT_ONLY=0; REFRESH=0
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --pilot) PILOT_ONLY=1;; --refresh-controls) REFRESH=1;; *) echo "!! 알 수 없는 인자 $1"; exit 1;; esac; shift; done
fail() { echo "!! $1"; exit 1; }
case "$SERVER" in s3|s4) ;; *) fail "QEGX switch 는 s3·s4 용 (이 서버: $SERVER)";; esac
[ -f work_dir/_pakd50/ledger.json ] || fail "work_dir/_pakd50/ledger.json 없음 — 이 서버는 pakd50_prepare 를 아직 안 했다"
grep -qx 'pakd50' work_dir/campaign_gates_enabled.txt 2>/dev/null || fail "work_dir/campaign_gates_enabled.txt 에 pakd50 이 없다 (편성 gate)"
CAMP=work_dir/_qegx; QUEUE="config/queues/qegx_${SERVER}.txt"; [ -f "$QUEUE" ] || fail "$QUEUE 없음 (git pull)"
mkdir -p "$CAMP" work_dir/_qegx_budget; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
PR3=PAKD50_J0_W104_D121_WV3_T0_S2026_FRESH50_v2
pilot_ce3() {   # s3: c_E3 가 없을 때만 산출; 있으면 출처만 대조 (감사 F02 '산출 후 고정')
  if [ -f "$CAMP/qec3_cE.json" ]; then "$PY" tools/qedge9_cue.py pilot --branch qegx --run "$PR3" --tag last 2>&1 | grep -v Warning | tail -1; return; fi
  if [ -f "work_dir/$PR3/last_meta.json" ] && [ -f "work_dir/$PR3/last/model.safetensors" ]; then
    "$PY" tools/qedge9_cue.py pilot --branch qegx --run "$PR3" --tag last 2>&1 | grep -v Warning | tail -2 || fail "QEC3 pilot c_E3 산출 실패"
  else
    echo "   pilot $PR3 의 exact50K last 없음 — QEC3 는 c_E3 가 생길 때까지 gate 가 건너뛴다 (대기자가 자동 산출; 수동은 ./tools/qegx_switch.sh --pilot)"
  fi
}
if [ "$PILOT_ONLY" = 1 ]; then [ "$SERVER" = s3 ] || fail "--pilot 은 s3 (J0 S2026 v2 exact50K) 에서"; echo "[qegx] QEC3 pilot"; pilot_ce3; exit 0; fi
echo "[qegx] ① $SERVER — unit gate (K01–K38; 임시 fixture — 운영 파일 기록 없음)"
"$PY" tools/pakd50_unit_tests.py > "$CAMP/gate_$SERVER.log" 2>&1 || { grep -v Warning "$CAMP/gate_$SERVER.log" | grep "FAIL" | head -5; fail "unit gate 실패 — $CAMP/gate_$SERVER.log"; }; tail -1 "$CAMP/gate_$SERVER.log"
echo "[qegx] ② cue 자산 verify (내부 일관성 · 이 서버 T0 aligner 로 96 base × 4 state 재계산)"
"$PY" tools/qedge9_cue.py verify --n 96 2>&1 | grep -v Warning | tail -1 || fail "cue verify 실패 — work_dir/_qedge9/verify_$SERVER.json"
echo "[qegx] ③ 명시 순서 config 검사 (생성기 == config/; 완료·실행 중 run 은 생략) + 기존 W104 control 완결 검증 (재사용 전제 §6; 보고 — 불통과면 --refresh-controls)"
mapfile -t ITEMS < <("$PY" -c "import sys; sys.path.insert(0, '.'); from tools.gen_pakd50_configs import priority_for, extra_priority; print('\n'.join(priority_for('$SERVER') + extra_priority()))")
"$PY" tools/gen_pakd50_configs.py --server "$SERVER" --cases "$(IFS=,; echo "${ITEMS[*]}")" --out-dir "$TMP/cfg" 2>&1 | grep -v Warning | tail -1 | cut -c1-120
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "config 생성 실패"
"$PY" - "$SERVER" "$TMP/cfg" "$DRY" "$REFRESH" <<'PYEOF' || fail "config/control 검사 실패"
import filecmp, json, os, subprocess, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal, complete
srv, tmp, dry, refresh = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4] == "1"; seed = G.SERVER_SEED[srv]; ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout; bad = []; ver = {}; need_refresh = []
for c in G.priority_for(srv) + G.extra_priority():
    run = G.to_tag(c, seed); f = os.path.join("config", run + ".yaml"); t = os.path.join(tmp, run + ".yaml"); br = G.branch_for(srv, c) or "E0"
    st = "완료/실패" if terminal(run) else ("실행 중" if any("main.py" in l and "--config" in l and run in l for l in ps.splitlines()) else ("대기" if G.cue_ready(c) else "대기(cue/c_E3 미준비)"))
    if complete(run):
        v = G.verified_complete(run); ver[run] = v; st += " · 검증 " + ("OK" if v["ok"] else "!! 불통과 " + str([k for k, x in v["checks"].items() if x is False]))
        if not v["ok"] and br == "E0" and G.case_of(c) in ("J0", "JQ", "XJ"):
            need_refresh.append(run)
    if br == "QEGX" and not (G.is_tag(c) or G.case_of(c) in G.QEGX_CASES):
        bad.append(f"{c}: QEGX 항목은 run 이름(seed·version 포함) 이어야 한다 — bare case 는 QEDGE9 자동 규칙으로 간다 (§12)")
    if st.startswith("완료") or st.startswith("실행 중"):
        print(f"   {c:<48} {br:<7} {st}"); continue
    if not os.path.exists(f):
        bad.append(f"{c}: config/{run}.yaml 없음 (git pull)"); continue
    same = filecmp.cmp(f, t, shallow=False); print(f"   {c:<48} {br:<7} {st} — config {'== 생성기' if same else '!= 생성기'}"); same or bad.append(f"{c}: config/{run}.yaml 가 생성기 출력과 다르다")
if srv == "s4":
    ctl = [G.to_tag(c, seed) for c in G.priority_for(srv) if G.branch_for(srv, c) is None and G.case_of(c) in ("J0", "JQ", "XJ")]
    miss = [r for r in ctl if not complete(r)]
    if miss or need_refresh:
        print(f"!! s4 재사용 전제(§6): 미완 {miss} · 검증 불통과 {need_refresh} — 대조 새로고침 필요: ./tools/qegx_switch.sh --refresh-controls (J0/JQ/XJ v3, 약 3h8m + XJ)")
        if not refresh:
            print("   (지금은 보고만 — 신규 14 run 은 그대로 편성된다; control 없이 최종 판정을 닫지 않는다)")
    else:
        print("   s4 control(J0/JQ/XJ@W104 S1234 v1) 완료·검증 OK — 재사용 (계획 §6 전제 1·3; 2·4 는 unit gate K27/K36)")
if refresh and srv == "s4" and not dry:
    xp = os.path.join(G.ROOT, G.EXTRA_PRIORITY_FILE); cur = G.extra_priority(); add = [r for r in G.QEGX_REFRESH_CONTROLS["s4"] if r not in cur]
    if add:
        G.generate("s4", add, os.path.join(G.ROOT, "config"), projected=None)
        with open(xp, "a") as fh:
            fh.write("# QEGX 대조 새로고침 (계획 §6: 재사용 전제 불통과 → J0/JQ/XJ v3, QEGX branch)\n" + "\n".join(add) + "\n")
        print(f"   refresh-controls: {add} → {G.EXTRA_PRIORITY_FILE} + config 생성")
if not dry:
    json.dump(ver, open(os.path.join(G.ROOT, "work_dir", "_qegx", "control_verification.json"), "w"), indent=1, ensure_ascii=False)
if bad:
    print("\n".join("!! " + b for b in bad)); sys.exit(1)
PYEOF
echo "[qegx] ④ 예약 표 (실측은 PAKD50 + QEDGE9 + QEGX ledger 통합; 계획 §9.2 s3 25.20 h / s4 26.28 h)"; "$PY" tools/gen_pakd50_configs.py --plan --server "$SERVER" 2>&1 | grep -v Warning
if [ "$DRY" = 1 ]; then
  "$PY" - "$SERVER" "$TMP" <<'PYEOF'
import sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
srv, tmp = sys.argv[1], sys.argv[2]; res = G.write_reservation_file(srv, G.priority_for(srv) + G.extra_priority(), G.measured_hours_all(srv), path=tmp + "/reservations.json")
print(f"   (dry-run) reservations 미리보기 {len(res['runs'])} run → 임시 경로 (운영 파일 미기록) · extra_priority 현재 {G.extra_priority() or '없음'}")
PYEOF
  echo "[qegx] --dry-run — 운영 파일·chain·cron 을 바꾸지 않았다"; exit 0
fi
echo "[qegx] ⑤ 서버 로컬 기본 묶음·예약 파일 · 옛 extra_priority 보존 분리"
"$PY" - "$SERVER" <<'PYEOF' || fail "로컬 파일 기록 실패"
import os, sys, time; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
srv = sys.argv[1]; xp = os.path.join(G.ROOT, G.EXTRA_PRIORITY_FILE); old = [x for x in G.extra_priority() if G.branch_for(srv, x) != "QEGX"]
if old:
    keep = xp.replace("extra_priority.txt", f"extra_priority.pre_qegx_{time.strftime('%m%d-%H%M')}.txt"); os.replace(xp, keep)
    open(xp, "w").write(f"# QEGX 전환({time.strftime('%Y-%m-%dT%H:%M')}): 이전 추가 편성 {len(old)} 항목은 {os.path.basename(keep)} 에 보존 — QEGX 의 추가 편성(대조 새로고침 v3 등) 만 여기에\n" + "\n".join(x for x in G.extra_priority() if G.branch_for(srv, x) == "QEGX") + "\n")
    print(f"   extra_priority {len(old)} 항목(비 QEGX) → {os.path.basename(keep)} (보존; 편성에서 제외)")
runs = G.write_mandatory_file(srv); m = G.measured_hours_all(srv); res = G.write_reservation_file(srv, G.priority_for(srv) + G.extra_priority(), m)
qx = [G.to_tag(it, G.SERVER_SEED[srv]) for it in G.priority_for(srv) + G.extra_priority() if G.branch_for(srv, it) == "QEGX"]; p = os.path.join(G.ROOT, G.QEGX_MANDATORY_FILE); os.makedirs(os.path.dirname(p), exist_ok=True)
open(p, "w").write(f"# {srv} QEGX run (자체 ledger {G.QEGX_LEDGER}; 상한 없음, required 경고만; 대기자 tools/qegx_waiter.sh 의 pending 목록) — tools/qegx_switch.sh\n" + "\n".join(qx) + "\n")
print(f"   {G.MANDATORY_FILE}: {len(runs)} run · {G.QEGX_MANDATORY_FILE}: {len(qx)} run · {G.RESERVATION_FILE}: {len(res['runs'])} run (실측 {m or '없음'})")
PYEOF
if [ "$SERVER" = s3 ]; then echo "[qegx] ⑥ QEC3 pilot c_E3 (§4.2; 없을 때만 산출, 있으면 출처 대조)"; pilot_ce3; else echo "[qegx] ⑥ (s4: QEC3 없음)"; fi
echo "[qegx] ⑦ chain: 상한 없음 — 마감 파일 제거; runner 는 그대로(현재 run·후처리 유지), 없으면 기동; 대기자 기동"
if [ -f work_dir/cases_deadline.txt ]; then cp work_dir/cases_deadline.txt "$CAMP/cases_deadline.before_qegx.txt"; rm -f work_dir/cases_deadline.txt; echo "   cases_deadline.txt 제거 (사본 $CAMP/cases_deadline.before_qegx.txt) — 새 run 시작에 절대 마감 없음"; fi
if ps -eo args | grep -q '[_]run_cases\.sh'; then
  echo "   chain 살아 있음 — 손대지 않는다. 현재 run 은 그 runner 가 끝내고(평가·업로드 포함) 다음 gate pass 부터 새 순서 (gate 는 매 pass tools/campaign_gate.py 를 새로 부른다)"
else
  ./tools/campaign_start.sh --queue "$QUEUE" --hours 24 --label "qegx-$SERVER-$(date +%m%d-%H%M)" || fail "기동 실패"; rm -f work_dir/cases_deadline.txt; ./tools/_watchdog.sh --install > /dev/null && echo "   chain 기동($QUEUE) · 마감 파일 제거 · 감시자 cron 등록"
fi
setsid nohup ./tools/qegx_waiter.sh >> "$CAMP/waiter.log" 2>&1 < /dev/null & sleep 1; echo "   대기자 pid $! ($CAMP/waiter.log · 상태 $CAMP/status.json)"
echo "[qegx] 완료 — $SERVER 명시 순서 (QEGX run 은 마감 admission 제외; gate/route run 은 cue 자산, QEC3 는 c_E3 가 있어야 열린다). 확인: tail -f work_dir/cases_chain.log · $CAMP/status.json"
