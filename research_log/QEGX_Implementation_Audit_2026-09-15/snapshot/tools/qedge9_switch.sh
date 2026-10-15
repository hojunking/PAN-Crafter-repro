#!/usr/bin/env bash
# QEDGE9 전환 (s5; s4 몫은 17:20 결정으로 s1 → tools/qedge9_prepare_s1.sh; 계획 research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md, 노트 research_log/2026-09-15_qedge9-implementation.md §8) — pull 뒤 한 번.
#   ./tools/qedge9_switch.sh --dry-run   # 검사만: gate(K01–K33) · cue verify · 명시 순서 config == 생성기 · 기존 W104 control 완결 검증 · 예약 표 — 운영 파일(mandatory/reservations/extra/마감/c_E) 은 **쓰지 않는다** (임시 경로)
#   ./tools/qedge9_switch.sh             # 적용: 위 검사 + 서버 로컬 mandatory(PAKD50 + QEDGE9)·reservations 기록 · 옛 extra_priority 보존 분리 · chain 마감 파일 제거(hard_deadline null)
#                                        #      · (s4) QEC pilot c_E 산출(없을 때만; 있으면 출처 대조) · chain 이 살아 있으면 **그대로 둔다**(현재 run·후처리·업로드는 그 runner 가 끝낸다; 다음 gate pass 부터 새 순서)
#                                        #      · chain 이 없으면(DONE) campaign_start · 대기자(tools/qedge9_waiter.sh) 기동: DONE-with-pending(QEC c_E 지연·gate pass 소진) 을 다시 연다
#   ./tools/qedge9_switch.sh --pilot     # (s4) c_E 만 (없을 때) — 대기자가 다음 tick 에 QEC 를 연다
# 감사 대응(2026-09-15): F01 runner 를 죽이지 않고 run 경계 인계 · dry-run 무기록 / F02 pilot identity·고정 / F05 대기자 / F06 control 완결 검증(보고) / F07 마감 파일 제거·실측 통합 / F08 extra 보존 분리
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; DRY=0; PILOT_ONLY=0
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --pilot) PILOT_ONLY=1;; *) echo "!! 알 수 없는 인자 $1"; exit 1;; esac; shift; done
fail() { echo "!! $1"; exit 1; }
case "$SERVER" in s5) ;; s4) fail "s4 의 QEDGE9 항목은 17:20 결정으로 s1 에 이관됐다 (s4 는 기존 E0 allocation 만) — 이 스크립트는 s5 용";; *) fail "QEDGE9 switch 는 s5 용 (이 서버: $SERVER; s1 은 tools/qedge9_prepare_s1.sh)";; esac
[ -f work_dir/_pakd50/ledger.json ] || fail "work_dir/_pakd50/ledger.json 없음 — 이 서버는 pakd50_prepare 를 아직 안 했다"
grep -qx 'pakd50' work_dir/campaign_gates_enabled.txt 2>/dev/null || fail "work_dir/campaign_gates_enabled.txt 에 pakd50 이 없다 (편성 gate)"
CAMP=work_dir/_qedge9; mkdir -p "$CAMP" work_dir/_qedge9_budget; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
pilot_ce() {   # s4: c_E 가 없을 때만 산출; 있으면 출처만 대조 (감사 F02 '산출 후 고정')
  PR=PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v1
  if [ -f work_dir/_qedge9/qec_cE.json ]; then "$PY" tools/qedge9_cue.py pilot --run "$PR" --tag last 2>&1 | grep -v Warning | tail -1; return; fi
  if [ -f "work_dir/$PR/last_meta.json" ] && [ -f "work_dir/$PR/last/model.safetensors" ]; then
    "$PY" tools/qedge9_cue.py pilot --run "$PR" --tag last 2>&1 | grep -v Warning | tail -2 || fail "QEC pilot c_E 산출 실패"
  else
    echo "   pilot $PR 의 exact50K last 없음 — QEC 는 c_E 가 생길 때까지 gate 가 건너뛴다 (뒤에 ./tools/qedge9_switch.sh --pilot)"
  fi
}
if [ "$PILOT_ONLY" = 1 ]; then [ "$SERVER" = s4 ] || fail "--pilot 은 s4 (W104 J0 S1234 exact50K) 에서"; echo "[qedge9] ③ QEC pilot"; pilot_ce; exit 0; fi
echo "[qedge9] ① $SERVER — unit gate (K01–K33; 임시 fixture 에서 — 운영 파일 기록 없음)"
"$PY" tools/pakd50_unit_tests.py > "$CAMP/gate_$SERVER.log" 2>&1 || { grep -v Warning "$CAMP/gate_$SERVER.log" | grep "FAIL" | head -5; fail "unit gate 실패 — $CAMP/gate_$SERVER.log"; }; tail -1 "$CAMP/gate_$SERVER.log"
echo "[qedge9] ② cue 자산 verify (내부 일관성 · 이 서버 T0 aligner 로 96 base × 4 state 재계산 · θq 밖 라벨 flip 0)"
"$PY" tools/qedge9_cue.py verify --n 96 2>&1 | grep -v Warning | tail -1 || fail "cue verify 실패 — work_dir/_qedge9/verify_$SERVER.json"
echo "[qedge9] ③ 명시 순서 config 검사 (생성기 == config/; 완료·실행 중 run 은 생략) + 기존 W104 control 완결 검증 (감사 F06; 보고만 — 재실행 여부는 사람이)"
"$PY" tools/gen_pakd50_configs.py --server "$SERVER" --cases "$("$PY" -c "from tools.gen_pakd50_configs import priority_for; print(','.join(priority_for('$SERVER')))")" --out-dir "$TMP/cfg" 2>&1 | grep -v Warning | tail -1 | cut -c1-120
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "config 생성 실패"
"$PY" - "$SERVER" "$TMP/cfg" "$DRY" <<'PYEOF' || fail "config/control 검사 실패"
import filecmp, json, os, subprocess, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal, complete
srv, tmp, dry = sys.argv[1], sys.argv[2], sys.argv[3] == "1"; seed = G.SERVER_SEED[srv]; ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout; bad = []; ver = {}
for c in G.priority_for(srv):
    run = G.to_tag(c, seed); f = os.path.join("config", run + ".yaml"); t = os.path.join(tmp, run + ".yaml"); br = G.branch_for(srv, c) or "E0"
    st = "완료/실패" if terminal(run) else ("실행 중" if any("main.py" in l and "--config" in l and run in l for l in ps.splitlines()) else ("대기" if G.cue_ready(c) else "대기(cue 미준비)"))
    if complete(run):
        v = G.verified_complete(run); ver[run] = v; st += " · 검증 " + ("OK" if v["ok"] else "!! 불통과 " + str([k for k, x in v["checks"].items() if x is False]))
    if st.startswith("완료") or st.startswith("실행 중"):
        print(f"   {c:<46} {br:<7} {st}"); continue
    if not os.path.exists(f):
        bad.append(f"{c}: config/{run}.yaml 없음 (git pull)"); continue
    same = filecmp.cmp(f, t, shallow=False); print(f"   {c:<46} {br:<7} {st} — config {'== 생성기' if same else '!= 생성기'}"); same or bad.append(f"{c}: config/{run}.yaml 가 생성기 출력과 다르다")
if not dry:
    json.dump(ver, open(os.path.join(G.ROOT, "work_dir", "_qedge9", "control_verification.json"), "w"), indent=1, ensure_ascii=False)
if bad:
    print("\n".join("!! " + b for b in bad)); sys.exit(1)
PYEOF
echo "[qedge9] ④ 예약 표 (실측은 PAKD50 + QEDGE9 ledger 통합)"; "$PY" tools/gen_pakd50_configs.py --plan --server "$SERVER" 2>&1 | grep -v Warning
if [ "$DRY" = 1 ]; then
  "$PY" - "$SERVER" "$TMP" <<'PYEOF'
import sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
srv, tmp = sys.argv[1], sys.argv[2]; res = G.write_reservation_file(srv, G.priority_for(srv) + G.extra_priority(), G.measured_hours_all(srv), path=tmp + "/reservations.json")
print(f"   (dry-run) reservations 미리보기 {len(res['runs'])} run → 임시 경로 (운영 파일 미기록) · extra_priority 현재 {G.extra_priority() or '없음'}")
PYEOF
  echo "[qedge9] --dry-run — 운영 파일·chain·cron 을 바꾸지 않았다"; exit 0
fi
echo "[qedge9] ⑤ 서버 로컬 기본 묶음·예약 파일 · 옛 extra_priority 보존 분리 (감사 F08)"
"$PY" - "$SERVER" <<'PYEOF' || fail "로컬 파일 기록 실패"
import os, sys, time; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
srv = sys.argv[1]; xp = os.path.join(G.ROOT, G.EXTRA_PRIORITY_FILE); old = G.extra_priority()
if old:
    keep = xp.replace("extra_priority.txt", f"extra_priority.pre_qedge9_{time.strftime('%m%d-%H%M')}.txt"); os.replace(xp, keep)
    open(xp, "w").write(f"# QEDGE9 전환({time.strftime('%Y-%m-%dT%H:%M')}): 이전 추가 편성 {len(old)} 항목은 {os.path.basename(keep)} 에 보존 — 새 branch 의 추가 편성은 여기에 명시 (계획 §10.3 seed 9091 은 별도 구현 뒤)\n")
    print(f"   extra_priority {len(old)} 항목 → {os.path.basename(keep)} (보존; 편성에서 제외)")
runs = G.write_mandatory_file(srv); m = G.measured_hours_all(srv); res = G.write_reservation_file(srv, G.priority_for(srv) + G.extra_priority(), m)
q9 = [G.to_tag(it, G.SERVER_SEED[srv]) for it in G.priority_for(srv) if G.branch_for(srv, it) == "QEDGE9"]; p = os.path.join(G.ROOT, G.QEDGE9_MANDATORY_FILE); os.makedirs(os.path.dirname(p), exist_ok=True)
open(p, "w").write(f"# {srv} QEDGE9 run (자체 ledger {G.QEDGE9_LEDGER} 의 remaining_mandatory; soft target {G.QEDGE9_SOFT_HOURS}h, 경고만; 대기자 tools/qedge9_waiter.sh 의 pending 목록) — tools/qedge9_switch.sh\n" + "\n".join(q9) + "\n")
print(f"   {G.MANDATORY_FILE}: {len(runs)} run · {G.QEDGE9_MANDATORY_FILE}: {len(q9)} run · {G.RESERVATION_FILE}: {len(res['runs'])} run (실측 {m or '없음'})")
PYEOF
if [ "$SERVER" = s4 ]; then echo "[qedge9] ⑥ QEC pilot c_E (§6.1; 없을 때만 산출, 있으면 출처 대조)"; pilot_ce; else echo "[qedge9] ⑥ (s5: QEC 없음)"; fi
echo "[qedge9] ⑦ chain: 소프트 시간 정책 — 마감 파일 제거; runner 는 그대로(현재 run·후처리 유지), 없으면 기동; 대기자 기동"
if [ -f work_dir/cases_deadline.txt ]; then cp work_dir/cases_deadline.txt "$CAMP/cases_deadline.before_qedge9.txt"; rm -f work_dir/cases_deadline.txt; echo "   cases_deadline.txt 제거 (사본 $CAMP/cases_deadline.before_qedge9.txt) — 새 run 시작에 절대 마감 없음"; fi
if ps -eo args | grep -q '[_]run_cases\.sh'; then
  echo "   chain 살아 있음 — 손대지 않는다. 현재 run 은 그 runner 가 끝내고(평가·업로드 포함) 다음 gate pass 부터 새 순서 (gate 는 매 pass tools/campaign_gate.py 를 새로 부른다)"
else
  QUEUE="config/queues/pakd50_${SERVER}_stage1.txt"; [ -f "$QUEUE" ] || fail "$QUEUE 없음"
  ./tools/campaign_start.sh --queue "$QUEUE" --hours 24 --label "qedge9-$SERVER-$(date +%m%d-%H%M)" || fail "기동 실패"; rm -f work_dir/cases_deadline.txt; ./tools/_watchdog.sh --install > /dev/null && echo "   chain 기동 · 마감 파일 제거 · 감시자 cron 등록"
fi
setsid nohup ./tools/qedge9_waiter.sh >> "$CAMP/waiter.log" 2>&1 < /dev/null & sleep 1; echo "   대기자 pid $! ($CAMP/waiter.log · 상태 $CAMP/status.json)"
echo "[qedge9] 완료 — $SERVER 명시 순서 (QEDGE9 run 은 마감 admission 제외; QE50/QES 는 cue 자산, QEC 는 c_E 가 있어야 열린다). 확인: tail -f work_dir/cases_chain.log · $CAMP/status.json"
