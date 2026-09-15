#!/usr/bin/env bash
# QEDGE9 전환 (s4/s5; 계획 research_log/PAN_QEDGE9_W104D121_S5_S4_Experiment_Plan_2026-09-15.md, 노트 research_log/2026-09-15_qedge9-implementation.md) — pull 뒤 한 번.
#   ./tools/qedge9_switch.sh [--dry-run]      # ① gate(K01–K28) ② cue 자산 verify(이 서버 T0 로 재계산 대조) ③ (s4) QEC pilot c_E: W104 J0 S1234 exact50K 가 있으면 산출 ④ 명시 순서 config == 생성기
#                                             # ⑤ 서버 로컬 mandatory(PAKD50 + QEDGE9)·reservations ⑥ 예약 표 ⑦ requeue: runner 만 교체(현재 학습은 끝까지) — chain 마감은 소프트 정책이라 넉넉히(72h)
#   ./tools/qedge9_switch.sh --pilot          # (s4) ③ 만 다시 (JQ/XJ/F0 뒤 QEC 차례 전에 pilot 이 생겼을 때) — gate 는 c_E 파일이 생기면 다음 pass 에 QEC 를 연다
# 시간 정책(§0.7·§9.3): QEDGE9 run 은 PAKD50 의 50h·09-16 11:22 마감을 상속하지 않는다(config 에 training_deadline 없음, required=True → 예산 경고만). NaN/오류/중복 run 보호는 그대로.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; DRY=0; PILOT_ONLY=0
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --pilot) PILOT_ONLY=1;; *) echo "!! 알 수 없는 인자 $1"; exit 1;; esac; shift; done
fail() { echo "!! $1"; exit 1; }
case "$SERVER" in s4|s5) ;; *) fail "QEDGE9 는 s4/s5 용 (이 서버: $SERVER)";; esac
[ -f work_dir/_pakd50/ledger.json ] || fail "work_dir/_pakd50/ledger.json 없음 — 이 서버는 pakd50_prepare 를 아직 안 했다"
grep -qx 'pakd50' work_dir/campaign_gates_enabled.txt 2>/dev/null || fail "work_dir/campaign_gates_enabled.txt 에 pakd50 이 없다 (편성 gate)"
mkdir -p work_dir/_qedge9 work_dir/_qedge9_budget
pilot_ce() {
  PR=PAKD50_J0_W104_D121_WV3_T0_S1234_FRESH50_v1
  if [ -f "work_dir/$PR/last_meta.json" ] && [ -f "work_dir/$PR/last/model.safetensors" ]; then
    "$PY" tools/qedge9_cue.py pilot --run "$PR" --tag last 2>&1 | grep -v Warning | tail -2 || fail "QEC pilot c_E 산출 실패"
  else
    echo "   pilot $PR 의 exact50K last 없음 — QEC 는 c_E 가 생길 때까지 gate 가 건너뛴다 (뒤에 ./tools/qedge9_switch.sh --pilot)"
  fi
}
if [ "$PILOT_ONLY" = 1 ]; then [ "$SERVER" = s4 ] || fail "--pilot 은 s4 (W104 J0 S1234 exact50K) 에서"; echo "[qedge9] ③ QEC pilot"; pilot_ce; exit 0; fi
echo "[qedge9] ① $SERVER — unit gate (K01–K28)"
"$PY" tools/pakd50_unit_tests.py > work_dir/_qedge9/gate_$SERVER.log 2>&1 || { grep -v Warning work_dir/_qedge9/gate_$SERVER.log | grep "FAIL" | head -5; fail "unit gate 실패 — work_dir/_qedge9/gate_$SERVER.log"; }; tail -1 work_dir/_qedge9/gate_$SERVER.log
echo "[qedge9] ② cue 자산 verify (이 서버 T0 aligner 로 96 base × 4 state 재계산 vs cache; θq 근처 라벨 차이 기록 §7.4)"
"$PY" tools/qedge9_cue.py verify --n 96 2>&1 | grep -v Warning | tail -1 || fail "cue verify 실패 (Teacher A hash / dataset sha / feeder 계약 / |Δq|) — work_dir/_qedge9/verify_$SERVER.json"
if [ "$SERVER" = s4 ]; then echo "[qedge9] ③ QEC pilot c_E (§6.1)"; pilot_ce; else echo "[qedge9] ③ (s5: QEC 없음)"; fi
echo "[qedge9] ④ 명시 순서의 config 검사 (생성기 == config/; 완료·실행 중 run 은 생략)"
TMP=$(mktemp -d); "$PY" tools/gen_pakd50_configs.py --server "$SERVER" --cases "$("$PY" -c "from tools.gen_pakd50_configs import priority_for; print(','.join(priority_for('$SERVER')))")" --out-dir "$TMP" 2>&1 | grep -v Warning | tail -1 | cut -c1-120
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "config 생성 실패"
"$PY" - "$SERVER" "$TMP" <<'PYEOF' || fail "config 검사 실패"
import filecmp, os, subprocess, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal
srv, tmp = sys.argv[1], sys.argv[2]; seed = G.SERVER_SEED[srv]; ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout; bad = []
for c in G.priority_for(srv):
    run = G.to_tag(c, seed); f = os.path.join("config", run + ".yaml"); t = os.path.join(tmp, run + ".yaml"); br = G.branch_for(srv, c) or "E0"
    st = "완료/실패" if terminal(run) else ("실행 중" if any("main.py" in l and "--config" in l and run in l for l in ps.splitlines()) else ("대기" if G.cue_ready(c) else "대기(cue 미준비)"))
    if st.startswith("완료") or st == "실행 중":
        print(f"   {c:<46} {br:<7} {st} — config 검사 생략"); continue
    if not os.path.exists(f):
        bad.append(f"{c}: config/{run}.yaml 없음 (git pull)"); continue
    same = filecmp.cmp(f, t, shallow=False); print(f"   {c:<46} {br:<7} {st} — config {'== 생성기' if same else '!= 생성기'}"); same or bad.append(f"{c}: config/{run}.yaml 가 생성기 출력과 다르다")
if bad:
    print("\n".join("!! " + b for b in bad)); sys.exit(1)
PYEOF
rm -rf "$TMP"
echo "[qedge9] ⑤ 서버 로컬 기본 묶음·예약 파일 (PAKD50 mandatory + QEDGE9 mandatory + reservations)"
"$PY" - "$SERVER" <<'PYEOF' || fail "로컬 파일 기록 실패"
import os, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
srv = sys.argv[1]; runs = G.write_mandatory_file(srv); m = G.measured_hours_from_ledger(srv); res = G.write_reservation_file(srv, G.priority_for(srv) + G.extra_priority(), m)
q9 = [G.to_tag(it, G.SERVER_SEED[srv]) for it in G.priority_for(srv) if G.branch_for(srv, it) == "QEDGE9"]; p = os.path.join(G.ROOT, G.QEDGE9_MANDATORY_FILE); os.makedirs(os.path.dirname(p), exist_ok=True)
open(p, "w").write(f"# {srv} QEDGE9 run (자체 ledger {G.QEDGE9_LEDGER} 의 remaining_mandatory; soft target {G.QEDGE9_SOFT_HOURS}h, 경고만) — tools/qedge9_switch.sh\n" + "\n".join(q9) + "\n")
print(f"   {G.MANDATORY_FILE}: {len(runs)} run · {G.QEDGE9_MANDATORY_FILE}: {len(q9)} run · {G.RESERVATION_FILE}: {len(res['runs'])} run (실측 {m or '없음'})")
PYEOF
echo "[qedge9] ⑥ 예약 표"; "$PY" tools/gen_pakd50_configs.py --plan --server "$SERVER" 2>&1 | grep -v Warning
if [ "$DRY" = 1 ]; then echo "[qedge9] --dry-run — 전환하지 않음"; exit 0; fi
echo "[qedge9] ⑦ requeue — runner 만 교체 (현재 학습은 그대로; 새 runner 는 학습이 끝날 때까지 기다린 뒤 gate 가 새 순서로 편성; chain 마감 72h = 소프트 정책)"
QUEUE="config/queues/pakd50_${SERVER}_stage1.txt"; [ -f "$QUEUE" ] || fail "$QUEUE 없음"
crontab -l 2>/dev/null | grep -v PANCRAFTER-WATCHDOG | crontab - || true
CH=$(ps -eo pid,args | awk '/bash \.\/tools\/_run_cases\.sh$/{print $1}'); TR=$(ps -eo pid,args | awk '/python .*main\.py --config .*PAKD50_/{print $1}' | head -1)
echo "   체인 runner [${CH:-없음}] 종료 · 현재 학습 [${TR:-없음}] 은 그대로"
if [ -n "$CH" ]; then kill $CH; for _ in 1 2 3 4 5 6 7 8 9 10; do ps -eo args | grep -q '[_]run_cases\.sh' || break; sleep 1; done; fi
ps -eo args | grep -q '[_]run_cases\.sh' && fail "runner 가 아직 살아 있다"
./tools/campaign_start.sh --queue "$QUEUE" --hours 72 --label "qedge9-$SERVER-$(date +%m%d-%H%M)" || fail "기동 실패"
./tools/_watchdog.sh --install > /dev/null && echo "   감시자 cron 재등록"
echo "[qedge9] 완료 — 다음 run 경계부터 $SERVER 명시 순서 (QEDGE9 run 은 마감 admission 제외; QE50/QES 는 cue 자산, QEC 는 c_E 가 있어야 열린다). 확인: tail -f work_dir/cases_chain.log"
