#!/usr/bin/env bash
# PAKD50 파생 실험 재배정 (research_log/PAN_PAKD50_S2_S4_S5_Derived_Run_Allocation_2026-09-15.md) — s2/s4/s5 에서 pull 뒤 한 번.
#   ./tools/pakd50_reallocate.sh                 # 명시 순서로 전환 (다음 run 경계에서; 진행 중 학습은 건드리지 않는다 — §9.2)
#   ./tools/pakd50_reallocate.sh --dry-run       # 검사·예약 표만
#   ./tools/pakd50_reallocate.sh --confirm <WIN>  # §7: 기본 단계 뒤 외부 분석의 WIN 이 오면 — 확인 seed(s4 3407 / s5 9091) config 생성 + extra_priority 등록 (최대 3 run)
#   ① gate 'pakd50' 켜짐 확인 · λE 사본 동기화 · unit gate ② 순서의 config 가 생성기와 같은지 검사 (config 는 git 으로 받는다 — 여기서 쓰지 않는다)
#   ③ 서버 로컬 mandatory_runs.txt(기본 묶음 = 명시 순서) · reservations.json(1.10×ref + 10/60) ④ 예약 표 ⑤ requeue: runner 만 교체 → 현재 학습이 끝나면 gate 가 새 순서로 편성
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; DRY=0; CONFIRM=""
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --confirm) CONFIRM="${2:-}"; shift;; *) echo "!! 알 수 없는 인자 $1"; exit 1;; esac; shift; done
fail() { echo "!! $1"; exit 1; }
case "$SERVER" in s2|s4|s5) ;; *) fail "재배정 대상은 s2/s4/s5 뿐이다 (이 서버: $SERVER; s1 은 분석용, s3 은 신규 배정 없음 — 계획 §1)";; esac
[ -f work_dir/_pakd50/ledger.json ] || fail "work_dir/_pakd50/ledger.json 없음 — 이 서버는 pakd50_prepare 를 아직 안 했다"
grep -qx 'pakd50' work_dir/campaign_gates_enabled.txt 2>/dev/null || fail "work_dir/campaign_gates_enabled.txt 에 pakd50 이 없다"
echo "[realloc] ① $SERVER — λE 사본 동기화 + unit gate"
"$PY" -c "import sys; sys.path.insert(0,'.'); from tools import gen_pakd50_configs as G; m, src = G.sync_calibration_from_assets(write=True); assert m.get('lambda_E'), 'λE 없음 — assets/pakd50/calibration_resolved.json 을 pull'; print('   calibration <-', src, {k: m.get(k) for k in ('tau_R','lambda_E')})" || fail "calibration"
"$PY" tools/pakd50_unit_tests.py 2>&1 | grep -v Warning | tail -3; [ "${PIPESTATUS[0]}" -eq 0 ] || fail "unit gate 실패"
if [ -n "$CONFIRM" ]; then
    echo "[realloc] --confirm $CONFIRM: §7 확인 seed 묶음"
    "$PY" - "$SERVER" "$CONFIRM" <<'PYEOF' || fail "확인 seed 준비 실패"
import os, subprocess, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
srv, win = sys.argv[1], sys.argv[2]; seed = G.CONFIRM_SEED.get(srv) or sys.exit(f"!! {srv} 에는 확인 seed 가 없다 (s4 3407 / s5 9091)")
cases = G.confirmation_cases(win); runs = [G.run_name(c, seed) for c in cases]
r = subprocess.run([sys.executable, "tools/gen_pakd50_configs.py", "--server", srv, "--seed", str(seed), "--cases", ",".join(cases)], capture_output=True, text=True); print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
r.returncode == 0 or sys.exit("!! config 생성 실패")
p = os.path.join(G.ROOT, G.EXTRA_PRIORITY_FILE); have = G.extra_priority(); os.makedirs(os.path.dirname(p), exist_ok=True)
with open(p, "a") as f:
    for rn in runs:
        if rn not in have:
            f.write(rn + "\n")
mp = os.path.join(G.ROOT, G.MANDATORY_FILE); cur = [l.strip() for l in open(mp)] if os.path.exists(mp) else []
with open(mp, "a") as f:
    for rn in runs:
        if rn not in cur:
            f.write(rn + "\n")
print(f"   확인 seed {seed}: {' → '.join(cases)} ({len(runs)} run; WIN {win} · control {G.BASELINE_OF[G.CASES[win][0]]} · F0) → {G.EXTRA_PRIORITY_FILE} + {G.MANDATORY_FILE}. 계수·release 는 여기서 고정 — 결과를 본 뒤 바꾸지 않는다 (§7)")
print("   config 는 이 서버 로컬 생성(미커밋) — 공유가 필요하면 커밋할 것. 예약 = 1.80h 가예약 (같은 case 실측이 있으면 그 값; §7)")
PYEOF
fi
echo "[realloc] ② 명시 순서의 config 검사 (생성기 == config/; 완료·실행 중 run 은 생략)"
TMP=$(mktemp -d); "$PY" tools/gen_pakd50_configs.py --server "$SERVER" --cases "$("$PY" -c "from tools.gen_pakd50_configs import priority_for; print(','.join(priority_for('$SERVER')))")" --out-dir "$TMP" 2>&1 | grep -v Warning | tail -1
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "config 생성 실패"
"$PY" - "$SERVER" "$TMP" <<'PYEOF' || fail "config 검사 실패"
import filecmp, os, subprocess, sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G; from tools.campaign_gate import terminal
srv, tmp = sys.argv[1], sys.argv[2]; seed = G.SERVER_SEED[srv]; ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
bad = []
for c in G.priority_for(srv):
    run = G.run_name(c, seed); f = os.path.join("config", run + ".yaml"); t = os.path.join(tmp, run + ".yaml")
    st = "완료/실패" if terminal(run) else ("실행 중" if any("main.py" in l and "--config" in l and run in l for l in ps.splitlines()) else "대기")
    if st != "대기":
        print(f"   {c:<12} {st} — config 검사 생략"); continue
    if not os.path.exists(f):
        bad.append(f"{c}: config/{run}.yaml 없음 (git pull)"); continue
    same = filecmp.cmp(f, t, shallow=False); print(f"   {c:<12} 대기 — config {'== 생성기' if same else '!= 생성기'}")
    same or bad.append(f"{c}: config/{run}.yaml 가 생성기 출력과 다르다 (release 불일치 — pull 또는 생성기 확인)")
if bad:
    print("\n".join("!! " + b for b in bad)); sys.exit(1)
PYEOF
rm -rf "$TMP"
echo "[realloc] ③ 서버 로컬 기본 묶음·예약 파일"
"$PY" - "$SERVER" <<'PYEOF' || fail "로컬 파일 기록 실패"
import sys; sys.path.insert(0, "."); from tools import gen_pakd50_configs as G
srv = sys.argv[1]; runs = G.write_mandatory_file(srv); m = G.measured_hours_from_ledger(srv)
res = G.write_reservation_file(srv, G.priority_for(srv) + G.extra_priority(), m)
print(f"   {G.MANDATORY_FILE}: {len(runs)} run · {G.RESERVATION_FILE}: {len(res['runs'])} run (실측 {m or '없음'})")
PYEOF
echo "[realloc] ④ 예약 표 (dry-run; 완료·실패 run 은 work_dir 판정)"; "$PY" tools/gen_pakd50_configs.py --plan --server "$SERVER" 2>&1 | grep -v Warning
if [ "$DRY" = 1 ]; then echo "[realloc] --dry-run — 전환하지 않음"; exit 0; fi
echo "[realloc] ⑤ requeue — runner 만 교체 (현재 학습은 그대로; 새 runner 는 학습이 끝날 때까지 기다린 뒤 gate 가 새 순서로 편성)"
./tools/pakd50_requeue.sh || fail "requeue 실패"
echo "[realloc] 완료 — 다음 run 경계부터 $SERVER 명시 순서. 확인: tail -f work_dir/cases_chain.log"
