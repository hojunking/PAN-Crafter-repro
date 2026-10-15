#!/usr/bin/env bash
# DCR12 준비·기동 (서버 공용; 2026-09-15 s2 이관용) — 계획 research_log/PAN_Consistency_Reconstruction_Quadrant_Validation_12H_2026-09-14.md, 노트 research_log/2026-09-15_dcr12-implementation.md §5
#   ./tools/dcr12_prepare.sh [--dry-run | --post]
#   ① 서버·큐·config · T0 자산·calibration · 이 서버 seed 의 B0(FQ) 완료본 ② bundle(work_dir/_dcr12_bundle) 이 있으면 verify → install(dry-run 은 verify 만)
#   ③ unit gate X01–X15 ④ registry 상태 · PAKD50 마감 여유 ⑤ runner detached (다른 chain 이 돌면 끝난 뒤 우리 큐를 기동한다 — tools/dcr12_run.sh)
#   --post : runner 가 이미 DONE 인데 bundle(다른 호스트 pair) 을 나중에 넣은 경우 — ①–④ 뒤 post 단계(D01→D02→D03→D04→RESULTS→REPORT) 만 새 phase 이름으로 다시 돈다 (증분; 기존 행은 갱신)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; DRY=0; POST=0
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --post) POST=1;; *) echo "!! 알 수 없는 인자 $1"; exit 1;; esac; shift; done
fail() { echo "!! $1"; exit 1; }
QUEUE="config/queues/dcr12_${SERVER}.txt"; CAMP="work_dir/_dcr12_${SERVER}_campaign"; BUNDLE="work_dir/_dcr12_bundle"; mkdir -p "$CAMP"
echo "[dcr12] ① 서버 $SERVER · 큐 $QUEUE"
[ -f "$QUEUE" ] || fail "큐 없음: $QUEUE (이 서버용 큐가 없다 — 노트 §5)"
for r in $(grep -vE '^[[:space:]]*(#|$)' "$QUEUE"); do [ -f "config/$r.yaml" ] || fail "config 없음: $r (git pull)"; echo "   큐: $r $([ -f work_dir/$r/results/full_best_hqnr.mat ] && echo '(완료)' || echo '(대기)')"; done
"$PY" - <<'PYEOF' || fail "자산 검사 실패"
import os, sys; sys.path.insert(0, "."); from tools.dcr12 import common as C
assert os.path.exists(os.path.join(C.T0_DIR, C.T0_TAG, "model.safetensors")), "T0 자산 없음 (assets/pakd50/T0_run)"
cal = C.calibration(); print(f"   T0 OK · τR {cal['tau_R']:.6g} · λE {cal['lambda_E']:.6g}")
seed = C.G.SERVER_SEED.get(C.SERVER)
if seed in C.SEEDS:
    ok = C.run_complete("B0", seed); print(f"   이 서버 seed {seed}: B0/FQ {'완료' if ok else '없음'} ({C.run_name('B0', seed)})")
    ok or sys.exit(f"!! B0/FQ S{seed} 완료본이 없다 — 이 서버의 PAKD50 FQ 를 먼저 끝내거나 큐에 넣는다")
else:
    print(f"   서버 seed {seed} 는 DCR12 seed {C.SEEDS} 가 아니다 — pair 는 전부 bundle 또는 큐로")
PYEOF
echo "[dcr12] ② bundle ($BUNDLE)"
if [ -f "$BUNDLE/bundle_manifest.json" ]; then
  "$PY" tools/dcr12_bundle.py verify --bundle "$BUNDLE" || fail "bundle verify 실패"
  if [ "$DRY" = 1 ]; then echo "   --dry-run: install 생략"; else "$PY" tools/dcr12_bundle.py install --bundle "$BUNDLE" || fail "bundle install 실패"; fi
else
  echo "   bundle 없음 — 다른 호스트 pair 는 install 전까지 registry 에서 incomplete (post 단계가 그 seed 를 unpaired 로 남긴다)"
fi
echo "[dcr12] ③ unit gate"
"$PY" tools/dcr12_unit_tests.py 2>&1 | grep -v Warning | tail -3; [ "${PIPESTATUS[0]}" -eq 0 ] || fail "unit gate 실패"
echo "[dcr12] ④ registry · 마감"
"$PY" - <<'PYEOF'
import sys; sys.path.insert(0, "."); from tools.dcr12 import common as C
for s in C.SEEDS:
    for ck in ("B0", "B1"):
        print(f"   {ck}/{C.CASES[ck]} S{s}: {'complete' if C.run_complete(ck, s) else 'incomplete'} · trained_on {C.trained_on(ck, s)}")
hp = C.host_pairing(); print(f"   host pairing: {hp['by_seed']} · seed×host 결합 {hp['seed_host_coupled']}")
h = C.G.hours_to_deadline(); print(f"   PAKD50 training_deadline 까지 {h:.2f} h" + (" — 큐 run 의 예상(4.0 h × margin 1.1 = 4.4 h) 보다 적으면 trainer 가 DEFERRED(rc=4) 한다: 노트 §5 의 마감 처리" if h is not None and h < 4.4 else "") if h is not None else "   (campaign_clock 없음)")
PYEOF
if [ "$DRY" = 1 ]; then echo "[dcr12] --dry-run — 기동하지 않음"; exit 0; fi
if [ "$POST" = 1 ]; then
  ps -eo args | grep -q '[d]cr12_run\.sh' && fail "runner 가 아직 돌고 있다 — 끝난 뒤(run_status.txt DONE/STOPPED) --post"
  PH="post$(date +%m%d%H%M)"; echo "[dcr12] ⑤ --post: D01→D02→D03→D04→RESULTS→REPORT (phase $PH)"
  for st in d01 d02 d03 d04 results report; do echo "[dcr12] === $st $(date -Iseconds) ==="; DCR12_PHASE="$PH" "$PY" tools/dcr12.py "$st" 8>&- || fail "$st 실패"; done
  echo "POST $PH DONE $(date -Iseconds)" >> "$CAMP/run_status.txt"; echo "[dcr12] --post 완료 — $CAMP/report.md"; exit 0
fi
echo "[dcr12] ⑤ runner 기동 (detached; 다른 chain 이 돌면 끝난 뒤 우리 큐 기동)"
setsid nohup ./tools/dcr12_run.sh >> "$CAMP/run.log" 2>&1 < /dev/null &
sleep 2; echo "   pid $! · 진행: tail -f $CAMP/run.log · 학습: tail -f work_dir/cases_chain.log"
