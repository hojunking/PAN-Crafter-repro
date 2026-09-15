#!/usr/bin/env bash
# SMEC12 준비 학습 기동 (s2) — 계획 research_log/PAN_SMEC12_MultiDataset_SampleMechanism_ExperimentPlan_2026-09-15.md §2–§3, 노트 research_log/2026-09-15_smec12-implementation.md
#   ./tools/smec12_prepare.sh [--dry-run] [--sensors qb,gf2]
#   ① 데이터(QB msfix + _pan.h5, GF2 + _pan.h5, mat20 FR) · unit gate ② config == 생성기 ③ 예산 ledger(40 h) ④ 현재 chain(PAKD50 재배정) 이 끝날 때까지 detached 로 기다렸다가 campaign_start
#   기존 gate 'pakd50' 는 큐 종료 뒤 옛 run 을 재주입하므로 SMEC12 기동 시점에 비운다(백업 work_dir/_smec12_gates_backup.txt).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; DRY=0; SENSORS=qb,gf2
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --sensors) SENSORS="$2"; shift;; *) echo "!! 알 수 없는 인자 $1"; exit 1;; esac; shift; done
fail() { echo "!! $1"; exit 1; }
QUEUE="config/queues/smec12_${SERVER}.txt"; LEDGER=work_dir/_smec12_budget/ledger.json; mkdir -p work_dir/_smec12_budget work_dir/_smec12_${SERVER}_campaign
echo "[smec12] ① 데이터 · unit gate ($SERVER, 센서 $SENSORS)"
"$PY" - "$SENSORS" <<'PYEOF' || fail "데이터 검사 실패"
import os, sys, h5py; sys.path.insert(0, "."); from tools.gen_smec12_bootstrap import SENSORS, ROOT
for s in sys.argv[1].split(","):
    X = SENSORS[s]; miss = []
    for k in ("train", "valid", "rr", "fr"):
        p = os.path.join(ROOT, X[k]); (miss.append(X[k]) if not os.path.exists(p) else None)
    for k in ("train", "valid"):
        pp = os.path.join(ROOT, X[k].replace(".h5", "_pan.h5"))
        if not os.path.exists(pp):
            miss.append(X[k].replace(".h5", "_pan.h5") + " (feeder 가 lpan 을 읽는다 — tools/repair_lpan.py / 기존 서버의 파일)")
    if miss:
        print(f"!! {X['S']} 데이터 없음 (blocker; 만들어 넣지 않는다 §2.3):", *miss, sep="\n   "); sys.exit(1)
    with h5py.File(os.path.join(ROOT, X["train"])) as f:
        n, b = f["pan"].shape[0], f["gt"].shape[1]; assert b == X["bands"], f"{X['S']} band {b} ≠ {X['bands']}"; print(f"   {X['S']}: train {n} patch · {b} band · max_pixel {X['max_pixel']} · {X['data_note']}")
PYEOF
"$PY" tools/smec12_unit_tests.py 2>&1 | grep -v Warning | tail -3; [ "${PIPESTATUS[0]}" -eq 0 ] || fail "unit gate 실패"
echo "[smec12] ② config == 생성기"
TMP=$(mktemp -d); "$PY" tools/gen_smec12_bootstrap.py --sensors "$SENSORS" --out-dir "$TMP" 2>&1 | grep -v Warning | tail -1 | cut -c1-100
"$PY" - "$TMP" "$QUEUE" <<'PYEOF' || fail "config 검사 실패"
import filecmp, os, sys; tmp, q = sys.argv[1], sys.argv[2]; bad = []
runs = [l.strip() for l in open(q) if l.strip() and not l.startswith("#")]
for r in runs:
    f = os.path.join("config", r + ".yaml"); t = os.path.join(tmp, r + ".yaml"); done = os.path.exists(os.path.join("work_dir", r, "results", "reduced_best_hqnr.mat"))
    if not os.path.exists(f):
        bad.append(f"{r}: config 없음 (git pull)"); continue
    same = filecmp.cmp(f, t, shallow=False); print(f"   {r:52s} {'완료' if done else '대기'} — config {'== 생성기' if same else '!= 생성기'}"); same or bad.append(f"{r}: config 가 생성기 출력과 다르다")
if bad:
    print("\n".join("!! " + b for b in bad)); sys.exit(1)
PYEOF
rm -rf "$TMP"
echo "[smec12] ③ 예산 ledger ($LEDGER; 40 h, reserve 2; 준비 학습은 required)"
"$PY" -c "
import json, os, time; p='$LEDGER'; d = json.load(open(p)) if os.path.exists(p) else dict(total_gpu_hours=40.0, entries={}); d['total_gpu_hours'] = 40.0; d.setdefault('start', time.strftime('%Y-%m-%dT%H:%M:%S')); json.dump(d, open(p, 'w'), indent=1); print('   ledger', p, 'entries', len(d['entries']))"
if [ "$DRY" = 1 ]; then echo "[smec12] --dry-run — 기동하지 않음"; exit 0; fi
echo "[smec12] ④ 기동 — 현재 chain 이 있으면 끝날 때까지 기다린 뒤(5 분 간격) gate 를 비우고 campaign_start (detached)"
cat > work_dir/_smec12_${SERVER}_campaign/launch_when_idle.sh <<EOS
#!/usr/bin/env bash
cd "$REPO"
while ps -eo args | grep -q '[_]run_cases\.sh' || ps -eo args | grep -q '[m]ain\.py --config'; do sleep 300; done
[ -f work_dir/campaign_gates_enabled.txt ] && cp work_dir/campaign_gates_enabled.txt work_dir/_smec12_gates_backup.txt; : > work_dir/campaign_gates_enabled.txt
./tools/campaign_start.sh --queue "$QUEUE" --hours 40 --label "smec12-$SERVER-\$(date +%m%d-%H%M)" && ./tools/_watchdog.sh --install
echo "[smec12] launched \$(date -Iseconds)" >> work_dir/_smec12_${SERVER}_campaign/launch.log
EOS
chmod +x work_dir/_smec12_${SERVER}_campaign/launch_when_idle.sh
setsid nohup work_dir/_smec12_${SERVER}_campaign/launch_when_idle.sh > work_dir/_smec12_${SERVER}_campaign/launch.log 2>&1 < /dev/null &
echo "[smec12] 대기·기동 waiter 시작 (pid $!; work_dir/_smec12_${SERVER}_campaign/launch.log). 진행: tail -f work_dir/cases_chain.log"
