#!/usr/bin/env bash
# EQREC4 prepare (s3 등 새 서버): registry bundle 설치 확인 → 데이터·DLPan → gate(tools/eqrec4_unit_tests.py) → G00 → runner 기동.
#   ./tools/eqrec4_prepare.sh [--no-start]
# 전제: s1 에서 `python tools/eqrec4_bundle.py pack` 한 work_dir/_eqrec4_bundle/ 을 이 서버의 같은 경로로 옮겨 두었다 (rsync/scp 는 사람이).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; START=1; [ "${1:-}" = "--no-start" ] && START=0
export PANCRAFTER_DLPAN="${PANCRAFTER_DLPAN:-/home/knuvi/Desktop/song/DLPan-Toolbox}"
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; CAMP="work_dir/_eqrec4_${SERVER}_campaign"; fail() { echo "!! $1"; exit 1; }
echo "[eqrec4] 서버 $SERVER · 출력 root $CAMP"
echo "[eqrec4] ① registry bundle"
if [ -f work_dir/_eqrec4_bundle/bundle_manifest.json ]; then "$PY" tools/eqrec4_bundle.py verify || fail "bundle sha 불일치 — 다시 전송"; "$PY" tools/eqrec4_bundle.py install || fail "bundle 설치 실패"; else echo "   bundle 없음 — 이 서버의 work_dir 에 registry 가 이미 있는지 확인"; fi
"$PY" -c "
import sys; sys.path.insert(0,'.'); from tools.eqrec4 import common as C
miss=[C.mkey(*t) for t in C.CORE+C.EXTRA+[C.PAIRS[p]['S'] for p in C.PAIRS] if C.ckpt_dir(*t) is None]; print('   registry 누락:', miss or '없음'); sys.exit(1 if miss else 0)" || fail "registry checkpoint 누락 — bundle 을 먼저"
echo "[eqrec4] ② 데이터·DLPan"; for f in data/PanCollection/WV3/train_wv3.h5 data/PanCollection/WV3/train_wv3_pan.h5 data/PanCollection/WV3/reduced_examples_h5/test_wv3_multiExm1.h5 data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5; do [ -f "$f" ] || fail "데이터 없음: $f"; done; [ -f "$PANCRAFTER_DLPAN/wald_utilities.py" ] || fail "DLPan 없음: $PANCRAFTER_DLPAN"
"$PY" tools/verify_metrics.py | tail -1
echo "[eqrec4] ③ gate"; "$PY" tools/eqrec4_unit_tests.py > "work_dir/_eqrec4_${SERVER}_gate.log" 2>&1 || { tail -3 "work_dir/_eqrec4_${SERVER}_gate.log"; fail "gate 실패 — work_dir/_eqrec4_${SERVER}_gate.log"; }; tail -1 "work_dir/_eqrec4_${SERVER}_gate.log"
echo "[eqrec4] ④ G00"; "$PY" tools/eqrec4.py g00 2>&1 | grep -v Warning | tail -2 | cut -c1-300; grep -q '"stage": "G00", "status": "done"' "$CAMP/time_ledger.jsonl" || fail "G00 실패"
"$PY" -c "import yaml; s=yaml.safe_load(open('$CAMP/protocol_resolved.yaml'))['status']; print('   G00 status', s); import sys; sys.exit(1 if s['implementation_invalid'] else 0)" || fail "G00 implementation_invalid"
if ps -eo args | grep -q '[_]run_cases\.sh'; then echo "[eqrec4] 주의: 이 서버에서 학습 체인(_run_cases.sh) 이 돌고 있다 — 같이 돌면 느려진다; 중단은 사람이 결정 (예: PAKD50 이면 runner·cron 정리)"; fi
[ "$START" = 1 ] || { echo "[eqrec4] 준비 완료 — 기동: setsid nohup ./tools/eqrec4_run.sh >> $CAMP/run.log 2>&1 < /dev/null &"; exit 0; }
setsid nohup ./tools/eqrec4_run.sh >> "$CAMP/run.log" 2>&1 < /dev/null & sleep 5; ps -eo args | grep -q '[e]qrec4_run' && echo "[eqrec4] 기동 완료 — 진행: tail -f $CAMP/run.log · 단계 ledger: $CAMP/time_ledger.jsonl"
