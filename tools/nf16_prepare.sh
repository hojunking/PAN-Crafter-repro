#!/usr/bin/env bash
# NF16 (N2 정합 능력 보존 + native fitting, 16 GPU-h) — s1: donor 확인 → gate → smoke/throughput → 예산 ledger → 체인 기동.
#   ./tools/nf16_prepare.sh [--no-start] [--hours 20]
# 명세 research_log/PAN_N2_NativeFitting_16GPUh_W112_D123_2026-09-11.md · 노트 research_log/2026-09-11_nf16-implementation.md
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
START=1; HOURS=20
while [ $# -gt 0 ]; do case "$1" in --no-start) START=0;; --hours) HOURS="$2"; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift; done
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null || PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python
QUEUE=config/queues/nf16_s1.txt; DONOR=PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; [ "$SERVER" = "s1" ] || echo "!! 명세: 주 서버는 s1 (현재 $SERVER)"
echo "[nf16] G0 donor: $DONOR/last (정확한 50K)"
"$PY" - <<PYEOF
import json, sys; sys.path.insert(0, ".")
from kdv.teacher_assets import load_donor_aligner, sha256_file
m = json.load(open("work_dir/$DONOR/last_meta.json")); assert m["step"] == 50000, m
al, man = load_donor_aligner("work_dir/$DONOR/last", 8); print(f"   step {m['step']} · aligner sha {man['aligner_tensors_sha256_16']} · file {man['file_sha256'][:16]} · head bias {al.fc2.bias.tolist()}")
PYEOF
echo "[nf16] 지표 이식"; "$PY" tools/verify_metrics.py | tail -1
echo "[nf16] A1–A3 / PO10 / KDV gate"; "$PY" tools/pa_unit_tests.py | tail -1; PO10_LEDGER=/tmp/_po10_gate_ledger.json "$PY" tools/po10_unit_tests.py | tail -1; "$PY" tools/kdv_unit_tests.py | tail -1
echo "[nf16] G1/G2 gate"; "$PY" tools/nf16_unit_tests.py | tail -1
echo "[nf16] config"; "$PY" tools/gen_nf16_configs.py | tail -1
CASES=$(grep -v '^#' "$QUEUE" | grep -v '^$' | tr '\n' ' ')
echo "[nf16] G3 smoke (실배치·시간)"
# shellcheck disable=SC2086
"$PY" tools/smoke_cases.py $CASES | tee work_dir/_nf16_smoke.log | tail -8
echo "[nf16] 예산 ledger (16 GPU-h) 초기화 + smoke 기반 예상 시간"
"$PY" - <<'PYEOF'
import json, os, re, time
p = "work_dir/_nf16_budget"; os.makedirs(p, exist_ok=True); lp = os.path.join(p, "ledger.json")
d = json.load(open(lp)) if os.path.exists(lp) else dict(total_gpu_hours=16.0, entries={})
log = open("work_dir/_nf16_smoke.log").read(); tn = re.findall(r"t_native (\d+)ms", log); tc = re.findall(r"t_corrupt (\d+)ms", log)
if tn:
    t_nat = float(tn[-1]) / 1000; t_cor = (float(tc[-1]) / 1000 if tc else t_nat)
    proj = 50000 * (t_nat + t_cor) / 2 / 3600 + 50 * 1.0 / 60        # 평가 50회 ≈ 1 min (PO10 실측 준용)
    d["throughput"] = dict(t_native_s=t_nat, t_offset_exercise_s=t_cor, projected_run_hours_50k=proj, measured_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
d["entries"].setdefault("gate", dict(kind="gate", hours=float(os.environ.get("NF16_GATE_HOURS", "0.5")), note="donor 확인·unit/smoke/prepare (명세 §10: 1.5h 예약)"))
json.dump(d, open(lp, "w"), indent=1); print("  ledger:", json.dumps(d.get("throughput")))
PYEOF
if ps -eo args | grep -q '[_]run_cases\.sh'; then echo "!! 다른 체인이 돌고 있다 — 끝난 뒤 다시 실행"; exit 1; fi
[ "$START" = 1 ] || { echo "[nf16] 준비 완료 — 기동: ./tools/campaign_start.sh --queue $QUEUE --hours $HOURS --label nf16-s1"; exit 0; }
./tools/campaign_start.sh --queue "$QUEUE" --hours "$HOURS" --label nf16-s1
./tools/_watchdog.sh --install && echo "  감시자 cron 등록"
