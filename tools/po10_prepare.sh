#!/usr/bin/env bash
# PO10 (PAN 추가 변위 + offset consistency, 10 GPU-h) — s1 에서 gate → throughput → 예산 ledger → 체인 기동.
#
#   ./tools/po10_prepare.sh                 # gate + 기동 (queue config/queues/po10_s1.txt: N1 → N2 → N3(예산 gate))
#   ./tools/po10_prepare.sh --no-start      # gate 만
# 명세 research_log/PAN_OffsetConsistency_10GPUh_W96_D124_2026-09-10.md §11–12 · 구현 노트 research_log/2026-09-10_po10-implementation.md
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
START=1; HOURS=12
while [ $# -gt 0 ]; do case "$1" in --no-start) START=0;; --hours) HOURS="$2"; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift; done
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null || PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python
QUEUE=config/queues/po10_s1.txt; CASES=$(grep -v '^#' "$QUEUE" | grep -v '^$' | tr '\n' ' ')
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; [ "$SERVER" = "s1" ] || echo "!! 명세 §7: 주 서버는 s1 (현재 $SERVER) — 세 case 를 같은 서버에서 돌려야 한다"
echo "[po10] G0 provenance: A1 run·init"
[ -f work_dir/PA_A1_REC_W96_D124_9CH_S2025/best_hqnr/model.safetensors ] || echo "  (A1 best 가중치 없음 — 배경 기준만 없을 뿐 실행은 가능)"
[ -f work_dir/_pa_init/init_unet_seed2025.pt ] && [ -f work_dir/_pa_init/init_aligner_seed2025.pt ] || { echo "!! work_dir/_pa_init seed 2025 init 파일 없음 (A1 이 만든 것)"; exit 1; }
echo "[po10] G1 scale: audit 부록 E WV3 train 전역 |δ| P90 0.250 LR px × 4 = 1.0 HR px"; grep -q "| WV3 | 199 | 0.072 | 0.250" research_log/2026-09-03_alignment-audit-s2-detail.md && echo "  확인 (LR px 표)" || { echo "!! audit 표를 찾지 못함"; exit 1; }
echo "[po10] 지표 이식"; "$PY" tools/verify_metrics.py | tail -1
echo "[po10] G2–G7 unit gate"; "$PY" tools/po10_unit_tests.py | tail -1
echo "[po10] A1–A3 gate (evaluator·selector 는 그대로 쓴다)"; "$PY" tools/pa_unit_tests.py | tail -1
echo "[po10] G8 smoke (실배치·throughput)"
# shellcheck disable=SC2086
"$PY" tools/smoke_cases.py $CASES | tee work_dir/_po10_smoke.log | tail -4
echo "[po10] 예산 ledger 초기화 (10 GPU-h, 이미 쓴 gate 시간 기록)"
"$PY" - <<PYEOF
import json, os, re, time
p = "work_dir/_po10_budget"; os.makedirs(p, exist_ok=True); lp = os.path.join(p, "ledger.json")
d = json.load(open(lp)) if os.path.exists(lp) else dict(total_gpu_hours=10.0, entries={})
log = open("work_dir/_po10_smoke.log").read()
tn = re.findall(r"t_native (\d+)ms", log); tc = re.findall(r"t_corrupt (\d+)ms", log)
if tn and tc:
    tn, tc = float(tn[-1]) / 1000, float(tc[-1]) / 1000
    proj = 50000 * (tn + tc) / 2 / 3600 + 50 * 1.0 / 60     # 50 회 평가 ≈ 1 min 씩 (A1 실측: 학습 46 min + 평가 ≈ 45 min → 1 h 31 m)
    d["throughput"] = dict(t_native_s=tn, t_corrupt_s=tc, projected_run_hours_50k=proj, measured_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
d["entries"].setdefault("gate", dict(kind="gate", hours=float(os.environ.get("PO10_GATE_HOURS", "0.3")), note="unit/smoke/prepare"))
json.dump(d, open(lp, "w"), indent=1); print("  ledger:", json.dumps(d.get("throughput"), indent=None))
PYEOF
if ps -eo args | grep -q '[_]run_cases\.sh'; then echo "!! 다른 체인이 돌고 있다 — 끝난 뒤 다시 실행"; exit 1; fi
[ "$START" = 1 ] || { echo "[po10] 준비 완료 — 기동: ./tools/campaign_start.sh --queue $QUEUE --hours $HOURS --label po10-s1"; exit 0; }
./tools/campaign_start.sh --queue "$QUEUE" --hours "$HOURS" --label po10-s1
./tools/_watchdog.sh --install && echo "  감시자 cron 등록"
