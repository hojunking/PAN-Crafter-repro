#!/usr/bin/env bash
# s2 W112·D124 KDV 캠페인 — gate → config → 캠페인 폴더 manifest → 체인 기동 (plan §14, §17.1, §20.2, §21.1).
#
#   ./tools/kdv_prepare.sh                 # gate + 기동 (queue config/queues/kdv_s2.txt: Q00 → Q01(Teacher) → Q02 … Q08)
#   ./tools/kdv_prepare.sh --no-start      # gate 만
#   ./tools/kdv_prepare.sh --hours 30      # 마감(기본 36h — 24h 는 review 시점이지 강제 종료가 아니다)
# 구현 노트 research_log/2026-09-10_s2-w112-kdv-implementation.md
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
START=1; HOURS=36; QUEUE=config/queues/kdv_s2.txt
while [ $# -gt 0 ]; do case "$1" in --no-start) START=0;; --hours) HOURS="$2"; shift;; --queue) QUEUE="$2"; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift; done
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null || PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; [ "$SERVER" = "s2" ] || echo "!! 계획: 주 서버는 s2 (현재 $SERVER) — 같은 서버에서 순차 실행해야 대응 비교가 된다"
CAMP=work_dir/_kdv_campaign/S2_W112_KDV_20260910; mkdir -p "$CAMP"
echo "[kdv] P0 환경 (§17.1) → $CAMP/environment_${SERVER}.json"
"$PY" - "$CAMP/environment_${SERVER}.json" <<PYEOF
import json, os, platform, subprocess, sys, torch
p = sys.argv[1]; g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
d = dict(hostname=platform.node(), server=open("gspread/server.txt").read().strip(), python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
         gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), gpu_uuid=g("nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1"),
         vram_mb=(torch.cuda.get_device_properties(0).total_memory // 2**20 if torch.cuda.is_available() else None), driver=g("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
         git_commit=g("git rev-parse HEAD"), git_dirty=bool(g("git status --porcelain -- ':!results_log'")), other_gpu_jobs=g("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"))
json.dump(d, open(p, "w"), indent=1); print("  ", json.dumps({k: d[k] for k in ('hostname','gpu','vram_mb','torch','cuda','git_commit')}))
PYEOF
echo "[kdv] 데이터·논문 세트"
for f in data/PanCollection/WV3/train_wv3.h5 data/PanCollection/WV3/valid_wv3.h5 data/PanCollection/WV3/reduced_examples_h5/test_wv3_multiExm1.h5 data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5; do
  [ -f "$f" ] || { echo "!! 없음: $f (논문 세트는 ./tools/metric_v2_prepare.sh)"; exit 1; }; done
echo "[kdv] donor aligner asset"
"$PY" - <<'PYEOF'
import json, hashlib
p = "assets/donor_aligner/PA_A1_REC_W96_D124_9CH_S2025_best_hqnr_aligner.pt"; m = json.load(open(p.replace(".pt", ".json")))
h = hashlib.sha256(open(p, "rb").read()).hexdigest(); assert h == m["file_sha256"], f"donor sha 불일치 {h} ≠ {m['file_sha256']}"
print(f"   {p} sha {h[:16]} (source {m['source_run']} step {m['selected_step']}, HQNR {m['selected_hqnr_raw_original']:.4f})")
PYEOF
echo "[kdv] 지표 이식"; "$PY" tools/verify_metrics.py | tail -1
echo "[kdv] A1–A3 gate (evaluator·selector 는 그대로 쓴다)"; "$PY" tools/pa_unit_tests.py | tail -1
echo "[kdv] KDV gate (M01–M04·W01/W03·L01–L04·R01·resolver·calibration)"; "$PY" tools/kdv_unit_tests.py | tail -1
echo "[kdv] config 생성 (Q00–Q08)"; "$PY" tools/gen_kdv_configs.py | tail -1
CASES=$(grep -v '^#' "$QUEUE" | grep -v '^$' | tr '\n' ' ')
echo "[kdv] P01 smoke — Teacher 가 아직 없는 Q03–Q08 은 체인이 각 case 직전에 다시 smoke 한다; 지금은 Q00·Q01·Q02 만"
FIRST=$(grep -v '^#' "$QUEUE" | grep -v '^$' | head -3 | tr '\n' ' ')
# shellcheck disable=SC2086
"$PY" tools/smoke_cases.py $FIRST | tee "$CAMP/smoke_${SERVER}.log" | tail -5
cp -f research_log/2026-09-10_s2-w112-kdv-implementation.md "$CAMP/implementation_map.md" 2>/dev/null || true
"$PY" - "$CAMP" "$QUEUE" <<'PYEOF'
import sys, yaml, os, time, json
camp, q = sys.argv[1], sys.argv[2]
runs = [l.strip() for l in open(q) if l.strip() and not l.startswith("#")]
yaml.safe_dump(dict(id="S2_W112_KDV_20260910", server=open("gspread/server.txt").read().strip(), version="v01", mode="prioritized_adaptive", target_review_hours=24, hard_time_limit_hours=None,
                    allow_overrun=True, continue_after_review=True, work_root=os.path.abspath("work_dir"), queue=q, runs=runs, plan="research_log/PAN_S2_W112_KD_Variance_Plan_and_References_2026-09-10/",
                    implementation_note="research_log/2026-09-10_s2-w112-kdv-implementation.md", created=time.strftime("%Y-%m-%dT%H:%M:%S")), open(os.path.join(camp, "campaign_manifest.yaml"), "w"), sort_keys=False)
print("   campaign_manifest.yaml:", len(runs), "runs")
PYEOF
if ps -eo args | grep -q '[_]run_cases\.sh'; then echo "!! 다른 체인이 돌고 있다 — 끝난 뒤 다시 실행"; exit 1; fi
[ "$START" = 1 ] || { echo "[kdv] 준비 완료 — 기동: ./tools/campaign_start.sh --queue $QUEUE --hours $HOURS --label kdv-s2"; exit 0; }
./tools/campaign_start.sh --queue "$QUEUE" --hours "$HOURS" --label kdv-s2
./tools/_watchdog.sh --install && echo "  감시자 cron 등록"
