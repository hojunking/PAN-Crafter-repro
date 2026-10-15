#!/usr/bin/env bash
# NA104 (W104·D122 no-align KD) — 실행 전 확인(§17.1·§18) → config 생성 → 캠페인 manifest·case 상태 → 체인 기동.
#
#   ./tools/na104_prepare.sh --no-start          # gate 만 (서버는 gspread/server.txt 로 판별)
#   ./tools/na104_prepare.sh                     # gate + 기동 (해당 서버 큐)
#   ./tools/na104_prepare.sh --server s3 --hours 200
#
# 계획 research_log/01_S2_FINAL_EXPERIMENT_PLAN.md · 02_S3_FINAL_EXPERIMENT_PLAN.md (2026-09-12 FINAL) · 원 계획 PAN_S2_W104_D122_NoAlign_KD_Experiment_Plan_2026-09-11.md · 노트 research_log/2026-09-11_na104-implementation.md
# 시간 제한을 두지 않는 캠페인이다 — --hours 는 체인 감시자의 마감일 뿐 실험 예산이 아니다.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
START=1; HOURS=2000; SERVER=""    # 시간 제한 없는 캠페인 — 마감은 체인 감시자용 상한일 뿐이다 (마감을 넘기면 미시작 run 은 실제로 건너뛴다)
while [ $# -gt 0 ]; do case "$1" in --no-start) START=0;; --hours) HOURS="$2"; shift;; --server) SERVER="$2"; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift; done
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
[ -n "$SERVER" ] || SERVER="$(tr -d '[:space:]' < gspread/server.txt)"
case "$SERVER" in s2|s3) ;; *) echo "!! 이 캠페인은 s2·s3 block 이다 (현재 $SERVER) — --server 로 지정하거나 gspread/server.txt 를 확인할 것"; exit 1;; esac
QUEUE="config/queues/na104_${SERVER}.txt"
CAMP="work_dir/_na104_campaign/S2_W104_D122_NOALIGN_KD_20260911"; mkdir -p "$CAMP"

echo "[na104] P0 실행 전 확인값 (§17.1) → $CAMP/environment_${SERVER}.json"
"$PY" - "$CAMP/environment_${SERVER}.json" <<'PYEOF'
import json, platform, subprocess, sys, torch, yaml, os
sys.path.insert(0, os.getcwd())
from main import import_class
p = sys.argv[1]; g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
tpl = yaml.safe_load(open("config/PA_A1_REC_W96_D124_9CH_S1234.yaml"))
ma = dict(tpl["model_args"], hidden_size=104, depth=[1, 2, 2]); m = import_class(tpl["model"])(**ma)
par = sum(q.numel() for q in m.parameters()); tr = sum(q.numel() for q in m.parameters() if q.requires_grad)
lat = {}; m.eval()
import time as _t
with torch.no_grad():
    for hr in (64, 512):                       # 복원망 forward 만 (PAN, LPAN, MS, mode weight) — CPU 기준 참고값
        x = (torch.randn(1, 1, hr, hr), torch.randn(1, 1, hr // 4, hr // 4), torch.randn(1, 8, hr // 4, hr // 4), torch.ones(1))
        for _ in range(2):
            m(*x)
        t0 = _t.time()
        for _ in range(3):
            m(*x)
        lat[f"cpu_forward_{hr}px_s"] = round((_t.time() - t0) / 3, 4)
d = dict(hostname=platform.node(), server=open("gspread/server.txt").read().strip(), python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
         cudnn=torch.backends.cudnn.version(), gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
         gpu_uuid=g("nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1"), vram_mb=(torch.cuda.get_device_properties(0).total_memory // 2**20 if torch.cuda.is_available() else None),
         driver=g("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"), git_commit=g("git rev-parse HEAD"), git_dirty=bool(g("git status --porcelain -- ':!results_log'")),
         other_gpu_jobs=g("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"),
         model=dict(class_path=tpl["model"], width=104, depth=[1, 2, 2], in_channels=9, out_channels=8, params_m=round(par / 1e6, 4), trainable_params_m=round(tr / 1e6, 4),
                    aligner_params=0, groupnorm=[(n, mm.num_groups, mm.num_channels) for n, mm in m.named_modules() if isinstance(mm, torch.nn.GroupNorm)][:4], **lat))
json.dump(d, open(p, "w"), indent=1)
print("  ", json.dumps({k: d[k] for k in ("hostname", "gpu", "vram_mb", "torch", "git_commit")}))
print("   model:", json.dumps(d["model"]))
PYEOF

echo "[na104] 데이터·논문 세트"
for f in data/PanCollection/WV3/train_wv3.h5 data/PanCollection/WV3/valid_wv3.h5 data/PanCollection/WV3/reduced_examples_h5/test_wv3_multiExm1.h5 data/PanCollection/WV3/full_examples_mat20/test_wv3_OrigScale_mat20.h5; do
  [ -f "$f" ] || { echo "!! 없음: $f (논문 세트는 ./tools/metric_v2_prepare.sh)"; exit 1; }; done
echo "[na104] 지표 이식"; "$PY" tools/verify_metrics.py | tail -1
echo "[na104] 기존 gate (평가기·선택기는 그대로 쓴다)"; "$PY" tools/pa_unit_tests.py | tail -1; "$PY" tools/kdv_unit_tests.py | tail -1
echo "[na104] NA104 gate (NA01–NA04·CK01·KD01–KD04·ST01–ST04·TRI01–TRI03·EV01·RS01·PERF01·CS01)"; "$PY" tools/na104_unit_tests.py | tail -1
echo "[na104] config 생성 ($SERVER block)"; "$PY" tools/gen_na104_configs.py --server "$SERVER" | tail -3

FIRST=$(grep -v '^#' "$QUEUE" | grep -v '^$' | head -2 | tr '\n' ' ')
echo "[na104] smoke — Teacher 가 아직 없는 case 는 체인이 각 case 직전에 다시 smoke 한다; 지금은 앞 2건 ($FIRST)"
# shellcheck disable=SC2086
"$PY" tools/smoke_cases.py $FIRST | tee "$CAMP/smoke_${SERVER}.log" | tail -5

"$PY" - "$CAMP" "$QUEUE" "$SERVER" <<'PYEOF'
import sys, yaml, os, time, csv, glob
sys.path.insert(0, os.getcwd())
from kdv.registry import resolve, describe
camp, q, srv = sys.argv[1], sys.argv[2], sys.argv[3]
runs = [l.strip() for l in open(q) if l.strip() and not l.startswith("#")]
yaml.safe_dump(dict(id="S2_W104_D122_NOALIGN_KD_20260911", server=srv, version="v1", protocol="NA-STRICT", wallclock_limit_hours=None,
                    run_kind="CONTROLLED", work_root=os.path.abspath("work_dir"), queue=q, runs=runs,
                    plan="research_log/PAN_S2_W104_D122_NoAlign_KD_Experiment_Plan_2026-09-11.md",
                    implementation_note="research_log/2026-09-11_na104-implementation.md", created=time.strftime("%Y-%m-%dT%H:%M:%S")),
               open(os.path.join(camp, f"campaign_manifest_{srv}.yaml"), "w"), sort_keys=False, allow_unicode=True)
# §19.2 상태: 지정한 유한 registry 의 진행 상태를 전부 남긴다 (실행하지 않은 case 에 성능값을 채우지 않는다)
rows = []
for f in sorted(glob.glob("config/NA104_*.yaml")):
    tag = os.path.basename(f)[:-5]; c = yaml.safe_load(open(f)); sp = resolve(c["kdv"])
    done = os.path.exists(os.path.join("work_dir", tag, "results", "reduced_best_val.mat")) or os.path.exists(os.path.join("work_dir", tag, "last_meta.json"))
    started = os.path.isdir(os.path.join("work_dir", tag))
    rows.append(dict(case_id=tag.split("_")[1], run_id=tag, server_block=(srv if tag in runs else ""), seed=c["seed"], updates=c["num_iter"],
                     status=("FULL_N" if done else ("RUNNING" if started else "SMOKE_PASS" if tag in runs[:2] else "UNIT_PASS")),
                     rec=sp["rec_case"], rec_control=sp["rec_control"], stat=("OFF" if not sp["stat_enabled"] else f"{sp['stat_key']}-{sp['stat_mode']}"),
                     tri=f"A{sp['tri']['a_mode']}/B{sp['tri']['b_mode']}/C{sp['tri']['c_mode']}", protocol=sp["na_protocol"],
                     teacher=("none" if not sp["has_teacher"] else ("eval_only" if sp["teacher_eval_only"] else "train")), setting=describe(sp)))
with open(os.path.join(camp, "case_status.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"   campaign_manifest_{srv}.yaml: {len(runs)} run · case_status.csv: {len(rows)} case")
PYEOF
cp -f research_log/2026-09-11_na104-implementation.md "$CAMP/implementation_map.md" 2>/dev/null || true

if ps -eo args | grep -q '[_]run_cases\.sh'; then echo "!! 다른 체인이 돌고 있다 — 끝난 뒤 다시 실행"; exit 1; fi
echo "[na104] 큐: $QUEUE ($(grep -vc '^#\|^$' "$QUEUE") run, FINAL 단계별: P0_DONE(완료분·건너뜀) → P1 → P2 → R1(seed777) → … ) · 구조·직접 대조: config/queues/na104_${SERVER}_stage_plan.json"
[ "$START" = 1 ] || { echo "[na104] 준비 완료 — 기동: ./tools/campaign_start.sh --queue $QUEUE --hours $HOURS --label na104-$SERVER"; exit 0; }
./tools/campaign_start.sh --queue "$QUEUE" --hours "$HOURS" --label "na104-$SERVER"
./tools/_watchdog.sh --install && echo "  감시자 cron 등록"
