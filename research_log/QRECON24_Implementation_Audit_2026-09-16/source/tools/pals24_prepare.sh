#!/usr/bin/env bash
# PALS24 (P2 기반 λ_off 비교, 24 GPU-h) — s1: donor 확인 → gate → metric gate(G-M0–G-M8 + 재사용 registry) → config → smoke/throughput → 예산 ledger → manifest → 체인 기동.
#   ./tools/pals24_prepare.sh [--no-start] [--hours 40]
# 계획 research_log/PAN_P2_P3_LambdaSweep_MetricAware_24GPUh_Plan_2026-09-12_v2.md · 노트 research_log/2026-09-12_pals24-implementation.md
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
START=1; HOURS=40
while [ $# -gt 0 ]; do case "$1" in --no-start) START=0;; --hours) HOURS="$2"; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift; done
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
QUEUE=config/queues/pals24_s1.txt; DONOR=PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT; CAMP=work_dir/_pals24_campaign; LEDGER=work_dir/_pals24_budget/ledger.json
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; [ "$SERVER" = "s1" ] || { echo "!! 계획: s1 단일 GPU 순차 실행 (현재 $SERVER)"; exit 1; }
T_GATE0=$(date +%s); mkdir -p "$CAMP" "$(dirname "$LEDGER")"
echo "[pals24] G0 donor: $DONOR/last (정확한 50K)"
"$PY" - <<PYEOF
import json, sys; sys.path.insert(0, ".")
from kdv.teacher_assets import load_donor_aligner
m = json.load(open("work_dir/$DONOR/last_meta.json")); assert m["step"] == 50000, m
al, man = load_donor_aligner("work_dir/$DONOR/last", 8); print(f"   step {m['step']} · aligner sha {man['aligner_tensors_sha256_16']} · file {man['file_sha256'][:16]} · head bias {al.fc2.bias.tolist()}")
PYEOF
echo "[pals24] 지표 이식"; "$PY" tools/verify_metrics.py | tail -1
echo "[pals24] 기존 gate (A1–A3 / PO10 / KDV / NF16)"; "$PY" tools/pa_unit_tests.py | tail -1; PO10_LEDGER=/tmp/_po10_gate_ledger.json "$PY" tools/po10_unit_tests.py | tail -1; "$PY" tools/kdv_unit_tests.py | tail -1; "$PY" tools/nf16_unit_tests.py | tail -1
echo "[pals24] PALS24 gate (PL01–PL12)"; "$PY" tools/pals24_unit_tests.py | tail -1
echo "[pals24] G-M0–G-M8 metric gate + 재사용 registry (NF16 P0/P1/P2/P3 best_hqnr 재평가, GPU)"
set +e; "$PY" tools/pals24_metric_gate.py > work_dir/_pals24_campaign/metric_gate.log 2>&1; rc=$?; set -e; grep -v "Warning\|torch.load\|obj = " work_dir/_pals24_campaign/metric_gate.log | tail -8
[ $rc -eq 0 ] || { echo "!! metric gate 실패 rc=$rc — work_dir/_pals24_campaign/metric_gate.log"; exit 1; }
"$PY" - <<'PYEOF'
import json; r = json.load(open("work_dir/_pals24_campaign/reuse_registry.json")); assert not r["rejected"] and len(r["approved"]) == 3, ("재사용 거부", r["rejected"]); print("   재사용 승인:", r["approved"])
PYEOF
echo "[pals24] 재사용 run 의 반응 진단을 PALS24 고정 probe 로 다시 (last·best_raw 반응 + donor 대비 drift; 보간/MS 대조·stress 는 기존 po10_diag_last.json 그대로) — 표 B/C 동일 정의"
# (pipefail 함정: 파이프 중간의 grep/head 가 먼저 닫히면 SIGPIPE(141) 로 스크립트가 죽는다 — 로그 파일에 쓰고 나서 읽는다)
for t in NF16_P2_W112_D123_WV3_S1234_N2LAST_v1 NF16_P3_W112_D123_WV3_S1234_N2LAST_v1 NF16_P1_W112_D123_WV3_S1234_N2LAST_v1; do
  for CK in last best_hqnr; do
    L="work_dir/$t/results/po10_diag_${CK}_pals24.log"
    set +e; "$PY" tools/po10_diag.py --run "$t" --ckpt "$CK" --out "po10_diag_${CK}_pals24" --probe-set pals24 --response-only --ref-run "$DONOR" --ref-ckpt last > "$L" 2>&1; rc=$?; set -e
    [ $rc -eq 0 ] || { echo "!! po10_diag($t, $CK) 실패 rc=$rc — $L"; grep -v Warning "$L" | tail -5; exit 1; }
    grep "fr512" "$L" | sed "s/^/   $t $CK: /" | cut -c1-200 || true
  done
done
L="work_dir/$DONOR/results/po10_diag_last_pals24.log"
set +e; "$PY" tools/po10_diag.py --run "$DONOR" --ckpt last --out po10_diag_last_pals24 --probe-set pals24 --response-only > "$L" 2>&1; rc=$?; set -e
[ $rc -eq 0 ] || { echo "!! po10_diag(donor) 실패 rc=$rc — $L"; exit 1; }
grep "fr512" "$L" | sed "s/^/   donor last: /" | cut -c1-200 || true
PROJ=$("$PY" -c "import json,os;p='$LEDGER';print((json.load(open(p)).get('throughput') or {}).get('projected_run_hours_50k') or '') if os.path.exists(p) else print('')" 2>/dev/null || echo "")
echo "[pals24] stage 1 config (run 당 예상 ${PROJ:-미측정 → block 예약 2.0} h)"; "$PY" tools/gen_pals24_configs.py ${PROJ:+--projected-hours "$PROJ"} | tail -2
CASES=$(grep -v '^#' "$QUEUE" | grep -v '^$' | tr '\n' ' ')
echo "[pals24] smoke (실배치·시간·FR 512² forward)"
# shellcheck disable=SC2086
set +e; "$PY" tools/smoke_cases.py $CASES > work_dir/_pals24_smoke.log 2>&1; rc=$?; set -e; tail -5 work_dir/_pals24_smoke.log
[ $rc -eq 0 ] || { echo "!! smoke 실패 rc=$rc — work_dir/_pals24_smoke.log"; exit 1; }
T_GATE=$(( $(date +%s) - T_GATE0 ))
echo "[pals24] 예산 ledger (24 GPU-h) 초기화 — gate 실측 ${T_GATE}s (예약 2.0h)"
"$PY" - "$T_GATE" <<'PYEOF'
import json, os, re, sys, time
lp = "work_dir/_pals24_budget/ledger.json"; d = json.load(open(lp)) if os.path.exists(lp) else dict(total_gpu_hours=24.0, entries={})
log = open("work_dir/_pals24_smoke.log").read(); tn = re.findall(r"t_native (\d+)ms", log); tc = re.findall(r"t_corrupt (\d+)ms", log)
if tn:
    t_nat = float(tn[-1]) / 1000; t_cor = (float(tc[-1]) / 1000 if tc else t_nat)
    proj = 50000 * (t_nat + t_cor) / 2 / 3600 + 25 * 67.0 / 3600           # 평가 25회 × 67s (NF16 실측: reduced 10s + full 53s + 여유)
    d["throughput"] = dict(t_native_s=t_nat, t_offset_exercise_s=t_cor, projected_run_hours_50k=proj, nf16_measured_run_hours=[1.43, 1.45, 1.47, 1.52], measured_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
gate_h = max(float(sys.argv[1]) / 3600.0, 0.05)
prev = d["entries"].get("gate", {}); d["entries"]["gate"] = dict(kind="gate", hours=gate_h + float(prev.get("hours") or 0.0), reserved_hours=2.0, note="metric provenance·donor/대조 재평가(G-M1)·재사용 run 반응 진단·unit/smoke·profiling (계획 §11.4 예약 2.0h)", finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
d.setdefault("reserved_plan", dict(gate=2.0, screen_seed1234=6.0, confirm_seed7777=6.0, confirm_seed2025=6.0, final_evaluation=3.0, buffer=1.0, total=24.0, margin=1.1, drop_order=["confirmation_seed2025", "confirmation_seed7777"]))
json.dump(d, open(lp, "w"), indent=1); print("  ledger:", json.dumps(d.get("throughput")), "| gate", round(d["entries"]["gate"]["hours"], 3), "h")
PYEOF
echo "[pals24] 캠페인 manifest·환경 기록 → $CAMP"
"$PY" - "$QUEUE" <<'PYEOF'
import json, os, platform, subprocess, sys, time, torch, yaml
sys.path.insert(0, os.getcwd()); from tools import gen_pals24_configs as G
from pa.evalviews import evaluator_hash
q = sys.argv[1]; runs = [l.strip() for l in open(q) if l.strip() and not l.startswith("#")]; g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
reg = json.load(open("work_dir/_pals24_campaign/reuse_registry.json"))
man = dict(campaign_id=G.CAMPAIGN_ID, protocol_id=G.PROTOCOL_ID, document_revision=G.DOC_REV, plan=G.PLAN, implementation_note=G.NOTE, server="s1", gpu_count=1, parallel_training_runs=1,
           budget_gpu_hours=G.TOTAL_HOURS, ledger=G.LEDGER, reserved=dict(initial_gate=2.0, final_evaluation=3.0, buffer=1.0, per_run=G.RUN_RESERVED_HOURS, margin=G.MARGIN, drop_order=["confirmation_seed2025", "confirmation_seed7777"]),
           training_updates_per_run=50000, screen_seed=G.SCREEN_SEED, screen_lambdas_new=[G.LAMBDA[c] for c in G.NEW_LAMBDAS], screen_control_lambdas=[0.0, 0.01], confirmation_seeds=G.CONFIRM_SEEDS, confirmation_cases=["CTRL-P0", "L000", "SELECTED_LAMBDA"],
           selected_lambda=None, selection_rule="raw best HQNR (best_raw raw_original mat20) -> fSCC (1e-4 band) -> smaller positive lambda; fixed once by tools/pals24_select_lambda.py", reselect_after_confirmation=False,
           termination_rule="all three new lambdas below seed-1234 P2 by more than 0.0031 -> confirmation blocks not opened (selected_lambda.json: TERMINATE); override: tools/pals24_select_lambda.py --policy continue",
           donor=dict(run=G.DONOR_RUN, kind="last", update=50000, copy="aligner_only", seed_fixed=2025), reuse=reg["approved"], reuse_registry="work_dir/_pals24_campaign/reuse_registry.json", metric_contract="work_dir/_pals24_campaign/metric_contract.json",
           stage1_queue=q, stage1_runs=runs, stage2="opened by tools/campaign_gate.py gate pals24 (work_dir/campaign_gates_enabled.txt) after A1-A3; order B1 P0/7777, B2 L000/7777, B3 λ*/7777, C1 L000/2025, C2 λ*/2025, C3 P0/2025",
           allow_unequal_update_budget=False, allow_automatic_new_lambda_or_architecture=False, evaluator_hash=evaluator_hash(),
           file_map=dict(config_resolved="work_dir/<run>/meta/config.yaml + kdv_config_resolved.json", provenance="init_and_teacher_hashes.json + architecture_manifest.json + dataset_hashes.json", metric_contract="work_dir/_pals24_campaign/metric_contract.json (campaign level)",
                         train_log="console.log + metrics.csv", gradient_diagnostics="gradient_diagnostics.jsonl (odd steps: grad_rec_A, grad_off_A, rho_g, cos_psi, loss_off_raw/weighted, off_unet_grad_absent)",
                         checkpoint_metrics="checkpoint_metrics.csv", scene_metrics="scene_metrics.csv", delta_predictions="delta_predictions.csv", offset_response="offset_response_<scale>_pals24_<ckpt>.csv + results/po10_diag_<ckpt>_pals24.json",
                         best_raw="best_hqnr/ (+ best_raw_meta.json, best_hqnr_meta.json)", last="last/ + last_meta.json (exact 50000)", best_aligned_diagnostic="best_aligned/ (diagnostic only)", budget="budget_status.json (+ campaign ledger)"),
           environment=dict(hostname=platform.node(), python=sys.version.split()[0], torch=str(torch.__version__), cuda=str(torch.version.cuda), cudnn=int(torch.backends.cudnn.version() or 0), gpu=(str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None),
                            driver=g("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"), git_commit=g("git rev-parse HEAD"), git_dirty=bool(g("git status --porcelain -- ':!results_log'")),
                            grid_sample_backward_nondeterministic_note="F.grid_sample backward on CUDA is not guaranteed deterministic (PyTorch docs); same seed does not imply bit-identical runs across platforms"),
           created=time.strftime("%Y-%m-%dT%H:%M:%S"))
yaml.safe_dump(man, open("work_dir/_pals24_campaign/campaign_manifest.yaml", "w"), sort_keys=False, allow_unicode=True)
print("   campaign_manifest.yaml:", len(runs), "stage-1 run ·", man["environment"]["gpu"], "·", man["environment"]["git_commit"][:10])
PYEOF
cp -f research_log/2026-09-12_pals24-implementation.md "$CAMP/implementation_note.md" 2>/dev/null || true
echo "pals24" > work_dir/campaign_gates_enabled.txt; echo "[pals24] 캠페인 gate 활성: work_dir/campaign_gates_enabled.txt = pals24 (stage 2 를 체인이 A1–A3 뒤에 연다; 캠페인이 끝나면 이 파일을 지울 것)"
if ps -eo args | grep -q '[_]run_cases\.sh'; then echo "!! 다른 체인이 돌고 있다 — 끝난 뒤 다시 실행"; exit 1; fi
[ "$START" = 1 ] || { echo "[pals24] 준비 완료 — 기동: ./tools/campaign_start.sh --queue $QUEUE --hours $HOURS --label pals24-s1"; exit 0; }
./tools/campaign_start.sh --queue "$QUEUE" --hours "$HOURS" --label pals24-s1
./tools/_watchdog.sh --install && echo "  감시자 cron 등록"
