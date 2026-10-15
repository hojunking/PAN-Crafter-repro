#!/usr/bin/env bash
# PALSV18 (L1E4 근방 미세조정·정합 능력 검증, 18 GPU-h) — s1: G0/G1 gate → G-M/G-S/G-C(재사용 registry·선택 격자·초기 tensor) → config → smoke/profiling → 예산 ledger → manifest → 체인 기동.
#   ./tools/palsv18_prepare.sh [--no-start] [--hours 30]
# 계획 research_log/PAN_L1E4_Refinement_AlignmentValidation_S1_18GPUh_2026-09-13.md · 노트 research_log/2026-09-13_palsv18-implementation.md
# V-pre(기존 대조군의 V0/V1/V2) 는 aligner 전용 추론이라 학습과 겹쳐 돌린다 — tools/palsv18_validate.sh (ledger 에 vpre 로 기록)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
START=1; HOURS=30
while [ $# -gt 0 ]; do case "$1" in --no-start) START=0;; --hours) HOURS="$2"; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift; done
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
QUEUE=config/queues/palsv18_s1.txt; DONOR=PO10_N2_OFFSG_W112_D123_WV3_S2025_R200_FRSTAT; CAMP=work_dir/_palsv18_campaign; LEDGER=work_dir/_palsv18_budget/ledger.json
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; [ "$SERVER" = "s1" ] || { echo "!! 계획: s1 단일 GPU 순차 실행 (현재 $SERVER)"; exit 1; }
T_GATE0=$(date +%s); mkdir -p "$CAMP" "$(dirname "$LEDGER")"
if ps -eo args | grep -q '[_]run_cases\.sh'; then echo "!! 다른 체인이 돌고 있다 (old pals24 종료 여부 확인, 계획 §4.2 G0) — 끝난 뒤 다시 실행"; exit 1; fi
grep -q "\[cases\] DONE" work_dir/cases_chain.log 2>/dev/null && echo "[palsv18] 이전 체인(PALS24) DONE 확인: $(grep '\[cases\] DONE' work_dir/cases_chain.log | tail -1)"
echo "[palsv18] G-A donor: $DONOR/last (정확한 50K)"
"$PY" - <<PYEOF
import json, sys; sys.path.insert(0, ".")
from kdv.teacher_assets import load_donor_aligner
m = json.load(open("work_dir/$DONOR/last_meta.json")); assert m["step"] == 50000, m
al, man = load_donor_aligner("work_dir/$DONOR/last", 8); assert man["aligner_tensors_sha256_16"] == "db638b55c62f8c62", man
print(f"   step {m['step']} · aligner sha {man['aligner_tensors_sha256_16']} · file {man['file_sha256'][:16]} · head bias {al.fc2.bias.tolist()}")
PYEOF
echo "[palsv18] 지표 이식"; "$PY" tools/verify_metrics.py | tail -1
echo "[palsv18] 기존 gate (A1–A3 / PO10 / KDV / NF16 / PALS24 PL01–PL12: G-G graph·G-W warp·G-R RNG)"; "$PY" tools/pa_unit_tests.py | tail -1; PO10_LEDGER=/tmp/_po10_gate_ledger.json "$PY" tools/po10_unit_tests.py | tail -1; "$PY" tools/kdv_unit_tests.py | tail -1; "$PY" tools/nf16_unit_tests.py | tail -1; "$PY" tools/pals24_unit_tests.py | tail -1
echo "[palsv18] PALSV18 gate (PV01–PV07: 편성·config 동치·예산·G-C 초기 tensor·G-S 격자·probe)"; "$PY" tools/palsv18_unit_tests.py | tail -1
echo "[palsv18] G-M metric gate + 재사용 registry (대조군 9벌 + N2 last + L1E2 best_hqnr 재평가, GPU) + selection_grid.json + initial_tensor_registry.json"
set +e; "$PY" tools/pals24_metric_gate.py --campaign palsv18 > "$CAMP/metric_gate.log" 2>&1; rc=$?; set -e; grep -v "Warning\|torch.load\|obj = " "$CAMP/metric_gate.log" | tail -14
[ $rc -eq 0 ] || { echo "!! metric gate 실패 rc=$rc — $CAMP/metric_gate.log"; exit 1; }
"$PY" - <<'PYEOF'
import json; r = json.load(open("work_dir/_palsv18_campaign/reuse_registry.json")); assert not r["rejected"] and len(r["approved"]) == 3, ("재사용 거부", r["rejected"])
i = json.load(open("work_dir/_palsv18_campaign/initial_tensor_registry.json")); assert all(v["all_match"] for v in i["seeds"].values()), i["seeds"]
print("   재사용 승인(case→마지막 seed 의 run):", r["approved"], "| 초기 tensor 3 seed 일치")
PYEOF
echo "[palsv18] config (run 당 예상 = max(1.7, smoke×1.1); 첫 생성은 1.7)"; "$PY" tools/gen_palsv18_configs.py --projected-hours 1.7 | tail -2
CASES=$(grep -v '^#' "$QUEUE" | grep -v '^$' | tr '\n' ' ')
echo "[palsv18] G-T smoke (실배치·시간·peak)"
set +e; "$PY" tools/smoke_cases.py $CASES > work_dir/_palsv18_smoke.log 2>&1; rc=$?; set -e; tail -4 work_dir/_palsv18_smoke.log
[ $rc -eq 0 ] || { echo "!! smoke 실패 rc=$rc — work_dir/_palsv18_smoke.log"; exit 1; }
PROJ=$("$PY" - <<'PYEOF'
import re; log = open("work_dir/_palsv18_smoke.log").read(); tn = re.findall(r"t_native (\d+)ms", log); tc = re.findall(r"t_corrupt (\d+)ms", log)
t_nat = float(tn[-1]) / 1000; t_cor = float(tc[-1]) / 1000 if tc else t_nat; proj = 50000 * (t_nat + t_cor) / 2 / 3600 + 25 * 67.0 / 3600
print(f"{max(1.7, 1.1 * proj):.3f}")
PYEOF
)
echo "[palsv18] config 재생성 (projected_hours $PROJ = max(1.7, smoke×1.1))"; "$PY" tools/gen_palsv18_configs.py --projected-hours "$PROJ" | tail -1
T_GATE=$(( $(date +%s) - T_GATE0 ))
echo "[palsv18] 예산 ledger (18 GPU-h) — gate 실측 ${T_GATE}s (예약 0.8h)"
"$PY" - "$T_GATE" "$PROJ" <<'PYEOF'
import json, os, re, sys, time
lp = "work_dir/_palsv18_budget/ledger.json"; d = json.load(open(lp)) if os.path.exists(lp) else dict(total_gpu_hours=18.0, entries={})
log = open("work_dir/_palsv18_smoke.log").read(); tn = re.findall(r"t_native (\d+)ms", log); tc = re.findall(r"t_corrupt (\d+)ms", log)
t_nat = float(tn[-1]) / 1000; t_cor = float(tc[-1]) / 1000 if tc else t_nat
d["throughput"] = dict(t_native_s=t_nat, t_offset_exercise_s=t_cor, projected_run_hours_50k=50000 * (t_nat + t_cor) / 2 / 3600 + 25 * 67.0 / 3600, projected_used_for_gate=float(sys.argv[2]), pals24_measured_run_hours=[1.46, 1.45, 1.45, 1.41, 1.41], measured_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
prev = d["entries"].get("gate", {}); d["entries"]["gate"] = dict(kind="gate", hours=max(float(sys.argv[1]) / 3600.0, 0.02) + float(prev.get("hours") or 0.0), reserved_hours=0.8, note="G0/G1: asset·metric contract·선택 격자·초기 tensor·gradient/warp/RNG gate·smoke/profiling (계획 §4.1 예약 0.8h)", finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
d["reserved_plan"] = dict(gate=0.8, pre_validation=2.0, new_training=10.2, post_validation=3.5, reporting=0.5, buffer=1.0, total=18.0, margin=1.1, per_run=1.7, drop_order=["seed2025_pair", "seed7777_pair"], include_previous_spent=False, auto_add_previous_unused=False)
json.dump(d, open(lp, "w"), indent=1); print("  ledger:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d["throughput"].items()}), "| gate", round(d["entries"]["gate"]["hours"], 3), "h")
PYEOF
echo "[palsv18] 캠페인 manifest → $CAMP"
"$PY" - "$QUEUE" <<'PYEOF'
import json, os, platform, subprocess, sys, time, torch, yaml
sys.path.insert(0, os.getcwd()); from tools import gen_palsv18_configs as V, gen_pals24_configs as G
from pa.evalviews import evaluator_hash
q = sys.argv[1]; runs = [l.strip() for l in open(q) if l.strip() and not l.startswith("#")]; g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
reg = json.load(open("work_dir/_palsv18_campaign/reuse_registry.json")); grid = json.load(open("work_dir/_palsv18_campaign/selection_grid.json"))
man = dict(campaign_id=V.CAMP["campaign_id"], training_semantics=V.CAMP["plan_protocol_id"], document_revision=V.CAMP["document_revision"], plan=V.CAMP["plan"], implementation_note=V.CAMP["note"], server="s1", gpu_count=1, parallel_training_runs=1,
           budget=dict(cap_gpu_hours=18.0, include_previous_spent_hours=False, auto_add_previous_unused_9p8_hours=False, gate_hours=0.8, pre_validation_hours=2.0, new_training_hours=10.2, post_validation_hours=3.5, reporting_hours=0.5, buffer_hours=1.0,
                       reservation_per_full_run_hours=1.7, eta_margin_fraction=0.10, drop_new_seed_blocks_in_order=[2025, 7777], allow_unequal_completed_updates=False, allow_new_unplanned_lambdas=False, ledger=V.CAMP["ledger"],
                       run_gate="used + 1.1*(this + same-seed other lambda) + 5.0 <= 18 (kdv.budget; DEFERRED_BUDGET exit 4)"),
           initialization=dict(donor_run=G.DONOR_RUN, donor_checkpoint_kind="last", donor_optimizer_update=50000, copy_aligner_only=True, donor_path=f"work_dir/{G.DONOR_RUN}/last/model.safetensors", donor_sha256=G.donor_identity()[0],
                               initial_unet_registry="work_dir/_palsv18_campaign/initial_tensor_registry.json", reuse_registry="work_dir/_palsv18_campaign/reuse_registry.json", resume_from_completed_L1E4=False),
           working_reference="L1E4 (lambda_off 1e-4)", new_runs=[dict(case_id=f"R-{c}-S{s}", run_id=V.run_name(c, s), seed=s, lambda_off=G.LAMBDA[c]) for c, s in V.CELLS], queue=q, queue_runs=runs, conditional_gate=None,
           metrics=dict(fr_manifest_id="fr_mat20", primary_view="raw_original", primary_checkpoint="best_raw", primary_metric="hqnr_raw_original", secondary_metric="fscc_raw_original", secondary_split="FR", secondary_reference="native_original_PAN",
                        evaluator_hash=evaluator_hash(), fscc_function_key="pa.evalviews.fscc_from_maps", selection_grid_path="work_dir/_palsv18_campaign/selection_grid.json", selection_grid_sha256=grid["sha256"], aggregate="mean_of_scene_products",
                        tie_band_hqnr=1e-4, empirical_hqnr_method_margin=0.0031, exact_last_update=50000, views=["raw_original", "raw_v64", "aligned_self_v64", "aligned_fixedN2_v64"], aligned_for_primary_selection=False),
           validation=dict(checkpoint_roles=["best_raw", "exact50k_last"], response_radii_hr=[0.25, 0.5, 1.0, 2.0], directions_per_radius=8, zero_probe_for_identity_only=True, response_modes=["pan_only", "ms_only", "common"],
                           stress_radii_hr=[0.5, 1.0, 2.0], stress_reference="native_fixed", stress_fr_margin_hr=96, tools="tools/po10_diag.py --probe-set palsv18 (V1/V2/V4 stress) · tools/palsv18_validate.py (V2 modality, V3, V4 interventions) · tools/palsv18_report.py"),
           environment=dict(hostname=platform.node(), python=sys.version.split()[0], torch=str(torch.__version__), cuda=str(torch.version.cuda), gpu=(str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None),
                            git_commit=g("git rev-parse HEAD"), git_dirty=bool(g("git status --porcelain -- ':!results_log'"))), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
yaml.safe_dump(man, open("work_dir/_palsv18_campaign/campaign_manifest.yaml", "w"), sort_keys=False, allow_unicode=True)
print("   campaign_manifest.yaml:", len(runs), "run ·", man["environment"]["gpu"], "·", man["environment"]["git_commit"][:10])
PYEOF
# 계획 §14.3: 종료된 pals24 token 정리 — PALSV18 은 조건부 gate 가 없다 (전부 무조건 큐)
if [ -f work_dir/campaign_gates_enabled.txt ]; then mv work_dir/campaign_gates_enabled.txt "$CAMP/campaign_gates_enabled.pals24.closed.txt"; echo "[palsv18] 캠페인 gate token 정리: pals24 → $CAMP/campaign_gates_enabled.pals24.closed.txt (PALSV18 은 gate 없음)"; fi
[ "$START" = 1 ] || { echo "[palsv18] 준비 완료 — 기동: ./tools/campaign_start.sh --queue $QUEUE --hours $HOURS --label palsv18-s1"; exit 0; }
./tools/campaign_start.sh --queue "$QUEUE" --hours "$HOURS" --label palsv18-s1
./tools/_watchdog.sh --install && echo "  감시자 cron 등록"
