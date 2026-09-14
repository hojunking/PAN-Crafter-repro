#!/usr/bin/env bash
# PAKD50 (L1E4 aligner 재사용·적응 × Q12/R1, raw HQNR 0.959–0.960, 50h 병렬) — 서버 공통 prepare: T0 자산 검증·재현 → gate → τR 고정값 대조 → stage 1 config → smoke·처리량 → ledger(46h 시계) → manifest → 체인 기동.
#   ./tools/pakd50_prepare.sh [--no-start] [--stage 1|2] [--hours 46]
# 계획 research_log/PAN_Integrated_50H_Experiment_Plan_HQNR959_960_2026-09-14.md · 노트 research_log/2026-09-14_pakd50-implementation.md
# s2/s3: git pull 로 같은 release(commit) 를 받고 실행. 학습 중인 checkout 을 pull 로 바꾸지 않는다 (계획 §13.4 — 별도 worktree 권장).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
START=1; HOURS=46; STAGE=1
while [ $# -gt 0 ]; do case "$1" in --no-start) START=0;; --hours) HOURS="$2"; shift;; --stage) STAGE="$2"; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift; done
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-}"; [ -n "$PY" ] || { [ -x /home/knuvi/miniconda3/envs/pancrafter/bin/python ] && PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python || PY=python; }
SERVER="$(tr -d '[:space:]' < gspread/server.txt)"; case "$SERVER" in s1|s2|s3|s4) ;; *) echo "!! s1/s2/s3/s4 전용 (현재 $SERVER)"; exit 1;; esac
QUEUE="config/queues/pakd50_${SERVER}_stage${STAGE}.txt"; CAMP=work_dir/_pakd50; LEDGER=work_dir/_pakd50_budget/ledger.json; mkdir -p "$CAMP" "$(dirname "$LEDGER")"; T0=$($PY -c "import sys; sys.path.insert(0,'.'); from tools import gen_pakd50_configs as G; print(G.t0_dir('$SERVER'))")
T_PREP0=$(date +%s); fail() { echo "!! $1"; exit 1; }
if ps -eo args | grep -q '[_]run_cases\.sh'; then fail "다른 체인이 돌고 있다 — 먼저 정리 (NA104 20H 등)"; fi
echo "[pakd50] ① T0 자산 검증 ($T0) + 기록 지표 재현 (계획 §14.1 'T0 재현')"
"$PY" - "$T0" "$SERVER" > "$CAMP/t0_verify.log" 2>&1 <<'PYEOF'
import json, os, sys, time, numpy as np, torch, yaml, h5py
sys.path.insert(0, os.getcwd()); from kdv.teacher_assets import sha256_file, load_run_model, freeze
from tools.eval_fr_paperset import build, h5_for, sensor_of, load_dlpan
from tools.metrics.eval_fr import d_lambda_k, d_s
from main import import_class
t0, srv = sys.argv[1], sys.argv[2]; ta = json.load(open("assets/pakd50/T0_run/teacher_assets.json"))
for f, sha in ta["files"].items():
    assert sha256_file(os.path.join(t0, f)) == sha, f"T0 파일 sha 불일치: {f}"
cfg = yaml.safe_load(open(os.path.join(t0, "meta", "config.yaml"))); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m, fwd, how = build(cfg, t0, "best_hqnr"); m = m.to(dev).eval(); Feeder = import_class(cfg["feeder"]); fa = dict(cfg["test_full_feeder_args"]); h5 = h5_for(sensor_of(cfg)); fa["dataroot"] = h5; ds = Feeder(**fa); mp = float(ds.max_pixel)
wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", "/home/knuvi/Desktop/song/DLPan-Toolbox")); srs = []
with torch.no_grad():
    for i in range(len(ds)):
        lms, ms, lpan, pan = (t.unsqueeze(0).to(dev) for t in ds[i]); y = fwd(pan, lpan, ms, lms); srs.append(((y.clip(-1, 1).float().cpu().numpy() + 1) / 2 * mp)[0])
with h5py.File(h5) as f:
    lms_all = np.asarray(f["lms"], dtype=np.float64).transpose(0, 2, 3, 1); pan_all = np.asarray(f["pan"], dtype=np.float64)[:, 0]
sensor = sensor_of(cfg); h = np.array([(1 - d_lambda_k(s.astype(np.float64).transpose(1, 2, 0), lms_all[i], sensor, 4, 32, wald)) * (1 - d_s(s.astype(np.float64).transpose(1, 2, 0), lms_all[i], pan_all[i], 4, 32, wald)) for i, s in enumerate(srs)])
rec = ta["recorded_hqnr_raw_original"]; diff = abs(float(h.mean()) - rec)
out = dict(server=srv, t0=t0, checkpoint_sha256=ta["checkpoint_sha256"], selected_update=ta["selected_update"], reproduced_hqnr_raw_original=float(h.mean()), recorded=rec, abs_diff=diff, per_scene=[float(x) for x in h], tolerance=1e-4, pass_=bool(diff <= 1e-4), at=time.strftime("%Y-%m-%dT%H:%M:%S"))
json.dump(out, open("work_dir/_pakd50/t0_reproduction.json", "w"), indent=1); print(f"   T0 raw HQNR 재현 {out['reproduced_hqnr_raw_original']:.6f} vs 기록 {rec:.6f} (Δ {diff:.1e}, 허용 1e-4) → {'PASS' if out['pass_'] else 'FAIL'}")
assert out["pass_"], "T0 재현 실패 — 평가기/자산 차이를 먼저 설명 (계획 §14.1)"
PYEOF
RC=$?; grep -v Warning "$CAMP/t0_verify.log" | tail -2; [ $RC -eq 0 ] || fail "T0 검증 실패 — $CAMP/t0_verify.log"
echo "[pakd50] ①' calibration 사본 → 로컬 (gate 가 로컬 τR 을 검사하므로 먼저; 감사 F01)"
"$PY" -c "import sys; sys.path.insert(0,'.'); from tools import gen_pakd50_configs as G; m, src = G.sync_calibration_from_assets(write=True); print('   calibration 로컬 <-', src, {k: m.get(k) for k in ('tau_R','lambda_E')})" || fail "calibration 사본 없음 — assets/pakd50/calibration_resolved.json (s1 이 만든 τR) 을 pull"
echo "[pakd50] ② 지표 이식·기존 gate"; "$PY" tools/verify_metrics.py | tail -1
for t in pa_unit_tests kdv_unit_tests nf16_unit_tests pals24_unit_tests; do "$PY" tools/$t.py > "$CAMP/gate_$t.log" 2>&1 || fail "$t 실패 — $CAMP/gate_$t.log"; tail -1 "$CAMP/gate_$t.log"; done
"$PY" tools/pakd50_unit_tests.py > "$CAMP/gate_pakd50.log" 2>&1 || { tail -3 "$CAMP/gate_pakd50.log"; fail "pakd50 gate 실패 — $CAMP/gate_pakd50.log"; }; tail -1 "$CAMP/gate_pakd50.log"
echo "[pakd50] ③ τR: git 의 고정값(work_dir/_pakd50/calibration_resolved.json 은 서버 로컬) — 이 서버에서 재계산해 대조"
"$PY" -c "import sys; sys.path.insert(0,'.'); from tools import gen_pakd50_configs as G; m, src = G.sync_calibration_from_assets(write=True); print('   calibration 로컬 <-', src, {k: m.get(k) for k in ('tau_R','lambda_E')})"   # 첫 prepare: 사본 복사 · 이후: λE 만 덧입힘
# 고정값(PIN) 의 출처는 **자산 사본**(assets/pakd50/calibration_resolved.json) 이다 — 로컬 파일은 한 번 실패하면 재계산값이 남아 '고정값' 행세를 한다 (s4 보고 P-5).
PIN=$($PY -c "import json; print(json.load(open('assets/pakd50/calibration_resolved.json')).get('tau_R'))" 2>/dev/null || echo "")
[ -n "$PIN" ] && [ "$PIN" != "None" ] || fail "자산 사본에 τR 고정값이 없다 — assets/pakd50/calibration_resolved.json (s1 이 만든다)"
"$PY" tools/pakd50_calibrate.py --tau --server "$SERVER" > "$CAMP/calibrate_tau.log" 2>&1 || fail "τR calibration 실패 — $CAMP/calibrate_tau.log"; grep "τR" "$CAMP/calibrate_tau.log" | tail -1
$PY - "$PIN" <<'PYEOF'
import json, sys; pin = float(sys.argv[1]); p = "work_dir/_pakd50/calibration_resolved.json"; d = json.load(open(p)); now = float(d["tau_R"]); rel = abs(now - pin) / pin
# 복원을 **판정보다 먼저** — 판정이 실패해도 로컬 파일에는 공통 고정값이 남아 λE 수신(sync 의 τR 1e-9 조건) 이 막히지 않는다 (s4 보고 P-5)
d.update(tau_R=pin, tau_R_local_recomputed=now, tau_R_local_reldiff=rel, tau_R_pinned_from="assets/pakd50/calibration_resolved.json"); json.dump(d, open(p, "w"), indent=1, ensure_ascii=False, default=str)
print(f"   τR 이 서버 {now:.9f} vs 고정 {pin:.9f} (상대차 {rel:.1e}, 허용 1e-3; config 에는 고정값이 들어간다)")
# 허용치는 상대 1e-3: 같은 patch·같은 T0 에서도 GPU/커널 차이로 분위수가 1e-6 수준(상대 ~2e-4) 흔들린다 (s4 실측, 보고 P-4). 그 이상이면 데이터/T0/평가 경로 차이
assert rel < 1e-3, f"τR 이 s1 고정값과 다르다 (상대차 {rel:.2e} ≥ 1e-3) — 데이터/T0/평가 경로 차이를 먼저 설명"
PYEOF
[ $? -eq 0 ] || fail "τR 대조 실패"
echo "[pakd50] ④ stage $STAGE config (seed $($PY -c "from tools.gen_pakd50_configs import SERVER_SEED as S; print(S['$SERVER'])" 2>/dev/null))"; "$PY" tools/gen_pakd50_configs.py --server "$SERVER" --stage "$STAGE" 2>&1 | grep -v Warning | tail -2; [ "${PIPESTATUS[0]}" -eq 0 ] || fail "config 생성 실패 (stage $STAGE — stage 2 는 λE 사본이 있어야 한다)"
CASES=$(grep -v '^#' "$QUEUE" | grep -v '^$' | tr '\n' ' ')
echo "[pakd50] ⑤ smoke (실배치·시간·peak — Teacher 포함)"; "$PY" tools/smoke_cases.py $CASES > "$CAMP/smoke_stage$STAGE.log" 2>&1 || { grep -v Warning "$CAMP/smoke_stage$STAGE.log" | tail -4; fail "smoke 실패"; }; grep "^\[smoke\]" "$CAMP/smoke_stage$STAGE.log" | tail -5
T_PREP=$(( $(date +%s) - T_PREP0 ))
echo "[pakd50] ⑥ ledger (서버 학습 46h + 감사 4h; 50h 시계 시작) — 준비 ${T_PREP}s"
"$PY" - "$T_PREP" "$SERVER" "$CAMP/smoke_stage$STAGE.log" <<'PYEOF'
import json, os, re, sys, time
lp = "work_dir/_pakd50_budget/ledger.json"; d = json.load(open(lp)) if os.path.exists(lp) else dict(total_gpu_hours=50.0, entries={})
d["total_gpu_hours"] = 50.0                                     # 50h 벽시계 = slot 1 의 GPU-h; reserve 4h(46–50h 감사) 는 config kdv.budget.reserve_hours → 학습 admission ≤ 46h (감사 F05)
per = {}
for m in re.finditer(r"^\[smoke\] OK\s+(\S+).*?t_native (\d+)ms(?:.*?t_corrupt (\d+)ms)?", open(sys.argv[3]).read(), re.M):
    t_nat = float(m.group(2)) / 1000; t_cor = float(m.group(3)) / 1000 if m.group(3) else t_nat
    per[m.group(1)] = dict(t_native_s=t_nat, t_offset_exercise_s=t_cor, projected_run_hours_50k=50000 * (t_nat + t_cor) / 2 / 3600 + 50 * 67.0 / 3600)
if per:                                                          # case 별 실측 (감사 F05: FR 의 native 와 JR 의 offset 을 섞지 않는다); 예산은 가장 느린 case
    worst = max(per.values(), key=lambda v: v["projected_run_hours_50k"])
    d["throughput"] = dict(worst, per_case=per, note="case 별 smoke 실측, 예산은 max; 50 평가 × 67s 포함 (eval_epoch 5) — 완료 run 이 생기면 trainer 는 완료 평균을 쓴다", measured_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
prev = d["entries"].get("gate", {}); d["entries"]["gate"] = dict(kind="gate", hours=float(sys.argv[1]) / 3600.0 + float(prev.get("hours") or 0.0), note="T0 검증·재현, gate, τR, smoke (계획 §9.3 0–4h)", finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
json.dump(d, open(lp, "w"), indent=1)
cp = "work_dir/_pakd50/ledger.json"; c = json.load(open(cp)) if os.path.exists(cp) else {}
ck = "assets/pakd50/campaign_clock.json"; clock = json.load(open(ck)) if os.path.exists(ck) else {}
if not c.get("start"):
    c.update(server=sys.argv[2], start=time.strftime("%Y-%m-%dT%H:%M:%S"), wall_hours=50, training_finish_target_hour=46, final_audit_hours=4, budget_kind="elapsed_wall_hours_parallel_servers")
c["local_prepared_at"] = c.get("local_prepared_at") or c["start"]; c["measured_run_hours"] = (d.get("throughput") or {}).get("projected_run_hours_50k") or c.get("measured_run_hours")
if clock.get("training_deadline"):                                # 세 서버 공통 절대 시계 (감사 F05): 서버별 50h 창이 아니라 s1 이 정한 하나의 start/deadline
    c.update(start=clock["start"], training_deadline=clock["training_deadline"], final_deadline=clock["final_deadline"], clock_source=ck)
else:
    t0 = time.mktime(time.strptime(c["start"][:19], "%Y-%m-%dT%H:%M:%S")); f = lambda h: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0 + h * 3600))
    c.update(training_deadline=f(46), final_deadline=f(50), clock_source="local")
json.dump(c, open(cp, "w"), indent=1); print("   50h 시계", c["start"], "→ 학습 마감", c["training_deadline"], f"({c['clock_source']}) | 예상 run", c["measured_run_hours"])
PYEOF
"$PY" - "$SERVER" "$QUEUE" <<'PYEOF'
import json, os, platform, subprocess, sys, time, torch, yaml
sys.path.insert(0, os.getcwd()); from tools import gen_pakd50_configs as G
srv, q = sys.argv[1:3]; g = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip(); runs = [l.strip() for l in open(q) if l.strip() and not l.startswith("#")]
cal = G.calibration(); ta = json.load(open("assets/pakd50/T0_run/teacher_assets.json")); led = json.load(open("work_dir/_pakd50/ledger.json"))
man = dict(campaign=dict(id=G.CAMPAIGN_ID, status="running", budget_kind="elapsed_wall_hours_parallel_servers", wall_hours=50, training_finish_target_hour=46, final_audit_hours=4, start_time=led["start"], server=srv, gpu_slots_per_server=1, measured_case_hours=led.get("measured_run_hours")),
           release=dict(git_full_sha=g("git rev-parse HEAD"), git_dirty=bool(g("git status --porcelain -- ':!results_log' ':!work_dir'")), plan=G.PLAN, summary=G.SUMMARY, note=G.NOTE),
           model=dict(width=112, depth=[1, 2, 3], input_channels=9, output_channels=8, output_kind="residual_plus_ms_base_once", frame="M", aligner_source_class="pa.aligner.PANGlobalAligner", warp=dict(mode="bicubic", padding_mode="border", align_corners=False), aligner_margin_hr=4, crop_before_zscore=True),
           teacher=dict(package_id="T0", run_id=G.T0_RUN, checkpoint_kind="best_raw", selected_update=ta["selected_update"], checkpoint_path=G.t0_dir(srv), checkpoint_sha256=ta["checkpoint_sha256"], aligner_tensors_sha256_16=ta["aligner_tensors_sha256_16"], recorded_hqnr=ta["recorded_hqnr_raw_original"], frozen_own_aligner=True, stationary_given_fixed_input=True),
           student=dict(protocol="FRESH50", seed=G.SERVER_SEED[srv], init_policy="copy_teacher_aligner_fresh_unet", unet_init_file=f"work_dir/_kdv_init_w112_d123/init_unet_seed{G.SERVER_SEED[srv]}.pt (trainer 가 첫 run 에서 만들고 hash 를 run 폴더에 기록)"),
           objective=dict(alpha=1.0, beta=0.1, eps=1e-6, lambda_off=1e-4, off_period=2, off_phase_zero_based=1, jitter_radius_hr=2.0, tau_R=cal.get("tau_R"), lambda_E=cal.get("lambda_E"), edge_r_grad=0.05, edge_definition="signed_bandwise_scharr32_reflect1_interior1"),
           calibration=dict(tau_teacher_package="T0", max_train_patches=3072, seed=1234, batch=48, geometric_augmentation=False, patch_manifest_sha256=cal.get("patch_manifest_sha256"), edge_pilot="J0_seed1234_exact50k", file="work_dir/_pakd50/calibration_resolved.json"),
           training=dict(total_updates=50000, effective_batch=48, optimizer="AdamW", lr_unet_peak=1e-4, lr_aligner_peak=1e-5, weight_decay=0.01, warmup_updates=100, schedule="cosine", source_config="config/PO10_N1_REC_W112_D123_WV3_S2025_R200_FRSTAT.yaml (template; resolved config per run in work_dir/<run>/meta)"),
           evaluation=dict(primary="hqnr_raw_original", fr_scene_set="WV3_paper_mat20", aggregation="mean_of_per_scene_products", candidate_grid_id=G.GRID_ID, candidate_updates="1010*k (k=1..49) + 50000", exact_last_update=50000, plateau_updates=[45450, 46460, 47470, 48480, 49490, 50000], selector="best_hqnr (running-max HQNR tie 1e-4 → fSCC)", target_lower=0.959, target_stretch=0.960,
                           evaluator_hash=__import__("pa.evalviews", fromlist=["evaluator_hash"]).evaluator_hash()),
           queue=dict(file=q, runs=runs, stage2="gate pakd50 (s1: λE 고정 후 생성; s2/s3: git pull 뒤 --stage 2 로 campaign_start)"),
           environment=dict(hostname=platform.node(), python=sys.version.split()[0], torch=str(torch.__version__), cuda=str(torch.version.cuda), gpu=(str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None)), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
yaml.safe_dump(man, open(f"work_dir/_pakd50/campaign_manifest_{srv}.yaml", "w"), sort_keys=False, allow_unicode=True); print("   campaign_manifest:", len(runs), "run · release", man["release"]["git_full_sha"][:10], "dirty" if man["release"]["git_dirty"] else "clean")
PYEOF
echo "pakd50" > work_dir/campaign_gates_enabled.txt; echo "[pakd50] ⑦ gate token = pakd50 (stage 2 는 λE 고정 뒤)"
# 체인 마감 = 공통 training_deadline (절대 시각). 재기동(--stage 2 등)이 마감을 늘리지 않는다 (계획 §11.2, 감사 F05). --hours 는 무시된다.
DL=$("$PY" -c "import json; print(json.load(open('work_dir/_pakd50/ledger.json'))['training_deadline'])")
REM=$("$PY" -c "import time; t=time.mktime(time.strptime('$DL'[:19],'%Y-%m-%dT%H:%M:%S')); print('%.2f' % max(0.0, (t-time.time())/3600))")
echo "[pakd50] 공통 학습 마감 $DL 까지 ${REM}h"
[ "$("$PY" -c "print(1 if float('$REM') > 0.5 else 0)")" = 1 ] || fail "46h 학습 창이 끝났다 — 새 run 을 시작하지 않는다"
HOURS=$("$PY" -c "import math; print(max(1, math.ceil(float('$REM'))))")
[ "$START" = 1 ] || { echo "[pakd50] 준비 완료 — 기동: ./tools/campaign_start.sh --queue $QUEUE --hours $HOURS --label pakd50-$SERVER"; exit 0; }
./tools/campaign_start.sh --queue "$QUEUE" --hours "$HOURS" --label "pakd50-$SERVER-stage$STAGE" || fail "기동 실패"
echo "$DL" > work_dir/cases_deadline.txt; echo "  체인 마감을 공통 training_deadline 으로 고정: $DL"
./tools/_watchdog.sh --install && echo "  감시자 cron 등록"
