#!/usr/bin/env python
"""NA104 20H 우선순위 gate (research_log/01_S2_20H_PRIORITY.md · 02_S3_20H_PRIORITY.md). 서버 상태와 무관한 fixture 로 판정 장치를 검사한다 (리뷰 P1-1). 하나라도 실패하면 exit 1.
    python tools/na104_20h_unit_tests.py"""
import csv, json, os, subprocess, sys, tempfile, time
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve
from kdv.calibration import load_lambda_from_run
from tools import na104_20h as H

FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)

# ---------------- H01 config·큐 (저장소 파일)
need = [H.RUN[c] for c in H.P1] + [H.RUN[("CF01", s)] for s in (777, 1234, 2026)] + [H.RUN[("X02", s)] for s in (777, 2026)]
missing = [r for r in need if not os.path.exists(os.path.join(ROOT, "config", r + ".yaml"))]
check("H01 P1 4 + CF01 3 + X02 2 config 존재 (계획 run ID 그대로)", not missing, f"missing {missing}")
check("H01 Q36 S1234 v1 · Q00 S1234 v1 · Teacher T00 config 는 기존 파일", all(os.path.exists(os.path.join(ROOT, "config", r + ".yaml")) for r in (H.RUN[("Q36", 1234)], H.RUN[("Q00", 1234)], H.TEACHER)))
for srv in ("s2", "s3"):
    q = os.path.join(ROOT, "config", "queues", f"na104_20h_{srv}.txt"); qs = [l.strip() for l in open(q) if l.strip() and not l.startswith("#")] if os.path.exists(q) else []
    check(f"H01 큐 {srv}: P1 순서 Q36-777 → Q00-2026 → Q36-2026 → Q12-2026", qs == [H.RUN[c] for c in H.P1])

# ---------------- H02 CF01 spec + λ 출처 검증 (리뷰 P2-4)
k = yaml.safe_load(open(os.path.join(ROOT, "config", H.RUN[("CF01", 777)] + ".yaml")))["kdv"]; sp = resolve(k)
check("H02 CF01 = rec N0 · stat GC/FIX window 5 · kd_weight 0.1 · lambda_from_run = Q36 S1234 v1 · Teacher 학습 필요(eval_only 없음) · TRI 없음 · λ scale 1",
      sp["rec_case"] == "N0" and sp["stat_key"] == "GC" and sp["stat_mode"] == "FIX" and k["stat"]["window"] == 5 and abs(k["stat"]["kd_weight"] - 0.1) < 1e-12 and sp["stat_lambda_from_run"] == H.RUN[("Q36", 1234)]
      and sp["needs_teacher"] and not sp["teacher_eval_only"] and not k.get("tri") and sp["stat_lambda_scale"] == 1.0)
q36 = yaml.safe_load(open(os.path.join(ROOT, "config", H.RUN[("Q36", 1234)] + ".yaml")))["kdv"]
check("H02 CF01 은 Q36 과 rec/window/eps/calibration 이 같고 stat.mode(H→FIX)·lambda_from_run·teacher.eval_only 만 다르다",
      k["rec"] == q36["rec"] and k["stat"]["window"] == q36["stat"]["window"] and k["calibration"] == q36["calibration"] and q36["stat"]["mode"] == "H" and q36["teacher"].get("eval_only") is True and "eval_only" not in k["teacher"])
try:
    resolve(dict(k, stat=dict(k["stat"], outer_weight=0.5))); check("H02 lambda_from_run + 숫자 outer_weight 는 거부", False)
except Exception:
    check("H02 lambda_from_run + 숫자 outer_weight 는 거부", True)
d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "work_dir", "SRC"))
def wsrc(j):
    json.dump(j, open(os.path.join(d, "work_dir", "SRC", "calibration_resolved.json"), "w"))
wsrc({"lambda": {"lambda_V_used": 0.42, "kind": "GC", "window": 5}, "stat": {"kind": "GC", "windows": [5], "transform": "none", "domain": "final_hrms"}})
lam, info = load_lambda_from_run("SRC", "GC", 5, root=d)
check("H02 λ 출처 검증: 정상 출처 → λ 0.42 + 파일 sha256·출처 통계 기록", lam == 0.42 and len(info["source_sha256"]) == 64 and info["source_stat"]["kind"] == "GC")
rej = []
for bad in ({"lambda": {"lambda_V_used": float("nan")}, "stat": {"kind": "GC", "windows": [5]}}, {"lambda": {"lambda_V_used": -3}, "stat": {"kind": "GC", "windows": [5]}}, {"lambda": {"lambda_V_used": 0.4}, "stat": {"kind": "IV", "windows": [7]}}, {"lambda": {"lambda_V_used": 0.4}, "stat": {"kind": "GC", "windows": [7]}}, {"stat": {"kind": "GC", "windows": [5]}}):
    wsrc(bad)
    try:
        load_lambda_from_run("SRC", "GC", 5, root=d); rej.append(False)
    except ValueError:
        rej.append(True)
check("H02 λ 출처 검증: NaN · 음수 · 종류(IV/창7) · 창 7 · 키 없음 → 전부 거부", all(rej), str(rej))
src = open(os.path.join(ROOT, "train_kdv.py")).read(); ssrc = open(os.path.join(ROOT, "tools", "smoke_cases.py")).read()
check("H02 trainer 는 load_lambda_from_run 으로 검증(CALIBRATION_SOURCE_INVALID gate) · smoke 도 같은 검증을 거친다", "load_lambda_from_run" in src and "CALIBRATION_SOURCE_INVALID" in src and "load_lambda_from_run" in ssrc)

# ---------------- fixture: 합성 run 들 (서버 상태와 독립)
FX = tempfile.mkdtemp(); os.makedirs(os.path.join(FX, "work_dir"), exist_ok=True); os.makedirs(os.path.join(FX, "config", "queues"), exist_ok=True)
H0, C0, R0 = H.ROOT, H.CAMP, dict(H.RUN); H.ROOT = FX; H.CAMP = os.path.join(FX, "work_dir", "_na104_20h"); os.makedirs(H.CAMP, exist_ok=True)
TEACH = "t0sha"; DATA = {"train": "d1", "valid": "d2"}; ARCH = {"width": 104, "depth": [1, 2, 2], "params_m": 2.0989, "class_path": "m", "in_channels": 9}
def mk(run, rows, best, seed=1234, init="i" + str(1234), finished_at="2026-09-13T21:00:00", started_at="2026-09-13T20:00:00", last_step=50000, teacher=TEACH, data=DATA, evaluator="ev1", roi="roi1", nan_steps=()):
    wd = os.path.join(FX, "work_dir", run); os.makedirs(os.path.join(wd, "results"), exist_ok=True); os.makedirs(os.path.join(wd, "meta"), exist_ok=True); os.makedirs(os.path.join(wd, "last"), exist_ok=True)
    with open(os.path.join(wd, "checkpoint_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "epoch", "raw_original.hqnr", "raw_original.fscc", "raw_original.d_lambda", "raw_original.d_s", "rr_ergas"]); w.writeheader()
        for st, ep, h in rows:
            w.writerow({"step": st, "epoch": ep, "raw_original.hqnr": ("nan" if st in nan_steps else h), "raw_original.fscc": 0.9, "raw_original.d_lambda": 0.02, "raw_original.d_s": 0.02, "rr_ergas": 2.0})
    with open(os.path.join(wd, "scene_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "scene", "view", "hqnr", "roi_hash"]); w.writeheader(); w.writerow({"step": rows[0][0], "scene": 0, "view": "raw_original", "hqnr": 0.9, "roi_hash": roi})
    json.dump(dict(step=best[0], hqnr=best[1], fscc=0.9), open(os.path.join(wd, "best_hqnr_meta.json"), "w")); json.dump(dict(step=last_step), open(os.path.join(wd, "last_meta.json"), "w"))
    open(os.path.join(wd, "last", "model.safetensors"), "w").write("x")
    for f in ("reduced_best_hqnr.mat", "full_best_hqnr.mat"):
        open(os.path.join(wd, "results", f), "w").write("x")
    open(os.path.join(wd, "meta", "started_at.txt"), "w").write(started_at)
    if finished_at:
        open(os.path.join(wd, "meta", "finished_at.txt"), "w").write(finished_at)
    json.dump(dict(init_hashes=dict(unet_init_sha256_16=init, seed=seed), teacher=dict(tensors_sha256_16=teacher)), open(os.path.join(wd, "init_and_teacher_hashes.json"), "w"))
    json.dump(data, open(os.path.join(wd, "dataset_hashes.json"), "w")); json.dump(ARCH, open(os.path.join(wd, "architecture_manifest.json"), "w")); json.dump(dict(evaluator_hash=evaluator), open(os.path.join(wd, "selector_state_raw.json"), "w"))
G10 = lambda base, slope=0.0003: [(2020 * i, 10 * i, base + slope * i) for i in range(1, 25)] + [(50000, 248, base + slope * 25)]
G5 = lambda base, slope=0.0003: [(1010 * i, 5 * i, base + slope * i / 2) for i in range(1, 50)] + [(50000, 248, base + slope * 25)]

# ---------------- H03 완료·적격·공통 격자 (리뷰 P2-3)
mk("A", G5(0.94), (50000, 0.9475)); mk("B", G10(0.941), (50000, 0.9485))
cg = H.common_grid_pair("A", "B"); check("H03 common grid = 두 run 모두 유효한 update 교집합(eval 5/10 혼재 → 25) 위에서 각각 재생", cg and cg["n_common"] == 25 and "valid both" in cg["grid_note"], str({k: cg[k] for k in ("n_common", "grid_note")}) if cg else "None")
mk("A2", G10(0.940, 0.0001), (50000, 0.9425), nan_steps=(50000,)); mk("B2", G10(0.940, 0.0001), (50000, 0.9425))
cg2 = H.common_grid_pair("A2", "B2"); check("H03 한쪽만 NaN 인 시점(50000) 은 양쪽 선택 기회에서 제외 → Δ 0 (한쪽만 뺐다면 Δ ≠ 0)", cg2 and cg2["n_common"] == 24 and abs(cg2["delta_common"]) < 1e-12, str({k: cg2[k] for k in ("n_common", "delta_common")}) if cg2 else "None")
mk("C", G10(0.94), (50000, 0.9475), teacher="OTHER"); ok, why = H.comparable("C", "B"); check("H03 Teacher sha 가 다르면 비교 불가 (사유 기록) · common_grid_pair None", not ok and "teacher" in " ".join(why) and H.common_grid_pair("C", "B") is None, str(why))
mk("D", G10(0.94), (50000, 0.9475), init="iOTHER"); check("H03 같은 seed 인데 초기 tensor 가 다르면 비교 불가", not H.comparable("D", "B")[0])
mk("E", G10(0.94), (50000, 0.9475), last_step=40000); ok, why = H.finished("E", True); check("H03 finished(): last step 40000 → 미완 (사유 명시)", not ok and any("50000" in x for x in why), str(why))
mk("F", G10(0.94)[:-1], (40400, 0.947)); ok, why = H.finished("F", True); check("H03 finished(): exact-50K FR 평가 없음 → 미완", not ok and any("50K" in x for x in why))
mk("Gx", G10(0.94), (50000, 0.9475), finished_at=None); check("H03 finished(): finished_at 없음(진행 중) → 미완", not H.finished("Gx"))

# ---------------- H04 3-seed·decide·순차 예산 (리뷰 P1-2, P2-5)
def fixture_core(q36_deltas, q12_deltas, seeds=(1234, 777, 2026)):
    for s_, dq, dr in zip(seeds, q36_deltas, q12_deltas):
        mk(f"Q00S{s_}", G10(0.945), (50000, 0.9525), seed=s_, init=f"i{s_}"); mk(f"Q36S{s_}", G10(0.945 + dq), (50000, 0.9525 + dq), seed=s_, init=f"i{s_}"); mk(f"Q12S{s_}", G10(0.945 + dr), (50000, 0.9525 + dr), seed=s_, init=f"i{s_}")
        H.RUN[("Q00", s_)] = f"Q00S{s_}"; H.RUN[("Q36", s_)] = f"Q36S{s_}"; H.RUN[("Q12", s_)] = f"Q12S{s_}"
    for s_ in (777, 1234, 2026):
        H.RUN[("CF01", s_)] = f"CF01S{s_}"
    for s_ in (777, 2026):
        H.RUN[("X02", s_)] = f"X02S{s_}"
fixture_core((0.002, -0.001, 0.004), (0.001, 0.001, -0.002))
t = H.three_seed("Q36"); check("H04 3-seed: Δ(+0.002, −0.001, +0.004) → 평균 > 0 · 양수 2/3 → pass", t["complete"] and t["passed"] is True and t["n_positive"] == 2)
t2 = H.three_seed("Q12"); check("H04 3-seed: Δ(+0.001, +0.001, −0.002) → 평균 0 → 불통과(False, 미완 아님)", t2["complete"] and t2["passed"] is False)
json.dump(dict(server="s3", start="2026-09-13T19:00:00", cap_hours=20.0, est_run_hours=1.0, post_run_overhead_hours=0.0, overhead_hours=17.0), open(os.path.join(H.CAMP, "ledger.json"), "w"))   # 사용 17h 반례
b = H.budget(); dec = H.decide("s3")
check("H04 예산 반례(사용 17h, run 1h): CF01 pilot 2 → 17+2.2=19.2 승인, X02 는 이미 승인분 포함 21.4 → STOP (중복 승인 없음)", abs(b["used_hours"] - 17.0) < 1e-6 and dec["open"] == ["CF01S777", "CF01S1234"] and not any("X02" in r for r in dec["open"]), str(dec["open"]) + " | " + " / ".join(dec["reasons"][-2:]))
fixture_core((0.002, -0.001, 0.004), (0.002, 0.001, 0.003)); json.dump(dict(server="s3", start="2026-09-13T19:00:00", cap_hours=20.0, est_run_hours=1.0, overhead_hours=0.0), open(os.path.join(H.CAMP, "ledger.json"), "w"))
dec = H.decide("s3"); check("H04 s3: Q36 pass · Q12 pass · 예산 충분 → CF01 pilot 2 + X02 2, 승인 예약 4.4h 누적", dec["open"] == ["CF01S777", "CF01S1234", "X02S777", "X02S2026"] and abs(dec["planned_hours"] - 4.4) < 1e-9)
dec2 = H.decide("s2"); check("H04 s2: 같은 조건이라도 s3 token 없으면 CF01 닫힘 (X02 만)", dec2["open"] == ["X02S777", "X02S2026"])
open(os.path.join(H.CAMP, "cf01_approved_by_s3.txt"), "w").write("ok"); dec3 = H.decide("s2"); check("H04 s2: token + 자기 Q36 통과 → CF01 3벌 (S777 → S1234 → S2026) 이 X02 앞에", dec3["open"][:3] == ["CF01S777", "CF01S1234", "CF01S2026"]); os.remove(os.path.join(H.CAMP, "cf01_approved_by_s3.txt"))
# pilot → S2026 → 최종 3-seed (vs N0 · vs Q36)
for s_, dq in ((777, 0.0015), (1234, 0.0012)):
    mk(f"CF01S{s_}", G10(0.945 + {777: -0.001, 1234: 0.002}[s_] + dq), (50000, 0.9525 + {777: -0.001, 1234: 0.002}[s_] + dq), seed=s_, init=f"i{s_}")
dec4 = H.decide("s3"); check("H04 s3: pilot 둘 다 Q36 대비 양성 → CF01 S2026 열림 + cf01_pilots_positive.json", "CF01S2026" in dec4["open"] and os.path.exists(os.path.join(H.CAMP, "cf01_pilots_positive.json")) and dec4["cf01_final"]["passed"] is None)
mk("CF01S2026", G10(0.945 + 0.004 - 0.003), (50000, 0.9525 + 0.004 - 0.003), seed=2026, init="i2026")
dec5 = H.decide("s3"); cf = dec5["cf01_final"]
check("H04 CF01 최종 3-seed: vs N0 (+0.0005, +0.0032, +0.001) pass · vs Q36 (+0.0015, +0.0012, −0.003) 평균 < 0 → 불통과 → 최종 False (pilot 통과와 분리)", cf["vs_N0"]["passed"] is True and cf["vs_Q36"]["passed"] is False and cf["passed"] is False, str(cf))
# budget: 전환 전 run 잔여·실패 시도·overhead
mk("PRE", G10(0.94), (50000, 0.947), started_at="2026-09-13T18:00:00", finished_at="2026-09-13T19:30:00"); mk("FAILED_TRY", G10(0.94)[:5], (2020, 0.94), started_at="2026-09-13T19:40:00", finished_at="2026-09-13T19:50:00", last_step=0)
json.dump(dict(server="s3", start="2026-09-13T19:00:00", cap_hours=20.0, est_run_hours=1.0, pre_switch_run="PRE", overhead_hours=0.25, post_run_overhead_hours=0.1), open(os.path.join(H.CAMP, "ledger.json"), "w"))
for r_ in list(os.listdir(os.path.join(FX, "work_dir"))):
    if r_ in ("PRE", "FAILED_TRY"):
        os.rename(os.path.join(FX, "work_dir", r_), os.path.join(FX, "work_dir", "NA104_" + r_))
H.RUN[("PRE", 0)] = "NA104_PRE"; json.dump(dict(server="s3", start="2026-09-13T19:00:00", cap_hours=20.0, est_run_hours=1.0, pre_switch_run="NA104_PRE", overhead_hours=0.25, post_run_overhead_hours=0.1), open(os.path.join(H.CAMP, "ledger.json"), "w"))
b2 = H.budget(); check("H04 예산 계상: 전환 전 run 잔여 0.5h + 실패 시도 0.167h + 준비 0.25h (+ whitelist 밖 실패도 포함) = 0.917h", abs(b2["used_hours"] - (0.5 + 10 / 60 + 0.25)) < 2e-3 and b2["items"]["NA104_PRE"]["kind"] == "pre_switch_remainder" and b2["items"]["NA104_FAILED_TRY"]["kind"] == "failed_or_partial", str(b2["used_hours"]))
H.RUN.clear(); H.RUN.update(R0); H.ROOT = H0; H.CAMP = C0

# ---------------- H05 격리·전환 스크립트 (리뷰 P1-1, P2-6)
r0 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "campaign_gate.py")], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PANCRAFTER_CAMPAIGN_GATES": ""})
check("H05 campaign gate 기본 닫힘 (token 없으면 아무 것도 열지 않음)", r0.stdout.strip() == "")
sw = open(os.path.join(ROOT, "tools", "na104_20h_switch.sh")).read()
check("H05 switch: smoke(P1)·unit test 실패면 기동하지 않음(exit 1) · 시계는 switch 시각 · 마감 = 남은 예산 · --reenter 로 DONE 뒤 재진입 · 감시자 재등록",
      all(x in sw for x in ('[ $RC_P1 -eq 0 ] ||', '[ $RC_UT -eq 0 ] ||', 'pre_switch_run', '--hours "$REM"', 'reenter', '_watchdog.sh --install')) and 'set -euo' not in sw)
print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
