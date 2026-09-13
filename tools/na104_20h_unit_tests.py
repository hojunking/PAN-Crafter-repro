#!/usr/bin/env python
"""NA104 20H 우선순위 gate (research_log/01_S2_20H_PRIORITY.md · 02_S3_20H_PRIORITY.md). 하나라도 실패하면 exit 1.   python tools/na104_20h_unit_tests.py"""
import csv, json, os, subprocess, sys, tempfile
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from kdv.registry import resolve
from tools import na104_20h as H

FAIL = []
def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {info}")
    if not cond: FAIL.append(name)

# H01 config 존재 (P1 4 + CF01 3 + X02 2) — 이름 규칙 v1/v2
need = [H.RUN[c] for c in H.P1] + [H.RUN[("CF01", s)] for s in (777, 1234, 2026)] + [H.RUN[("X02", s)] for s in (777, 2026)]
missing = [r for r in need if not os.path.exists(os.path.join(ROOT, "config", r + ".yaml"))]
check("H01 P1 4 + CF01 3 + X02 2 config 존재 (계획 run ID 그대로)", not missing, f"missing {missing}")
check("H01 Q36 S1234 v1 · Q00 S1234 v1 · Teacher T00 · pilot 은 기존 파일 (재생성 안 함)", all(os.path.exists(os.path.join(ROOT, "config", r + ".yaml")) for r in (H.RUN[("Q36", 1234)], H.RUN[("Q00", 1234)], H.TEACHER)))
for srv in ("s2", "s3"):
    q = os.path.join(ROOT, "config", "queues", f"na104_20h_{srv}.txt"); qs = [l.strip() for l in open(q) if l.strip() and not l.startswith("#")] if os.path.exists(q) else []
    check(f"H01 큐 {srv}: P1 순서 Q36-777 → Q00-2026 → Q36-2026 → Q12-2026", qs == [H.RUN[c] for c in H.P1])
# H02 CF01 spec: N0 · GC-FIX · kd 0.1 · λ_C from Q36 · Teacher 학습 필요(eval_only 아님) · window 5 · 새 gate 없음
k = yaml.safe_load(open(os.path.join(ROOT, "config", H.RUN[("CF01", 777)] + ".yaml")))["kdv"]; sp = resolve(k)
check("H02 CF01 = rec N0 · stat GC/FIX window 5 · kd_weight 0.1 · lambda_from_run = Q36 S1234 v1 · Teacher 필요(eval_only 없음) · TRI 없음 · λ scale 1",
      sp["rec_case"] == "N0" and sp["stat_key"] == "GC" and sp["stat_mode"] == "FIX" and k["stat"]["window"] == 5 and abs(k["stat"]["kd_weight"] - 0.1) < 1e-12 and sp["stat_lambda_from_run"] == H.RUN[("Q36", 1234)]
      and sp["needs_teacher"] and not sp["teacher_eval_only"] and not k.get("tri") and sp["stat_lambda_scale"] == 1.0)
q36 = yaml.safe_load(open(os.path.join(ROOT, "config", H.RUN[("Q36", 1234)] + ".yaml")))["kdv"]
check("H02 CF01 은 Q36 과 rec/window/eps/calibration 이 같고 stat.mode(H→FIX)·lambda_from_run·teacher.eval_only 만 다르다",
      k["rec"] == q36["rec"] and k["stat"]["window"] == q36["stat"]["window"] and k["calibration"] == q36["calibration"] and q36["stat"]["mode"] == "H" and q36["teacher"].get("eval_only") is True and "eval_only" not in k["teacher"])
try:
    resolve(dict(k, stat=dict(k["stat"], outer_weight=0.5))); check("H02 lambda_from_run + 숫자 outer_weight 는 거부", False)
except Exception:
    check("H02 lambda_from_run + 숫자 outer_weight 는 거부", True)
src = open(os.path.join(ROOT, "train_kdv.py")).read()
check("H02 trainer: lambda_from_run → calibration_resolved.json 의 lambda.lambda_V_used, 없으면 CALIBRATION_SOURCE_MISSING gate", "CALIBRATION_SOURCE_MISSING" in src and 'source="from_run"' in src)
# H03 common-grid 재생·3-seed 기준 (합성)
d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "work_dir"), exist_ok=True)
def mk(run, rows, best):
    os.makedirs(os.path.join(d, "work_dir", run, "results"), exist_ok=True); os.makedirs(os.path.join(d, "work_dir", run, "meta"), exist_ok=True)
    with open(os.path.join(d, "work_dir", run, "checkpoint_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "epoch", "raw_original.hqnr", "raw_original.fscc", "raw_original.d_lambda", "raw_original.d_s", "rr_ergas"]); w.writeheader()
        for st, ep, h in rows:
            w.writerow({"step": st, "epoch": ep, "raw_original.hqnr": h, "raw_original.fscc": 0.9, "raw_original.d_lambda": 0.02, "raw_original.d_s": 0.02, "rr_ergas": 2.0})
    json.dump(dict(step=best[0], hqnr=best[1], fscc=0.9), open(os.path.join(d, "work_dir", run, "best_hqnr_meta.json"), "w")); json.dump(dict(step=50000), open(os.path.join(d, "work_dir", run, "last_meta.json"), "w"))
    open(os.path.join(d, "work_dir", run, "results", "reduced_best_hqnr.mat"), "w").write("x"); open(os.path.join(d, "work_dir", run, "meta", "started_at.txt"), "w").write("2026-09-13T20:00:00"); open(os.path.join(d, "work_dir", run, "meta", "finished_at.txt"), "w").write("2026-09-13T21:00:00")
H0 = H.ROOT; H.ROOT = d
A = [(1010 * i, 5 * i, 0.94 + 0.0002 * i) for i in range(1, 11)] + [(50000, 248, 0.9419)]          # eval 5 격자 (11 시점)
B = [(2020 * i, 10 * i, 0.941 + 0.0003 * i) for i in range(1, 6)] + [(50000, 248, 0.9420)]          # eval 10 격자 (6 시점) → 공통 = 짝수 epoch 5개 + 50000
mk("A", A, (10100, 0.942)); mk("B", B, (10100, 0.9425))
cg = H.common_grid_pair("A", "B")
check("H03 common grid = 두 run 의 평가 update 교집합(6) 위에서 각각 BestSelector 재생 (A 의 eval-5 전용 시점은 제외)", cg and cg["n_common"] == 6 and cg["a_best"]["step"] in {2020, 4040, 6060, 8080, 10100, 50000}, str({k: cg[k] for k in ("n_common", "a_best", "b_best", "delta_common")}) if cg else "None")
# 3-seed: Q36 vs Q00 — RUN 표를 임시로 합성 run 으로 바꾼다
R0 = dict(H.RUN)
for s_, (hq, hn) in zip(H.SEEDS, ((0.950, 0.948), (0.951, 0.952), (0.953, 0.949))):
    mk(f"Q36S{s_}", [(2020 * i, 10 * i, hq - 0.001 * (5 - i)) for i in range(1, 6)] + [(50000, 248, hq)], (50000, hq)); mk(f"Q00S{s_}", [(2020 * i, 10 * i, hn - 0.001 * (5 - i)) for i in range(1, 6)] + [(50000, 248, hn)], (50000, hn))
    H.RUN[("Q36", s_)] = f"Q36S{s_}"; H.RUN[("Q00", s_)] = f"Q00S{s_}"
t = H.three_seed("Q36")
check("H03 3-seed 기준: Δ = (+0.002, −0.001, +0.004) → 평균 > 0 · 양수 2/3 → pass", t["complete"] and t["passed"] is True and t["n_positive"] == 2 and abs(t["mean_delta"] - 0.0016667) < 1e-6)
H.RUN[("Q36", 2026)] = "NOPE"; t2 = H.three_seed("Q36"); check("H03 seed 하나 미완이면 passed=None (판정 보류, 미실험을 실패로 쓰지 않음)", t2["passed"] is None and t2["n_ok"] == 2)
H.RUN.clear(); H.RUN.update(R0); H.ROOT = H0
# H04 예산 guard·decide 닫힘 (이 서버엔 run 이 없다)
dec = H.decide("s3"); check("H04 s3 decide: Q36 미완 → CF01 닫힘, Q12 미완 → X02 닫힘, open 없음", dec["open"] == [] and dec["q36"]["passed"] is None and dec["q12"]["passed"] is None)
dec2 = H.decide("s2"); check("H04 s2 decide: 닫힘 + s3 token 규칙 문구", dec2["open"] == [] and any("token" in r or "닫힘" in r for r in dec2["reasons"]))
b = H.budget(); check("H04 budget: 시작 전 used 0 · cap 20", b["cap_hours"] == 20.0 and b["used_hours"] == 0.0)
# H05 campaign gate 격리: 기본 닫힘, na104_20h 는 s1 에서 닫힘
r0 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "campaign_gate.py")], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PANCRAFTER_CAMPAIGN_GATES": ""})
r1 = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "campaign_gate.py")], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PANCRAFTER_CAMPAIGN_GATES": "na104_20h"})
check("H05 campaign gate 기본 닫힘 · na104_20h 는 s2/s3 전용 (s1 닫힘, stdout 비어 있음)", r0.stdout.strip() == "" and r1.returncode == 0 and r1.stdout.strip() == "" and "NA104-20H" in r1.stderr)
check("H05 switch 스크립트: 감시자 해제 → runner 만 종료 → 완료 대기·업로드 → 시계 → smoke(P1+CF01) → token → 큐 기동 → 감시자 재등록", all(x in open(os.path.join(ROOT, "tools", "na104_20h_switch.sh")).read() for x in ("PANCRAFTER-WATCHDOG", "_run_cases", "smoke_cases.py", "campaign_gates_enabled.txt", "campaign_start.sh", "_watchdog.sh --install")))
print(f"\n{'FAIL ' + str(FAIL) if FAIL else 'ALL OK'} ({len(FAIL)} failed)"); sys.exit(1 if FAIL else 0)
