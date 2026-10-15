#!/usr/bin/env python
"""QRECON24 run 뒤처리 (case 경계; tools/_upload.sh 가 부른다 — ADJ-R1 계획 §7.2·§8.1·§11):
  1) 이 run 의 **공식 RR selector** (tools/qrecon24_select.py --official --threshold 0.9585 --device cuda; 같은 checkpoint 의 raw HQNR ≥ 하한 후보만 재추론) → results/qrecon24_target_selection.json
  2) --backlog: 이 서버의 완료 QRC24 run 중 공식 선택이 안 끝난 것(선택 파일 없음 · official false · incomplete_official_rr) 도 같은 자리에서 — 예: s1 G23 S1234, s4 H31 S1234 (§7.2)
  3) 버전 감사(§8.1; tools/qrecon24_version_audit.py) → results/qrecon24_version_audit.json
  4) 걸린 GPU 시간은 QRC24 ledger 에 select_<run>(kind diag) 으로 따로 — Train(h)·run 당 10 분 후처리 가정에 섞지 않는다 (§7.2·§10)

    python tools/qrecon24_postrun.py <run> [--backlog] [--device cuda] [--threshold 0.9585] [--force]
    python tools/qrecon24_postrun.py --backlog            # run 없이 backlog 만 (switch 가 GPU 가 빌 때 부른다)

학습(main.py --config) 이 돌고 있으면 GPU 를 겹치지 않으려고 **건너뛴다**(--force 로 강제) — 다음 case 경계에서 다시 시도된다.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import gen_pakd50_configs as G  # noqa: E402
from tools.qrecon24_version_audit import audit_run, completed_qrc24_runs  # noqa: E402

THRESHOLD = 0.9585
DONE_STATUS = ("official", "no_eligible")          # 공식 선택이 끝난 상태 (적격 후보가 없으면 재추론할 것이 없다)


BUSY_MARKERS = ("main.py --config", "eval_fr_paperset.py", "smoke_cases.py", "qrecon24_select.py")     # 학습 · FR 평가 · smoke · 다른 selector — GPU 를 겹치지 않는다


def training_running():
    """학습(main.py --config) 또는 다른 GPU 작업이 돌고 있으면 True (자기 자신의 명령줄은 pkill/pgrep 자기매칭 함정 때문에 pid 로 제외)."""
    me = os.getpid(); ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for l in ps.splitlines()[1:]:
        pid, _, args = l.strip().partition(" ")
        if pid.isdigit() and int(pid) != me and any(m in args for m in BUSY_MARKERS):
            return True
    return False


def selection_state(run):
    p = os.path.join(ROOT, "work_dir", run, "results", "qrecon24_target_selection.json")
    if not os.path.exists(p):
        return "missing", {}
    try:
        d = json.load(open(p))
    except Exception:
        return "unreadable", {}
    st = d.get("target_status"); return (("done" if (d.get("official") and st in DONE_STATUS) or st == "no_eligible" else "pending"), d)


def needs_official(run):
    st, _ = selection_state(run); return st != "done"


def run_selector(run, device, threshold):
    cmd = [sys.executable, os.path.join(ROOT, "tools", "qrecon24_select.py"), run, "--official", "--threshold", str(threshold), "--device", device]
    t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT); sec = time.time() - t0
    tail = "\n".join([l for l in (r.stdout + r.stderr).splitlines() if "Warning" not in l][-4:])
    return r.returncode, sec, tail


def ledger_add(run, sec, note):
    lp = os.path.join(ROOT, G.QRC24_LEDGER)
    if not os.path.exists(lp):
        return
    with open(lp + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.load(open(lp)); ent = d.setdefault("entries", {}); prev = ent.get(f"select_{run}", {})
        ent[f"select_{run}"] = dict(kind="diag", hours=float(prev.get("hours") or 0.0) + sec / 3600.0, runs=int(prev.get("runs", 0)) + 1, note=note, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
        tmp = lp + ".tmp"; json.dump(d, open(tmp, "w"), indent=1); os.replace(tmp, lp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?"); ap.add_argument("--backlog", action="store_true"); ap.add_argument("--device", default="cuda"); ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--force", action="store_true", help="학습이 돌고 있어도 GPU 재평가를 강행 (기본: 건너뜀)"); ap.add_argument("--dry-run", action="store_true", help="대상만 나열")
    a = ap.parse_args()
    targets = []
    if a.run:
        if needs_official(a.run) or a.force:
            targets.append(a.run)
        else:
            print(f"[qrecon24-postrun] {a.run}: 공식 선택이 이미 끝났다({selection_state(a.run)[1].get('target_status')}) — 건너뜀 (runner 의 '완료됨 — 업로드만 확인' 재방문; --force 로 다시)")
    if a.backlog:
        targets += [r for r in completed_qrc24_runs() if r not in targets and needs_official(r)]
    if not targets:
        print("[qrecon24-postrun] 대상 없음 (공식 선택이 전부 끝났다)"); return 0
    print(f"[qrecon24-postrun] 대상 {len(targets)}: " + ", ".join(t[len('PAKD50_QRC24_'):] for t in targets))
    if a.dry_run:
        return 0
    if training_running() and not a.force:
        print("[qrecon24-postrun] 학습(main.py)/FR 평가/다른 selector 가 돌고 있다 — GPU 를 겹치지 않으려고 건너뛴다(다음 case 경계에서 재시도; --force 로 강제)"); return 0
    os.makedirs(os.path.join(ROOT, "work_dir", "_qrecon24"), exist_ok=True); lk = open(os.path.join(ROOT, "work_dir", "_qrecon24", ".postrun.lock"), "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)                          # switch ⑦ 과 _upload.sh 가 같은 경계에서 겹치면 하나만 (sidecar/선택 파일 동시 기록 방지)
    except OSError:
        print("[qrecon24-postrun] 다른 postrun 이 진행 중 — 건너뛴다(그쪽이 backlog 를 처리한다)"); return 0
    rc_all = 0
    for run in targets:
        rd = os.path.join(ROOT, "work_dir", run)
        if not (os.path.exists(os.path.join(rd, "results", "reduced_best_hqnr.mat")) and os.path.exists(os.path.join(rd, "checkpoint_metrics.csv"))):
            print(f"  {run}: 완료 산출물(results mat / checkpoint_metrics.csv) 없음 — 건너뜀"); continue
        rc, sec, tail = run_selector(run, a.device, a.threshold); st, d = selection_state(run)
        print(f"  {run}: selector rc {rc} · {sec:.0f}s · status {d.get('target_status')} official {d.get('official')} eligible {d.get('n_eligible')} target {(d.get('target') or {}).get('step')}" + (f"\n    {tail}" if rc else ""))
        if rc == 0:
            ledger_add(run, sec, f"qrecon24_select --official (threshold {a.threshold}, {a.device}); status {d.get('target_status')}")
        else:
            rc_all = 1
        v = audit_run(run, write=True); print(f"    version audit: {v['verdict']} (factor {v['evidence'].get('numerator_factor')}, uniform {v['evidence'].get('uniform_weight_manifest')}, λE {v['evidence'].get('lambda_E_spec')}, release {(v['evidence'].get('release_sha') or '?')[:12]})")
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
