#!/usr/bin/env python
"""평가 phase 제어 — 학습을 죽이지 않고 '새 학습만' 멈춘 뒤 원래 큐 그대로 복귀
(계획 research_log/PAN_AllServers_StudentEval_AlignerAnalysis_CurrentMethod_Integrated_2026-09-18.md §2, protocol PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918).

    python tools/eval_phase.py status
    python tools/eval_phase.py hold [--reason ...]      # 복귀 manifest 저장 + hold 켜기 (진행 중 학습은 그대로 끝난다)
    python tools/eval_phase.py release                  # hold 해제 (MAIN_RESUME)
    python tools/eval_phase.py resume                   # hold 해제 + 저장된 큐로 chain/대기자 복원 (필요할 때만)

hold 는 **파일 하나**(work_dir/_eval_phase/hold.json)다. 새 학습을 시작할 수 있는 네 주체가 각자 그 파일을 본다:
  · 돌고 있는 runner  — case 마다 새로 띄우는 `tools/smoke_cases.py` 가 rc 2(일시 사유) 로 빠져 그 case 를 **원장 없이** 건너뛴다.
    (runner 스크립트 자체는 삭제된 inode 로 실행 중이라 고쳐도 이번 chain 에 반영되지 않는다 — 그래서 자식 프로세스에 건다.)
  · `tools/qrecon24_waiter.sh` — 매 loop 새 python 에서 gen_pakd50_configs 를 import 하므로 즉시 순응한다.
  · cron `tools/_watchdog.sh` — 매번 새로 실행되므로 스크립트의 검사가 바로 먹는다.
  · 사람이 부르는 `tools/qrecon24_switch.sh` — hold 중이면 기동을 거부한다.

hold 는 **진행 중 학습을 죽이지 않는다**(계획 §2.3). 지금 도는 case 는 원 정의로 끝나고, 그 다음 case 부터 멈춘다.
복귀 manifest 에 큐 파일 내용과 sha256, revision, 다음 미완 case, mandatory/held/reservation 을 그대로 담아 복귀 시 원위치를 보장한다(§2.2).
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PHASE_DIR = os.path.join("work_dir", "_eval_phase")
HOLD = os.path.join(PHASE_DIR, "hold.json")
RESUME = os.path.join(PHASE_DIR, "resume_manifest.json")
PROTOCOL_ID = "PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918"
PHASES = ("MAIN_RUNNING", "DRAIN_CURRENT_CASE", "LOCAL_STUDENT_EVAL", "LOCAL_EVAL_UPLOAD_CHECK", "S1_ALIGNER_ANALYSIS", "S1_ANALYSIS_COMPLETE", "MAIN_RESUME")
QUEUE_FILES = ("work_dir/cases_queue.txt", "work_dir/cases_queue_handover.txt", "work_dir/_qrecon24/queue_effective.txt", "work_dir/_qrecon24/queue_active.txt",
               "work_dir/_qrecon24/mandatory_runs.txt", "work_dir/_pakd50/mandatory_runs.txt", "work_dir/_pakd50/extra_priority.txt", "work_dir/cases_deadline.txt")
JSON_FILES = ("work_dir/_qrecon24/queue_revision.json", "work_dir/_qrecon24/held_runs.json", "work_dir/_pakd50/reservations.json", "work_dir/_qrecon24/status.json")


def _sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else str(b).encode()).hexdigest()


def _read(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    b = open(p, "rb").read()
    return dict(path=rel, sha256=_sha(b), bytes=len(b), text=b.decode("utf-8", "replace"))


def hold_state():
    """{} 또는 hold 내용. 다른 도구(gen_pakd50_configs·smoke_cases·watchdog)가 이 함수를 쓴다."""
    p = os.path.join(ROOT, HOLD)
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p))
    except Exception:
        return dict(phase="LOCAL_STUDENT_EVAL", reason="hold.json 을 읽지 못했다 — 안전을 위해 hold 로 본다")


def is_held():
    return bool(hold_state())


def chain_alive():
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any("tools/_run_cases.sh" in l for l in ps.splitlines()), [l for l in ps.splitlines() if "main.py" in l and "--config" in l]


def next_unfinished():
    """저장된 활성 큐에서 아직 완료·실패가 아닌 첫 항목 (복귀 지점)."""
    from tools.campaign_gate import terminal
    for rel in ("work_dir/_qrecon24/queue_effective.txt", "work_dir/cases_queue.txt"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        items = [l.strip() for l in open(p) if l.strip() and not l.startswith("#")]
        pend = [r for r in items if not terminal(r)]
        if items:
            return dict(queue=rel, n_items=len(items), n_pending=len(pend), next_case=(pend[0] if pend else None), pending=pend)
    return dict(queue=None, n_items=0, n_pending=0, next_case=None, pending=[])


def capture(reason, phase="LOCAL_STUDENT_EVAL"):
    from tools import gen_pakd50_configs as G
    srv = G.server_id(open(os.path.join(ROOT, "gspread", "server.txt")).read())
    alive, training = chain_alive()
    man = dict(protocol_id=PROTOCOL_ID, server=srv, captured_at=time.strftime("%Y-%m-%dT%H:%M:%S"), phase=phase, reason=reason,
               source_train_server=srv, chain_alive=alive, training_now=[t[:160] for t in training],
               approved_main_queue_revision=(json.load(open(os.path.join(ROOT, "work_dir", "_qrecon24", "queue_revision.json"))).get("queue_revision")
                                             if os.path.exists(os.path.join(ROOT, "work_dir", "_qrecon24", "queue_revision.json")) else None),
               generator_revision=G.QRC24_ADJ_REVISION, queue_files={rel: _read(rel) for rel in QUEUE_FILES}, state_files={rel: _read(rel) for rel in JSON_FILES},
               resume=next_unfinished(), watchdog_cron=subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout.count("PANCRAFTER-WATCHDOG"),
               main_worktree=subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               main_recipe_lock_id=None, restore_main_forward_mode="A_ON", preempt_running_training=False)
    os.makedirs(os.path.join(ROOT, PHASE_DIR), exist_ok=True)
    json.dump(man, open(os.path.join(ROOT, RESUME), "w"), indent=1, ensure_ascii=False)
    return man


def cmd_hold(a):
    if is_held():
        print("[phase] 이미 hold 중:", json.dumps(hold_state(), ensure_ascii=False)[:200]); return 0
    man = capture(a.reason, phase=a.phase)
    h = dict(protocol_id=PROTOCOL_ID, server=man["server"], phase=a.phase, reason=a.reason, created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
             allow_new_main_training=False, preempt_running_training=False, resume_manifest=RESUME,
             note="진행 중 학습은 원 정의로 끝난다. 새 case 는 smoke_cases rc2(원장 기록 없음) 로 건너뛴다.")
    tmp = os.path.join(ROOT, HOLD + ".tmp"); json.dump(h, open(tmp, "w"), indent=1, ensure_ascii=False); os.replace(tmp, os.path.join(ROOT, HOLD))
    print(f"[phase] hold 켜짐 ({a.phase}) — 복귀 manifest {RESUME}")
    print(f"   chain 살아 있음 {man['chain_alive']} · 학습 중 {len(man['training_now'])} 건(그대로 끝난다) · 다음 미완 case {man['resume']['next_case']}")
    print(f"   저장한 큐: " + ", ".join(k for k, v in man["queue_files"].items() if v))
    return 0


def cmd_release(a):
    if not is_held():
        print("[phase] hold 아님"); return 0
    h = hold_state(); p = os.path.join(ROOT, HOLD)
    os.replace(p, os.path.join(ROOT, PHASE_DIR, "hold.released.%s.json" % time.strftime("%m%d-%H%M%S")))
    print(f"[phase] hold 해제 (phase 였음: {h.get('phase')}) — 이제 runner/대기자/watchdog 가 다음 case 를 시작할 수 있다")
    man = json.load(open(os.path.join(ROOT, RESUME))) if os.path.exists(os.path.join(ROOT, RESUME)) else {}
    nxt = (man.get("resume") or {}).get("next_case")
    print(f"   복귀 대상: {nxt or '없음(큐 완료)'} · 저장 revision {man.get('approved_main_queue_revision')}")
    return 0


def cmd_resume(a):
    """hold 해제 + 필요하면 chain 재기동. chain 이 살아 있으면 건드리지 않는다(다음 case 부터 자동 진행)."""
    cmd_release(a)
    alive, training = chain_alive()
    if alive:
        print("[phase] chain 이 살아 있다 — 재기동하지 않는다. 다음 case 부터 원래 큐로 진행한다."); return 0
    man = json.load(open(os.path.join(ROOT, RESUME))) if os.path.exists(os.path.join(ROOT, RESUME)) else {}
    nxt = (man.get("resume") or {}).get("next_case")
    if not nxt:
        print("[phase] 저장된 큐에 미완 case 가 없다 — 재기동하지 않는다 (필요하면 tools/qrecon24_switch.sh)"); return 0
    if a.dry_run:
        print(f"[phase] --dry-run: 재기동하지 않았다. 사람이 실행할 명령: ./tools/qrecon24_switch.sh  (다음 case {nxt})"); return 0
    print(f"[phase] chain 없음 · 미완 {nxt} — 기존 전환 스크립트로 복원한다")
    r = subprocess.run(["./tools/qrecon24_switch.sh"], cwd=ROOT, text=True)
    return r.returncode


def cmd_status(a):
    h = hold_state(); alive, training = chain_alive(); nxt = next_unfinished()
    print(f"[phase] hold: {'켜짐 — ' + h.get('phase', '?') + ' (' + str(h.get('reason'))[:60] + ')' if h else '꺼짐 (MAIN_RUNNING)'}")
    print(f"   chain {'살아 있음' if alive else '없음'} · 학습 중 {len(training)} 건" + (f" [{os.path.basename(training[0].split('--config')[1].strip())[:60]}]" if training else ""))
    print(f"   활성 큐 {nxt['queue']} · 항목 {nxt['n_items']} · 미완 {nxt['n_pending']} · 다음 {nxt['next_case']}")
    coh = os.path.join(ROOT, PHASE_DIR, "cohort.json")
    if os.path.exists(coh):
        c = json.load(open(coh)); done = len(glob.glob(os.path.join(ROOT, PHASE_DIR, "records", "*.json")))
        print(f"   cohort {c['id']} · 대상 {len(c['records'])} · 평가 레코드 {done}")
    if os.path.exists(os.path.join(ROOT, RESUME)):
        m = json.load(open(os.path.join(ROOT, RESUME)))
        print(f"   복귀 manifest {m['captured_at']} · revision {m.get('approved_main_queue_revision')} · 저장 큐 {sum(1 for v in m['queue_files'].values() if v)} 개")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.set_defaults(fn=cmd_status)
    h = sub.add_parser("hold"); h.add_argument("--reason", default="NOA/A_ON 전수 Student 평가 (PAN_ALLSERVER_NOA_AUDIT_METHOD_v2_20260918 §2)"); h.add_argument("--phase", default="LOCAL_STUDENT_EVAL", choices=PHASES); h.set_defaults(fn=cmd_hold)
    r = sub.add_parser("release"); r.set_defaults(fn=cmd_release)
    q = sub.add_parser("resume"); q.add_argument("--dry-run", action="store_true"); q.set_defaults(fn=cmd_resume)
    a = ap.parse_args()
    if a.cmd == "release":
        a.dry_run = False
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
