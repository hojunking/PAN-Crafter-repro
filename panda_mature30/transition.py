"""Finish one exact old case, request graceful stop, then launch reviewed queue.

No SIGKILL, no container stop/removal, no signal on incomplete/failed cases.
Run outside the sandbox PID namespace. The readiness file is an explicit
operator-owned marker written only after new code validation.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process_identity(pid):
    path = Path("/proc") / str(pid)
    try:
        stat = (path / "stat").read_text()
        start = stat.rsplit(")", 1)[1].split()[19]
        cmd = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        return {"pid": pid, "start_ticks": start, "cmdline": cmd}
    except FileNotFoundError:
        return None


def completed(directory, run_id):
    path = Path(directory)
    state_file = path / "run_state.json"
    if not state_file.exists():
        return False
    state = json.loads(state_file.read_text())
    if state.get("run_id") != run_id:
        raise ValueError("Transition target run identity changed")
    if state.get("status") != "COMPLETED" or state.get("step") != 50000:
        return False
    model = path / "models" / "step_0050000.pt"
    evaluation = path / "evaluation_0050000.json"
    sidecar = model.with_suffix(".json")
    if not all(p.is_file() and p.stat().st_size for p in (model, evaluation, sidecar)):
        raise ValueError("Completion without retained final checkpoint/evaluation")
    row = json.loads(sidecar.read_text())
    if (row["artifact"]["completed_update"] != 50000
            or row["config"]["run_id"] != run_id
            or row["evaluation"] != json.loads(evaluation.read_text())):
        raise ValueError("Final checkpoint sidecar/evaluation mismatch")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    initial = process_identity(args.pid)
    if initial is None or initial["cmdline"] != "python -m panda_band30 run --server s1":
        raise ValueError("Expected exact existing s1 trainer process; refusing to arm")
    print(json.dumps({"status": "ARMED_FINISH_CURRENT_CASE", "process": initial,
                      "run_id": args.run_id}), flush=True)
    while not completed(args.run_directory, args.run_id):
        if process_identity(args.pid) != initial:
            raise RuntimeError("Old trainer exited/replaced before case completion; no new launch")
        time.sleep(.2)
    if process_identity(args.pid) == initial:
        os.kill(args.pid, signal.SIGTERM)
        print('{"status":"CURRENT_CASE_COMPLETED_GRACEFUL_STOP_SENT"}', flush=True)
    while process_identity(args.pid) == initial:
        time.sleep(1)
    print('{"status":"OLD_TRAINER_EXITED_WAITING_VALIDATED_READY_MARKER"}', flush=True)
    while not args.ready.is_file():
        time.sleep(5)
    ready = json.loads(args.ready.read_text())
    if ready.get("server") != "s1" or ready.get("validated") is not True:
        raise ValueError("Reviewed s1 readiness receipt required")
    # Pin precisely the reviewed code files before the next process starts.
    import hashlib
    if not ready.get("code_sha256"):
        raise ValueError("Readiness receipt lacks reviewed source hashes")
    for relative, expected in ready["code_sha256"].items():
        path = (args.repository / relative).resolve()
        if not path.is_relative_to(args.repository.resolve()):
            raise ValueError("Readiness code path outside repository")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reviewed code changed: {relative}")
    subprocess.run(["bash", "scripts/run_panda_mature30_docker.sh", "s1"],
                   cwd=args.repository, check=True)
    print('{"status":"NEW_CAMPAIGN_LAUNCH_COMMAND_SUCCEEDED"}', flush=True)


if __name__ == "__main__":
    main()
