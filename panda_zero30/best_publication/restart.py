"""Replace only this server/campaign's Sheet watcher; never signal a trainer."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import POLICY, R, legacy, project

MODULE = "panda_zero30.best_publication"
WRITERS = ("panda_zero30.publication", MODULE)


def matches_writer(argv, root, server):
    def option(name):
        return argv[argv.index(name) + 1] if argv.count(name) == 1 and argv.index(name) + 1 < len(argv) else None
    return (option("-m") in WRITERS and option("--server") == server
            and option("--root") == str(root) and "--apply" in argv and "--watch" in argv)


def stop_writer(pid, argv, root, server):
    process = Path(f"/proc/{pid}")
    # Some existing Python 3.9 builds do not expose pidfd_open. Retain the
    # original process generation and recheck it immediately before SIGTERM.
    generation = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
    supported = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
    fd = os.pidfd_open(pid) if supported else None
    try:
        current = (process / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        again = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
        if current != argv or generation != again or not matches_writer(current, root, server):
            raise ValueError("Writer identity changed; no signal sent")
        if supported:
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    finally:
        if fd is not None:
            os.close(fd)


def find_writers(root, server):
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            argv = (path / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        except (OSError, UnicodeError):
            continue
        if matches_writer(argv, root, server):
            found.append((int(path.name), argv))
    return found


def restart(server, *, replace_running=False):
    from sheet_publish_v2.transport import GoogleSheets
    from .transport import prepare
    repo = legacy.ROOT
    if (repo / "gspread/server.txt").read_text().strip() != server:
        raise ValueError("Local server marker mismatch; no process changed")
    root = (repo / "work_dir/panda_zero30" / R.CAMPAIGN_ID / server / "WV3").resolve(strict=True)
    # Validate local evidence + authentication before replacing any watcher.
    value = project(root, server)
    if value["conflicts"]:
        raise ValueError("Projection conflicts; existing watcher untouched")
    adapter = GoogleSheets(str(repo / "gspread/account.json"))
    # Read-only native schema/ownership preflight. A bad header/credential must
    # not stop the existing writer. Also verifies the expanded-column fix.
    prepare(adapter.snapshot(value["destination"]), value, server)
    state = root / "sheet_publish_panda_zero30"
    state.mkdir(exist_ok=True)
    with (state / "publisher_switch.lock").open("a+") as switch:
        fcntl.flock(switch, fcntl.LOCK_EX | fcntl.LOCK_NB)
        found = find_writers(root, server)
        if len(found) > 1:
            raise ValueError("Multiple matching writers; inspect before replacing")
        if found and MODULE in found[0][1] and not replace_running:
            print(R.canonical_json({"action": "BEST_WRITER_ALREADY_RUNNING", "pid": found[0][0]}))
            return 0
        for pid, argv in found:
            stop_writer(pid, argv, root, server)
        deadline = time.monotonic() + 20
        with (state / ("writer_" + server + ".lock")).open("a+") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise ValueError("Writer lock still held; nothing else started")
                    time.sleep(.2)
        log = repo / "work_dir/panda_zero30" / f"publisher-{server}-best-hqnr-v1.log"
        args = [sys.executable, "-u", "-m", MODULE, "--root", str(root), "--server", server,
                "--apply", "--watch", "--poll-seconds", "60"]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        with log.open("ab") as stream:
            child = subprocess.Popen(args, cwd=repo, env=env, start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        R.atomic_json(state / "publisher_switch.json", {
            "at": R.utc_now(), "policy": POLICY, "pid": child.pid,
            "previous_pids": [p for p, _ in found], "log": str(log),
            "training_untouched": True, "replace_running": replace_running})
        print(R.canonical_json({"action": "BEST_WRITER_STARTED", "pid": child.pid, "log": str(log)}))
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, choices=("s1", "s3"))
    parser.add_argument("--restart", action="store_true", dest="replace_running",
                        help="Reload an already-running matching Sheet watcher only")
    args = parser.parse_args()
    raise SystemExit(restart(args.server, replace_running=args.replace_running))

