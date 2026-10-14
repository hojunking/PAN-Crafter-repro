"""Standalone qualified-steps publisher; dry-run/local-only by default."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from .transport import GoogleSheets, TransportError, atomic_json, code_revision, reconcile


def legacy_publishers(server):
    """Read-only PID/argv proof. Never send signals to any process."""
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = [p.decode("utf-8", "replace") for p in path.read_bytes().split(b"\0") if p]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        is_module = any(argv[i:i + 3] == ["-m", "edgeres48", "publish"]
                        for i in range(max(0, len(argv) - 2)))
        is_server = any(argv[i:i + 2] == ["--server", server]
                        for i in range(max(0, len(argv) - 1)))
        if is_module and is_server:
            found.append({"pid": int(path.parent.name), "argv": argv})
    return found


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", choices=("s1", "s2", "s3"), required=True)
    parser.add_argument("--root", required=True, help="Existing campaign output root (all runs/outbox)")
    parser.add_argument("--state-dir", help="Defaults to ROOT/sheet_publish_v2 (separate from training)")
    parser.add_argument("--credentials", default="gspread/account.json")
    parser.add_argument("--apply", action="store_true", help="Explicitly permit owned sheet view replacement")
    parser.add_argument("--google-read-only", action="store_true", help="Validate live native cells without writing")
    parser.add_argument("--watch", action="store_true", help="Poll; requires --apply or --google-read-only")
    parser.add_argument("--poll-seconds", type=float, default=120)
    parser.add_argument("--audit-seconds", type=float, default=600,
                        help="Re-read an unchanged live view at least this often")
    parser.add_argument("--max-polls", type=int, default=0, help="0 means no poll-count limit")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.watch and not (args.apply or args.google_read_only):
        raise SystemExit("--watch requires explicit --apply or --google-read-only")
    if args.poll_seconds < 10 or args.audit_seconds < args.poll_seconds or args.max_polls < 0:
        raise SystemExit("Require 10 <= poll-seconds <= audit-seconds and max-polls >=0")
    root = Path(args.root).resolve(strict=True)
    state = Path(args.state_dir).resolve() if args.state_dir else root / "sheet_publish_v2"
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / (args.server + ".lock")).open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit("Another projector owns this server/state-directory lock")
    atomic_json(state / "process.json", {"pid": os.getpid(), "argv": sys.argv,
                "server": args.server, "root": str(root), "apply": args.apply,
                "revision": code_revision()})
    adapter = None
    polls, failures = 0, 0
    verified_hash, verified_time = None, 0.0
    while True:
        try:
            from .selection import project_campaign
            project = project_campaign(root, args.server)
            project["legacy_publishers_observed"] = legacy_publishers(args.server)
            atomic_json(state / "projection.json", project)
            if project.get("conflicts"):
                raise TransportError("Projection source conflicts; inspect projection.json")
            if args.apply and project["legacy_publishers_observed"]:
                raise TransportError("Legacy publisher still running; only operator may stop its verified PID")
            if args.apply or args.google_read_only:
                if (args.apply and project["view_hash"] == verified_hash
                        and time.monotonic() - verified_time < args.audit_seconds):
                    report = {"action": "UNCHANGED_LOCAL_HASH", "new_row_count": len(project["rows"]),
                              "view_hash": project["view_hash"], "fresh_readback": False,
                              "last_verified_path": str(state / "last_verified.json")}
                    atomic_json(state / "last_poll.json", report)
                else:
                    if adapter is None:
                        adapter = GoogleSheets(args.credentials)
                    report = reconcile(adapter, project, state, apply=args.apply)
                    if report["readback_verified"]:
                        verified_hash, verified_time = project["view_hash"], time.monotonic()
            else:
                report = {"action": "LOCAL_DRY_RUN", "view_hash": project["view_hash"],
                          "destination": project["destination"], "new_row_count": len(project["rows"]),
                          "source_digest": project.get("source_digest"),
                          "raw_count": project.get("raw_count"),
                          "summary_count": project.get("summary_count"),
                          "runs": project.get("runs"), "selected": project.get("selected"),
                          "legacy_publishers_observed": project["legacy_publishers_observed"],
                          "projector_revision": code_revision()}
                atomic_json(state / "last_dry_run.json", report)
            print(json.dumps({"action": report["action"], "rows": report["new_row_count"],
                              "view_hash": report["view_hash"], "state_dir": str(state)},
                             ensure_ascii=False), flush=True)
            failures = 0
        except Exception as exc:
            failures += 1
            error = {"error": type(exc).__name__ + ": " + str(exc),
                     "consecutive_failures": failures, "at_unix": time.time(),
                     "training_untouched": True}
            atomic_json(state / "last_error.json", error)
            print(json.dumps(error, ensure_ascii=False), file=sys.stderr, flush=True)
            if not args.watch or failures >= 5:
                lock.close()
                return 1
        polls += 1
        if not args.watch or (args.max_polls and polls >= args.max_polls):
            lock.close()
            return 0
        time.sleep(min(60, 5 * 2 ** (failures - 1)) if failures else args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
