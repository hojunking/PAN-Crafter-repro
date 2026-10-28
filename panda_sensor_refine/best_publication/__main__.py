"""Stage1 RRE + all joint passes and a representative for every evaluated run."""
import argparse
import fcntl
import math
from pathlib import Path
import time

from . import POLICY, R, legacy, project
from .transport import reconcile_rows


def publish(root, server, *, credentials="gspread/account.json", apply=False,
            watch=False, poll_seconds=60):
    from sheet_publish_v2.transport import GoogleSheets
    root = Path(root).resolve(strict=True)
    state = root / "sheet_publish_panda_sensor_refine"
    state.mkdir(parents=True, exist_ok=True)
    adapter = GoogleSheets(credentials) if apply else None
    banner_verified = False
    # One writer for this campaign/server. Other campaigns keep their own locks
    # and disjoint rows; Google allocates append ranges atomically.
    with (state / ("writer_" + server + ".lock")).open("a+") as lock:
        if apply:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        terminal_seen = None
        while True:
            try:
                terminal_before = legacy._terminal_state(root)
                value = project(root, server, require_cost=apply)
                R.atomic_json(state / "projection.json", value)
                if value["conflicts"]:
                    raise ValueError("Evidence/cost conflicts; see projection.json")
                if apply and not banner_verified:
                    from .header import reconcile_header
                    reconcile_header(adapter, value["destination"], state, server, apply=True)
                    banner_verified = True
                report = (reconcile_rows(adapter, value, state, server, apply=True) if apply
                          else {"action": "LOCAL_DRY_RUN", "policy": POLICY,
                                "rows": len(value["rows"]), "readback_verified": False})
                R.atomic_json(state / "last_poll.json", report)
                print(R.canonical_json(report), flush=True)
                if (watch and apply and terminal_before is not None
                        and legacy.terminal_upload_complete(root, value, report, terminal_before)):
                    marker = (R.canonical_json(terminal_before), value["source_digest"])
                    if marker == terminal_seen:
                        R.atomic_json(state / "terminal_upload.json", {
                            "campaign_status": terminal_before["status"], "policy": POLICY,
                            "source_digest": value["source_digest"],
                            "raw_count": value["raw_count"], "row_count": len(value["rows"]),
                            "readback_verified": True, "at": R.utc_now()})
                        return 0
                    terminal_seen = marker
                else:
                    terminal_seen = None
            except Exception as exc:
                terminal_seen = None
                R.atomic_json(state / "last_error.json", {
                    "error": repr(exc), "at": R.utc_now(), "training_untouched": True})
                print("[SensorRefine qualified-ERGAS publisher] " + repr(exc), flush=True)
                if not watch:
                    return 1
            if not watch:
                return 0
            time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--server", required=True, choices=("s1", "s3"))
    parser.add_argument("--credentials", default="gspread/account.json")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    return publish(**vars(args))


if __name__ == "__main__":
    raise SystemExit(main())
