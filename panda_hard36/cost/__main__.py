"""Prepare all local native cost sidecars before starting/resuming training."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time


def durable_elapsed(state, now):
    """Use the larger durable or wall-clock duration, including after rollback."""
    started = state.get("started_unix")
    elapsed = state.get("elapsed_seconds", 0.)
    if (type(started) not in (int, float) or not math.isfinite(started)
            or type(now) not in (int, float) or not math.isfinite(now)
            or started > now):
        raise ValueError("Invalid durable campaign budget clock")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid durable elapsed time")
    return max(now - started, elapsed)


def profiling_deadline(state, now, seconds):
    """Preserve measured elapsed time and reserve the campaign shutdown interval."""
    from panda_hard36.contract import BUDGET
    elapsed = durable_elapsed(state, now)
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid profiling duration")
    remaining = BUDGET["total_seconds"] - BUDGET["closeout_seconds"] - elapsed
    if remaining <= 0:
        raise ValueError("Campaign profiling budget exhausted; no GPU work started")
    return now + min(seconds, remaining)


def persist_profiling_elapsed(state_path, state, *, elapsed_at_open, monotonic_at_open,
                               now=None, monotonic_now=None):
    """Retain actual profiling time on success or failure without resetting status."""
    from . import R
    now = time.time() if now is None else now
    monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in
            (elapsed_at_open, monotonic_at_open, monotonic_now))
            or elapsed_at_open < 0 or monotonic_now < monotonic_at_open):
        raise ValueError("Invalid profiling monotonic clock")
    state["elapsed_seconds"] = max(durable_elapsed(state, now),
                                   elapsed_at_open + monotonic_now - monotonic_at_open)
    R.atomic_json(state_path, state)
    return state["elapsed_seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", choices=("s1", "s3"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    from panda_hard36 import plan
    from panda_method.edge_data import EdgeDataSources
    from panda_stage2_condq.training import configure_precision
    from .profiling import measure, registered_profiles
    from . import R
    marker = plan.ROOT / "gspread/server.txt"
    if marker.read_text().strip() != args.server:
        raise ValueError("Local server marker mismatch")
    if args.seconds < 120:
        raise ValueError("At least 120 seconds required for full isolated profiling")
    configure_precision()
    root = args.root or plan.ROOT / "work_dir/panda_hard36" / plan.CAMPAIGN_ID / args.server / "WV3"
    root.mkdir(parents=True, exist_ok=True)
    # Same local controller lock; never profile concurrently with this campaign.
    with (root / ".controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        catalog = plan.ROOT / "ablr2/sensor_sources.json"
        cfg = plan.default_config(args.server)
        if hashlib.sha256(catalog.read_bytes()).hexdigest() != cfg["baseline_resolved"]["source_registry_sha256"]:
            raise ValueError("Registered data catalog SHA changed")
        state_path = root / "campaign_state.json"
        if not state_path.exists():
            R.atomic_json(state_path, {"started_unix": time.time(), "created_at": R.utc_now(),
                "runs": [], "groups": [], "timing": {}, "donors": {},
                "status": "PREPARING_DEPLOYMENT_COST"})
        # Cost preparation counts against the same persistent 36-hour campaign.
        state = json.loads(state_path.read_text())
        opened_wall, opened_monotonic = time.time(), time.monotonic()
        elapsed_at_open = durable_elapsed(state, opened_wall)
        deadline = profiling_deadline(state, opened_wall, min(args.seconds, 1200))
        monotonic_deadline = opened_monotonic + deadline - opened_wall

        def persist_elapsed():
            return persist_profiling_elapsed(state_path, state,
                elapsed_at_open=elapsed_at_open, monotonic_at_open=opened_monotonic)

        try:
            sources = EdgeDataSources(catalog, root=plan.ROOT, sensor="WV3")
            profiles = []
            for cfg, step in registered_profiles(args.server):
                persist_elapsed()
                profiling_left = monotonic_deadline - time.monotonic()
                if profiling_left <= 0:
                    raise ValueError("Campaign profiling budget exhausted before next deployment")
                deadline = profiling_deadline(state, time.time(), profiling_left)
                result = measure(cfg, step, sources, root, deadline_unix=deadline)
                profiles.append({"phase": cfg["phase"], "geometry": result["signature"]["geometry_key"],
                                 "sampling": result["signature"]["reconstructor_sampling"]["variant"],
                                 "architecture_key": result["architecture_key"],
                                 "profile_sha256": result["profile_sha256"]})
                print(json.dumps({"profile": result["architecture_key"], "phase": cfg["phase"],
                    "geometry": result["signature"]["geometry_key"],
                    "sampling": result["signature"]["reconstructor_sampling"]["variant"],
                    "sheet_fields": result["measurement"]["sheet_cost"]}), flush=True)
            persist_elapsed()
            if time.monotonic() >= monotonic_deadline:
                raise ValueError("Campaign profiling budget exhausted before completion")
            profiling_deadline(state, time.time(), 1)
            R.atomic_json(root / "cost_profiles/ready.json", {
                "schema": "PANDA_HARD36_ALL_DEPLOYMENTS_COSTED_V1", "server": args.server,
                "profiles": profiles, "completed_at": R.utc_now()})
        finally:
            # The controller lock is still held; preserve elapsed even if a
            # profile failed or the wall clock moved backward during its work.
            persist_elapsed()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
