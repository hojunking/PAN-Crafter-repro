"""Prepare all local native cost sidecars before starting/resuming training."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", choices=("s1", "s3"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    from panda_scale30 import plan
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
    root = args.root or plan.ROOT / "work_dir/panda_scale30" / plan.CAMPAIGN_ID / args.server / "WV3"
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
        state = json.loads(state_path.read_text())
        deadline = min(time.time() + args.seconds,
                       state["started_unix"] + cfg["budget"]["total_seconds"]
                       - cfg["budget"]["closeout_seconds"])
        if deadline - time.time() < 120:
            raise TimeoutError("Insufficient original campaign budget for isolated profiling")
        sources = EdgeDataSources(catalog, root=plan.ROOT, sensor="WV3")
        profiles = []
        for cfg, step in registered_profiles(args.server):
            result = measure(cfg, step, sources, root, deadline_unix=deadline)
            profiles.append({"phase": cfg["phase"], "geometry": result["signature"]["geometry_key"],
                             "architecture_key": result["architecture_key"],
                             "profile_sha256": result["profile_sha256"]})
            print(json.dumps({"profile": result["architecture_key"], "phase": cfg["phase"],
                "geometry": result["signature"]["geometry_key"],
                "sheet_fields": result["measurement"]["sheet_cost"]}), flush=True)
        R.atomic_json(root / "cost_profiles/ready.json", {
            "schema": "PANDA_SCALE30_ALL_DEPLOYMENTS_COSTED_V1", "server": args.server,
            "profiles": profiles, "completed_at": R.utc_now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

