"""Finish an authorized old sensor run with its unchanged runtime and admission.

The successor controller safely stops the previous trainer first. This adapter
resumes only that run and, when it is a Stage1 run, its registered first Student.
All omitted experiments remain explicit omissions in the successor's accounting.
"""
import argparse
import ast
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import signal
import time
import traceback

from directshift48.reporting import atomic_json
from panda_sensor_refine import early_stop
from panda_sensor_refine.orchestration import (
    authenticate_stop, final_evidence, local, read, sha,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("panda_sensor_refine", "panda_s1loss9", "panda_sensor5")
PAUSED = "PAUSED_FOR_SENSOR_FOCUS"
STAGE1_SUPPORT = "SUPPORT_VIOLATION: every predicted PAN view must support fixed ROI8/stencil7"
STAGE2_SUPPORT = "SUPPORT_VIOLATION: Lanczos4 Student / Lanczos4 Teacher violates fixed ROI8; "


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def registered_configs(package, server):
    if package not in PACKAGES or server not in ("s1", "s3"):
        raise ValueError("Unregistered sensor boundary package/server")
    plan = importlib.import_module(package + ".plan")
    return [plan.resolve_case(server, case, replica)
            for group in plan.groups(server) for case, replica in group["runs"]]


def select_configs(configs, state, run_ids):
    """An existing run is mandatory; only its immediate donor child may be new."""
    expected = {cfg["run_id"]: cfg for cfg in configs}
    rows = {row["run_id"]: row for row in state.get("runs", [])}
    if (len(expected) != len(configs) or len(rows) != len(state.get("runs", []))
            or not set(rows) <= set(expected)):
        raise ValueError("Unknown or duplicate original registry run")
    if (not run_ids or len(set(run_ids)) != len(run_ids)
            or not set(run_ids) <= set(expected)):
        raise ValueError("Boundary run IDs must be unique registered entries")
    chosen = [expected[key] for key in run_ids]
    first = chosen[0]
    if first["run_id"] not in rows:
        raise ValueError("Boundary must start with an already-started run")
    allowed = [first["run_id"]]
    if first["phase"] == "donor":
        children = [cfg for cfg in configs
                    if cfg["phase"] == "stage2" and cfg.get("donor_version") == first["case"]]
        if not children:
            raise ValueError("Started donor has no registered dependent Student")
        allowed.append(children[0]["run_id"])
    if run_ids != allowed:
        raise ValueError("Boundary must contain only the started run and its first donor child")
    active = [row for row in rows.values() if row.get("status") in ("RUNNING", "STOPPED")]
    if (len(active) > 1 or any(row["run_id"] not in run_ids for row in active)
            or any(row["status"] == "RUNNING" for row in active)):
        raise ValueError("Boundary requires one safely stopped trainer, never an active/unselected run")
    return chosen


def authorization(server, package, run_ids, receipt_path):
    from .queue import controller_pin
    directory = ROOT / "work_dir/panda_sensor_focus_launchers" / server
    receipt_path = Path(receipt_path)
    if (not receipt_path.is_absolute() or receipt_path.parent.resolve() != directory.resolve()
            or receipt_path.is_symlink() or receipt_path.name in ("takeover.json", "boundary_plan.json")):
        raise ValueError("Boundary receipt must be an owned file inside its launcher directory")
    paths = [directory / "takeover.json", directory / "boundary_plan.json"]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("Owned frozen takeover and boundary plan are required")
    takeover, boundary = [read(path) for path in paths]
    if (takeover.get("server") != server or takeover.get("controller_pin") != controller_pin()
            or boundary.get("server") != server):
        raise ValueError("Frozen boundary server or controller source changed")
    entries = boundary.get("entries", [])
    if sum(entry.get("package") == package and entry.get("run_ids") == run_ids
           for entry in entries) != 1:
        raise ValueError("Boundary IDs are not uniquely authorized by the frozen plan")
    proof = {"takeover_sha256": sha(paths[0]), "boundary_plan_sha256": sha(paths[1]),
             "controller_pin_sha256": digest(takeover["controller_pin"])}
    return proof


def known_support(message):
    if message == STAGE1_SUPPORT:
        return True
    if not message.startswith(STAGE2_SUPPORT):
        return False
    try:
        payload = json.loads(message[len(STAGE2_SUPPORT):])
    except (TypeError, ValueError):
        return False
    return payload.get("action") == "STOP_BEFORE_UPDATE_NO_CLAMP_OR_ADAPTIVE_ROI"


def row_support_message(row):
    error = row.get("error", "")
    if not (error.startswith("ValueError(") and error.endswith(")")):
        return None
    try:
        message = ast.literal_eval(error[len("ValueError("):-1])
    except (ValueError, SyntaxError):
        return None
    return message if isinstance(message, str) and known_support(message) else None


def failed_proof(row, cfg):
    message = row_support_message(row)
    if (row.get("status") != "FAILED" or message is None
            or row.get("poisoned") is not False or row.get("stop_checkpoint_error")):
        raise ValueError("Only known, unpoisoned support failures can be isolated")
    return {"status": "FAILED_SUPPORT", "error": message,
            "safe_checkpoint": authenticate_stop(row, cfg, ROOT)}


def row_proof(camp, cfg):
    row = next((r for r in camp.state["runs"] if r["run_id"] == cfg["run_id"]), None)
    if row is None:
        return None
    if row["status"] == "COMPLETED":
        if row.get("step") != cfg["planned_horizon"]:
            raise ValueError("Completed boundary run has the wrong horizon")
        return {"status": "COMPLETED", "evidence": final_evidence(camp.root, cfg)}
    if row["status"] == early_stop.STATUS:
        return {"status": early_stop.STATUS, "evidence": early_stop.verify(row, cfg, ROOT)}
    if row["status"] == "FAILED":
        return failed_proof(row, cfg)
    if row["status"] == "STOPPED":
        authenticate_stop(row, cfg, ROOT)
        return None
    raise ValueError("Unexpected boundary run status: " + str(row.get("status")))


def preflight_failure(camp, cfg, exc):
    if not isinstance(exc, ValueError) or not known_support(str(exc)):
        return None
    if (any(row["run_id"] == cfg["run_id"] for row in camp.state["runs"])
            or (camp.root / "runs" / cfg["run_id"]).exists()):
        return None
    camp.verify_code()
    donor = None
    if cfg["phase"] == "stage2":
        source = camp.state["donors"].get(cfg["donor_version"])
        if source is None:
            raise ValueError("Preflight support failure lacks its frozen donor")
        checkpoint = local(source["checkpoint_path"], ROOT)
        selector = local(source["selector_receipt_path"], ROOT)
        if (checkpoint.is_symlink() or selector.is_symlink()
                or sha(checkpoint) != source["checkpoint_sha256"]):
            raise ValueError("Preflight donor evidence changed")
        donor = {"checkpoint_sha256": source["checkpoint_sha256"],
                 "selector_file_sha256": sha(selector),
                 "selector_sha256": source["selector_sha256"]}
    return {"status": "FAILED_PREFLIGHT_SUPPORT", "error": str(exc),
            "run_id": cfg["run_id"], "requested_config_sha256": digest(cfg),
            "code_pin_sha256": digest(camp.code), "production_updates": 0,
            "production_run_absent": True, "donor": donor}


def run_boundary(server, package, run_ids, receipt_path):
    """Resume a frozen authorized boundary; return its authenticated receipt."""
    run_ids = list(run_ids)
    if (ROOT / "gspread/server.txt").read_text().strip() != server:
        raise ValueError("Local server marker mismatch")
    frozen = authorization(server, package, run_ids, receipt_path)
    configs = registered_configs(package, server)
    runtime = importlib.import_module(package + ".runtime")
    camp = runtime.Campaign(server)
    admission_path = camp.root / "admission.json"
    if not admission_path.is_file() or admission_path.is_symlink():
        raise ValueError("Boundary requires the original admitted campaign")
    admission_sha = sha(admission_path)
    old_check, next_check = camp.check, [0.]

    def check(**kwargs):
        old_check(**kwargs)
        if time.monotonic() >= next_check[0]:
            if authorization(server, package, run_ids, receipt_path) != frozen:
                raise RuntimeError("Boundary controller or authorized selection changed")
            if sha(admission_path) != admission_sha:
                raise RuntimeError("Original boundary admission changed")
            next_check[0] = time.monotonic() + 20.

    camp.check = check
    with (camp.root / ".controller.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Reload after locking; no stale inventory may authorize a lost optimizer.
        camp.state = read(camp.state_path)
        chosen = select_configs(configs, camp.state, run_ids)
        for cfg in chosen:
            row_proof(camp, cfg)
        original = {sig: signal.signal(sig, lambda *args: setattr(camp, "stop_requested", True))
                    for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            camp.admission(ROOT / "ablr2/sensor_sources.json")
            if sha(admission_path) != admission_sha:
                raise ValueError("Original admission bytes changed during resume")
            if package == "panda_sensor_refine":
                external = camp.migration.get("external_donor", {})
                if external.get("status") == "COMPLETED":
                    camp.register_external_donor()
                elif external.get("status") != "FAILED_SUPPORT":
                    raise ValueError("Original external donor admission is not authenticated")
            control = camp.state.setdefault("sensor_focus_boundary", {})
            identity = {"package": package, "run_ids": run_ids, **frozen,
                        "original_admission_sha256": admission_sha}
            if control and control.get("identity") != identity:
                raise ValueError("Existing boundary control identity changed")
            control.setdefault("identity", identity)
            outcomes = control.setdefault("outcomes", {})
            failed_donor = None
            for cfg in chosen:
                camp.check()
                key = cfg["run_id"]
                if failed_donor is not None and cfg.get("donor_version") == failed_donor:
                    if any(row["run_id"] == key for row in camp.state["runs"]):
                        raise ValueError("A started Student cannot be hidden by a failed-donor hold")
                    result = {"status": "HELD_FAILED_DONOR", "donor_case": failed_donor}
                else:
                    result = row_proof(camp, cfg)
                    saved = outcomes.get(key)
                    if saved and saved.get("status") == "FAILED_PREFLIGHT_SUPPORT":
                        recreated = preflight_failure(camp, cfg, ValueError(saved["error"]))
                        if recreated != saved:
                            raise ValueError("Boundary preflight failure evidence changed")
                        result = saved
                    if result is None:
                        if early_stop.finish_stopped(camp, cfg):
                            result = row_proof(camp, cfg)
                        else:
                            camp.admit_group({"name": "focus_boundary_" + cfg["case"],
                                              "runs": [(cfg["case"], cfg["replica"])]})
                            try:
                                camp.validate_preflight(cfg)
                            except Exception as exc:
                                result = preflight_failure(camp, cfg, exc)
                                if result is None:
                                    raise
                            if result is None:
                                try:
                                    early_stop.run_case(camp, cfg, runtime.BudgetExpired)
                                except ValueError:
                                    result = row_proof(camp, cfg)
                                    if result is None or result["status"] != "FAILED_SUPPORT":
                                        raise
                                if result is None:
                                    result = row_proof(camp, cfg)
                    if result is None:
                        raise RuntimeError("Boundary failed to reach an explicit terminal outcome")
                    if cfg["phase"] == "donor":
                        if result["status"] == "COMPLETED":
                            camp.register_donor(cfg["case"])
                        elif result["status"] in ("FAILED_SUPPORT", "FAILED_PREFLIGHT_SUPPORT"):
                            failed_donor = cfg["case"]
                        else:
                            raise ValueError("Unexpected terminal Stage1 boundary state")
                if key in outcomes and outcomes[key] != result:
                    raise ValueError("Existing boundary outcome changed")
                outcomes[key] = result
                camp.persist()
            if authorization(server, package, run_ids, receipt_path) != frozen:
                raise ValueError("Boundary authorization changed before completion")
            camp.verify_code()
            camp.state["status"] = PAUSED
            camp.persist()
            receipt = {"schema": "SENSOR_FOCUS_BOUNDARY_COMPLETION_V1", "server": server,
                       **identity, "status": PAUSED, "outcomes": outcomes,
                       "original_training_registry_unchanged": True,
                       "all_original_runs_completed": False}
            path = Path(receipt_path)
            if path.exists() and read(path) != receipt:
                raise ValueError("Existing boundary completion receipt changed")
            atomic_json(path, receipt)
            return receipt
        except runtime.BudgetExpired as exc:
            camp.state.update(status="STOPPED_SAFE", reason=str(exc))
            raise
        except BaseException as exc:
            camp.state.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            camp.persist()
            camp.check = old_check
            for sig, handler in original.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, choices=("s1", "s3"))
    parser.add_argument("--package", required=True, choices=PACKAGES)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu("cuda")
    run_boundary(args.server, args.package, args.run_id, args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
