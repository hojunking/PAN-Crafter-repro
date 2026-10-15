"""Crash recovery from durable evaluations/selectors without new training.

Call once under the campaign controller lock before resuming the queue. Never
run concurrently with checkpoint creation: this is not a publisher-side repair.
Existing immutable event timestamps and identities are never regenerated.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

from . import publication as P, reporting as R
from .plan import _record_valid


def _file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _checkpoint_path(root, cfg, declared):
    raw = Path(declared)
    marker = ("runs", cfg["run_id"], "attempt_%02d" % cfg["attempt"])
    offsets = [i for i in range(len(raw.parts)-2) if raw.parts[i:i+3] == marker]
    if len(offsets) != 1:
        raise ValueError("Recovery checkpoint run/attempt lineage mismatch")
    path = root.joinpath(*raw.parts[offsets[0]:]).resolve()
    run = root.joinpath(*marker).resolve()
    if root not in run.parents or run not in path.parents:
        raise ValueError("Recovery checkpoint path escapes owning run")
    return path


def _selectors(root, directory, cfg):
    path = directory / "models/selection.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if (value.get("schema") != "DONOR30_ATOMIC_SELECTORS_V1"
            or value.get("run_id") != cfg["run_id"] or value.get("threshold") != cfg["hqnr_min"]
            or not isinstance(value.get("saved"), list)):
        raise ValueError("Recovery selector schema/run/gate mismatch")
    selected = {}
    for saved in value["saved"]:
        if not _record_valid(saved):
            raise ValueError("Recovery found unverified selector record")
        expected = {"run_id": cfg["run_id"], "attempt": cfg["attempt"], "seed": cfg["seed"],
                    "server": cfg["server"], "dataset": cfg["dataset"], "phase": cfg["phase"]}
        if any(saved.get(key) != val for key, val in expected.items()):
            raise ValueError("Recovery saved selector run identity mismatch")
        path = _checkpoint_path(root, cfg, saved["checkpoint_path"])
        if not R._sha(saved.get("file_sha256")) or _file_sha(path) != saved["file_sha256"]:
            raise ValueError("Recovery checkpoint file SHA mismatch")
        step = saved["completed_update"]
        if step in selected and selected[step] != saved:
            raise ValueError("Recovery conflicting selector states at one step")
        selected[step] = dict(saved, checkpoint_path=str(path))
    for key in ("best_hqnr", "best_gate_ergas"):
        pointer = value.get(key)
        if pointer is not None:
            recorded = selected.get(pointer.get("completed_update"))
            if recorded is None or any(pointer.get(k) != recorded.get(k) for k in
                ("state_hash", "file_sha256", "run_id", "attempt", "HQNR", "ERGAS", "SCC")):
                raise ValueError("Recovery best pointer absent from saved audit")
    return selected


def recover_campaign(root, server=None):
    root = Path(root).resolve(strict=True)
    scan = P.scan_campaign(root)
    if scan["conflicts"]:
        raise ValueError("Cannot recover conflicting/torn immutable ledger: " + R.canonical_json(scan["conflicts"]))
    records = {}
    restored = 0
    # A crash after durable ledger append but before outbox rename is safe to
    # restore using the original envelope, without appending a duplicate event.
    with (root / ".enqueue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for record in scan["records"]:
            target = root / "outbox" / (record["record_id"] + ".json")
            if not target.exists():
                R.atomic_json(target, record)
                restored += 1
            if record["record_type"] != "EVAL_SUMMARY":
                continue
            ident = record["identity"]
            key = (ident["run"], ident["attempt"], ident["selected_step"])
            if key in records and records[key] != record:
                raise ValueError("Recovery cannot choose between conflicting same-step envelopes")
            records[key] = record
    observations, emitted, verified = [], 0, 0
    for cfg_path in sorted(root.glob("runs/*/attempt_*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        R._config(cfg)
        if server is not None and cfg["server"] != server:
            raise ValueError("Recovery encountered foreign server")
        directory = cfg_path.parent
        expected = root / "runs" / cfg["run_id"] / ("attempt_%02d" % cfg["attempt"])
        if directory.resolve() != expected.resolve() or root not in directory.resolve().parents:
            raise ValueError("Recovery config directory/identity mismatch")
        saved = _selectors(root, directory, cfg)
        observed_steps = set()
        for path in sorted(directory.glob("evaluation_*.json")):
            ev = json.loads(path.read_text())
            step = ev["state_before"]["completed_update"]
            if path.name != "evaluation_%07d.json" % step:
                raise ValueError("Recovery evaluation filename/step mismatch")
            observed_steps.add(step)
            item = saved.get(step)
            artifact = None
            if item is not None:
                if (item["state_hash"] != ev["state_before"]["model_state_sha256"]
                        or item["HQNR"] != ev["fr"]["metrics"]["HQNR"]
                        or item["ERGAS"] != ev["rr"]["metrics"]["ERGAS"]
                        or item["SCC"] != ev["rr"]["metrics"]["SCC"]):
                    raise ValueError("Recovery selector/evaluation state or metric mismatch")
                artifact = {"weights_saved": True, "checkpoint_path": item["checkpoint_path"],
                    "file_sha256": item["file_sha256"], "model_state_sha256": item["state_hash"],
                    "identity": ev["state_before"], "checkpoint_verified": True,
                    "predictions_reverified": item.get("predictions_reverified", False),
                    "verification": "RECOVERED_STRICT_RELOAD_SELECTOR_AND_FILE_SHA",
                    "selection_reason": "RECOVERED_SAVED_SELECTOR"}
            key = cfg["run_id"], cfg["attempt"], step
            existing = records.get(key)
            candidate = existing or R.make_record(cfg, ev, artifact, root=root)
            validated = P._candidate(candidate, root, cfg["server"])
            if existing is not None and item is not None:
                if existing["identity"].get("checkpoint") != item["file_sha256"]:
                    raise ValueError("Recovery existing envelope conflicts with saved selector artifact")
                P._artifact(validated, root)
            if existing is None:
                R.LocalOutbox(root).enqueue(candidate)
                emitted += 1
            verified += 1
            if item is not None:
                observations.append({**item, "evaluation_path": str(path), "phase": cfg["phase"],
                    "server": cfg["server"], "dataset": cfg["dataset"], "attempt": cfg["attempt"]})
        if set(saved) - observed_steps:
            raise ValueError("Recovery saved selector missing physical evaluation")
    return {"recovered_evaluations": emitted, "recovered_outbox_files": restored,
            "verified_evaluations": verified, "observations": observations}
