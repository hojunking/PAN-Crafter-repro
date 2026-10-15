"""Immutable, reload-verified research checkpoints and atomic selector pointers.

Only locally-created checkpoint payloads are loaded here. The evaluator remains
responsible for official full20 protocol validation; this layer additionally
binds evaluation identity to the exact paused live trainer being serialized.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import copy, deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from . import plan

SCHEMA = "DONOR30_CHECKPOINT_V1"
SELECTOR_SCHEMA = "DONOR30_ATOMIC_SELECTORS_V1"


def _model_hash(stage):
    from .evaluation import model_state_hash
    return model_state_hash(stage)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        result = copy(value)
        for key, item in value.items():
            result[key] = _cpu(item)
        # PyTorch state_dict OrderedDict carries per-module serialization
        # versions outside the dictionary entries.
        if hasattr(value, '_metadata'):
            result._metadata = deepcopy(value._metadata)
        return result
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return deepcopy(value)


def _same(first, second):
    if isinstance(first, torch.Tensor):
        return (isinstance(second, torch.Tensor) and first.dtype == second.dtype
                and first.shape == second.shape and torch.equal(first.cpu(), second.cpu()))
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and first.dtype == second.dtype and np.array_equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and set(first) == set(second) and all(_same(first[k], second[k]) for k in first)
    if isinstance(first, (tuple, list)):
        return type(first) is type(second) and len(first) == len(second) and all(_same(a, b) for a, b in zip(first, second))
    return type(first) is type(second) and first == second


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state().clone(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _fsync_directory(directory):
    fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path, payload):
    data = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode()
    fd, temp = tempfile.mkstemp(prefix=".selectors-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def _locked_models(run_root):
    models = Path(run_root) / "models"
    models.mkdir(parents=True, exist_ok=True)
    if models.is_symlink():
        raise ValueError("Checkpoint directory must not be a symlink")
    with (models / ".artifact.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield models


def read_selection(run_root):
    path = Path(run_root) / "models" / "selection.json"
    if not path.exists():
        return {"schema": SELECTOR_SCHEMA, "best_hqnr": None, "best_gate_ergas": None, "saved": []}
    result = json.loads(path.read_text())
    if result.get("schema") != SELECTOR_SCHEMA:
        raise ValueError("Unknown selector registry schema")
    for key in ("best_hqnr", "best_gate_ergas"):
        if result.get(key) is not None and not plan._record_valid(result[key]):
            raise ValueError("Selector points to an unverified checkpoint")
    if not isinstance(result.get("saved"), list):
        raise ValueError("Saved checkpoint audit missing")
    return result


def _evaluation_record(evaluation, *, cfg=None):
    gate = evaluation.get("qualification", {})
    if not all(gate.get(k) is True for k in ("same_state_verified", "metrics_complete", "metrics_finite")):
        raise ValueError("Only complete finite same-state official evaluations can save a candidate")
    identities = [evaluation.get(key) for key in ("state_before", "rr_state", "fr_state", "state_after")]
    if not isinstance(identities[0], dict) or any(i != identities[0] for i in identities[1:]):
        raise ValueError("RR/FR/live state identity mismatch")
    identity = identities[0]
    if cfg is not None and any(identity.get(k) != cfg[k] for k in ("run_id", "attempt", "seed")):
        raise ValueError("Evaluated state does not belong to this run")
    h = evaluation.get("fr", {}).get("metrics", {}).get("HQNR")
    e = evaluation.get("rr", {}).get("metrics", {}).get("ERGAS")
    s = evaluation.get("rr", {}).get("metrics", {}).get("SCC")
    # Placeholder path is used for rank validation only; it is never returned
    # as checkpoint verification evidence or written as a selector.
    record = {"run_id": identity.get("run_id"), "attempt": identity.get("attempt"),
        "seed": identity.get("seed"), "completed_update": identity.get("completed_update"),
        "state_hash": identity.get("model_state_sha256"), "HQNR": h, "ERGAS": e, "SCC": s,
        "checkpoint_path": "UNSAVED_RANK_ONLY", "checkpoint_verified": True,
        "metrics_complete": True, "metrics_finite": True, "same_state_verified": True}
    if not plan._record_valid(record):
        raise ValueError("Invalid evaluated metric or model identity")
    return record


def _candidate_reasons(record, selection, threshold):
    reasons = []
    if selection["best_hqnr"] is None or plan.best_hqnr_key(record) < plan.best_hqnr_key(selection["best_hqnr"]):
        reasons.append("BEST_HQNR")
    if record["HQNR"] >= threshold and (selection["best_gate_ergas"] is None
            or plan.selection_key(record, threshold) < plan.selection_key(selection["best_gate_ergas"], threshold)):
        reasons.append("HQNR_GATE_ERGAS_BEST")
    return reasons


def candidate_reasons(evaluation, run_root, threshold):
    return _candidate_reasons(_evaluation_record(evaluation), read_selection(run_root), threshold)


def _prediction_signature(evaluation):
    signature = {}
    for split in ("rr", "fr"):
        rows = evaluation.get(split, {}).get("per_scene", [])
        if len(rows) != 20:
            raise ValueError("Prediction verification requires all20 RR and FR scenes")
        entries = [(r.get("scene_id"), r.get("raw_prediction_sha256")) for r in rows]
        if len({name for name, _ in entries}) != 20 or any(not name or not isinstance(sha, str) or len(sha) != 64 for name, sha in entries):
            raise ValueError("Prediction verification needs unique scene IDs and raw output hashes")
        signature[split] = sorted(entries)
    return signature


def _save_immutable(path, payload):
    fd, temp = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # link is atomic and fails instead of replacing an existing checkpoint.
        os.link(temp, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save_candidate(trainer, stream, evaluation, run_root, cfg, *, reload_factory,
                   verify_evaluate=None, state_hash_fn=None):
    """Save an improvement, verify actual reload, then move both atomic pointers.

    reload_factory(cfg) MUST return a fresh CPU trainer, not the live trainer.
    Optional verify_evaluate(reloaded) runs the official full40 prediction path;
    without it, verification explicitly covers exact model/optimizer/stream
    bytes and metadata, not a newly measured set of metric values.
    Returns None for non-improvements, otherwise {artifact, record, reasons}.
    """
    if not callable(reload_factory):
        raise TypeError("A real fresh-trainer reload factory is required")
    state_hash_fn = state_hash_fn or _model_hash
    record = _evaluation_record(evaluation, cfg=cfg)
    if getattr(trainer, "poisoned", False) or trainer.updates != record["completed_update"]:
        raise ValueError("Live trainer is failed or advanced since evaluation")
    if state_hash_fn(trainer.stage) != record["state_hash"]:
        raise ValueError("Live model no longer matches the evaluated checkpoint")
    rng = capture_rng()
    try:
        with _locked_models(run_root) as models:
            selection = read_selection(run_root)
            threshold = cfg["hqnr_min"]
            if selection.get("run_id", cfg["run_id"]) != cfg["run_id"] or selection.get("threshold", threshold) != threshold:
                raise ValueError("Selector registry belongs to another run or HQNR gate")
            reasons = _candidate_reasons(record, selection, threshold)
            if not reasons:
                return None
            path = (models / ("step_%07d.pt" % trainer.updates)).absolute()
            payload = {"schema": SCHEMA, "config": deepcopy(cfg), "evaluation": deepcopy(evaluation),
                "identity": deepcopy(evaluation["state_before"]), "state_hash": record["state_hash"],
                "trainer": _cpu(trainer.state_dict()), "stream": _cpu(stream.state_dict()),
                "rng": _cpu(rng), "selection_reasons": reasons,
                "scheduler": {"family": "warmup_cosine", "planned_horizon": cfg["planned_horizon"],
                              "completed_update": trainer.updates, "indexing": "ONE_BASED"}}
            if path.exists():
                if path.is_symlink():
                    raise ValueError("Immutable checkpoint path is a symlink")
            else:
                _save_immutable(path, payload)
            # This is our local checkpoint, not an untrusted downloaded pickle.
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if not _same(payload, loaded):
                raise ValueError("Immutable checkpoint exists with different state or serialized content differs")
            reloaded = reload_factory(deepcopy(cfg))
            if reloaded is trainer or reloaded.stage is trainer.stage:
                raise ValueError("Reload factory must create an independent trainer")
            if any(p.device.type != "cpu" for p in reloaded.stage.parameters()):
                raise ValueError("Reload factory must construct on CPU")
            reloaded.load_state_dict(loaded["trainer"])
            if (state_hash_fn(reloaded.stage) != record["state_hash"]
                    or not _same(_cpu(reloaded.state_dict()), loaded["trainer"])):
                raise ValueError("Fresh trainer strict reload differs from saved state")
            predictions_verified = False
            if verify_evaluate is not None:
                rerun = verify_evaluate(reloaded)
                if _prediction_signature(rerun) != _prediction_signature(evaluation):
                    raise ValueError("Reloaded raw predictions differ from evaluated model")
                predictions_verified = True
            if trainer.updates != record["completed_update"] or state_hash_fn(trainer.stage) != record["state_hash"]:
                raise ValueError("Live model changed during artifact verification")
            if not _same(stream.state_dict(), loaded["stream"]):
                raise ValueError("Data stream advanced during artifact verification")
            sha = file_sha256(path)
            record.update(checkpoint_path=str(path), checkpoint_verified=True, file_sha256=sha,
                server=cfg["server"], dataset=cfg["dataset"], phase=cfg["phase"],
                predictions_reverified=predictions_verified)
            if cfg["phase"] == "stage2":
                record.update({key: deepcopy(cfg[key]) for key in
                    ("donor_run_id", "donor_completed_update", "donor_state_hash", "donor_selection_status", "donor_hqnr_qualified")})
            artifact = {"weights_saved": True, "checkpoint_path": str(path), "file_sha256": sha,
                "model_state_sha256": record["state_hash"], "identity": deepcopy(evaluation["state_before"]),
                "checkpoint_verified": True, "predictions_reverified": predictions_verified,
                "verification": "STRICT_CPU_RELOAD_MODEL_OPTIMIZER_STREAM_IDENTITY",
                "selection_reason": "+".join(reasons), "selection_reasons": reasons}
            if "BEST_HQNR" in reasons:
                selection["best_hqnr"] = deepcopy(record)
            if "HQNR_GATE_ERGAS_BEST" in reasons:
                selection["best_gate_ergas"] = deepcopy(record)
            selection["run_id"] = cfg["run_id"]
            selection["threshold"] = threshold
            if not any(r["checkpoint_path"] == record["checkpoint_path"] for r in selection["saved"]):
                selection["saved"].append(deepcopy(record))
            _atomic_json(models / "selection.json", selection)
            return {"artifact": artifact, "record": record, "selection_record": record, "reasons": reasons}
    finally:
        restore_rng(rng)


def save_verified(trainer, stream, evaluation, run_root, *, cfg=None, reload_factory,
                  verify_evaluate=None, state_hash_fn=None):
    """Alias for callers already holding a complete evaluated candidate."""
    if cfg is None:
        cfg = evaluation.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("Explicit resolved case configuration required")
    return save_candidate(trainer, stream, evaluation, run_root, cfg,
        reload_factory=reload_factory, verify_evaluate=verify_evaluate, state_hash_fn=state_hash_fn)


def save_resume(trainer, stream, run_root, cfg, *, state_hash_fn=None, reason="SAFE_STOP"):
    """Emergency complete state; never qualified or a donor-selector candidate.

    No live trainer mutation and no expensive fresh-model build at deadline.
    Actual file round-trip is checked. A new attempt still starts fresh unless
    the caller explicitly implements validated resume of optimizer/data/RNG.
    """
    state_hash_fn = state_hash_fn or _model_hash
    if getattr(trainer, "poisoned", False):
        raise ValueError("Cannot checkpoint a partially failed optimizer update")
    rng = capture_rng()
    try:
        state_hash = state_hash_fn(trainer.stage)
        with _locked_models(run_root) as models:
            path = (models / ("resume_step_%07d_%d.pt" % (trainer.updates, time.time_ns()))).absolute()
            identity = {key: cfg[key] for key in ("run_id", "attempt", "seed")}
            identity.update(completed_update=trainer.updates, model_state_sha256=state_hash)
            payload = {"schema": SCHEMA, "config": deepcopy(cfg), "trainer": _cpu(trainer.state_dict()),
                "stream": _cpu(stream.state_dict()), "rng": _cpu(rng), "evaluation": None,
                "identity": identity, "state_hash": state_hash, "selection_reasons": [],
                "resume_only": True, "reason": str(reason),
                "scheduler": {"family": "warmup_cosine", "planned_horizon": cfg["planned_horizon"],
                              "completed_update": trainer.updates, "indexing": "ONE_BASED"}}
            _save_immutable(path, payload)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if not _same(payload, loaded) or state_hash_fn(trainer.stage) != state_hash:
                raise ValueError("Resume file round-trip/live state mismatch")
            result = {"checkpoint_path": str(path), "file_sha256": file_sha256(path),
                "model_state_sha256": state_hash, "completed_update": trainer.updates,
                "resume_only": True, "donor_eligible": False, "reason": str(reason),
                "verification": "SERIALIZED_STATE_ROUNDTRIP_ONLY"}
            _atomic_json(models / "latest_resume.json", result)
            return result
    finally:
        restore_rng(rng)
