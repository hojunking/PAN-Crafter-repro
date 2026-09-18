"""Local exact-50K FH12 Teacher calibration; no legacy T0 fallback.

The reference is published last, after the full training four-view AXIS16 cache
and unaugmented, train-only 3072-patch final-output tau calibration succeed.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fh12.data import (AUGMENTATION, RECIPE, build_dataset, canonical_sha,
                       check_deadline, sha256_file, write_immutable_json)
from fh12.common import ROOT, source_identity
from pa.warp import warp_pan


RADII = (0.25, 0.5, 1.0, 2.0)
AXIS16 = tuple((dy, dx) for radius in RADII for dy, dx in
               ((radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)))
CONSTANT_ALIGNER_Q = 0.46875


def _batch(dataset, indices, device, rot=None):
    rows = [dataset.base(int(i)) if rot is None else dataset.get_view(int(i), int(rot), augment=True)
            for i in indices]
    if not dataset.has_gt:
        raise ValueError("Calibration requires train GT, never FR data")
    # Legacy tuple order: GT, LMS, MS, LP, PAN, metadata.
    return tuple(torch.stack([row[j] for row in rows]).to(device) for j in range(5))


@torch.no_grad()
def axis16_q(model, pan, ms):
    """Return raw per-patch q, four per-radius q values, and native c.

    P/M-only aligner probes; reconstruction U and LP/H are not involved.
    mean over xy implements 1/(2K), not 1/K, without any temperature.
    """
    base = F.interpolate(ms.float(), scale_factor=4, mode="bicubic", align_corners=False)
    native = model.predict_delta(pan.float(), base)
    probes = []
    for shift in AXIS16:
        epsilon = pan.new_tensor(shift, dtype=torch.float32).expand(len(pan), 2)
        shifted = warp_pan(pan.float(), epsilon)
        predicted = model.predict_delta(shifted, base)
        probes.append((predicted + epsilon - native).abs().mean(dim=1))
    values = torch.stack(probes, dim=1)
    by_radius = values.reshape(len(pan), len(RADII), 4).mean(dim=2)
    return values.mean(dim=1), by_radius, native


@torch.no_grad()
def compute_calibration(model, dataset, indices, device="cuda", batch_size=64, deadline_utc=None):
    """Numerical core, also used by tiny synthetic CPU tests."""
    if batch_size < 1:
        raise ValueError("Calibration batch size must be positive")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("Calibration subset must be nonempty unique base-patch indices")
    if indices.min() < 0 or indices.max() >= len(dataset):
        raise ValueError("Calibration indices outside training dataset")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    errors = []
    for start in range(0, len(indices), batch_size):
        check_deadline(deadline_utc)
        gt, _, ms, lp, pan = _batch(dataset, indices[start:start + batch_size], device)
        output = model(pan, ms, lp)["y"]
        e_t = (output.float() - gt.float()).abs().mean(dim=1)
        if not torch.isfinite(e_t).all():
            raise ValueError("Non-finite Teacher final-output calibration error")
        errors.append(e_t.cpu().numpy().reshape(-1))
    pooled_errors = np.concatenate(errors)
    tau_r = max(float(np.median(pooled_errors)), 1e-6)
    raw_q = np.empty((len(dataset), 4), dtype=np.float32)
    per_radius = np.empty((len(dataset), 4, len(RADII)), dtype=np.float32)
    native_delta = np.empty((len(dataset), 4, 2), dtype=np.float32)
    for rot in range(4):
        for start in range(0, len(dataset), batch_size):
            check_deadline(deadline_utc)
            ids = np.arange(start, min(start + batch_size, len(dataset)))
            _, _, ms, _, pan = _batch(dataset, ids, device, rot)
            q, qr, delta = axis16_q(model, pan, ms)
            if not torch.isfinite(q).all() or (q < 0).any() or not torch.isfinite(delta).all():
                raise ValueError("Invalid AXIS16 q or native offset")
            raw_q[ids, rot] = q.cpu().numpy()
            per_radius[ids, rot] = qr.cpu().numpy()
            native_delta[ids, rot] = delta.cpu().numpy()
    q_ref = float(np.median(raw_q[indices, :]))
    if not np.isfinite(q_ref) or q_ref <= 0:
        raise ValueError(f"Invalid q_ref={q_ref}; do not repair zero with epsilon")
    radius_mean = per_radius.mean(axis=(0, 1), dtype=np.float64)
    q_spread = float(np.std(raw_q, dtype=np.float64))
    calibration = {
        "tau_R": tau_r, "q_ref": q_ref, "calibration_n": int(len(indices)),
        "tau_input": "train-only unaugmented final-output band-mean absolute error",
        "tau_min": 1e-6, "q_formula": "mean_j(mean_xy(abs(c_shift+eps-c_native)))",
        "weight_formula": "q_ref/(q_ref+q)", "q_ref_views": "subset indices x all four fixed-HV rotations",
        "calibration_indices_sha256": canonical_sha(indices.tolist()),
        "q_shape": list(raw_q.shape), "radii": list(RADII), "AXIS16": [list(x) for x in AXIS16],
        "diagnostics": {
            "q_min": float(raw_q.min()), "q_max": float(raw_q.max()),
            "q_mean": float(raw_q.mean(dtype=np.float64)), "q_std": q_spread,
            "q_median_full_train": float(np.median(raw_q)),
            "per_radius_q_mean": radius_mean.tolist(),
            "per_radius_q_over_radius": (radius_mean / np.asarray(RADII)).tolist(),
            "radius_q_linear_slope": float(np.polyfit(RADII, radius_mean, 1)[0]),
            "native_delta_mean_dydx": native_delta.mean(axis=(0, 1), dtype=np.float64).tolist(),
            "native_delta_abs_max": float(np.abs(native_delta).max()),
            "calibration_output_error_quantiles": np.quantile(pooled_errors, [0, .25, .5, .75, 1]).tolist(),
            "constant_aligner_expected_q": CONSTANT_ALIGNER_Q,
            "constant_q_warning": bool(q_spread < 1e-7),
            "near_constant_aligner_q_warning": bool(abs(q_ref - CONSTANT_ALIGNER_Q) < 1e-5),
            "note": "q is equivariance consistency, not physical/native alignment GT error. Warnings do not gate Students.",
        },
    }
    return calibration, {"q": raw_q, "per_radius_q": per_radius, "native_delta": native_delta,
                         "calibration_indices": indices}


def _teacher_identity(teacher_run, root, server=None):
    root = Path(root).resolve()
    teacher_run = Path(teacher_run)
    if not teacher_run.is_absolute():
        teacher_run = root / (teacher_run if len(teacher_run.parts) > 1 else Path("work_dir") / teacher_run)
    teacher_run = teacher_run.resolve()
    config_path = teacher_run / "meta" / "config.resolved.yaml"
    candidate = teacher_run / "candidates" / "50000"
    weights = candidate / "model.safetensors"
    state_path = candidate / "training_state.pt"
    identity_path = candidate / "identity.json"
    for path in (config_path, weights, state_path, identity_path):
        if not path.is_file():
            raise FileNotFoundError(f"FH12 exact50K reference is incomplete: {path}")
    cfg = yaml.safe_load(config_path.read_text())
    f = cfg["fh12"]
    if f["role"] != "T" or (server is not None and f["server_id"] != server):
        raise ValueError("FH12 reference must be this server's explicit fresh Teacher")
    checkpoint_identity = json.loads(identity_path.read_text())
    release = source_identity(root)
    if checkpoint_identity.get("source_identity") != release:
        raise ValueError("Teacher calibration source/runtime differs from checkpoint release")
    state_sha = sha256_file(state_path)
    if checkpoint_identity.get("training_state_sha256") != state_sha:
        raise ValueError("Teacher training-state checksum mismatch before loading")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("source_identity") != release:
        raise ValueError("Teacher training-state source/runtime differs from checkpoint release")
    if int(state.get("update", -1)) != 50000:
        raise ValueError("FH12 Teacher reference must be exact update 50000")
    config_sha = canonical_sha(cfg)
    if state.get("config_sha256") != config_sha:
        raise ValueError("Teacher checkpoint/config provenance mismatch")
    model_sha = sha256_file(weights)
    if state.get("model_sha256") != model_sha:
        raise ValueError("Teacher model bytes/training-state provenance mismatch")
    if checkpoint_identity.get("config_sha256") != config_sha or checkpoint_identity.get("model_sha256") != model_sha:
        raise ValueError("Teacher exact50K checkpoint identity mismatch")
    identity = {
        "teacher_run": str(teacher_run), "teacher_run_id": teacher_run.name,
        "teacher_checkpoint": str(weights), "teacher_checkpoint_sha": model_sha,
        "teacher_checkpoint_sha256": model_sha,
        "teacher_training_state": str(state_path), "teacher_training_state_sha256": state_sha,
        "teacher_config": str(config_path), "teacher_config_sha256": config_sha,
        "teacher_config_file_sha256": sha256_file(config_path),
        "teacher_checkpoint_identity": str(candidate / "identity.json"),
        "teacher_checkpoint_identity_sha256": sha256_file(candidate / "identity.json"),
        "teacher_layout": f["input_layout"], "teacher_update": 50000,
        "server": f["server_id"],
        "source_identity": release,
    }
    return cfg, identity


def _write_npz_immutable(path, arrays):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError(f"Refusing to replace different q cache: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".q_cache-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp, 0o444)
        os.link(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def calibrate(teacher_run, root: Path, server: str, device="cuda", deadline_utc=None, batch_size=64):
    """Build an immutable, local exact50K reference or verify and reuse it."""
    root = Path(root).resolve()
    check_deadline(deadline_utc)
    cfg, identity = _teacher_identity(teacher_run, root, server)
    asset_dir = root / "work_dir" / "_fh12" / server
    data_path = asset_dir / "dataset_manifest.json"
    lp_manifest_path = asset_dir / "lpan_manifest.json"
    data = json.loads(data_path.read_text())
    ref_dir = asset_dir / "references" / identity["teacher_run_id"]
    ref_path = ref_dir / "reference_manifest.json"
    if ref_path.exists():
        load_reference(ref_path, expected_teacher_run=identity["teacher_run"], data_manifest=data, root=root)
        return ref_path
    dataset = build_dataset(data, "train")
    if len(dataset) < 3072:
        raise ValueError("FH12 production calibration requires 3072 distinct train base patches")
    indices = np.random.default_rng(1234).choice(len(dataset), 3072, replace=False)
    from fh12.model import build_model
    ma = cfg["model_args"]
    model, _ = build_model(layout=identity["teacher_layout"], width=ma["hidden_size"],
                           depth=ma["depth"], seed=cfg["seed"], role="T")
    model.load_state_dict(load_file(identity["teacher_checkpoint"], device="cpu"), strict=True)
    model = model.to(device)
    calibration, arrays = compute_calibration(model, dataset, indices, device=device,
                                              batch_size=batch_size, deadline_utc=deadline_utc)
    check_deadline(deadline_utc)
    # Ensure neither the Teacher nor original data changed during computation.
    _, checked_identity = _teacher_identity(teacher_run, root, server)
    if checked_identity != identity or sha256_file(dataset.raw_h5_path) != data["splits"]["train"]["sha256"]:
        raise ValueError("Teacher or train source changed during calibration")
    q_path = ref_dir / "q_cache.npz"
    _write_npz_immutable(q_path, arrays)
    cal_path = ref_dir / "calibration.json"
    calibration.update({"schema": "FH12_CALIBRATION_v1", "calibration_seed": 1234, **identity})
    write_immutable_json(cal_path, calibration)
    manifest = {
        "schema": "FH12_REFERENCE_v1", **identity,
        "tau_R": calibration["tau_R"], "q_ref": calibration["q_ref"],
        "calibration_path": str(cal_path), "calibration_sha256": sha256_file(cal_path),
        "q_cache_path": str(q_path), "q_cache_sha256": sha256_file(q_path),
        "q_shape": calibration["q_shape"],
        "dataset_manifest_path": str(data_path), "dataset_manifest_sha256": sha256_file(data_path),
        "lpan_manifest_path": str(lp_manifest_path), "lpan_manifest_sha256": sha256_file(lp_manifest_path),
        "train_sha256": data["splits"]["train"]["sha256"],
        "train_lpan_sha256": data["splits"]["train"]["lpan_sha256"],
        "train_sample_order_sha256": data["splits"]["train"]["sample_order_sha256"],
        "LP_recipe": RECIPE, "augmentation": AUGMENTATION,
        "augmentation_sha256": canonical_sha(AUGMENTATION),
        "calibration_indices_sha256": calibration["calibration_indices_sha256"],
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": str(device)},
    }
    write_immutable_json(ref_path, manifest)
    return ref_path


def load_reference(path, expected_teacher_run=None, data_manifest=None, root=ROOT):
    """Validate provenance before a Student can consume tau/q or clone its A."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "FH12_REFERENCE_v1" or manifest.get("teacher_update") != 50000:
        raise ValueError("Not a FH12 exact50K reference")
    if manifest.get("source_identity") != source_identity(root):
        raise ValueError("FH12 reference source/runtime differs from current release")
    if expected_teacher_run is not None:
        wanted = str(expected_teacher_run)
        if wanted not in (manifest["teacher_run"], manifest["teacher_run_id"]):
            raise ValueError("Student points at a different Teacher reference")
    pairs = [("teacher_checkpoint", "teacher_checkpoint_sha256"),
             ("teacher_training_state", "teacher_training_state_sha256"),
             ("teacher_config", "teacher_config_file_sha256"),
             ("teacher_checkpoint_identity", "teacher_checkpoint_identity_sha256"),
             ("calibration_path", "calibration_sha256"), ("q_cache_path", "q_cache_sha256"),
             ("dataset_manifest_path", "dataset_manifest_sha256"), ("lpan_manifest_path", "lpan_manifest_sha256")]
    for file_key, sha_key in pairs:
        if sha256_file(manifest[file_key]) != manifest[sha_key]:
            raise ValueError(f"FH12 reference {file_key} changed")
    if canonical_sha(yaml.safe_load(Path(manifest["teacher_config"]).read_text())) != manifest["teacher_config_sha256"]:
        raise ValueError("Teacher resolved config content identity changed")
    calibration = json.loads(Path(manifest["calibration_path"]).read_text())
    for key in ("tau_R", "q_ref", "teacher_checkpoint_sha256", "teacher_config_sha256",
                "teacher_layout", "teacher_update", "calibration_indices_sha256", "q_shape", "source_identity"):
        if calibration[key] != manifest[key]:
            raise ValueError(f"Reference manifest differs from calibrated {key}")
    if manifest["augmentation_sha256"] != canonical_sha(AUGMENTATION) or manifest["LP_recipe"] != RECIPE:
        raise ValueError("FH12 augmentation/LP recipe changed after calibration")
    if data_manifest is None:
        data_manifest = json.loads(Path(manifest["dataset_manifest_path"]).read_text())
    if isinstance(data_manifest, (str, Path)):
        data_manifest = json.loads(Path(data_manifest).read_text())
    item = data_manifest["splits"]["train"]
    for key, expected in (("sha256", "train_sha256"), ("lpan_sha256", "train_lpan_sha256"),
                          ("sample_order_sha256", "train_sample_order_sha256")):
        if item[key] != manifest[expected]:
            raise ValueError("Student training-data identity differs from calibrated Teacher")
    if sha256_file(item["dataroot"]) != item["sha256"] or sha256_file(item["lpan_path"]) != item["lpan_sha256"]:
        raise ValueError("Training H5/LP content changed after calibration")
    with np.load(manifest["q_cache_path"], allow_pickle=False) as cache:
        q = cache["q"].copy()
        indices = cache["calibration_indices"]
        if (indices.ndim != 1 or len(indices) == 0 or len(np.unique(indices)) != len(indices)
                or indices.min() < 0 or indices.max() >= item["count"]):
            raise ValueError("Invalid cached calibration base-patch indices")
        if canonical_sha(indices.tolist()) != manifest["calibration_indices_sha256"]:
            raise ValueError("Calibration subset/cache mismatch")
    if list(q.shape) != manifest["q_shape"] or q.shape != (item["count"], 4):
        raise ValueError("FH12 q cache must cover every training patch and four rotations")
    if not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("Invalid raw q cache values")
    if not np.isfinite(manifest["tau_R"]) or manifest["tau_R"] < 1e-6 or not np.isfinite(manifest["q_ref"]) or manifest["q_ref"] <= 0:
        raise ValueError("Invalid tau_R or q_ref (no silent epsilon repair)")
    if float(np.median(q[indices])) != manifest["q_ref"]:
        raise ValueError("q_ref is not the median of the declared calibration subset/views")
    return manifest, q
