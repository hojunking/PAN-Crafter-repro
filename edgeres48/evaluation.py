"""Paused native full20 evaluation and target-only artifact retention.

Official metric formulas remain the audited ta2 evaluator, including original
PAN/native LMS FR references, unmasked full512, and all20 RR dim_cut21. No model
or prediction files are needed to evaluate a failing snapshot.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from ta2.evaluation import (JQM_VARIANT, _save_array, atomic_json, digest,
                            evaluator_identity, file_sha, fr_scene, rr_scene)
from panda_method.calibration import canonical_sha256, state_dict_sha256
from .plan import METHOD_CONTRACT, qualify_evaluation, validate_method_config


def model_state_hash(stage):
    """A/T tensor identity only, distinct from checkpoint-file SHA or RNG."""
    return canonical_sha256({"A": state_dict_sha256(stage.aligner),
                             "T": state_dict_sha256(stage.teacher)})


def expected_ids(sources, split):
    if split not in ("rr", "fr"):
        raise ValueError("Official evaluation uses RR/FR only")
    return tuple(f"{sources.sensor}:{split}:{i}" for i in range(20))


def source_manifest(sources):
    return {"sensor": sources.sensor, "registry_path": str(sources.catalog_path),
            "registry_sha256": sources.catalog_sha256, "registered_sources": sources.entry,
            "lms_overrides": sources.lms_overrides,
            "split_manifest_sha256": {s: sources.manifest_sha(s) for s in ("train", "val", "rr", "fr")},
            "scene_mapping_required": False, "prior_asset_required": False}


def _optimizer_signature(optimizer):
    if optimizer is None:
        return None
    versions = []
    for parameter, state in optimizer.state.items():
        for name, value in sorted(state.items()):
            versions.append((id(parameter), name, id(value), value._version,
                             float(value) if value.numel() == 1 else None)
                            if isinstance(value, torch.Tensor) else (id(parameter), name, repr(value)))
    groups = [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups]
    return versions, repr(groups)


@contextmanager
def paused_model(trainer, *, gradients=False):
    """Preserve mode/RNG/gradients; never step, save or alter live model state."""
    if getattr(trainer, "poisoned", False):
        raise RuntimeError("Cannot evaluate a failed optimizer state")
    stage = trainer.stage
    state, updates = model_state_hash(stage), trainer.updates
    policy = digest(stage.pipeline_manifest())
    modes = [(m, m.training) for m in stage.modules()]
    grads = [(p, None if p.grad is None else p.grad.detach().clone()) for p in stage.parameters()]
    extra = copy.deepcopy(stage.get_extra_state())
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    optimizer_before = _optimizer_signature(getattr(trainer, "optimizer", None))
    try:
        stage.eval()
        with torch.set_grad_enabled(gradients):
            yield state
    finally:
        for m, mode in modes:
            m.training = mode
        for p, grad in grads:
            p.grad = grad
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.random.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        try:
            stage.set_extra_state(extra)
        except Exception as exc:
            trainer.poisoned = True
            raise RuntimeError("Inference policy/state changed during paused evaluation") from exc
        if (model_state_hash(stage) != state or trainer.updates != updates
                or digest(stage.pipeline_manifest()) != policy
                or _optimizer_signature(getattr(trainer, "optimizer", None)) != optimizer_before):
            trainer.poisoned = True
            raise RuntimeError("A/T, optimizer, update or inference policy changed during paused evaluation")


def identity(trainer, config, sources, evaluator_revision, evaluator_pin):
    validate_method_config(config)
    pipeline = trainer.stage.pipeline_manifest()
    if digest(trainer.stage.matcher.manifest()) != config["edge_protocol_digest"]:
        raise ValueError("Live EdgeMatch policy differs from the predeclared campaign config")
    return {**METHOD_CONTRACT, "run_id": config["run_id"], "attempt": config["attempt"],
            "seed": config["seed"], "completed_update": trainer.updates,
            "model_state_sha256": model_state_hash(trainer.stage),
            "inference_policy_digest": digest({"pipeline": pipeline, "eval": evaluator_pin,
                 "references": "original_PAN/native_LMS-full_frame-no-mask",
                 "rr_source": sources.manifest_sha("rr"), "fr_source": sources.manifest_sha("fr")}),
            "evaluator_revision": evaluator_revision,
            "edge_protocol_digest": digest(trainer.stage.matcher.manifest())}


evaluation_context = paused_model


def metric_functions_for(sources, evaluator_pin):
    """Only the existing evaluator; no formulas copied into this campaign."""
    if evaluator_identity() != evaluator_pin:
        raise ValueError("Audited official evaluator differs from admission pin")
    from tools.metrics.eval_fr import load_dlpan
    root = Path(__file__).resolve().parents[1]
    wald = load_dlpan(os.environ.get("PANCRAFTER_DLPAN", str(root.parent / "DLPan-Toolbox")))
    return {"rr": lambda p, a: rr_scene(p, a["gt"], sources.sensor),
            "fr": lambda p, a: fr_scene(p, a["pan"], a["lms"], a["ms"], sources.sensor, wald)}


def metric_values(metrics, split, bands):
    keys = (("ergas", "scc", "sam", "psnr", "ssim", "q8" if bands == 8 else "q4", "rmse", "cc")
            if split == "rr" else ("hqnr", "d_s", "d_lambda", "jqm"))
    renamed = {"d_s": "D_s", "d_lambda": "D_lambda"}
    values = {renamed.get(k, k.upper()): float(metrics[k]) for k in keys}
    if not np.isfinite(list(values.values())).all():
        raise FloatingPointError("Nonfinite official metric; no zero filling or skipped images")
    return values


def official_evaluation(trainer, config, sources, *, device, evaluator_pin,
                        evaluator_revision=None, metric_functions=None,
                        on_scene=None, deadline_unix=None):
    """Evaluate RR20+FR20 on one live state, without creating any artifacts.

    Predictions are released after each scene. Qualified retention recomputes
    them AFTER saving/validating the qualified checkpoint, then verifies their
    hashes. Empty raw_outputs preserves the controller's two-return interface.
    ``metric_functions`` injection is explicitly test-only.
    """
    if evaluator_revision is None:
        evaluator_revision = digest(evaluator_pin)
    injected = metric_functions is not None
    if injected and evaluator_revision != "TEST":
        raise ValueError("Injected metrics are allowed only in synthetic TEST evaluations")
    metric_functions = metric_functions or metric_functions_for(sources, evaluator_pin)
    started = time.monotonic()
    record = {"record_type": "EVAL_SUMMARY", "full_method_eligible": not config.get("disposable_preflight", False),
              "rr": {}, "fr": {}, "test_aware": True, "jqm_variant": JQM_VARIANT,
              "eval_protocol_id": digest(evaluator_pin), "metric_function_injection": injected,
              "official_reference": "original_PAN/native_LMS; FR full512 unmasked; RR all20 dim_cut21",
              "measurement_domain": "OFFICIAL_FULL20_NATIVE", "weights_saved": False,
              "checkpoint_path": None, "weights_file_sha256": None,
              "inference_pipeline": copy.deepcopy(trainer.stage.pipeline_manifest())}
    with paused_model(trainer):
        record["state_before"] = identity(trainer, config, sources, evaluator_revision, evaluator_pin)
        for split in ("rr", "fr"):
            rows, tick = [], time.monotonic()
            for i, sample_id in enumerate(expected_ids(sources, split)):
                if deadline_unix is not None and time.time() >= deadline_unix:
                    raise TimeoutError("EVAL_INCOMPLETE: campaign deadline before full RR/FR evaluation")
                batch = sources.batch(split, [i], device=device)
                if tuple(batch.lineage.sample_ids) != (sample_id,):
                    raise ValueError("Official H5 row and registered evaluation image identity differ")
                output = trainer.stage.native_prediction(batch)
                raw = ((output["y"][0].float() + 1) * (sources.peak / 2)).cpu().numpy()
                arrays = {k: v[0] for k, v in sources.raw(split, [i]).items()}
                metrics = metric_functions[split](raw, arrays)
                diag = output["match_diagnostics"]
                row = {"scene_id": sample_id, "evaluation_image_id": sample_id, "sample_id": sample_id,
                       "metrics": metric_values(metrics, split, sources.bands),
                       "correction_dy_dx": output["u"][0].cpu().tolist(),
                       "b_dy_dx": output["b"][0].cpu().tolist(), "delta_dy_dx": output["delta"][0].cpu().tolist(),
                       "raw_prediction_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                       "raw_min": float(raw.min()), "raw_max": float(raw.max()),
                       "jqm_weights": np.asarray(metrics.get("jqm_weights", [])).tolist(),
                       "jqm_weight_source": metrics.get("jqm_weight_source"),
                       "edge_boundary_hit": bool(diag["boundary_hit"][0]),
                       "edge_tie_count": int(diag["tie_count"][0])}
                rows.append(row)
                if on_scene is not None:
                    on_scene({"branch": split, "identity": record["state_before"], **row})
                del raw, output, arrays, batch
            record[split] = {"per_scene": rows,
                "metrics": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in rows[0]["metrics"]},
                "std": {k: float(np.std([r["metrics"][k] for r in rows], ddof=1)) for k in rows[0]["metrics"]},
                "elapsed_seconds": time.monotonic() - tick}
            record[split + "_state"] = identity(trainer, config, sources, evaluator_revision, evaluator_pin)
        record["state_after"] = identity(trainer, config, sources, evaluator_revision, evaluator_pin)
    if not injected and evaluator_identity() != evaluator_pin:
        raise ValueError("Official evaluator changed during evaluation")
    record["elapsed_seconds"] = time.monotonic() - started
    record["qualification"] = qualify_evaluation(record, server=config["server"],
        expected_rr_ids=expected_ids(sources, "rr"), expected_fr_ids=expected_ids(sources, "fr"))
    print(f"[EdgeRes48] step={trainer.updates} HQNR={record['fr']['metrics']['HQNR']:.6f} "
          f"SCC={record['rr']['metrics']['SCC']:.6f} ERGAS={record['rr']['metrics']['ERGAS']:.6f}", flush=True)
    return record, {"rr": [], "fr": []}


def _atomic_checkpoint(path, payload):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Never overwrite an existing qualified checkpoint")
    fd, temp = tempfile.mkstemp(prefix=".qualified-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
        loaded = torch.load(temp, map_location="cpu", weights_only=False)
        for role in ("aligner", "teacher"):
            if set(loaded[role]) != set(payload[role]):
                raise ValueError("Qualified checkpoint key roundtrip failed")
            for key, value in payload[role].items():
                if not torch.equal(loaded[role][key].cpu(), value.cpu()):
                    raise ValueError("Qualified checkpoint tensor roundtrip failed")
        os.link(temp, path)
    finally:
        os.unlink(temp)
    return file_sha(path)


def export_qualified(path, trainer, sources, evaluation, settings, *, device, deadline_unix=None):
    """Recompute all40 native predictions only after target-qualified save."""
    from PIL import Image
    path, files = Path(path), {}
    rgb = settings["rgb_indices"]
    if (tuple(rgb) != tuple(trainer.stage.matcher.edges.rgb_indices)
            or settings["display_range_DN"] != [0, sources.peak] or settings["roi"] != "FULL_SCENE"):
        raise ValueError("Qualified display must use pinned sensor RGB/full scene/fixed DN range")
    def png(name, array):
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".qualified-png-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(handle, format="PNG")
                handle.flush(); os.fsync(handle.fileno())
            if target.exists():
                if file_sha(temporary) != file_sha(target):
                    raise ValueError("Existing qualified display differs; refusing overwrite")
            else:
                os.link(temporary, target)
        finally:
            os.unlink(temporary)
        files[name] = file_sha(target)
    with paused_model(trainer):
        for split in ("rr", "fr"):
            for i, row in enumerate(evaluation[split]["per_scene"]):
                if deadline_unix is not None and time.time() >= deadline_unix:
                    raise TimeoutError("Qualified export deadline reached; keep QUAL_PENDING")
                batch = sources.batch(split, [i], device=device)
                output = trainer.stage.native_prediction(batch)
                raw = ((output["y"][0].float() + 1) * (sources.peak / 2)).cpu().numpy()
                if hashlib.sha256(raw.tobytes()).hexdigest() != row["raw_prediction_sha256"]:
                    raise ValueError("Qualified re-inference differs from evaluated snapshot")
                arrays = {k: v[0] for k, v in sources.raw(split, [i]).items()}
                prefix = f"{split}/image_{i:02d}"
                name = prefix + "/prediction_unclipped_DN.npy"
                files[name] = _save_array(path / name, raw)
                view = lambda a: a[list(rgb)].transpose(1, 2, 0) / sources.peak
                png(prefix + "/prediction.png", view(raw))
                reference = arrays["gt"] if split == "rr" else arrays["lms"]
                png(prefix + ("/GT.png" if split == "rr" else "/native_LMS.png"), view(reference))
                png(prefix + "/MS.png", view(arrays["ms"]))
                png(prefix + "/PAN.png", np.repeat((arrays["pan"][0] / sources.peak)[..., None], 3, axis=2))
                if split == "rr":
                    png(prefix + "/absolute_error.png", view(np.abs(raw-reference)) * settings["error_gain"])
    atomic_json(path / "files.json", files, immutable=True)
    return files


def retain_qualified(root, trainer, config, sources, evaluation, raw_outputs=None, *,
                     stream_states, provenance, device=None, deadline_unix=None):
    """Call only AFTER durable numeric evaluation journaling; no failed writes."""
    validate_method_config(config)
    gate = qualify_evaluation(evaluation, server=config["server"],
        expected_rr_ids=expected_ids(sources, "rr"), expected_fr_ids=expected_ids(sources, "fr"))
    if not gate["joint_pass"]:
        raise ValueError("STRICT_TARGET_ONLY: failed/incomplete/DIAG_ONLY weights and images cannot be stored")
    ident = evaluation["state_after"]
    if (model_state_hash(trainer.stage) != ident["model_state_sha256"]
            or trainer.updates != ident["completed_update"]
            or any(config[key] != ident[key] for key in ("run_id", "attempt", "seed"))
            or digest(trainer.stage.pipeline_manifest()) != digest(evaluation["inference_pipeline"])
            or digest(trainer.stage.matcher.manifest()) != ident["edge_protocol_digest"]):
        raise ValueError("Live state moved after same-state native evaluation")
    directory = Path(root) / config["run_id"] / f"attempt_{config['attempt']}" / f"step_{trainer.updates}" / ident["model_state_sha256"]
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"aligner": trainer.stage.aligner.state_dict(), "teacher": trainer.stage.teacher.state_dict(),
               "training_state": trainer.state_dict(), "data_stream_states": stream_states,
               "config": config, "identity": ident, "provenance": provenance,
               "pipeline": trainer.stage.pipeline_manifest()}
    sha = _atomic_checkpoint(directory / "teacher_aligner.pt", payload)
    atomic_json(directory / "source_manifest.json", source_manifest(sources), immutable=True)
    atomic_json(directory / "metrics.json", evaluation, immutable=True)
    manifest = {"weights_saved": True, "file_sha256": sha, "model_state_sha256": ident["model_state_sha256"],
                "identity": ident, "qualification": gate, "provenance": provenance, "config": config,
                "status": "QUAL_PENDING", "checkpoint_path": str(directory / "teacher_aligner.pt")}
    atomic_json(directory / "artifact_manifest.json", manifest)
    if device is None:
        device = next(trainer.stage.parameters()).device
    try:
        files = export_qualified(directory / "qualitative", trainer, sources, evaluation,
                                 config["baseline_resolved"]["qual_export"], device=device,
                                 deadline_unix=deadline_unix)
        manifest.update(status="QUAL_COMPLETE", qualitative_files=files)
    except Exception as exc:
        manifest.update(status="QUAL_PENDING", export_error=f"{type(exc).__name__}: {exc}")
    atomic_json(directory / "artifact_manifest.json", manifest)
    return {"artifact_path": str(directory), "checkpoint_file_sha256": sha, **manifest}


def retry_qualified(path, sources, build_trainer, *, device, deadline_unix=None):
    """Retry QUAL_PENDING using ONLY an already-qualified authenticated file.

    ``build_trainer(saved_config)`` must return a fresh compatible trainer on
    ``device``. No live training model is mutated and no failed snapshot can be
    reconstructed through this entrypoint. Exact source/policy/output hashes
    are rechecked before the incomplete export is marked complete.
    """
    directory = Path(path).resolve(strict=True)
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    evaluation = json.loads((directory / "metrics.json").read_text())
    source_pin = json.loads((directory / "source_manifest.json").read_text())
    checkpoint = directory / "teacher_aligner.pt"
    if (manifest.get("weights_saved") is not True
            or manifest.get("status") not in ("QUAL_PENDING", "QUAL_COMPLETE")
            or file_sha(checkpoint) != manifest.get("file_sha256")):
        raise ValueError("Only intact previously qualified checkpoint files may be retried")
    config = manifest["config"]
    validate_method_config(config)
    gate = qualify_evaluation(evaluation, server=config["server"],
        expected_rr_ids=expected_ids(sources, "rr"), expected_fr_ids=expected_ids(sources, "fr"))
    if not gate["joint_pass"] or source_pin != source_manifest(sources):
        raise ValueError("Qualified evaluation/source identity no longer verifies")
    # The exact local serializer's authenticated file was checked above; this
    # is not an arbitrary supplied pickle or a path to any unqualified state.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ident = evaluation["state_after"]
    if (digest(payload.get("identity")) != digest(ident) or digest(manifest.get("identity")) != digest(ident)
            or digest(payload.get("config")) != digest(config)
            or digest(payload.get("pipeline")) != digest(evaluation["inference_pipeline"])):
        raise ValueError("Saved checkpoint/evaluation/policy identities disagree")
    trainer = build_trainer(copy.deepcopy(config))
    trainer.load_state_dict(payload["training_state"])
    if (model_state_hash(trainer.stage) != ident["model_state_sha256"]
            or trainer.updates != ident["completed_update"]
            or trainer.stage.pipeline_manifest() != payload["pipeline"]):
        raise ValueError("Reloaded qualified trainer does not reproduce the saved model identity")
    try:
        files = export_qualified(directory / "qualitative", trainer, sources, evaluation,
                                 config["baseline_resolved"]["qual_export"], device=device,
                                 deadline_unix=deadline_unix)
        manifest.update(status="QUAL_COMPLETE", qualitative_files=files)
        manifest.pop("export_error", None)
    except Exception as exc:
        manifest.update(status="QUAL_PENDING", export_error=f"{type(exc).__name__}: {exc}")
    atomic_json(manifest_path, manifest)
    return {"artifact_path": str(directory), "checkpoint_file_sha256": manifest["file_sha256"], **manifest}
