"""Read-only direct-shift numeric probes on predeclared inputs and offsets.

Synthetic response is not physical registration accuracy. Nothing here updates
weights, consumes the training epsilon stream, writes images or qualifies a
checkpoint. All temporary modes/RNG/gradient state use the evaluator guard.
"""
from __future__ import annotations

import copy
import json
import math
import time

import numpy as np
import torch

from pa.warp import warp_pan, warp_support_mask
from ta2.evaluation import digest
from .evaluation import model_state_hash, paused_model
from .method import epsilon_loss, sample_epsilon


PROTOCOL = "DIRECTSHIFT48_FIXED_VIEW_NUMERIC_PROBES_V1"
RADII = (.25, .5, 1., 2., 3.)


def axis20():
    return [{"radius": radius, "axis": axis, "epsilon_dy_dx": [radius*y, radius*x],
             "offset_type": "integer" if radius in (1., 2., 3.) else "fractional"}
            for radius in RADII
            for axis, y, x in (("+y", 1, 0), ("-y", -1, 0), ("+x", 0, 1), ("-x", 0, -1))]


def response_statistics(rows):
    if not rows:
        raise ValueError("No response observations; do not fabricate zero metrics")
    eps = np.asarray([row["epsilon_dy_dx"] for row in rows], dtype=np.float64)
    du = np.asarray([row["du"] for row in rows], dtype=np.float64)
    if any(a.shape != (len(rows), 2) or not np.isfinite(a).all() for a in (eps, du)):
        raise ValueError("Finite paired Nx2 perturbations and direct responses required")
    denominator = float(np.square(eps).sum())
    if denominator <= 0:
        raise ValueError("Nonzero fixed perturbations required")
    error = du + eps
    return {"n": len(rows), "gain_denominator": denominator,
            "gain": float(-(du * eps).sum() / denominator),
            "EPE": float(np.linalg.norm(error, axis=1).mean()),
            "MAE": float(np.abs(error).mean()), "MAE_dy_dx": np.abs(error).mean(0).tolist()}


def response_summary(rows):
    return {"all": response_statistics(rows),
            "by_radius": {str(radius): response_statistics([r for r in rows if r["radius"] == radius])
                          for radius in RADII if any(r["radius"] == radius for r in rows)},
            "by_offset_type": {kind: response_statistics([r for r in rows if r["offset_type"] == kind])
                               for kind in ("integer", "fractional")
                               if any(r["offset_type"] == kind for r in rows)}}


def validate_probe_spec(spec):
    if not isinstance(spec, dict) or "rows" not in spec or "max_wall_seconds" not in spec:
        raise ValueError("Explicit fixed probe rows and wall budget required")
    if set(spec["rows"]) != {"train", "val", "rr", "fr"}:
        raise ValueError("Pin train/validation/RR/FR probe rows separately")
    for split, indices in spec["rows"].items():
        if (not isinstance(indices, list) or not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or i < 0 for i in indices)):
            raise ValueError("Unique nonnegative integer H5 rows required")
        if split in ("rr", "fr") and max(indices) >= 20:
            raise ValueError("Official test probes are fixed subsets of full20")
    budget = spec["max_wall_seconds"]
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("Finite positive diagnostic wall budget required")
    if "response_radii" in spec and tuple(spec["response_radii"]) != RADII:
        raise ValueError("Direct R3 response probes require .25/.5/1/2/3 radii")
    if "clean_cnn_view_margin" in spec and spec["clean_cnn_view_margin"] != 8:
        raise ValueError("Diagnostics must use the same native/epsilon input crop8")
    if "response_domains" in spec and spec["response_domains"] not in (["fixed_clean_input_view"], ["normal"]):
        raise ValueError("No alternate CNN view or matching decomposition in direct-shift probes")
    if "gradient_rows" in spec:
        micro = spec.get("gradient_microbatch")
        if type(micro) is not int or micro < 1 or type(spec.get("probe_seed")) is not int:
            raise ValueError("Explicit gradient microbatch and diagnostic epsilon seed required")
        groups = spec["gradient_rows"]
        if not isinstance(groups, list) or not groups:
            raise ValueError("Predeclared gradient row groups required")
        for group in groups:
            if (not isinstance(group, list) or not group or len(group) % micro
                    or len(set(group)) != len(group) or any(type(i) is not int or i < 0 for i in group)):
                raise ValueError("Unique fixed gradient RR rows divisible by microbatch required")
    return copy.deepcopy(spec)


def _check_deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("DIAGNOSTIC_BUDGET_EXHAUSTED")


def _geometry(stage, batch, native):
    u = native["u"]
    support = warp_support_mask(*batch.pan.shape[-2:], u)
    margin = stage.roi_margin - 1
    support_ok = bool(support[..., margin:-margin, margin:-margin].all())
    row = {"sample_id": batch.lineage.sample_ids[0], "u": u[0].cpu().tolist(),
           "warp_valid_fraction": float(support.float().mean()),
           "fixed_discrepancy_roi_margin": stage.roi_margin,
           "spatial_support_ok": support_ok, "discrepancy": {}}
    for name, shift in (("ZERO", torch.zeros_like(u)), ("CURRENT", u)):
        # If current correction violates the fixed domain, explicitly record
        # absence instead of shrinking the ROI or labeling extrapolation valid.
        if name == "CURRENT" and not support_ok:
            row["discrepancy"][name] = {"status": "INVALID_SUPPORT", "RGB_LMS": None,
                                         **({"RGB_GT": None} if batch.gt is not None else {})}
            continue
        corrected = warp_pan(batch.pan, shift)
        measures = {"status": "VALID_FIXED_SUPPORT", "RGB_LMS": float(stage.spatial.absolute_loss(
            corrected, batch.native_lms, roi_margin=stage.roi_margin))}
        if batch.gt is not None:
            measures["RGB_GT"] = float(stage.spatial.absolute_loss(corrected, batch.gt, roi_margin=stage.roi_margin))
        row["discrepancy"][name] = measures
    return row


def gradient_probe(trainer, sources, *, rows, microbatch, seed, device, deadline_unix=None):
    """Exact RR term gradients, diagnostic RNG, no optimizer or FR gradients."""
    stage = trainer.stage
    params = list(stage.aligner.parameters()) + list(stage.teacher.parameters())
    n_a = len(list(stage.aligner.parameters()))
    epsilon_rng = torch.Generator(device="cpu").manual_seed(seed)
    total = sum(len(group) for group in rows)
    accumulated = {key: [None]*len(params) for key in ("rec", "abs_rr", "eps_rr")}
    observed = []
    with paused_model(trainer, gradients=True):
        for group in rows:
            for start in range(0, len(group), microbatch):
                _check_deadline(deadline_unix)
                indices = group[start:start+microbatch]
                rr = sources.batch("train", indices, device=device)
                rr.validate(stage.teacher.spec.bands, training=True, require_gt=True)
                stage.sources.check(rr)
                native = stage.native_prediction(rr)
                stage._native_support(rr.pan, native["u"])
                rec = (native["y"]-rr.gt.detach()).abs().mean()
                absolute = stage.spatial.absolute_loss(native["pan_aligned"], rr.gt.detach(),
                                                       roi_margin=stage.roi_margin)
                with torch.no_grad():
                    eps = sample_epsilon(len(rr.pan), epsilon_rng).to(rr.pan.device)
                    pan_eps = warp_pan(rr.pan, eps)
                    support = warp_support_mask(*rr.pan.shape[-2:], eps)
                shifted = stage.predict_shift(pan_eps, rr.native_lms, resolution="RR", pan_valid_mask=support)
                terms = {"rec": rec, "abs_rr": absolute, "eps_rr": epsilon_loss(shifted["u"], native["u"], eps)}
                for name, loss in terms.items():
                    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    for i, grad in enumerate(grads):
                        if grad is not None:
                            value = grad.detach() * (len(indices)/total)
                            if accumulated[name][i] is None:
                                accumulated[name][i] = value.clone()
                            else:
                                accumulated[name][i].add_(value)
                observed.extend(rr.lineage.sample_ids)
    accumulated["weighted_abs"] = [None if g is None else g * stage.lambda_abs for g in accumulated["abs_rr"]]
    accumulated["weighted_eps_rr"] = [None if g is None else g * stage.lambda_eps for g in accumulated["eps_rr"]]
    return {"status": "COMPLETE", "train_sample_ids": observed, "n_inputs": total,
            "optimizer_updates": 0, "uses_evaluation_inputs": False,
            "epsilon_seed": seed, "gradient_parameter_state_update": trainer.updates,
            "loss_coefficients": [1., stage.lambda_abs, stage.lambda_eps],
            "term_gradient_norm": {name: trainer._gradient_summary(grads, n_a)
                                   for name, grads in accumulated.items()},
            "initialization_note": "zero FC2 blocks initial upstream A gradients; zero Teacher output initially blocks rec-to-A"}


def run_probes(trainer, sources, probe_spec, *, device="cpu", evaluator_pin=None,
               deadline_unix=None):
    spec = validate_probe_spec(probe_spec)
    started = time.monotonic()
    deadline = min(time.time()+spec["max_wall_seconds"], deadline_unix if deadline_unix is not None else float("inf"))
    stage = trainer.stage
    result = {"protocol": PROTOCOL, "record_type": "DIAGNOSTICS_NUMERIC", "status": "RUNNING",
              "completed_update": trainer.updates, "model_state_sha256": model_state_hash(stage),
              "fixed_probe_spec": spec, "measurement_domain": "DIAG_ONLY_FIXED_CLEAN_VIEW",
              "official_selection_input": False, "full_method_eligible": False,
              "images_written": False, "optimizer_updates": 0,
              "physical_displacement_ground_truth": False,
              "alignment_protocol_digest": digest(stage.alignment_manifest()),
              "source_manifest_sha256": {split: sources.manifest_sha(split) for split in spec["rows"]},
              "metric_evaluator_invoked": False, "evaluator_pin": evaluator_pin,
              "response": {}, "geometry": {}, "completed_input_count": 0,
              "expected_input_count": sum(len(rows) for rows in spec["rows"].values()),
              "n_offsets_per_input": 20}
    try:
        if "gradient_rows" in spec:
            result["gradient"] = gradient_probe(trainer, sources, rows=spec["gradient_rows"],
                microbatch=spec["gradient_microbatch"], seed=spec["probe_seed"], device=device,
                deadline_unix=deadline)
        with paused_model(trainer):
            for split, indices in spec["rows"].items():
                responses, geometry = [], []
                result["response"][split] = {"rows": responses, "summary": None}
                result["geometry"][split] = {"rows": geometry}
                for index in indices:
                    _check_deadline(deadline)
                    batch = sources.batch(split, [index], device=device)
                    native = stage.predict_shift(batch.pan, batch.native_lms, resolution=batch.lineage.resolution)
                    geometry.append(_geometry(stage, batch, native))
                    for point in axis20():
                        _check_deadline(deadline)
                        epsilon = batch.pan.new_tensor(point["epsilon_dy_dx"])[None]
                        pan_eps = warp_pan(batch.pan, epsilon)
                        support = warp_support_mask(*batch.pan.shape[-2:], epsilon)
                        shifted = stage.predict_shift(pan_eps, batch.native_lms,
                            resolution=batch.lineage.resolution, pan_valid_mask=support)
                        responses.append({**point, "sample_id": batch.lineage.sample_ids[0],
                            "du": (shifted["u"]-native["u"])[0].cpu().tolist(),
                            "measurement_domain": "DIAG_ONLY_FIXED_CLEAN_VIEW",
                            "synthetic_warp_valid_fraction": float(support.float().mean()),
                            "fixed_view_support_ok": True})
                    result["completed_input_count"] += 1
                result["response"][split]["summary"] = response_summary(responses)
        _check_deadline(deadline)
        result["status"] = "COMPLETE"
    except TimeoutError as exc:
        result.update(status="INCOMPLETE", incomplete_reason=str(exc))
    result["elapsed_seconds"] = time.monotonic() - started
    result["model_state_sha256_after"] = model_state_hash(stage)
    if result["model_state_sha256_after"] != result["model_state_sha256"]:
        raise RuntimeError("Diagnostic changed model weights")
    json.dumps(result, allow_nan=False)
    return result


def benchmark_inference(trainer, sources, *, device, warmup, repeats, rows, deadline_unix=None):
    """Batch1 resident-input full-pipeline timing, not sum of component timings."""
    if (type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1
            or set(rows) != {"rr", "fr"} or any(type(v) is not int or not 0 <= v < 20 for v in rows.values())):
        raise ValueError("Pin warmup, positive repeats and one official row per grid")
    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
    def measure(call):
        for _ in range(warmup):
            _check_deadline(deadline_unix)
            call()
        values = []
        for _ in range(repeats):
            _check_deadline(deadline_unix)
            sync()
            tick = time.perf_counter()
            call()
            sync()
            values.append((time.perf_counter()-tick)*1000)
        return {"median_ms": float(np.median(values)), "mean_ms": float(np.mean(values)),
                "p90_ms": float(np.quantile(values, .9)), "samples_ms": values}
    stage, measurements = trainer.stage, {}
    result = {"record_type": "COST", "status": "RUNNING", "measurement_domain": "DIAG_ONLY_NATIVE_INFERENCE_TIMING",
        "model_state_sha256": model_state_hash(stage), "device": str(device),
        "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
        "precision": "FP32", "warmup": warmup, "repeats": repeats, "batch": 1,
        "cache": "none; direct Aligner recomputed every full-pipeline call",
        "cold_start_claim": "none; first_call_in_probe is not a fresh process/device cold start",
        "Infer_ms_scope": "resident stored-LMS input full pipeline; acquisition+H2D separate",
        "observations": measurements, "optimizer_updates": 0,
        "trainable_params": sum(p.numel() for p in stage.parameters() if p.requires_grad),
        "aligner_params": sum(p.numel() for p in stage.aligner.parameters()),
        "teacher_params": sum(p.numel() for p in stage.teacher.parameters()),
        "Mem_MB": None, "Mem_reason": "isolated allocation not measured during paused training",
        "FLOPs": None, "FLOPs_reason": "not measured; no copied architecture estimate"}
    try:
        with paused_model(trainer):
            for split, row in rows.items():
                _check_deadline(deadline_unix)
                sync()
                tick = time.perf_counter()
                batch = sources.batch(split, [row], device=device)
                sync()
                load_ms = (time.perf_counter()-tick)*1000
                tick = time.perf_counter()
                native = stage.native_prediction(batch)
                sync()
                first_ms = (time.perf_counter()-tick)*1000
                measurements[split] = {"sample_id": batch.lineage.sample_ids[0], "shape": list(batch.pan.shape),
                    "observation_read_normalize_H2D_ms": load_ms, "first_call_in_probe_ms": first_ms,
                    "full_pipeline": measure(lambda: stage.native_prediction(batch)),
                    "A_only_prepared_inputs": measure(lambda: stage.aligner(batch.pan, batch.native_lms)),
                    "T_only_prepared_pan": measure(lambda: stage.teacher(native["pan_aligned"], batch.ms, batch.native_lms)),
                    "final_warp_only": measure(lambda: warp_pan(batch.pan, native["u"]))}
        _check_deadline(deadline_unix)
        result["status"] = "COMPLETE"
    except TimeoutError as exc:
        result.update(status="INCOMPLETE", incomplete_reason=str(exc))
    result["model_state_sha256_after"] = model_state_hash(stage)
    json.dumps(result, allow_nan=False)
    return result
