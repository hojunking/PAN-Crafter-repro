"""Fixed, numeric-only Stage-1 probes; never official selection or training.

The frozen inference interventions are not retrained ablations.  The spatial
discrepancy shares the training descriptor and is explicitly not independent
geometric ground truth.  The only autograd probes use registered *training*
observations; evaluation observations are never sent to an objective/backward.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import time

import torch

from pa.offset import sample_offsets
from pa.warp import warp_pan, warp_support_mask
from panda_method.reconstruction import aligned_observations
from .plan import PROBE_SEED


PROTOCOL = "TEACHER48_FIXED_NUMERIC_PROBES_V1"
RADII = (0.25, 0.5, 1., 2.)
AXES = (("+y", (1., 0.)), ("-y", (-1., 0.)),
        ("+x", (0., 1.)), ("-x", (0., -1.)))


def axis16():
    """Sixteen fixed PAN-pixel perturbations, component order (dy,dx)."""
    return [{"radius": radius, "axis": axis,
             "epsilon_dy_dx": [radius * d[0], radius * d[1]]}
            for radius in RADII for axis, d in AXES]


def response_statistics(rows):
    """Gain= -sum(delta*eps)/sum(eps²), ideal response delta=-eps."""
    if not rows:
        raise ValueError("At least one response row required")
    delta = torch.tensor([r["delta_dy_dx"] for r in rows], dtype=torch.float64)
    epsilon = torch.tensor([r["epsilon_dy_dx"] for r in rows], dtype=torch.float64)
    if delta.shape != epsilon.shape or delta.ndim != 2 or delta.shape[1] != 2:
        raise ValueError("Response and epsilon must be equal Nx2 tensors")
    if not bool(torch.isfinite(delta).all() and torch.isfinite(epsilon).all()):
        raise FloatingPointError("Nonfinite probe response")
    residual = delta + epsilon
    numerator, denominator = float(-(delta * epsilon).sum()), float(epsilon.square().sum())
    output = {"n": len(rows), "gain": numerator / denominator if denominator > 0 else None,
              "gain_numerator": numerator, "gain_denominator": denominator,
              "component_mae": float(residual.abs().mean()),
              "epe": float(torch.linalg.vector_norm(residual, dim=1).mean())}
    for i, axis in enumerate(("y", "x")):
        den = float(epsilon[:, i].square().sum())
        output[f"gain_{axis}"] = float(-(delta[:, i] * epsilon[:, i]).sum()) / den if den > 0 else None
        output[f"mae_{axis}"] = float(residual[:, i].abs().mean())
    return output


def _indices(sources, group, requested, *, field):
    rows = sources.samples[group]
    available = {row[field]: i for i, row in enumerate(rows)}
    if len(available) != len(rows):
        raise ValueError(f"Duplicate {field} in {group}")
    if not requested or len(set(requested)) != len(requested):
        raise ValueError(f"Explicit nonempty unique {group} probe IDs required")
    missing = set(requested) - available.keys()
    if missing:
        raise ValueError(f"Unregistered fixed {group} probe IDs: {sorted(missing)}")
    return [available[item] for item in requested]


def validate_probe_spec(spec, sources):
    """Pin IDs and a per-branch sample cap before observing probe outcomes."""
    required = ("rr_eval_ids", "fr_eval_ids", "rr_train_sample_id", "fr_train_sample_id", "max_samples")
    if not isinstance(spec, dict) or any(key not in spec for key in required):
        raise ValueError("Explicit fixed probe IDs and max_samples are required")
    if type(spec["max_samples"]) is not int or spec["max_samples"] <= 0:
        raise ValueError("max_samples must be a pinned positive per-branch count")
    indices = {}
    for branch in ("rr", "fr"):
        ids = spec[f"{branch}_eval_ids"]
        if (not isinstance(ids, (tuple, list)) or not all(isinstance(v, str) and v for v in ids)
                or len(ids) > spec["max_samples"]):
            raise ValueError("Evaluation probe IDs exceed the explicit budget or are invalid")
        indices[f"{branch}_eval"] = _indices(sources, f"{branch}_eval", ids, field="scene_id")
        sample_id = spec[f"{branch}_train_sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Explicit actual training probe sample ID required")
        indices[f"{branch}_train"] = _indices(sources, f"{branch}_train", [sample_id], field="sample_id")
    return indices


def _summary(rows):
    return {"all_radii": response_statistics(rows),
            "by_radius": [{"radius": radius,
                           **response_statistics([r for r in rows if r["radius"] == radius])}
                          for radius in RADII],
            "by_axis": [{"axis": axis,
                         **response_statistics([r for r in rows if r["axis"] == axis])}
                        for axis, _ in AXES],
            "by_radius_axis": [{"radius": radius, "axis": axis,
                                **response_statistics([r for r in rows if r["radius"] == radius
                                                       and r["axis"] == axis])}
                               for radius in RADII for axis, _ in AXES]}


def run_probes(trainer, sources, baseline, device="cpu"):
    """Return JSON-compatible diagnostic records without any file/network I/O.

    ``baseline['probes']`` pins rr/fr_eval_ids, rr/fr_train_sample_id and the
    per-branch max_samples cap.  All images remain transient.  The entire call
    preserves global/training RNG and module modes; trainer detects A/T state
    mutation and gradient_diagnostics uses autograd.grad, not backward/step.
    """
    started = time.perf_counter()
    spec = baseline.get("probes")
    indices = validate_probe_spec(spec, sources)
    stage = trainer.stage
    response, discrepancy, interventions = {}, [], []
    with trainer.evaluation_context() as state_hash:
        for branch in ("rr", "fr"):
            response_rows = []
            group = f"{branch}_eval"
            for index in indices[group]:
                source = sources.samples[group][index]
                batch = sources.batch(group, [index], device=device)
                batch.validate(stage.teacher.spec.bands, require_gt=branch == "rr")
                if batch.lineage.split == "train" or len(batch.pan) != 1:
                    raise ValueError("Inference probe must use one registered held-out observation")
                with torch.autocast(device_type=batch.pan.device.type, enabled=False):
                    native = stage.aligner(batch.pan.float(), batch.native_lms.float(), batch.prior_dy_dx)
                    if native.shape != (1, 2) or not bool(torch.isfinite(native).all()):
                        raise FloatingPointError("Invalid native probe correction")
                    for point in axis16():
                        epsilon = batch.pan.new_tensor(point["epsilon_dy_dx"], dtype=torch.float32)[None]
                        shifted = warp_pan(batch.pan.float(), epsilon)
                        # Keep the exact native scene prior, including under epsilon.
                        shifted_c = stage.aligner(shifted, batch.native_lms.float(), batch.prior_dy_dx)
                        response_rows.append({**point, "scene_id": source["scene_id"],
                            "sample_id": source["sample_id"], "source_scene_id": source["source_scene_id"],
                            "native_correction_dy_dx": native[0].cpu().tolist(),
                            "shifted_correction_dy_dx": shifted_c[0].cpu().tolist(),
                            "delta_dy_dx": (shifted_c - native)[0].cpu().tolist(),
                            "warp_support_fraction": float(warp_support_mask(*batch.pan.shape[-2:], epsilon).float().mean())})
                    reference = stage.spatial.rgb_luminance(batch.gt if branch == "rr" else batch.native_lms)
                    corrections = {"zero": torch.zeros_like(native), "prior_only": batch.prior_dy_dx,
                                   "current": native}
                    for name, correction in corrections.items():
                        pan, low = aligned_observations(batch.pan, correction,
                            lpan=batch.lpan, need_lpan=stage.teacher.spec.uses_lpan)
                        spatial_pan = batch.pan.float() if name == "zero" else pan
                        value = stage.spatial.reduce(stage.spatial.discrepancy(spatial_pan, reference))
                        discrepancy.append({"branch": branch.upper(), "scene_id": source["scene_id"],
                            "mode": "before" if name == "zero" else name,
                            "reference": "RR_GT_RGB_MEAN" if branch == "rr" else "FR_NATIVE_LMS_RGB_MEAN",
                            "absolute_gradient_l1": float(value),
                            "fixed_roi_margin": stage.spatial.spec.roi_margin,
                            "same_descriptor_as_training": True,
                            "independent_geometric_evidence": False})
                        prediction = stage.teacher(pan, batch.ms, batch.native_lms, aligned_lpan=low)["y"]
                        target = batch.gt if branch == "rr" else batch.native_lms
                        mse = float((prediction - target).square().mean())
                        interventions.append({"branch": branch.upper(), "scene_id": source["scene_id"],
                            "mode": name, "correction_dy_dx": correction[0].cpu().tolist(),
                            "metric_name": "MSE_RR_GT_NORMALIZED" if branch == "rr" else "MSE_FR_NATIVE_LMS_RESIDUAL_NORMALIZED",
                            "value": mse, "official_metric": False,
                            "interpretation": "FROZEN_T_INFERENCE_INTERVENTION_NOT_RETRAINED_ABLATION",
                            "fr_reference_is_ground_truth": False,
                            "output_clipping": "none_diagnostic_raw_prediction"})
            response[branch] = {"n_scenes": len(indices[group]), "n_offsets_per_scene": 16,
                                "units": "current_PAN_pixels_dy_dx", "probe": "AXIS16",
                                "prior_unchanged_under_epsilon": True,
                                "rows": response_rows, "summary": _summary(response_rows)}
        rr = sources.batch("rr_train", indices["rr_train"], device=device)
        fr = sources.batch("fr_train", indices["fr_train"], device=device)
        # These two disk samples are deterministic diagnostic-only draws.  They
        # never consume either live training epsilon stream.
        generator = torch.Generator(device="cpu").manual_seed(PROBE_SEED)
        epsilon_rr = sample_offsets(len(rr.pan), 2., generator).to(device)
        epsilon_fr = sample_offsets(len(fr.pan), 2., generator).to(device)
        gradients = trainer.gradient_diagnostics(rr, fr, epsilon_rr=epsilon_rr, epsilon_fr=epsilon_fr)
    result = {"protocol": PROTOCOL, "record_type": "DIAGNOSTICS_NUMERIC",
        "completed_update": trainer.updates, "model_state_sha256": state_hash,
        "probe_seed": PROBE_SEED, "fixed_probe_spec": dict(spec),
        "source_group_counts": {group: len(index) for group, index in indices.items()},
        "response": response, "spatial_discrepancy": discrepancy,
        "spatial_descriptor": asdict(stage.spatial.spec), "inference_interventions": interventions,
        "gradient_diagnostics": {**gradients, "rr_train_sample_id": spec["rr_train_sample_id"],
            "fr_train_sample_id": spec["fr_train_sample_id"],
            "epsilon_rr_dy_dx": epsilon_rr.cpu().tolist(), "epsilon_fr_dy_dx": epsilon_fr.cpu().tolist(),
            "epsilon_distribution": "UNIFORM_AREA_DISK_RADIUS_2", "evaluation_data_in_backward": False},
        "official_selection_input": False, "images_written": False,
        "physical_displacement_ground_truth": False, "elapsed_seconds": time.perf_counter() - started}
    json.dumps(result, allow_nan=False)  # Fail, rather than publishing NaN diagnostics.
    return result
