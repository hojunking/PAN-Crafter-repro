"""Validation-only geometric donor selection, separate from immutable old records.

RRE is an independently estimated proxy, not a physical displacement ground
truth. Every checkpoint is scored on the same GT-only eligible population.
Estimator failures receive 12 px, above the largest searchable vector, never 0.
"""
from __future__ import annotations

from . import plan
from collections import Counter
from copy import deepcopy
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from directshift48.reporting import atomic_json
from panda_method.calibration import canonical_sha256, state_dict_sha256
from .geometry import warp_pan, warp_support_mask, support
from panda_unified.core import base_from_batch, reconstruct_with_base
from ta2.evaluation import file_sha
from ta2.registration import _features, register_band


SELECTOR = "FIXED50K_VALIDATION_DIAGNOSTIC_ONLY"
SCHEMA = "PANDA_TAB3_RGBN_RRE_SELECTION_V1"
def protocol(sensor, reference_mode):
    return {
        "schema": "PANDA_TAB3_RGBN_VAL_RRE_V1",
        "sensor": sensor, "reference_mode": reference_mode,
        "split": "registered_RR_validation_only",
        "population": "all_registered_validation_patches",
        "target": "Scharr_of_" + reference_mode + "_mean_GT",
        "reference_band_indices": list(plan.reference_bands(sensor, reference_mode)),
        "reference_unit": "stored_DN_div_sensor_peak",
        "shift_unit": "current_RR_PAN_pixel", "pan_policy": "full_uncapped_predicted_u",
        "warp": "LANCZOS4_WHOLE_SAMPLE_REFLECTION", "descriptor": "gradient",
        "sigma_p": .8, "sigma_r": .8, "bound": 8, "margin": 19,
        "target_rms_threshold": 1e-4,
        "eligibility": "GT_only_centered_gradient_RMS_at_fixed_metric_search_ROI",
        "source_support_margin": 7,
        "source_support_derivation": "metric_ROI19_minus_search8_minus_Gaussian3_minus_Scharr1",
        "invalid_registration_penalty_px": 12., "quantile_method": "numpy_linear",
        "rank": [], "checkpoint_selection": "FIXED50K; validation diagnostics only",
        "epsilon_vectors_dy_dx": [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
        "epsilon_error": "norm(A(W(P,eps),LMS)+eps-A(P,LMS))",
        "epsilon_gate": "finite_only; diagnostic_not_posthoc_score_threshold",
        "val_ergas": "mean_patch_100_over_4_sqrt_mean_band_masked_MSE_over_masked_GTmean_squared",
        "val_ergas_domain": "COMMON_LANCZOS4_SUPPORT_ERODED_SCHARR1",
        "val_ergas_clipping": "none", "physical_displacement_ground_truth": False,
        "uses_test_for_selection": False,
    }



def target_eligibility(reference):
    feature = _features(np.asarray(reference, np.float32), .8, "gradient")
    roi = feature[19:-19, 19:-19]
    centered = roi - roi.mean(axis=(0, 1), keepdims=True)
    rms = float(np.sqrt(np.mean(np.square(centered, dtype=np.float64))))
    return rms >= 1e-4, rms


def validation_inventory(sources, reference_mode="RGBN"):
    if sources.sensor not in ("WV3", "QB") or sources.bands != {"WV3":8, "QB":4}[sources.sensor]:
        raise ValueError("Registered WV3/QB validation required")
    protocol_value = protocol(sources.sensor, reference_mode)
    path = sources._verify("val")
    with h5py.File(path, "r") as handle:
        count = len(handle["pan"])
        if count <= 0:
            raise ValueError("Nonempty full validation inventory required")
        eligible, target_rms = [], []
        for index in range(count):
            gt = np.asarray(handle["gt"][index], np.float32)
            if gt.shape != (sources.bands, 64, 64) or not np.isfinite(gt).all():
                raise ValueError("Fixed finite 64x64 multiband validation GT required")
            reference = gt[protocol_value["reference_band_indices"]].mean(axis=0) / sources.peak
            ok, rms = target_eligibility(reference)
            eligible.append(bool(ok));target_rms.append(rms)
    if not any(eligible):
        raise ValueError("No GT-texture-eligible validation patches")
    result = {"schema": "PANDA_TAB3_RGBN_VAL_INVENTORY_V1", "sensor":sources.sensor,
        "reference_mode":reference_mode, "bands":sources.bands, "protocol":protocol_value,
        "source_manifest_sha256": sources.manifest_sha("val"),
        "source_h5_sha256": sources.entry["splits"]["val"]["sha256"],
        "catalog_sha256": sources.catalog_sha256,
        "sample_ids": [f"{sources.sensor}:val:{i}" for i in range(count)],
        "eligible": eligible, "target_rms": target_rms,
        "sample_count": count, "eligible_count": sum(eligible),
        "excluded_gt_low_texture_count": count - sum(eligible),
        "selector_protocol_sha256": canonical_sha256(protocol_value)}
    result["inventory_sha256"] = canonical_sha256(result)
    sources._verify("val")
    return result


def validate_inventory(inventory, sources=None, reference_mode=None):
    digest = inventory.get("inventory_sha256")
    plain = {k: v for k, v in inventory.items() if k != "inventory_sha256"}
    if digest != canonical_sha256(plain):
        raise ValueError("Validation inventory content/digest mismatch")
    sensor = inventory.get("sensor");mode = inventory.get("reference_mode")
    expected = protocol(sensor, mode)
    count = inventory.get("sample_count")
    if (inventory.get("schema") != "PANDA_TAB3_RGBN_VAL_INVENTORY_V1"
            or (reference_mode is not None and mode != reference_mode)
            or inventory.get("protocol") != expected
            or inventory.get("selector_protocol_sha256") != canonical_sha256(expected)
            or type(count) is not int or count <= 0
            or len(inventory.get("eligible", [])) != count
            or len(inventory.get("target_rms", [])) != count
            or inventory.get("eligible_count") != sum(inventory["eligible"])
            or inventory.get("sample_ids") != [f"{sensor}:val:{i}" for i in range(count)]):
        raise ValueError("Validation population or selector protocol mismatch")
    if sources is not None:
        sources._verify("val")
        if sensor != sources.sensor or inventory["source_manifest_sha256"] != sources.manifest_sha("val"):
            raise ValueError("Validation inventory belongs to different sources")


def aggregate_rows(rows, inventory):
    validate_inventory(inventory)
    if len(rows) != inventory["sample_count"]:
        raise ValueError("Every fixed validation sample needs a recorded result")
    errors, failures = [], Counter()
    for index, row in enumerate(rows):
        if row.get("sample_id") != inventory["sample_ids"][index]:
            raise ValueError("Validation rows were reordered or removed")
        if row.get("eligible") is not inventory["eligible"][index]:
            raise ValueError("Checkpoint-dependent validation eligibility forbidden")
        if not row["eligible"]:
            continue
        value = float(row["rre_px"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Finite nonnegative RRE required")
        if not row["valid"]:
            if value != inventory["protocol"]["invalid_registration_penalty_px"]:
                raise ValueError("Failed registration must receive explicit penalty")
            failures[row["failure_reason"]] += 1
        errors.append(value)
    if not errors:
        raise ValueError("Empty eligible validation population")
    return {"RRE50": float(np.quantile(errors, .5, method="linear")),
        "RRE90": float(np.quantile(errors, .9, method="linear")),
        "failure_count": sum(failures.values()),
        "failure_fraction": sum(failures.values()) / len(errors),
        "failure_reasons": dict(failures), "eligible_count": len(errors),
        "sample_count": len(rows),
        "excluded_gt_low_texture_count": inventory["excluded_gt_low_texture_count"],
        "validation_inventory_sha256": inventory["inventory_sha256"],
        "selector_protocol_sha256": canonical_sha256(inventory["protocol"])}


def _modules(value, teacher):
    if hasattr(value, "stage"):
        value = value.stage
    if hasattr(value, "aligner"):
        return value.aligner, teacher if teacher is not None else getattr(value, "teacher", None)
    return value, teacher


@torch.no_grad()
def collect_validation(value, sources, *, teacher=None, inventory=None,
                       batch_size=48, device="cpu", include_ergas=False,
                       epsilon=True, on_progress=None):
    """Score an Aligner/trainer/LoadedDonor without touching optimizer or RNG."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive validation batch size required")
    cfg = (getattr(value, "cfg", None) or
           getattr(value, "checkpoint_metadata", {}).get("source_config", {}))
    mode = cfg.get("reference_mode", inventory.get("reference_mode") if inventory else None)
    if mode not in ("RGB", "RGBN"):
        raise ValueError("Validation requires explicit matching RGB/RGBN reference mode")
    inventory = validation_inventory(sources, mode) if inventory is None else deepcopy(inventory)
    validate_inventory(inventory, sources, mode)
    protocol_value = inventory["protocol"]
    aligner, teacher = _modules(value, teacher)
    outputs = getattr(aligner, "outputs", getattr(getattr(aligner, "fc2", None), "out_features", None))
    if outputs != 2:
        raise ValueError("Global two-coordinate Aligner required for reference-matched validation")
    if include_ergas and teacher is None:
        raise ValueError("Validation ERGAS requires exact paired Teacher")
    modules = [aligner] + ([teacher] if teacher is not None else [])
    modes = [module.training for module in modules]
    before = [state_dict_sha256(module) for module in modules]
    rows, epsilon_errors, epsilon_gains, ergas = [], [], [], []
    for module in modules:
        module.eval()
    try:
        for start in range(0, inventory["sample_count"], batch_size):
            indexes = list(range(start, min(start + batch_size, inventory["sample_count"])))
            batch = sources.batch("val", indexes, device=device)
            if (batch.lineage.split != "validation" or batch.lineage.resolution != "RR"
                    or any(a != (False, False, 0) for a in batch.lineage.augmentations)):
                raise ValueError("Unaugmented registered RR validation required")
            u = aligner(batch.pan, batch.native_lms)
            shifted = warp_pan(batch.pan, u)
            support_ok = warp_support_mask(64, 64, u)[..., 7:-7, 7:-7].flatten(1).all(1)
            pans = ((shifted[:, 0] + 1) / 2).cpu().numpy()
            references = ((batch.gt[:, protocol_value["reference_band_indices"]].mean(1) + 1) / 2).cpu().numpy()
            for local, index in enumerate(indexes):
                eligible = inventory["eligible"][index]
                entry = {"sample_id": inventory["sample_ids"][index], "eligible": eligible,
                         "predicted_u": u[local].cpu().tolist(),
                         "valid": False, "rre_px": None, "failure_reason": None}
                if eligible:
                    record = None
                    if not bool(support_ok[local]):
                        entry["failure_reason"] = "WARP_SEARCH_SUPPORT_VIOLATION"
                    else:
                        record = register_band(pans[local], references[local],
                            descriptor="gradient", sigma_p=.8, sigma_r=.8,
                            bound=8, margin=19, target_rms_threshold=1e-4)
                        entry["valid"] = bool(record["valid"])
                        entry["failure_reason"] = record["fallback_reason"]
                    entry["registration"] = record
                    entry["rre_px"] = (float(np.linalg.norm(record["shift"]))
                        if entry["valid"] else inventory["protocol"]["invalid_registration_penalty_px"])
                rows.append(entry)
            if epsilon:
                for vector in protocol_value["epsilon_vectors_dy_dx"]:
                    eps = torch.tensor(vector, device=batch.pan.device, dtype=torch.float32).expand_as(u)
                    ue = aligner(warp_pan(batch.pan, eps), batch.native_lms)
                    epsilon_errors.extend(torch.linalg.vector_norm(ue + eps - u, dim=1).cpu().tolist())
                    epsilon_gains.extend((-(ue - u).mul(eps).sum(1)).cpu().tolist())
            if include_ergas:
                y = reconstruct_with_base(teacher, shifted,
                    base_from_batch(batch, "interp23"), batch.ms)["y"]
                mask = support(batch.pan, u).double()
                counts = mask.sum((-2, -1))
                target = ((batch.gt + 1) / 2).double()
                error = ((y - batch.gt) / 2).double()
                means = (target * mask).sum((-2, -1)) / counts
                if bool((means <= 0).any()):
                    raise ValueError("Nonpositive GT band mean in validation ERGAS")
                mse = (error.square() * mask).sum((-2, -1)) / counts
                er = 25 * torch.sqrt((mse / means.square()).mean(1))
                ergas.extend(er.cpu().tolist())
            if on_progress:
                on_progress({"seen": len(rows), "expected": inventory["sample_count"]})
        summary = aggregate_rows(rows, inventory)
        if epsilon:
            if not all(math.isfinite(v) for v in epsilon_errors + epsilon_gains):
                raise ValueError("Nonfinite epsilon response")
            summary["epsilon"] = {"count": len(epsilon_errors),
                "EPE50": float(np.median(epsilon_errors)),
                "EPE90": float(np.quantile(epsilon_errors, .9)),
                "gain_mean": float(np.mean(epsilon_gains)),
                "gain_median": float(np.median(epsilon_gains))}
        summary["Val_ERGAS"] = float(np.mean(ergas)) if include_ergas else None
        return {**summary, "per_sample": rows, "protocol": deepcopy(protocol_value)}
    finally:
        for module, mode, sha in zip(modules, modes, before):
            module.train(mode)
            if state_dict_sha256(module) != sha:
                raise ValueError("Validation mutated A/T model state")
        validate_inventory(inventory, sources)
