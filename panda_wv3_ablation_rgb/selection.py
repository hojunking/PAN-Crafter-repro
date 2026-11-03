"""Validation-only geometric donor selection, separate from immutable old records.

RRE is an independently estimated proxy, not a physical displacement ground
truth. Every checkpoint is scored on the same GT-only eligible population.
Estimator failures receive 12 px, above the largest searchable vector, never 0.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from directshift48.reporting import atomic_json
from panda_method.calibration import canonical_sha256, state_dict_sha256
from pa.warp import warp_pan, warp_support_mask
from panda_band30.reconstruction import base_from_batch, reconstruct_with_base
from ta2.evaluation import file_sha
from ta2.registration import _features, register_band


SELECTOR = "BEST_VAL_RRE"
SCHEMA = "WV3_RGB_ABLATION_RRE_SELECTION_V1"
PROTOCOL = {
    "schema": "WV3_RGB_ABLATION_VAL_RRE_V1",
    "split": "registered_RR_validation_only",
    "population": "all_1080_stored_WV3_validation_patches",
    "target": "Scharr_of_RGB_mean_GT",
    "reference_band_indices": [4, 2, 1],
    "reference_unit": "stored_DN_div_2047",
    "shift_unit": "current_RR_PAN_pixel",
    "pan_policy": "full_uncapped_predicted_u",
    "descriptor": "gradient",
    "sigma_p": .8,
    "sigma_r": .8,
    "bound": 8,
    "margin": 16,
    "target_rms_threshold": 1e-4,
    "eligibility": "GT_only_centered_gradient_RMS_at_fixed_ROI",
    "source_support_margin": 4,
    "invalid_registration_penalty_px": 12.,
    "quantile_method": "numpy_linear",
    "rank": ["RRE50_ASC", "RRE90_ASC", "VAL_ERGAS_ASC", "UPDATE_ASC"],
    "epsilon_vectors_dy_dx": [[1., 0.], [-1., 0.], [0., 1.], [0., -1.]],
    "epsilon_error": "norm(A(W(P,eps),LMS)+eps-A(P,LMS))",
    "epsilon_gate": "finite_only; diagnostic_not_posthoc_score_threshold",
    "val_ergas": "mean_patch_100_over_4_sqrt_mean_band_MSE_over_GTmean_squared",
    "val_ergas_roi_margin": 8,
    "val_ergas_clipping": "none",
    "physical_displacement_ground_truth": False,
    "uses_test_for_selection": False,
}


def target_eligibility(reference):
    feature = _features(np.asarray(reference, np.float32), .8, "gradient")
    roi = feature[16:-16, 16:-16]
    centered = roi - roi.mean(axis=(0, 1), keepdims=True)
    rms = float(np.sqrt(np.mean(np.square(centered, dtype=np.float64))))
    return rms >= PROTOCOL["target_rms_threshold"], rms


def validation_inventory(sources):
    if sources.sensor != "WV3" or sources.bands != 8:
        raise ValueError("Mature30 selector is registered WV3 validation only")
    path = sources._verify("val")
    with h5py.File(path, "r") as handle:
        count = len(handle["pan"])
        if count != 1080:
            raise ValueError("Expected all 1080 registered validation patches")
        eligible, target_rms = [], []
        for index in range(count):
            gt = np.asarray(handle["gt"][index], np.float32)
            if gt.shape != (8, 64, 64) or not np.isfinite(gt).all():
                raise ValueError("Fixed 8x64x64 finite validation GT required")
            ok, rms = target_eligibility(gt[[4, 2, 1]].mean(axis=0) / sources.peak)
            eligible.append(bool(ok))
            target_rms.append(rms)
    if not any(eligible):
        raise ValueError("No GT-texture-eligible validation patches")
    result = {"schema": "WV3_RGB_ABLATION_VAL_INVENTORY_V1",
        "source_manifest_sha256": sources.manifest_sha("val"),
        "source_h5_sha256": sources.entry["splits"]["val"]["sha256"],
        "catalog_sha256": sources.catalog_sha256,
        "sample_ids": [f"WV3:val:{i}" for i in range(count)],
        "eligible": eligible, "target_rms": target_rms,
        "sample_count": count, "eligible_count": sum(eligible),
        "excluded_gt_low_texture_count": count - sum(eligible),
        "selector_protocol_sha256": canonical_sha256(PROTOCOL)}
    result["inventory_sha256"] = canonical_sha256(result)
    sources._verify("val")
    return result


def validate_inventory(inventory, sources=None):
    digest = inventory.get("inventory_sha256")
    plain = {k: v for k, v in inventory.items() if k != "inventory_sha256"}
    if digest != canonical_sha256(plain):
        raise ValueError("Validation inventory content/digest mismatch")
    if (inventory.get("selector_protocol_sha256") != canonical_sha256(PROTOCOL)
            or inventory.get("sample_count") != 1080
            or len(inventory.get("eligible", [])) != 1080
            or inventory.get("eligible_count") != sum(inventory["eligible"])
            or inventory.get("sample_ids") != [f"WV3:val:{i}" for i in range(1080)]):
        raise ValueError("Validation population or selector protocol mismatch")
    if sources is not None:
        sources._verify("val")
        if inventory["source_manifest_sha256"] != sources.manifest_sha("val"):
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
            if value != PROTOCOL["invalid_registration_penalty_px"]:
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
        "selector_protocol_sha256": canonical_sha256(PROTOCOL)}


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
    inventory = validation_inventory(sources) if inventory is None else deepcopy(inventory)
    validate_inventory(inventory, sources)
    aligner, teacher = _modules(value, teacher)
    if getattr(aligner, "outputs", None) != 2:
        raise ValueError("Global2 Aligner required for RGB-reference validation, not bandwise head")
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
            support = warp_support_mask(64, 64, u)[..., 4:-4, 4:-4].flatten(1).all(1)
            pans = ((shifted[:, 0] + 1) / 2).cpu().numpy()
            references = ((batch.gt[:, [4, 2, 1]].mean(1) + 1) / 2).cpu().numpy()
            for local, index in enumerate(indexes):
                eligible = inventory["eligible"][index]
                entry = {"sample_id": inventory["sample_ids"][index], "eligible": eligible,
                         "predicted_u": u[local].cpu().tolist(),
                         "valid": False, "rre_px": None, "failure_reason": None}
                if eligible:
                    record = None
                    if not bool(support[local]):
                        entry["failure_reason"] = "WARP_SEARCH_SUPPORT_VIOLATION"
                    else:
                        record = register_band(pans[local], references[local],
                            descriptor="gradient", sigma_p=.8, sigma_r=.8,
                            bound=8, margin=16, target_rms_threshold=1e-4)
                        entry["valid"] = bool(record["valid"])
                        entry["failure_reason"] = record["fallback_reason"]
                    entry["registration"] = record
                    entry["rre_px"] = (float(np.linalg.norm(record["shift"]))
                        if entry["valid"] else PROTOCOL["invalid_registration_penalty_px"])
                rows.append(entry)
            if epsilon:
                for vector in PROTOCOL["epsilon_vectors_dy_dx"]:
                    eps = torch.tensor(vector, device=batch.pan.device, dtype=torch.float32).expand_as(u)
                    ue = aligner(warp_pan(batch.pan, eps), batch.native_lms)
                    epsilon_errors.extend(torch.linalg.vector_norm(ue + eps - u, dim=1).cpu().tolist())
                    epsilon_gains.extend((-(ue - u).mul(eps).sum(1)).cpu().tolist())
            if include_ergas:
                y = reconstruct_with_base(teacher, shifted,
                    base_from_batch(batch, "interp23"), batch.ms)["y"]
                target = ((batch.gt + 1) / 2)[..., 8:-8, 8:-8].double()
                error = ((y - batch.gt) / 2)[..., 8:-8, 8:-8].double()
                means = target.mean((-2, -1))
                if bool((means <= 0).any()):
                    raise ValueError("Nonpositive GT band mean in validation ERGAS")
                er = 25 * torch.sqrt((error.square().mean((-2, -1)) / means.square()).mean(1))
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
        return {**summary, "per_sample": rows, "protocol": deepcopy(PROTOCOL)}
    finally:
        for module, mode, sha in zip(modules, modes, before):
            module.train(mode)
            if state_dict_sha256(module) != sha:
                raise ValueError("Validation mutated A/T model state")
        validate_inventory(inventory, sources)
