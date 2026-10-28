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
from directshift48.method import RGB_INDICES
from panda_method.calibration import canonical_sha256, state_dict_sha256
from .warp import warp_pan, warp_support_mask
from panda_sensor_refine.base_reconstruction import base_from_batch, reconstruct_with_base
from ta2.evaluation import file_sha
from ta2.registration import _features, register_band

from .contract import CAMPAIGN_ID, LANES, validate_loss_pair_config

SELECTOR = "BEST_VAL_RRE"
SCHEMA = "PANDA_SENSOR_REFINE_RGB_L4_RRE_SELECTION_V2"
PROTOCOL = {
    "schema": "PANDA_SENSOR_REFINE_RGB_L4_VAL_RRE_V2",
    "split": "registered_RR_validation_only",
    "population": "all_registered_sensor_validation_patches",
    "target": "RGB_mean_GT",
    "target_bands_R_G_B": [2, 1, 0],
    "reference_unit": "stored_DN_div_sensor_peak_GF2_1023_QB_2047",
    "shift_unit": "current_RR_PAN_pixel",
    "pan_policy": "full_uncapped_predicted_u",
    "descriptor": "gradient",
    "sigma_p": .8,
    "sigma_r": .8,
    "bound": 8,
    "margin": 20,
    "target_rms_threshold": 1e-4,
    "eligibility": "GT_only_centered_gradient_RMS_at_fixed_ROI",
    "source_support_margin": 8,
    "source_support_derivation": "20_ROI_minus_8_search_minus_3_Gaussian_minus_1_Scharr",
    "warp": "lanczos4_whole_sample_reflection",
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
    margin = PROTOCOL['margin']
    roi = feature[margin:-margin, margin:-margin]
    centered = roi - roi.mean(axis=(0, 1), keepdims=True)
    rms = float(np.sqrt(np.mean(np.square(centered, dtype=np.float64))))
    return rms >= PROTOCOL["target_rms_threshold"], rms


def validation_inventory(sources):
    if sources.sensor not in ("QB", "GF2") or sources.bands != 4:
        raise ValueError("Mature30 selector is registered QB/GF2 validation only")
    path = sources._verify("val")
    with h5py.File(path, "r") as handle:
        count = len(handle["pan"])
        if count <= 0:
            raise ValueError("Expected nonempty full registered validation inventory")
        eligible, target_rms = [], []
        for index in range(count):
            gt = np.asarray(handle["gt"][index], np.float32)
            if gt.shape != (4, 64, 64) or not np.isfinite(gt).all():
                raise ValueError("Fixed 4x64x64 finite validation GT required")
            ok, rms = target_eligibility(gt[list(RGB_INDICES[sources.sensor])].mean(axis=0) / sources.peak)
            eligible.append(bool(ok))
            target_rms.append(rms)
    if not any(eligible):
        raise ValueError("No GT-texture-eligible validation patches")
    result = {"schema": "PANDA_SENSOR_REFINE_VAL_INVENTORY_V1",
        "sensor": sources.sensor, "bands": sources.bands, "peak": sources.peak,
        "source_manifest_sha256": sources.manifest_sha("val"),
        "source_h5_sha256": sources.entry["splits"]["val"]["sha256"],
        "catalog_sha256": sources.catalog_sha256,
        "sample_ids": [f"{sources.sensor}:val:{i}" for i in range(count)],
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
            or type(inventory.get("sample_count")) is not int or inventory["sample_count"] <= 0
            or inventory.get("sensor") not in ("QB", "GF2")
            or inventory.get("bands") != 4
            or inventory.get("peak") != (1023 if inventory["sensor"] == "GF2" else 2047)
            or len(inventory.get("eligible", [])) != inventory["sample_count"]
            or inventory.get("eligible_count") != sum(inventory["eligible"])
            or inventory.get("sample_ids") != [f"{inventory['sensor']}:val:{i}" for i in range(inventory["sample_count"])]):
        raise ValueError("Validation population or selector protocol mismatch")
    if sources is not None:
        sources._verify("val")
        if (inventory["source_manifest_sha256"] != sources.manifest_sha("val")
                or inventory["sensor"] != sources.sensor):
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
        raise ValueError("RGB-reference global2 Aligner required, not bandwise head")
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
            margin = PROTOCOL['source_support_margin']
            support = warp_support_mask(64, 64, u)[..., margin:-margin, margin:-margin].flatten(1).all(1)
            pans = ((shifted[:, 0] + 1) / 2).cpu().numpy()
            references = ((batch.gt[:, RGB_INDICES[sources.sensor]].mean(1) + 1) / 2).cpu().numpy()
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
                            bound=PROTOCOL['bound'], margin=PROTOCOL['margin'], target_rms_threshold=1e-4)
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


def geometry_rank(row):
    score = row.get("validation", row)
    values = tuple(score[k] for k in ("RRE50", "RRE90"))
    if (any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values)
            or values[1] < values[0]):
        raise ValueError("Finite validation RRE50/RRE90 required")
    return values


def rank(row):
    values = geometry_rank(row)
    er = row["validation"].get("Val_ERGAS")
    if type(er) not in (float, int) or not math.isfinite(er) or er < 0:
        raise ValueError("Validation ERGAS required for complete selection rank")
    return (*values, er, row["completed_update"])


def validate_receipt(receipt):
    if (receipt.get("schema") != SCHEMA or receipt.get("selector") != SELECTOR
            or receipt.get("campaign_id") != CAMPAIGN_ID
            or receipt.get("selector_protocol") != PROTOCOL
            or receipt.get("server") not in ("s1", "s3")):
        raise ValueError("Unexpected RRE selection receipt")
    if receipt.get("receipt_sha256") != canonical_sha256(
            {k: v for k, v in receipt.items() if k != "receipt_sha256"}):
        raise ValueError("Selector receipt SHA mismatch")
    validate_inventory(receipt["validation_inventory"])
    if receipt["validation_inventory"]["sensor"] != LANES[receipt["server"]]["dataset"]:
        raise ValueError("Selection sensor/server mismatch")
    candidates = receipt["candidates"]
    if not candidates or receipt["selected"] not in candidates:
        raise ValueError("Selected checkpoint missing from candidate inventory")
    selected = receipt["selected"]
    best_geometry = min(map(geometry_rank, candidates))
    tied = [r for r in candidates if geometry_rank(r) == best_geometry]
    if geometry_rank(selected) != best_geometry or selected != min(tied, key=rank):
        raise ValueError("Selected checkpoint violates RRE50/RRE90/ERGAS/update ranking")
    if any(r["validation"]["validation_inventory_sha256"] !=
           receipt["validation_inventory"]["inventory_sha256"] for r in candidates):
        raise ValueError("Candidates do not share fixed validation inventory")
    for row in candidates:
        validate_loss_pair_config(row["source_config"])
        if row["source_config"].get("run_id") != row.get("source_run_id"):
            raise ValueError("Selection source config/run identity differs")
        score = row["validation"]
        recomputed = aggregate_rows(score["per_sample"], receipt["validation_inventory"])
        if any(score.get(key) != value for key, value in recomputed.items()):
            raise ValueError("Candidate aggregate differs from fixed per-sample evidence")
        if (row["source_config"]["server"] != receipt["server"]
                or row["source_run_id"] != receipt["source_history"]["source_run_id"]
                or row["source_config"]["campaign_id"] != row["source_campaign_id"]):
            raise ValueError("Selection mixes source campaigns or servers")
    return receipt


def select_checkpoints(paths, sources, *, server, output=None, device="cpu",
                       batch_size=48, on_progress=None):
    """Reselect complete local Stage1 history; old metadata remains read-only."""
    from .donor import load_donor, checkpoint_evidence, complete_history
    from panda_stage2_condq.sources import RegisteredRRTrain
    paths = [Path(path).resolve(strict=True) for path in paths]
    history = complete_history(paths, server=server)
    inventory, registered = validation_inventory(sources), RegisteredRRTrain(sources)
    candidates = []
    for path in paths:
        evidence = checkpoint_evidence(path, server=server)
        donor = load_donor(path, evidence["file_sha256"], registered,
                           server=server, device=device)
        score = collect_validation(donor, sources, inventory=inventory,
            batch_size=batch_size, device=device, include_ergas=False, epsilon=True)
        row = {**evidence, "validation": score}
        candidates.append(row)
        del donor
        if on_progress:
            on_progress({"checkpoint": str(path), "completed_update": row["completed_update"],
                         "RRE50": score["RRE50"], "RRE90": score["RRE90"]})
    best_geometry = min(map(geometry_rank, candidates))
    for row in candidates:
        if geometry_rank(row) == best_geometry:
            donor = load_donor(row["checkpoint_path"], row["file_sha256"], registered,
                               server=server, device=device)
            result = collect_validation(donor, sources, inventory=inventory,
                batch_size=batch_size, device=device, include_ergas=True, epsilon=False)
            if geometry_rank(result) != geometry_rank(row):
                raise ValueError("Validation RRE changed during lazy ERGAS evaluation")
            row["validation"]["Val_ERGAS"] = result["Val_ERGAS"]
            del donor
    selected = min((r for r in candidates if geometry_rank(r) == best_geometry), key=rank)
    if complete_history(paths, server=server) != history:
        raise ValueError("Source history changed while selecting its donor")
    receipt = {"schema": SCHEMA, "selector": SELECTOR, "campaign_id": CAMPAIGN_ID,
        "server": server, "selector_protocol": deepcopy(PROTOCOL),
        "validation_inventory": inventory, "source_history": history,
        "candidates": candidates, "selected": selected}
    receipt.update({k: selected[k] for k in ("source_root", "source_run_dir", "source_config",
        "source_evaluation_path", "source_state", "checkpoint_path", "file_sha256")})
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    validate_receipt(receipt)
    if output is not None:
        path = Path(output)
        if path.exists():
            raise FileExistsError("Immutable separate selection receipt already exists")
        atomic_json(path, receipt)
    return receipt
