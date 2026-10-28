"""Exact RR-train ECDFs for positive PAN scales, reusable across Hard/LR choices.

Only the geometry and frozen local donor determine these reference statistics;
alpha, gamma and Student LR never enter calibration or its cache identity.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time

from directshift48.method import ScharrAbsolute
from panda_sensor_refine.base_reconstruction import BandReconstructionSpec, upsampler_manifest
from .calibration_core import CalibrationBuilder, CalibrationStats, validate_binding
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.sources import validate_training_batch
from panda_stage2_condq.spec import MASK2_POLICY
from panda_stage2_condq.training import configure_precision
from panda_method.calibration import canonical_sha256
from ta2.evaluation import file_sha
from .spec import MatureSpec, geometry_manifest
from .contract import CAMPAIGN_ID, WARP_MANIFEST, LOSS_PAIRS

# The inherited exact collector estimates 3.344 GiB for GF2's 19,809 RR
# patches. Four GiB admits that complete population without approximation.
HOST_CALIBRATION_BUDGET_BYTES = 4 * 1024**3


def base_protocol():
    return {
        "schema": "PANDA_NEXT30_BASE_BINDING_V1",
        "student_warp": deepcopy(WARP_MANIFEST),
        "teacher_warp": "STAGE1_LANCZOS4_REFLECT_FULL_U",
        "teacher_policy": "FIXED_STAGE1_FULL_U_SAME_CHECKPOINT_A_T",
        "geometry_unit": "CURRENT_PAN_PIXEL_NO_X4",
        "base": upsampler_manifest("interp23"),
        "gt_and_output_coordinates": "ORIGINAL_MS",
        "q_target": "ORIGINAL_RGB_MEAN_2_1_0",
        "q_pan": "ACTUAL_STUDENT_GEOMETRY",
        "loss_mask_policy": MASK2_POLICY,
        "loss_mask_policy_token": "LEGACY_REDUCTION_COMPATIBILITY_ID",
        "actual_support_operator": "TEACHER_STUDENT_LANCZOS4_INTERSECTION",
        "alignment_target": "SCHARR_OF_RGB_MEAN_GT",
    }


def validate_base_binding(binding, spec=None):
    if spec is not None and not isinstance(spec, MatureSpec):
        raise TypeError("Explicit MatureSpec required")
    value = validate_binding(binding)
    if value["sensor"] not in ("QB", "GF2") or (spec is not None and value["sensor"] != spec.sensor):
        raise ValueError("Matched four-band donor/sensor required")
    from .contract import validate_source_teacher
    validate_source_teacher(value)
    recon = BandReconstructionSpec(4, 112, (1, 2, 3), "P0", "BICUBIC_MS", 1).manifest()
    if canonical_sha256(value.get("band30_reconstruction_spec")) != canonical_sha256(recon):
        raise ValueError("Truthful four-band W112/D123/PAN1 reconstruction binding required")
    if any(key in value for key in ("band30_geometry", "band30_q_reference", "mature30_geometry")):
        raise ValueError("Unbound donor required; cannot relabel pre-bound applied geometry")
    if "mature30_base" in value and value["mature30_base"] != base_protocol():
        raise ValueError("Incompatible pre-bound mature base protocol")
    value["mature30_base"] = base_protocol()
    return value


def binding_for_geometry(binding, geometry_key):
    value = validate_base_binding(binding)
    value["mature30_geometry"] = geometry_manifest(geometry_key)
    return value


def bind_donor(donor, spec, geometry_key=None):
    if not isinstance(donor, LoadedDonor) or not isinstance(spec, MatureSpec):
        raise TypeError("Pinned LoadedDonor and MatureSpec required")
    donor.assert_frozen()
    binding = validate_base_binding(donor.binding, spec)
    metadata = deepcopy(donor.checkpoint_metadata)
    metadata["mature30_base"] = base_protocol()
    if geometry_key is not None:
        if geometry_key not in spec.geometry_keys:
            raise ValueError("Calibration geometry is not used by this registered policy")
        binding = binding_for_geometry(binding, geometry_key)
        metadata["mature30_geometry"] = geometry_manifest(geometry_key)
    return replace(donor, binding=binding, checkpoint_metadata=metadata)


def validate_stats_bank(stats_bank, binding, spec, *, full_hash=False):
    if not isinstance(spec, MatureSpec) or not isinstance(stats_bank, dict):
        raise TypeError("MatureSpec and explicit geometry-keyed calibration bank required")
    base = validate_base_binding(binding, spec)
    if set(stats_bank) != set(spec.geometry_keys):
        raise ValueError("Exact calibration bank must contain every and only required phase geometry")
    for key, stats in stats_bank.items():
        if not isinstance(stats, CalibrationStats):
            raise TypeError("Every phase requires new exact CalibrationStats")
        stats.validate_binding(binding_for_geometry(base, key))
        if (stats.mask_policy != spec.mask_policy or stats.schema != spec.calibration_schema
                or stats.calibration_kind != "exact_condq"):
            raise ValueError("Exact full-train MASK2 conditional-q required for every phase")
        stats.assert_integrity(full_hash=full_hash)
    return dict(stats_bank)


def code_pin():
    # Reuse shared dependency hashes, but never import old runtime campaign
    # configuration/state. It is safe to depend on frozen implementation files.
    from panda_s1loss9.runtime import code_pin as shared
    root = Path(__file__).resolve().parents[1]
    return {**shared(), **{str(p.relative_to(root)): file_sha(p)
                          for p in (root / "panda_sensor_refine").glob("*.py")}}


def calibrate_geometry(donor, registered, *, spec, geometry_key, output, k,
                       batch_size, memory_budget_bytes, device="cpu", on_progress=None):
    from .method import frozen_predictions
    path = Path(output)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Immutable mature calibration destination already exists")
    configure_precision()
    registered.verify()
    donor = bind_donor(donor, spec, geometry_key)
    if donor.teacher is None:
        raise ValueError("Calibration requires the same-checkpoint frozen Teacher")
    implementation = code_pin()
    collector = CalibrationBuilder(registered.sample_ids, donor.binding, k=k,
        memory_budget_bytes=memory_budget_bytes, include_condq=True, mask_policy=MASK2_POLICY)
    scharr = ScharrAbsolute(spec.sensor).to(device)
    started, seen = time.monotonic(), 0
    for rr in registered.calibration_batches(batch_size, device=device):
        validate_training_batch(rr, donor.binding, expected_sample_ids=registered.sample_ids,
                                require_unaugmented=True)
        f = frozen_predictions(donor.aligner, donor.teacher, rr, geometry_key=geometry_key)
        e, g, q = reference_maps(scharr, f["teacher_y"], f["gt_loss"], f["pan_loss"])
        collector.add(rr.lineage.sample_ids, e, g, q, sensor=rr.lineage.sensor,
            split=rr.lineage.split, resolution=rr.lineage.resolution,
            unaugmented=all(a == (False, False, 0) for a in rr.lineage.augmentations),
            support_ok=bool(f["spatial_support_ok"].all()), loss_mask=f["loss_mask"])
        seen += len(rr.pan)
        if on_progress:
            on_progress({"geometry_key": geometry_key, "seen": seen,
                         "expected": len(registered.sample_ids),
                         "elapsed_seconds": time.monotonic() - started})
    donor.assert_frozen()
    registered.verify()
    if code_pin() != implementation:
        raise ValueError("Implementation changed during mature exact calibration")
    stats = collector.finalize()
    stats.assert_integrity(full_hash=True)
    receipt = stats.save(path)
    return {**receipt, "kind": stats.calibration_kind, "tau_e": stats.tau_e,
            "mask_policy": stats.mask_policy, "calibration_schema": stats.schema,
            "sample_count": seen, "n_e": stats.metadata["n_e"], "n_q": stats.metadata["n_q"],
            "estimated_peak_host_bytes": collector.estimated_peak_bytes,
            "elapsed_seconds": time.monotonic() - started, "donor_binding": donor.binding,
            "geometry_key": geometry_key, "mature30_geometry": geometry_manifest(geometry_key),
            "implementation_pin": implementation}


def load_stats_bank(donor, spec, artifacts, *, memory_budget_bytes):
    """Load exact immutable files; account for resident tables plus load workspace."""
    if (not isinstance(artifacts, dict) or set(artifacts) != set(spec.geometry_keys)
            or type(memory_budget_bytes) is not int or memory_budget_bytes <= 0):
        raise ValueError("Complete explicit phase-artifact map and positive memory budget required")
    binding = validate_base_binding(donor.binding, spec)
    bank, resident = {}, 0
    for key in spec.geometry_keys:
        record = artifacts[key]
        if not isinstance(record, dict) or not record.get("path") or not record.get("file_sha256"):
            raise ValueError("Every calibration file requires an explicit path and file SHA256")
        remaining = memory_budget_bytes - resident
        if remaining <= 0:
            raise MemoryError("Exact phase bank exceeds host memory budget")
        bank[key] = CalibrationStats.load(
            record["path"], expected_binding=binding_for_geometry(binding, key),
            expected_file_sha256=record["file_sha256"], memory_budget_bytes=remaining,
            expected_mask_policy=spec.mask_policy)
        resident += bank[key].memory_bytes
    return validate_stats_bank(bank, binding, spec, full_hash=True)
