"""Fresh exact RR-train calibration bound to midpoint target transport."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import time

from directshift48.method import ScharrAbsolute
from panda_stage2_condq.calibration import CalibrationBuilder, CalibrationStats
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.losses import reference_maps
from panda_stage2_condq.sources import validate_training_batch
from panda_stage2_condq.training import configure_precision
from panda_stage2_condq.spec import MASK2_POLICY
from ta2.evaluation import file_sha
from .method import frozen_midpoint_predictions, validate_midpoint_binding
from .spec import geometry_manifest


def bind_donor(donor):
    """New immutable binding, same frozen A/T objects and original checkpoint."""
    if not isinstance(donor, LoadedDonor):
        raise TypeError("Strictly imported local LoadedDonor required")
    donor.assert_frozen()
    binding = deepcopy(donor.binding)
    old = binding.get("midpoint_geometry")
    if old is not None and old != geometry_manifest():
        raise ValueError("Incompatible midpoint donor geometry")
    binding["midpoint_geometry"] = geometry_manifest()
    validate_midpoint_binding(binding)
    metadata = deepcopy(donor.checkpoint_metadata)
    metadata["midpoint_geometry"] = geometry_manifest()
    return replace(donor, binding=binding, checkpoint_metadata=metadata)


def code_pin():
    from panda_stage2_condq.runtime import code_pin as inherited
    root = Path(__file__).resolve().parents[1]
    files = list((root / "panda_midpoint").glob("*.py"))
    files += [root / "align/__init__.py", root / "align/resample.py"]
    return {**inherited(), **{str(p.relative_to(root)): file_sha(p) for p in files}}


def calibrate(donor, registered, *, output, k, batch_size, memory_budget_bytes,
              include_condq, device="cpu", on_progress=None, mask_policy=MASK2_POLICY):
    path = Path(output)
    if path.exists():
        raise FileExistsError("Immutable midpoint calibration destination already exists")
    if mask_policy != MASK2_POLICY:
        raise ValueError("Midpoint calibration preserves MASK2 reduction")
    configure_precision()
    registered.verify()
    donor = bind_donor(donor)
    implementation = code_pin()
    if donor.teacher is None:
        raise ValueError("Midpoint calibration requires original frozen Teacher")
    collector = CalibrationBuilder(registered.sample_ids, donor.binding, k=k,
        memory_budget_bytes=memory_budget_bytes, include_condq=include_condq,
        mask_policy=mask_policy)
    scharr = ScharrAbsolute(registered.sensor).to(device) if include_condq else None
    started, seen = time.monotonic(), 0
    for rr in registered.calibration_batches(batch_size, device=device):
        validate_training_batch(rr, donor.binding, expected_sample_ids=registered.sample_ids,
                                require_unaugmented=True)
        out = frozen_midpoint_predictions(donor.aligner, donor.teacher, rr, sensor=registered.sensor)
        if include_condq:
            e_t, g, q = reference_maps(scharr, out["teacher_y"], out["gt_midpoint"], out["pan_aligned"])
        else:
            e_t = (out["teacher_y"] - out["gt_midpoint"]).abs().mean(1, keepdim=True).detach()
            g = q = None
        collector.add(rr.lineage.sample_ids, e_t, g, q, sensor=rr.lineage.sensor,
            split=rr.lineage.split, resolution=rr.lineage.resolution,
            unaugmented=all(a == (False, False, 0) for a in rr.lineage.augmentations),
            support_ok=bool(out["spatial_support_ok"].all()), loss_mask=out["loss_mask"])
        seen += len(rr.pan)
        if on_progress is not None:
            on_progress({"seen": seen, "expected": len(registered.sample_ids),
                         "elapsed_seconds": time.monotonic() - started})
    donor.assert_frozen()
    registered.verify()
    if code_pin() != implementation:
        raise ValueError("Implementation changed during midpoint calibration")
    stats = collector.finalize()
    stats.assert_integrity(full_hash=True)
    receipt = stats.save(path)
    return {**receipt, "kind": stats.calibration_kind, "tau_e": stats.tau_e,
            "mask_policy": stats.mask_policy, "calibration_schema": stats.schema,
            "sample_count": seen, "n_e": stats.metadata["n_e"], "n_q": stats.metadata["n_q"],
            "estimated_peak_host_bytes": collector.estimated_peak_bytes,
            "elapsed_seconds": time.monotonic() - started, "donor_binding": donor.binding,
            "midpoint_geometry": geometry_manifest()}
