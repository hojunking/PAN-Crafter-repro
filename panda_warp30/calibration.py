"""Exact full RR-train calibration, separately bound to each geometry policy."""
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
from .method import frozen_predictions, validate_warp_binding
from .spec import geometry_manifest


def bind_donor(donor, spec):
    if not isinstance(donor, LoadedDonor):
        raise TypeError('Strict fresh Warp30 LoadedDonor required')
    donor.assert_frozen()
    binding = deepcopy(donor.binding)
    old = binding.get('warp30_geometry')
    if old is not None and old != geometry_manifest(spec):
        raise ValueError('Incompatible pre-bound Warp30 geometry')
    binding['warp30_geometry'] = geometry_manifest(spec)
    validate_warp_binding(binding, spec)
    metadata = deepcopy(donor.checkpoint_metadata)
    metadata['warp30_geometry'] = geometry_manifest(spec)
    return replace(donor, binding=binding, checkpoint_metadata=metadata)


def code_pin():
    from panda_stage2_condq.runtime import code_pin as inherited
    root = Path(__file__).resolve().parents[1]
    files = list((root/'panda_warp30').glob('*.py'))
    files += [root/'panda_midpoint/resampling.py', root/'panda_midpoint/contract.py',
              root/'align/__init__.py', root/'align/resample.py']
    return {**inherited(), **{str(p.relative_to(root)): file_sha(p) for p in files}}


def calibrate(donor, registered, *, spec, output, k, batch_size, memory_budget_bytes,
              include_condq=True, device='cpu', on_progress=None, mask_policy=MASK2_POLICY):
    path = Path(output)
    if path.exists():
        raise FileExistsError('Immutable Warp30 calibration path already exists')
    if mask_policy != MASK2_POLICY or spec.mask_policy != mask_policy:
        raise ValueError('Warp30 requires MASK2 calibration')
    configure_precision()
    registered.verify()
    donor = bind_donor(donor, spec)
    implementation = code_pin()
    if donor.teacher is None:
        raise ValueError('Warp30 calibration requires frozen fresh Stage 1 Teacher')
    collector = CalibrationBuilder(registered.sample_ids, donor.binding, k=k,
        memory_budget_bytes=memory_budget_bytes, include_condq=include_condq,
        mask_policy=mask_policy)
    scharr = ScharrAbsolute(spec.sensor).to(device) if include_condq else None
    started, seen = time.monotonic(), 0
    for rr in registered.calibration_batches(batch_size, device=device):
        validate_training_batch(rr, donor.binding, expected_sample_ids=registered.sample_ids,
                                require_unaugmented=True)
        f = frozen_predictions(donor.aligner, donor.teacher, rr, spec=spec)
        if include_condq:
            e, g, q = reference_maps(scharr, f['teacher_y'], f['gt_loss'], f['pan_loss'])
        else:
            e = (f['teacher_y'] - f['gt_loss']).abs().mean(1, keepdim=True).detach()
            g = q = None
        collector.add(rr.lineage.sample_ids, e, g, q, sensor=rr.lineage.sensor,
            split=rr.lineage.split, resolution=rr.lineage.resolution,
            unaugmented=all(a == (False, False, 0) for a in rr.lineage.augmentations),
            support_ok=bool(f['spatial_support_ok'].all()), loss_mask=f['loss_mask'])
        seen += len(rr.pan)
        if on_progress:
            on_progress({'seen': seen, 'expected': len(registered.sample_ids),
                         'elapsed_seconds': time.monotonic()-started})
    donor.assert_frozen()
    registered.verify()
    if code_pin() != implementation:
        raise ValueError('Warp30 implementation changed during calibration')
    stats = collector.finalize()
    stats.assert_integrity(full_hash=True)
    receipt = stats.save(path)
    return {**receipt, 'kind': stats.calibration_kind, 'tau_e': stats.tau_e,
            'mask_policy': stats.mask_policy, 'calibration_schema': stats.schema,
            'sample_count': seen, 'n_e': stats.metadata['n_e'], 'n_q': stats.metadata['n_q'],
            'estimated_peak_host_bytes': collector.estimated_peak_bytes,
            'elapsed_seconds': time.monotonic()-started, 'donor_binding': donor.binding,
            'warp30_geometry': geometry_manifest(spec)}
