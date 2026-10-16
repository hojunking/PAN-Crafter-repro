"""Explicit calibration and short diagnostics, NOT an approved campaign runner."""
from dataclasses import asdict
from pathlib import Path
import time

import torch

from directshift48.method import ScharrAbsolute
from directshift48.reporting import atomic_json, append_jsonl
from directshift48.runtime import environment_manifest
from ta2.evaluation import file_sha
from pa.warp import warp_support_mask
from .calibration import CalibrationBuilder, CalibrationStats
from .method import frozen_donor_predictions, ConditionalQStage2
from .losses import reference_maps
from .sources import validate_training_batch
from .stream import TrainStream
from .training import StudentTrainer, configure_precision
from .checkpoint import save_checkpoint
from .spec import MASK_POLICY, MASK2_POLICY


def code_pin():
    from donor24.runtime import code_pin as inherited
    root = Path(__file__).resolve().parents[1]
    return {**inherited(), **{str(p.relative_to(root)): file_sha(p)
                             for p in (root/'panda_stage2_condq').glob('*.py')}}


def calibrate(donor, registered, *, output, k, batch_size, memory_budget_bytes,
              include_condq, device='cpu', on_progress=None, mask_policy=MASK_POLICY):
    """All registered train rows or failure; never partial-statistics success."""
    path = Path(output)
    if path.exists():
        raise FileExistsError('Immutable calibration destination already exists')
    configure_precision()
    registered.verify()
    donor.assert_frozen()
    implementation = code_pin()
    if donor.teacher is None:
        raise ValueError('Calibration requires the frozen Teacher from the bound checkpoint')
    collector = CalibrationBuilder(registered.sample_ids, donor.binding, k=k,
        memory_budget_bytes=memory_budget_bytes, include_condq=include_condq,
        mask_policy=mask_policy)
    scharr = ScharrAbsolute(registered.sensor).to(device) if include_condq else None
    started = time.monotonic()
    seen = 0
    for rr in registered.calibration_batches(batch_size, device=device):
        validate_training_batch(rr, donor.binding, expected_sample_ids=registered.sample_ids,
                                require_unaugmented=True)
        out = frozen_donor_predictions(donor.aligner, donor.teacher, rr, sensor=registered.sensor)
        if include_condq:
            e_t, g, q = reference_maps(scharr, out['teacher_y'], rr.gt, out['pan_aligned'])
        else:
            e_t = (out['teacher_y'] - rr.gt).abs().mean(1, keepdim=True).detach()
            g = q = None
        collector.add(rr.lineage.sample_ids, e_t, g, q, sensor=rr.lineage.sensor,
            split=rr.lineage.split, resolution=rr.lineage.resolution,
            unaugmented=all(a == (False, False, 0) for a in rr.lineage.augmentations),
            support_ok=bool(out['spatial_support_ok'].all()),
            loss_mask=warp_support_mask(*rr.pan.shape[-2:], out['correction'].detach())
                      if mask_policy == MASK2_POLICY else None)
        seen += len(rr.pan)
        if on_progress is not None:
            on_progress({'seen': seen, 'expected': len(registered.sample_ids),
                         'elapsed_seconds': time.monotonic() - started})
    donor.assert_frozen()
    registered.verify()
    if code_pin() != implementation:
        raise ValueError('Implementation changed during calibration; no artifact published')
    stats = collector.finalize()
    # Capture lifecycle audit before any immutable publication.
    stats.assert_integrity(full_hash=True)
    receipt = stats.save(path)
    return {**receipt, 'kind': stats.calibration_kind, 'tau_e': stats.tau_e,
        'mask_policy': stats.mask_policy, 'calibration_schema': stats.schema,
        'sample_count': seen, 'n_e': stats.metadata['n_e'], 'n_q': stats.metadata['n_q'],
        'estimated_peak_host_bytes': collector.estimated_peak_bytes,
        'elapsed_seconds': time.monotonic() - started, 'donor_binding': donor.binding}


def diagnostic(donor, registered, spec, training, stream_spec, *, output,
               stats=None, device='cpu', updates=2, calibration_device_budget_bytes=None,
               hqnr_target=None):
    """One/two optimizer updates; isolated outputs cannot be mistaken for 50K."""
    if type(updates) is not int or updates not in (1, 2):
        raise ValueError('Only 1-2 diagnostic updates; formal campaign requires a separate plan')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Diagnostic run directory already exists; no overwrite/resume here')
    donor.assert_frozen()
    registered.verify()
    stage = ConditionalQStage2(spec, donor.aligner, donor.teacher,
                              donor_binding=donor.binding, stats=stats).to(device)
    if stats is not None:
        stage.move_calibration(device, memory_budget_bytes=calibration_device_budget_bytes)
    engine = StudentTrainer(stage, training, hqnr_target=hqnr_target)
    stream = TrainStream(registered, stream_spec, microbatch_size=training.microbatch_size,
                        effective_batch_size=training.effective_batch_size, device=device)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'DIAGNOSTIC_RUNNING', 'official_experiment': False, 'method': spec.manifest(),
        'training': asdict(training), 'stream': asdict(stream_spec), 'donor': donor.checkpoint_metadata,
        'contract': engine.contract(), 'code_pin': code_pin(), 'environment': environment_manifest(device),
        'calibration_augmentation': 'NONE', 'training_augmentation': asdict(stream_spec),
        'no_sheet_upload': True, 'requested_updates': updates}
    atomic_json(output/'manifest.json', manifest)
    tick = time.monotonic()
    try:
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(updates):
            if code_pin() != manifest['code_pin']:
                raise ValueError('Implementation changed during diagnostic; no mixed-code continuation')
            started = time.monotonic()
            report = engine.step_accumulated(stream.next_update())
            if str(device).startswith('cuda'):
                torch.cuda.synchronize(device)
            report['elapsed_seconds'] = time.monotonic() - started
            append_jsonl(output/'train.jsonl', report)
            print(f"[DIAGNOSTIC:{spec.variant}] update={engine.updates} loss={report['loss']:.7g} "
                  f"hard={report['hard_weighted']:.7g} soft={report['soft_weighted']:.7g}", flush=True)
        if code_pin() != manifest['code_pin']:
            raise ValueError('Implementation changed before diagnostic checkpoint')
        artifact = save_checkpoint(output/'diagnostic_checkpoint.pt', engine, stream_state=stream.state_dict())
        donor.assert_frozen()
        manifest.update(status='DIAGNOSTIC_COMPLETE', completed_updates=engine.updates,
                        elapsed_seconds=time.monotonic()-tick, checkpoint=artifact,
                        calibration_table_bytes=0 if stats is None else stats.memory_bytes,
                        cuda_peak_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None)
    except Exception as exc:
        manifest.update(status='DIAGNOSTIC_FAILED', error=repr(exc), completed_updates=engine.updates)
        atomic_json(output/'manifest.json', manifest)
        raise
    atomic_json(output/'manifest.json', manifest)
    return manifest
