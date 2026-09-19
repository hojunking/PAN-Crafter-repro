"""Fail-soft P1 reconstruction-error distributions; never fit or modify tau.

Automatic publication uses a bounded diagnostic budget. Partial measurements
declare actual IDs/counts and cannot masquerade as the full train3072/val audit.
"""
from pathlib import Path
import time

import numpy as np
import torch

from qg40.common import atomic_json, check_deadline, object_sha
from qg40.reference_parity import isolated_teacher

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def _summary(parts, threshold):
    if not parts:
        return dict(actual_samples=0, status='NOT_MEASURED')
    errors = np.concatenate([p['errors'] for p in parts])
    texture = np.concatenate([p['texture'] for p in parts])
    ids = np.concatenate([p['ids'] for p in parts])
    per_band = {k: np.concatenate([p[k] for p in parts]) for k in ('l1', 'mse', 'bias')}
    def selected(mask):
        if not mask.any():
            return dict(samples=0, error_quantiles=None, per_band=None)
        return dict(samples=int(mask.sum()), error_quantiles=dict(zip(
            ('p'+str(p) for p in PERCENTILES), np.percentile(errors[mask], PERCENTILES).tolist())),
            per_band={k: a[mask].mean(0, dtype=np.float64).tolist() for k, a in per_band.items()})
    return dict(actual_samples=len(ids), indices=ids.tolist(), indices_sha256=object_sha(ids.tolist()),
        full_support=True, error_definition='mean_band(abs(final_HRMS - GT)); normalized image units',
        error_pixels=int(errors.size), all=selected(np.ones(len(ids), dtype=bool)),
        low_texture=selected(texture <= threshold), high_texture=selected(texture > threshold),
        texture_values=texture.tolist(), texture_quantiles=np.percentile(texture, PERCENTILES).tolist())


def measure_error_distribution(model, train, val, calibration_indices, tau_R, *,
                               raw_calibration_median, device='cuda', batch_size=32,
                               deadline=None, reference_identity=None, synthetic_test=False):
    """Alternate train/val batches; a short budget still observes both domains."""
    if batch_size < 1 or not train.has_gt or not val.has_gt:
        raise ValueError('P1 error diagnostics require positive batch size and real train/val GT')
    from qg40.calibration import select_calibration_indices
    train_ids = np.asarray(calibration_indices)
    if not synthetic_test and not np.array_equal(train_ids, select_calibration_indices(train.base_count)):
        raise ValueError('P1 train IDs must be the immutable calibration subset')
    if train_ids.ndim != 1 or not np.issubdtype(train_ids.dtype, np.integer):
        raise ValueError('Invalid P1 calibration IDs')
    selections = {'train_calibration': train_ids, 'validation': np.arange(val.base_count)}
    datasets = {'train_calibration': train, 'validation': val}
    cursors, parts = dict.fromkeys(selections, 0), {key: [] for key in selections}
    start, limited = time.monotonic(), False
    with isolated_teacher(model):
        try:
            while any(cursors[key] < len(ids) for key, ids in selections.items()):
                for key, ids in selections.items():
                    if cursors[key] >= len(ids):
                        continue
                    check_deadline(deadline)
                    selected = ids[cursors[key]:cursors[key]+batch_size]
                    rows = [datasets[key].base(int(index)) for index in selected]
                    gt, ms, lp, pan = [torch.stack([row[j] for row in rows]).to(device)
                                       for j in (0, 2, 3, 4)]
                    output = model(pan, ms, lp)['y']
                    residual = output.float() - gt.float()
                    if residual.shape != gt.shape or residual.shape[1] != 4 or not torch.isfinite(residual).all():
                        raise ValueError('Invalid final C4 output in reconstruction-error diagnostics')
                    # Fixed, descriptive texture proxy in normalized native PAN units.
                    texture = (pan[..., 1:, :]-pan[..., :-1, :]).abs().mean((1, 2, 3))
                    texture += (pan[..., :, 1:]-pan[..., :, :-1]).abs().mean((1, 2, 3))
                    r64 = residual.double()
                    parts[key].append(dict(ids=selected, errors=residual.abs().mean(1).cpu().numpy(),
                        texture=texture.cpu().numpy(), l1=r64.abs().mean((2, 3)).cpu().numpy(),
                        mse=r64.square().mean((2, 3)).cpu().numpy(), bias=r64.mean((2, 3)).cpu().numpy()))
                    cursors[key] += len(selected)
        except TimeoutError:
            limited = True
    training_texture = np.concatenate([p['texture'] for p in parts['train_calibration']]) if parts['train_calibration'] else []
    threshold = float(np.median(training_texture)) if len(training_texture) else None
    reports = {key: _summary(value, threshold) for key, value in parts.items()}
    difference = None
    if all(r['actual_samples'] for r in reports.values()):
        difference = {metric: (np.asarray(reports['validation']['all']['per_band'][metric]) -
                               np.asarray(reports['train_calibration']['all']['per_band'][metric])).tolist()
                      for metric in ('l1', 'mse', 'bias')}
    return dict(schema='QG40_RECONSTRUCTION_DISTRIBUTION_v1',
        status='PARTIAL_BUDGET' if limited else 'MEASURED', p0_gate=False, diagnostic_only=True,
        synthetic_test=bool(synthetic_test), reference_identity=reference_identity,
        requested_counts={k: len(v) for k, v in selections.items()}, splits=reports,
        tau_R=tau_R, tau_raw_pooled_median=raw_calibration_median,
        tau_floor_used=bool(raw_calibration_median < 1e-6), tau_refitted=False,
        sampling='fixed train calibration order and validation source order; no augmentation',
        texture_proxy='mean_abs_vertical_difference + mean_abs_horizontal_difference of native PAN',
        texture_threshold=threshold, texture_threshold_source='median of measured train calibration patches only',
        strata_use='diagnostic only; no sample reweighting or calibration/loss changes',
        validation_minus_train_per_band=difference,
        e_student_minus_teacher='measured separately in Student diagnostics; unavailable for Teacher calibration',
        seconds=time.monotonic()-start)


def write_error_distribution(path, model, train, val, calibration_indices, tau_R, **kwargs):
    """P1 failures leave an explicit receipt, never invalidate a P0 reference."""
    try:
        report = measure_error_distribution(model, train, val, calibration_indices, tau_R, **kwargs)
    except Exception as error:
        report = dict(schema='QG40_RECONSTRUCTION_DISTRIBUTION_v1', status='DIAGNOSTIC_FAILED',
                      p0_gate=False, tau_R=tau_R, tau_refitted=False,
                      reason=f'{type(error).__name__}: {error}')
        print('QG40 P1 reconstruction diagnostics failed: ' + report['reason'], flush=True)
    try:
        atomic_json(Path(path), report)
    except Exception as error:
        report = dict(report, publication_error=f'{type(error).__name__}: {error}')
        print('QG40 P1 diagnostic receipt could not be saved: ' + report['publication_error'], flush=True)
    return report
