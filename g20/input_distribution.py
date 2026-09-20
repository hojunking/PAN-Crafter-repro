"""Bounded split-local P1 input diagnostics; never fitted normalization or a gate."""
from datetime import datetime, timedelta, timezone
import hashlib
import logging

import h5py
import numpy as np

from g20.common import atomic_json, camp, check_deadline, object_sha, read, utcnow

QUANTILES = (.01, .05, .50, .95, .99)


def _radiometry(source, key, ids, maximum, seed, max_quantile_pixels, deadline):
    array = source[key]
    channels = array.shape[1]
    count = np.zeros(channels, np.int64)
    total = np.zeros(channels, np.float64)
    square = np.zeros(channels, np.float64)
    zeros, center, saturation, invalid = [np.zeros(channels, np.int64) for _ in range(4)]
    minimum, maximum_seen = np.full(channels, np.inf), np.full(channels, -np.inf)
    covariance_sum, covariance_cross = np.zeros(channels), np.zeros((channels, channels))
    covariance_count = 0
    quantile_values, positions = [], hashlib.sha256()
    rng = np.random.default_rng(seed)
    per_patch = max(1, max_quantile_pixels // len(ids))
    for index in ids:
        check_deadline(deadline)
        value = np.asarray(array[int(index)], np.float64).reshape(channels, -1)
        finite = np.isfinite(value)
        clean = np.where(finite, value, 0.)
        count += finite.sum(1)
        invalid += (~finite).sum(1)
        zeros += (value == 0).sum(1)
        center += (value == maximum / 2).sum(1)
        saturation += (value == maximum).sum(1)
        total += clean.sum(1)
        square += np.square(clean).sum(1)
        minimum = np.minimum(minimum, np.where(finite, value, np.inf).min(1))
        maximum_seen = np.maximum(maximum_seen, np.where(finite, value, -np.inf).max(1))
        common = value[:, finite.all(0)]
        covariance_sum += common.sum(1)
        covariance_cross += common @ common.T
        covariance_count += common.shape[1]
        pixels = np.sort(rng.choice(value.shape[1], min(per_patch, value.shape[1]), replace=False))
        positions.update(np.asarray([index], dtype='<i8').tobytes())
        positions.update(np.asarray(pixels, dtype='<i8').tobytes())
        quantile_values.append(value[:, pixels])
    if not count.all() or not covariance_count:
        raise ValueError(f'No finite radiometry support for {key}')
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean ** 2, 0))
    covmean = covariance_sum / covariance_count
    covariance = covariance_cross / covariance_count - np.outer(covmean, covmean)
    sampled = np.concatenate(quantile_values, axis=1)
    q = np.stack([np.quantile(band[np.isfinite(band)], QUANTILES) for band in sampled])
    if not np.isfinite(q).all():
        raise ValueError(f'Nonfinite quantile report for {key}')
    def values(scale, offset, zero_count):
        return dict(mean=(scale * mean + offset).tolist(), std=(scale * std).tolist(),
            minimum=(scale * minimum + offset).tolist(), maximum=(scale * maximum_seen + offset).tolist(),
            quantiles={f'p{int(p * 100)}': (scale * q[:, j] + offset).tolist()
                       for j, p in enumerate(QUANTILES)},
            covariance=(scale ** 2 * covariance).tolist(), zero_count=zero_count.tolist(),
            invalid_count=invalid.tolist(), saturation_count=saturation.tolist())
    return dict(raw_DN=values(1., 0., zeros), normalized=values(2. / maximum, -1., center),
        finite_pixels_per_band=count.tolist(), population_pixels_per_band=int(len(ids) * np.prod(array.shape[2:])),
        moments_scope='all pixels of selected base patches; population covariance (ddof=0)',
        covariance_complete_case_pixels=int(covariance_count),
        quantile_scope='exact on reproducible stratified pixel subsample; no population-fit claim',
        quantile_covers_all_selected_pixels=sampled.shape[1] == int(len(ids) * np.prod(array.shape[2:])),
        quantile_samples_per_band=sampled.shape[1], quantile_pixel_positions_sha256=positions.hexdigest(),
        quantile_seed=seed, quantile_sampling='local default_rng; sorted choice without replacement per selected patch',
        raw_saturation_threshold_DN=maximum, normalized_saturation_threshold=1.)


def _energies(source, lp_source, ids, maximum, deadline, low_texture_threshold):
    import torch
    from torch.nn import functional as F
    from pa.losses import scharr
    rows = []
    for index in ids:
        check_deadline(deadline)
        pan_dn = torch.from_numpy(np.asarray(source['pan'][int(index)], np.float32))[None]
        pan = pan_dn * (2. / maximum) - 1
        low_lr = torch.from_numpy(np.asarray(lp_source['lpan'][int(index)], np.float32))[None]
        low = F.interpolate(low_lr * (2. / maximum) - 1,
                            size=pan_dn.shape[-2:], mode='bicubic', align_corners=False)
        signals = dict(PAN=pan, LP=low, HP=pan - low)
        energy = {}
        for name, signal in signals.items():
            gx, gy = scharr(signal)
            energy[name] = dict(mean_absolute=float(signal.abs().mean()),
                                mean_square=float(signal.square().mean()),
                                edge_mean_absolute=float((.5 * (gx.abs() + gy.abs())).mean()))
        rows.append(dict(sample_id=int(index), energy=energy,
                         low_texture=energy['PAN']['edge_mean_absolute'] <= low_texture_threshold))
    return dict(rows=rows, low_texture_ratio=float(np.mean([r['low_texture'] for r in rows])),
        low_texture_rule=f'normalized native PAN mean(0.5*(abs(Scharr_x)+abs(Scharr_y))) <= {low_texture_threshold}',
        low_texture_threshold_fixed=True, threshold_fitted=False,
        signal_scope='native unaligned PAN, bicubic(LPAN,align_corners=False), signed PAN-LP; normalized units',
        edge_definition='existing pa.losses.scharr, including its padding; no GT used',
        means={name: {metric: float(np.mean([r['energy'][name][metric] for r in rows]))
                      for metric in ('mean_absolute', 'mean_square', 'edge_mean_absolute')}
               for name in ('PAN', 'LP', 'HP')})


def capture_input_distributions(manifest, *, deadline_utc=None, max_samples=128,
                                seed=5678, max_quantile_pixels=65536,
                                low_texture_threshold=1e-3):
    """Read bounded patch samples independently per split. No cross-split fit."""
    if max_samples < 1 or max_quantile_pixels < max_samples or low_texture_threshold < 0:
        raise ValueError('Invalid bounded diagnostic sampling policy')
    report = dict(schema='G20_INPUT_DISTRIBUTIONS_v1', status='MEASURED', complete=True,
        p0_gate=False, dataset_manifest_sha256=object_sha(manifest), sensor=manifest['sensor'],
        max_DN=manifest['max_pixel'], band_order=manifest['band_order'], sampling_seed=seed,
        max_samples_per_split=max_samples, max_quantile_pixels_per_band=max_quantile_pixels,
        sampling_policy=dict(max_samples=max_samples, seed=seed, max_quantile_pixels=max_quantile_pixels,
                             low_texture_threshold=low_texture_threshold),
        normalization_fitted=False, train_test_mixed=False, splits={})
    for split in ('train', 'val', 'rr', 'fr'):
        check_deadline(deadline_utc)
        item = manifest['splits'][split]
        with h5py.File(item['dataroot'], 'r') as source, h5py.File(item['lpan_path'], 'r') as lp:
            n = len(source['pan'])
            if n < 1:
                raise ValueError('Empty diagnostic source')
            ids = np.sort(np.random.default_rng(seed).choice(n, min(n, max_samples), replace=False))
            keys = ['pan', 'ms', 'lms'] + ([] if split == 'fr' else ['gt'])
            radiometry = {key: _radiometry(source, key, ids, manifest['max_pixel'], seed,
                            max_quantile_pixels, deadline_utc) for key in keys}
            radiometry['lpan'] = _radiometry(lp, 'lpan', ids, manifest['max_pixel'], seed,
                                            max_quantile_pixels, deadline_utc)
            energy = _energies(source, lp, ids, manifest['max_pixel'], deadline_utc, low_texture_threshold)
            report['splits'][split] = dict(source_sha256=item['sha256'], lpan_sha256=item['lpan_sha256'],
                sample_ids=ids.tolist(), sample_ids_sha256=object_sha(ids.tolist()),
                population_count=n, selected_count=len(ids), all_patches=len(ids) == n,
                radiometry=radiometry, energy=energy,
                has_reconstruction_GT=split != 'fr')
    return report


def record_input_distributions(root, server, manifest, *, deadline_utc=None, budget_seconds=120, **kwargs):
    """P1 best-effort publication. All diagnostic failures remain nonblocking."""
    path = camp(root, server) / 'diagnostics/input_distribution.json'
    try:
        policy = dict(max_samples=128, seed=5678, max_quantile_pixels=65536, low_texture_threshold=1e-3)
        policy.update(kwargs)
        if budget_seconds <= 0:
            raise TimeoutError('P1 distribution time budget is exhausted')
        stop = datetime.now(timezone.utc) + timedelta(seconds=budget_seconds)
        if deadline_utc:
            campaign_stop = datetime.fromisoformat(str(deadline_utc).replace('Z', '+00:00'))
            if campaign_stop.tzinfo is None:
                raise ValueError('P1 deadline requires a timezone')
            stop = min(stop, campaign_stop)
        deadline_utc = stop.isoformat()
        previous = read(path)
        if (previous.get('complete') and previous.get('dataset_manifest_sha256') == object_sha(manifest)
                and previous.get('sampling_policy') == policy):
            return previous
        report = capture_input_distributions(manifest, deadline_utc=deadline_utc, **kwargs)
        if manifest['sensor'] == 'QB':
            from qg40.phase_audit import qb_phase_residuals
            try:
                report['qb_phase_residuals'] = qb_phase_residuals(manifest, deadline_utc=deadline_utc)
            except Exception as error:
                report['qb_phase_residuals'] = dict(status='INCOMPLETE', p0_gate=False, reason=str(error))
    except Exception as error:
        report = dict(schema='G20_INPUT_DISTRIBUTIONS_v1', status='INCOMPLETE', complete=False,
                      p0_gate=False, dataset_manifest_sha256=object_sha(manifest), reason=str(error))
    report['at_utc'] = utcnow()
    report['max_p1_seconds'] = budget_seconds
    if not report.get('complete'):
        logging.getLogger(__name__).warning('G20 P1 input distribution incomplete: %s', report.get('reason'))
    phase = report.get('qb_phase_residuals', {})
    if phase.get('status') == 'INCOMPLETE':
        logging.getLogger(__name__).warning('G20 P1 QB phase residual analysis incomplete: %s', phase.get('reason'))
    try:
        atomic_json(path, report)
    except Exception as error:
        report['publication_error'] = str(error)
        logging.getLogger(__name__).warning('G20 P1 input distribution receipt unavailable at %s: %s', path, error)
    return report
