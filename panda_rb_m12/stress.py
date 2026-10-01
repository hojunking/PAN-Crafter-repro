"""M12 fixed160/fixed192 response diagnostics; not checkpoint selection.

The frozen B01 numerical primitives remain untouched.  The new modes, fixed
ROIs, coverage aggregation and protocol identifiers belong only to M12.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from panda_rb.stress import (array_digest, validate_grid, shifted_inputs,
                             relative_response, geometric_coverage,
                             signed_edge_error, warp_convention_test, RAW_SUBSET)
from tools.metrics.eval_rr import ergas, sam
from tools.eval_dlpan import psnr_global

MODES = ('A_ON', 'A_ZERO_INFERENCE_ONLY', 'A_NATIVE_FIXED', 'KNOWN_SHIFT_INVERSE')
STRESS_KEYS = ('ergas', 'psnr', 'sam', 'edge_error_dn')
ROIS = {'fixed160': 48, 'fixed192': 32}
PROTOCOL_ID = 'PANDA_M12_RR20_FIXED160_PRIMARY_FIXED192_AUX_v1'


@torch.no_grad()
def stress_forward(model, pan, ms, native_lp, epsilon, mode, native_correction=None):
    if mode not in MODES:
        raise ValueError('Unregistered M12 inference mode')
    shifted, lp = shifted_inputs(pan, native_lp, epsilon)
    base = F.interpolate(ms.float(), scale_factor=4, mode='bicubic', align_corners=False)
    native = model.predict_delta(pan, base) if native_correction is None else native_correction
    if native.shape != epsilon.shape:
        raise ValueError('Native correction and epsilon must be corresponding HR (dy,dx)')
    estimated = native if torch.count_nonzero(epsilon).item() == 0 else model.predict_delta(shifted, base)
    applied = {'A_ON': estimated, 'A_ZERO_INFERENCE_ONLY': torch.zeros_like(estimated),
               'A_NATIVE_FIXED': native, 'KNOWN_SHIFT_INVERSE': native - epsilon}[mode]
    out = model(shifted, ms, lp, delta_override=applied)
    if not torch.allclose(out['delta'], applied.float(), atol=0, rtol=0, equal_nan=True):
        raise ValueError('Synchronized PAN/LP/HP frontend ignored the complete correction override')
    if all(k in out for k in ('pan_aligned', 'L', 'H')):
        if not torch.allclose(out['H'], out['pan_aligned'] - out['L'], atol=0, rtol=0, equal_nan=True):
            raise ValueError('Signed HP is not aligned PAN minus aligned LP')
    return out, estimated, shifted, lp


def stress_metrics(pred, truth, roi='fixed160'):
    if roi not in ROIS:
        raise ValueError('M12 permits only preregistered fixed160/fixed192 ROIs')
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != (8, 256, 256) or truth.shape != pred.shape:
        raise ValueError('Both ROI scores must use the same full256 WV3 prediction')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise FloatingPointError('Nonfinite full prediction/reference, including discarded borders')
    margin = ROIS[roi]
    a = pred[:, margin:-margin, margin:-margin].astype(np.float64)
    b = truth[:, margin:-margin, margin:-margin].astype(np.float64)
    ah, bh = a.transpose(1, 2, 0), b.transpose(1, 2, 0)
    return dict(ergas=float(ergas(ah, bh)), psnr=float(psnr_global(ah, bh, 2047.)),
                sam=float(sam(ah, bh)), edge_error_dn=signed_edge_error(a, b))


def point_valid(row):
    return (row.get('status') == 'ok' and row.get('coverage_all_pan_paths') == 1.
            and all(row.get(key) is not None and np.isfinite(row[key]) for key in STRESS_KEYS))


def summarize_curve(rows, shifts, roi='fixed160'):
    """Never discard a bad direction or scene from a clean Student mean."""
    shifts = validate_grid(shifts)
    if roi not in ROIS:
        raise ValueError('Unregistered ROI')
    if len(rows) != len(shifts) * 20 or any(row.get('roi') != roi for row in rows):
        raise ValueError('A curve must contain every 49 x 20 observation in one fixed ROI')
    points = []
    for shift in shifts:
        part = [r for r in rows if r['shift_id'] == shift['id']]
        if len(part) != 20 or sorted(r['scene_index'] for r in part) != list(range(20)):
            raise ValueError('Stress curve has a missing or duplicated RR scene')
        valid = sum(point_valid(r) for r in part)
        item = dict(shift, roi=roi, n_scenes=20, n_expected=20, n_valid=valid, n_failures=20-valid,
                    clean=valid == 20, status='CLEAN' if valid == 20 else 'PROCESSED_WITH_FLAGS')
        for key in (*STRESS_KEYS, 'relative_response_l1_sum', 'coverage_all_pan_paths'):
            values = [r.get(key) for r in part]
            finite = all(v is not None and np.isfinite(v) for v in values)
            item[key] = float(np.mean(values)) if finite else None
            item[key + '_scene_std'] = float(np.std(values, ddof=1)) if finite else None
            item['clean_' + key] = item[key] if valid == 20 else None
        points.append(item)
    zero = points[0]
    for point in points:
        for key in STRESS_KEYS:
            point['delta_from_zero_' + key] = (None if point[key] is None or zero[key] is None
                                               else point[key] - zero[key])
    radii = []
    for radius in (0., .25, .5, 1., 2., 3., 4.):
        part = [r for r in points if r['radius_hr'] == radius]
        expected_directions = 1 if radius == 0 else 8
        if len(part) != expected_directions:
            raise ValueError('Incomplete preregistered radius directions')
        valid = sum(r['n_valid'] for r in part)
        expected = expected_directions * 20
        summary = dict(radius_hr=radius, roi=roi, n_directions=expected_directions,
                       n_expected=expected, n_valid=valid, n_failures=expected-valid,
                       n_independent_students=1, clean=valid == expected)
        for key in (*STRESS_KEYS, 'relative_response_l1_sum', 'coverage_all_pan_paths'):
            values = [r[key] for r in part]
            finite = all(v is not None and np.isfinite(v) for v in values)
            summary['flagged_all_' + key] = float(np.mean(values)) if finite else None
            summary['clean_' + key] = summary['flagged_all_' + key] if valid == expected else None
        radii.append(summary)
    return dict(points=points, radii=radii, statistical_unit='Student; scene then direction means',
                missing_excluded=False, roi=roi, protocol_id=PROTOCOL_ID)
