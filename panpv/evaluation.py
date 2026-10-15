"""Private-data diagnostic protocol, explicitly NOT the benchmark RR20 wrapper.

All methods share complete valid 32x32 windows anchored to the supplied source
coordinate grid. Metrics are window-macro averages, not full-scene metrics.
Stored model/Brovey predictions remain float; analysis clips all methods to the
same declared DN range. Only Q8's pinned primitive rounds internally to uint16.
No FR/HQNR claim is made by this RR-only evaluator.
"""
from __future__ import annotations

import warnings
import numpy as np

from deploy.pandep_shared.project.src.pan_shared.metrics.eval_rr import ergas, sam
from deploy.pandep_shared.project.src.pan_shared.metrics.extras import (
    psnr_global, scc_dlpan, ssim_skimage,
)
from deploy.pandep_shared.project.src.pan_shared.metrics.q2n import q2n

PROTOCOL = {
    'id': 'PANPV_RR_VALID32_WINDOW_MACRO_v1',
    'scope': 'provided_RR_spatial_audit_windows_NOT_full_scene',
    'window_size': 32, 'stride': 32, 'window_anchor': 'global_source_MS_origin_0',
    'incomplete_or_partly_invalid_windows': 'exclude_for_ALL_metrics_and_methods',
    'aggregation': 'arithmetic_mean_of_per_window_metrics; never_patch_or_scene_claim',
    'output_analysis': 'clip_prediction_DN_0_to_2047; no_rounding_except_Q8',
    'target': 'provided_uint16_RR_gt; no_second_degradation',
    'psnr_peak': 2047, 'ergas_ratio': 4, 'q8': 'pinned_q2n_uint16_block32_shift32',
    'scc': 'pinned_DLPan_Sobel_zero_boundary_per_window',
    'ssim': 'pinned_Gaussian11_sigma1.5_population_per_window',
    'cc': 'flattened_8band_Pearson_per_window',
    'undefined_value': 'null_with_count; any_undefined_makes_aggregate_null',
    'hqnr': None, 'jqm': None,
}


def valid_windows(mask, origin=(0, 0)):
    """origin=(row,column) in the RR target/MS source grid; no local phase reset."""
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2 or not np.isin(mask, [False, True]).all():
        raise ValueError('A finite binary HW mask is required')
    if len(origin) != 2 or any(int(x) != x for x in origin):
        raise ValueError('Integer source-grid row/column origin required')
    size = 32
    for row in range((-int(origin[0])) % size, mask.shape[0] - size + 1, size):
        for col in range((-int(origin[1])) % size, mask.shape[1] - size + 1, size):
            region = np.s_[row:row + size, col:col + size]
            if mask[region].all():
                yield region


class RRAuditAccumulator:
    """Bounded-memory, same-support streaming diagnostic accumulator."""
    keys = ('ergas', 'scc', 'sam', 'psnr', 'ssim', 'q8', 'rmse', 'cc')

    def __init__(self):
        self.sums = dict.fromkeys(self.keys, 0.0)
        self.undefined = dict.fromkeys(self.keys, 0)
        self.windows = self.samples = self.input_pixels = self.mask_pixels = 0
        self.out_of_range = self.output_elements = 0
        self.seen = set()

    def add(self, prediction, target, mask, *, sample_id, origin=(0, 0)):
        # Explicit CHW inputs prevent silently permuting the eight-band axis.
        pred, truth = np.asarray(prediction), np.asarray(target)
        if pred.ndim != 3 or pred.shape[0] != 8 or truth.shape != pred.shape:
            raise ValueError('Matching 8xHxW prediction and actual RR target required')
        if not np.isfinite(pred).all() or not np.isfinite(truth).all():
            raise ValueError('Nonfinite RR arrays, including outside-mask values')
        mask = np.asarray(mask)
        if mask.shape == (1,) + pred.shape[1:]:
            mask = mask[0]
        if mask.shape != pred.shape[1:]:
            raise ValueError('Mask and target grids differ')
        if sample_id in self.seen:
            raise ValueError('Duplicate RR sample ID')
        regions = list(valid_windows(mask, origin))
        self.seen.add(sample_id)
        self.samples += 1
        self.input_pixels += int(mask.size)
        self.mask_pixels += int(mask.sum())
        self.output_elements += int(pred.size)
        self.out_of_range += int(((pred < 0) | (pred > 2047)).sum())
        pred = np.clip(pred.astype(np.float64), 0, 2047).transpose(1, 2, 0)
        truth = truth.astype(np.float64).transpose(1, 2, 0)
        for region in regions:
            p, g = pred[region], truth[region]
            with warnings.catch_warnings(), np.errstate(divide='ignore', invalid='ignore'):
                warnings.simplefilter('ignore', RuntimeWarning)
                values = dict(ergas=ergas(p, g, 4), scc=scc_dlpan(p, g), sam=sam(p, g),
                              psnr=psnr_global(p, g, 2047), ssim=ssim_skimage(p, g, 2047),
                              q8=float(q2n(g, p, 32, 32)[0]),
                              rmse=float(np.sqrt(np.square(p - g).mean())),
                              cc=float(np.corrcoef(p.ravel(), g.ravel())[0, 1]))
            self.windows += 1
            for key, value in values.items():
                if np.isfinite(value):
                    self.sums[key] += float(value)
                else:
                    self.undefined[key] += 1

    def result(self):
        return {
            'protocol': PROTOCOL,
            'status': 'RR_DIAGNOSTIC_COMPLETE' if self.windows else 'NO_VALID_WINDOWS',
            'metrics': {key: self.sums[key] / self.windows
                        if self.windows and not self.undefined[key] else None for key in self.keys},
            'undefined_window_counts': dict(self.undefined),
            'samples': self.samples, 'valid_windows': self.windows,
            'evaluated_pixels': self.windows * 32 * 32,
            'input_pixels': self.input_pixels, 'mask_pixels': self.mask_pixels,
            'prediction_out_of_range_elements': self.out_of_range,
            'prediction_elements': self.output_elements,
            'fr_hqnr': None, 'fr_jqm': None,
        }
