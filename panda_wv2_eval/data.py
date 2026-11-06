"""Native WV2 RR20/FR20, with truthful WV2 lineage and no GT in inference."""
import h5py
import numpy as np
import torch
from .common import ROOT, PLAN, digest, sha


def register_sensor():
    # This registration affects only this evaluation process. No source files change.
    from panda_method import global_data
    from ta2 import evaluation
    for sensors in (global_data.SENSORS, evaluation.SENSORS):
        if sensors.get('WV2', (8, 2047.)) != (8, 2047.):
            raise ValueError('Conflicting WV2 sensor definition')
        sensors['WV2'] = (8, 2047.)
    global_data.BAND_ORDER['WV2'] = ('Coastal', 'B', 'G', 'Y', 'R', 'RE', 'NIR1', 'NIR2')


class Sources:
    sensor, peak, bands = 'WV2', 2047., 8

    def __init__(self):
        register_sensor()
        self.paths = {}
        for split, item in PLAN['data'].items():
            p = (ROOT / item['path']).resolve(strict=True)
            if sha(p) != item['sha256']:
                raise ValueError('WV2 file differs from registered input: ' + split)
            self.paths[split] = p
            self.raw(split, [0])

    def manifest_sha(self, split):
        return digest({'sensor': 'WV2', 'split': split, 'source': PLAN['data'][split],
                       'band_order': PLAN['band_order'], 'radiometry': '2*DN/2047-1'})

    def raw(self, split, rows):
        if split not in ('rr', 'fr') or not rows or any(type(i) is not int or not 0 <= i < 20 for i in rows):
            raise ValueError('Only WV2 test scenes 0..19 are registered')
        side = 256 if split == 'rr' else 512
        shapes = {'pan': (20, 1, side, side), 'ms': (20, 8, side//4, side//4),
                  'lms': (20, 8, side, side)}
        if split == 'rr': shapes['gt'] = (20, 8, side, side)
        result = {}
        with h5py.File(self.paths[split]) as f:
            if split == 'fr' and 'gt' in f:
                raise ValueError('FR must not invent or include HRMS GT')
            for key, expected in shapes.items():
                d = f[key]
                if d.shape != expected or d.is_virtual or d.external or not isinstance(f.get(key, getlink=True), h5py.HardLink):
                    raise ValueError('Unexpected WV2 dataset shape/storage')
                result[key] = np.stack([d[i] for i in rows])
                if not np.isfinite(result[key]).all(): raise ValueError('Nonfinite WV2 input')
        return result

    def batch(self, split, rows, *, device='cpu'):
        from panda_method.edge_data import EdgeBatchLineage, EdgeObservationBatch
        raw = self.raw(split, rows)
        inputs = {k: torch.as_tensor(raw[k].astype(np.float32), device=device) * (2./self.peak) - 1
                  for k in ('pan', 'ms', 'lms')}
        lineage = EdgeBatchLineage(split='inference', resolution=split.upper(),
            source_manifest_sha256=self.manifest_sha(split), sample_ids=tuple(f'WV2:{split}:{i}' for i in rows),
            sensor='WV2', grid_name='PAN_' + split.upper(), augmentations=((False, False, 0),)*len(rows))
        return EdgeObservationBatch(inputs['pan'], inputs['ms'], inputs['lms'], lineage, gt=None).validate(8)
