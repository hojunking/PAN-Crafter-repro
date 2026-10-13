"""Stored-LMS observations for input-specific EdgeMatch, with no prior assets.

Keep the source registry, QB original-LMS decision and observation scaling.
Matching occurs after online image augmentation; no precomputed vector is read
or transformed. Official RR/FR arrays retain their original whole-image grids.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import copy
import json

import h5py
import numpy as np
import torch

from .calibration import canonical_sha256
from .data import require_sha256
from .global_data import BAND_ORDER, SENSORS, file_sha, stamp


@dataclass(frozen=True)
class EdgeBatchLineage:
    split: str
    resolution: str
    source_manifest_sha256: str
    sample_ids: tuple
    sensor: str
    grid_name: str
    augmentations: tuple
    reference: str = 'PROVIDED_STORED_LMS'
    geometry_evidence: str = 'Stored H5 coordinates; offline augmentation unknown'

    def __post_init__(self):
        if self.split not in ('train', 'validation', 'test', 'inference'):
            raise ValueError('Explicit split required')
        if self.sensor not in SENSORS or self.resolution not in ('RR', 'FR'):
            raise ValueError('Explicit sensor and RR/FR resolution required')
        if self.grid_name != 'PAN_' + self.resolution:
            raise ValueError('Current input PAN grid required')
        if self.reference != 'PROVIDED_STORED_LMS':
            raise ValueError('This adapter requires the provided stored LMS')
        require_sha256(self.source_manifest_sha256, 'source manifest')
        if (not self.sample_ids or len(self.sample_ids) != len(self.augmentations)
                or any(not isinstance(v, str) or not v for v in self.sample_ids)):
            raise ValueError('Sample identity and augmentation count mismatch')
        for aug in self.augmentations:
            if (len(aug) != 3 or type(aug[0]) is not bool or type(aug[1]) is not bool
                    or type(aug[2]) is not int or aug[2] not in range(4)):
                raise ValueError('Explicit Hflip/Vflip/CCW rotation required')
        if not isinstance(self.geometry_evidence, str) or not self.geometry_evidence.strip():
            raise ValueError('Geometry evidence required')
        object.__setattr__(self, 'sample_ids', tuple(self.sample_ids))
        object.__setattr__(self, 'augmentations', tuple(tuple(a) for a in self.augmentations))

    def require_training(self):
        if self.split != 'train' or self.resolution != 'RR':
            raise ValueError('Only registered RR train observations may update Stage 1')


@dataclass(frozen=True)
class EdgeObservationBatch:
    pan: torch.Tensor
    ms: torch.Tensor
    native_lms: torch.Tensor
    lineage: EdgeBatchLineage
    gt: torch.Tensor | None = None

    def validate(self, bands, *, training=False, require_gt=False):
        if not isinstance(self.lineage, EdgeBatchLineage):
            raise TypeError('Edge input lineage required, never legacy scene/global prior lineage')
        p, m, lms = self.pan, self.ms, self.native_lms
        if p.ndim != 4 or p.shape[0] == 0 or p.shape[1] != 1:
            raise ValueError('Nonempty PAN N1HW required')
        n, _, h, w = p.shape
        if min(h, w) < 8 or h % 4 or w % 4:
            raise ValueError('PAN spatial sides >=8 and divisible by four required')
        if (bands != SENSORS[self.lineage.sensor][0]
                or m.shape != (n, bands, h//4, w//4)
                or lms.shape != (n, bands, h, w)
                or len(self.lineage.sample_ids) != n):
            raise ValueError('PAN/MS/provided LMS/band correspondence mismatch')
        if require_gt and self.gt is None:
            raise ValueError('RR supervised Stage 1 requires GT')
        if self.gt is not None and self.gt.shape != lms.shape:
            raise ValueError('GT shape mismatch')
        if self.lineage.resolution == 'FR' and self.gt is not None:
            raise ValueError('FR inference has no HRMS GT; do not fabricate one')
        for t in (p, m, lms, self.gt):
            if t is not None and (t.dtype not in (torch.float32, torch.float64)
                                  or t.device != p.device or t.dtype != p.dtype
                                  or not bool(torch.isfinite(t).all())):
                raise ValueError('Finite FP32/FP64 observations on one dtype/device required')
        if training:
            self.lineage.require_training()
        return self


def augment_edge_batch(batch, *, hflip, vflip, rot90):
    """Transform observations once. EdgeMatch is recomputed afterwards."""
    batch.validate(batch.ms.shape[1])
    if type(hflip) is not bool or type(vflip) is not bool or type(rot90) is not int:
        raise ValueError('Explicit bool flips and integer rotation required')
    if any(a != (False, False, 0) for a in batch.lineage.augmentations):
        raise ValueError('Online augmentation must be applied once')
    def change(t):
        if t is None:
            return None
        if hflip:
            t = t.flip(-1)
        if vflip:
            t = t.flip(-2)
        return torch.rot90(t, rot90, (-2, -1))
    lineage = replace(batch.lineage,
        augmentations=((hflip, vflip, rot90 % 4),) * len(batch.pan),
        geometry_evidence=batch.lineage.geometry_evidence +
            f'; online H={hflip},V={vflip},CCW={rot90%4}; matching after augmentation')
    return replace(batch, pan=change(batch.pan), ms=change(batch.ms),
                   native_lms=change(batch.native_lms), gt=change(batch.gt),
                   lineage=lineage).validate(batch.ms.shape[1])


class EdgeDataSources:
    """Actual registered PAN/MS/LMS/GT, no b_global, scene mapping or FR train.

    QB uses original stored LMS plus separately registered repaired LRMS. Their
    consumed PAN/GT rows must be equal. This records the hybrid source contract;
    it does not assert LMS=U_phase(repaired MS) was independently established.
    """
    def __init__(self, catalog_path, *, root, sensor, lms_overrides=None):
        if sensor not in SENSORS:
            raise ValueError('Explicit WV3/QB/GF2 sensor required')
        self.root, self.sensor = Path(root).resolve(), sensor
        self.bands, self.peak = SENSORS[sensor]
        self.catalog_path = Path(catalog_path).resolve(strict=True)
        before = stamp(self.catalog_path)
        data = json.loads(self.catalog_path.read_text())
        self.catalog_sha256 = file_sha(self.catalog_path)
        if stamp(self.catalog_path) != before:
            raise ValueError('Source registry changed during read')
        if data.get('schema') != 'ABLR2_SENSOR_SOURCES_v1':
            raise ValueError('Existing sensor source registry required')
        self.entry = copy.deepcopy(data['sensors'][sensor])
        if tuple(self.entry['band_order']) != BAND_ORDER[sensor]:
            raise ValueError('Band order differs from sensor convention')
        self.lms_overrides = copy.deepcopy(lms_overrides or {})
        if sensor == 'QB':
            if (set(self.lms_overrides) != {'train', 'val'}
                    or any(not r.get('reason') for r in self.lms_overrides.values())):
                raise ValueError('QB requires explicit original LMS train/val overrides')
        elif self.lms_overrides:
            raise ValueError('Unrequested sensor LMS override')
        self._files = {}
        self._verify('train')
        if 'train' in self.lms_overrides:
            self._verify('train', lms=True)

    def _verify(self, split, *, lms=False):
        if split not in ('train', 'val', 'rr', 'fr'):
            raise ValueError('Only registered train/val/rr/fr splits allowed')
        ref = self.lms_overrides[split] if lms else self.entry['splits'][split]
        path = (self.root / ref['path']).resolve(strict=True)
        require_sha256(ref['sha256'], split + ' file SHA')
        now, identity = stamp(path), (split, lms)
        if identity in self._files:
            if self._files[identity] != (path, now):
                raise ValueError('Pinned H5 changed after verification')
        else:
            if file_sha(path) != ref['sha256'] or stamp(path) != now:
                raise ValueError('Source H5 SHA mismatch/changed during verification')
            self._files[identity] = (path, now)
        return path

    def manifest_sha(self, split):
        if split not in self.entry['splits']:
            raise ValueError('Unregistered source split')
        return canonical_sha256({'schema': 'PANDA_EDGE_OBSERVATIONS_V1',
            'sensor': self.sensor, 'registry_sha256': self.catalog_sha256,
            'split': split, 'source': self.entry['splits'][split],
            'band_order': self.entry['band_order'], 'radiometry': '2*stored_DN/peak-1',
            'aligner_reference': 'PROVIDED_STORED_LMS',
            'lms_override': self.lms_overrides.get(split),
            'reconstruction_ms': 'unchanged_registered_source',
            'lms_reproduction_from_active_ms': 'not_claimed',
            'prior_asset': None, 'scene_mapping': None,
            'offline_augmentation': 'unknown_stored_coordinates',
            'source_provenance': self.entry.get('source_provenance', {})})

    def training_sources(self):
        from .edge_guided import EdgeTrainingSources
        return EdgeTrainingSources(sensor=self.sensor, rr_manifest_sha256=self.manifest_sha('train'))

    def raw(self, split, rows):
        path = self._verify(split)
        if not rows or any(type(i) is not int or i < 0 for i in rows):
            raise ValueError('Nonempty explicit H5 row indices required')
        side = {'train': 64, 'val': 64, 'rr': 256, 'fr': 512}[split]
        keys = ('pan', 'ms', 'lms') + (() if split == 'fr' else ('gt',))
        with h5py.File(path, 'r') as f:
            n = f['pan'].shape[0]
            if (split in ('rr', 'fr') and n != 20) or max(rows) >= n:
                raise ValueError('Invalid row or official evaluation is not all20')
            expected = {'pan': (n, 1, side, side), 'ms': (n, self.bands, side//4, side//4),
                        'lms': (n, self.bands, side, side), 'gt': (n, self.bands, side, side)}
            result = {key: self._read(f, path, key, rows, expected[key]) for key in keys}
        self._verify(split)
        for key in ('pan', 'gt'):
            if key in result and (result[key].min() < 0 or result[key].max() > self.peak):
                raise ValueError('PAN/GT outside nominal sensor DN range')
        if split in self.lms_overrides:
            lms_path = self._verify(split, lms=True)
            with h5py.File(lms_path, 'r') as f:
                for key in ('pan', 'gt', 'lms'):
                    a = self._read(f, lms_path, key, rows, expected[key])
                    if key in ('pan', 'gt') and not np.array_equal(a, result[key]):
                        raise ValueError('Original LMS and active source have different PAN/GT rows')
                    if key == 'lms':
                        result['lms'] = a
            self._verify(split, lms=True)
        return result

    @staticmethod
    def _read(f, path, key, rows, shape):
        if not isinstance(f.get(key, getlink=True), h5py.HardLink):
            raise ValueError('Stored arrays must not be external/soft H5 links')
        d = f[key]
        if (not isinstance(d, h5py.Dataset) or d.is_virtual or d.external
                or Path(d.file.filename).resolve() != path or d.shape != shape
                or d.dtype.kind not in 'uif'):
            raise ValueError('Stored H5 array shape/storage/dtype mismatch')
        a = np.stack([d[i] for i in rows])
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite source observation')
        return a

    def batch(self, split, rows, *, device='cpu', augmentation=(False, False, 0)):
        arrays = self.raw(split, rows)
        tensor = {k: torch.as_tensor(v.astype(np.float32), device=device) * (2./self.peak) - 1.
                  for k, v in arrays.items()}
        resolution = 'FR' if split == 'fr' else 'RR'
        lineage = EdgeBatchLineage(
            split={'train': 'train', 'val': 'validation', 'rr': 'test', 'fr': 'test'}[split],
            resolution=resolution, source_manifest_sha256=self.manifest_sha(split),
            sample_ids=tuple(f'{self.sensor}:{split}:{i}' for i in rows), sensor=self.sensor,
            grid_name='PAN_' + resolution, augmentations=((False, False, 0),) * len(rows))
        return augment_edge_batch(EdgeObservationBatch(tensor['pan'], tensor['ms'],
            tensor['lms'], lineage, tensor.get('gt')), hflip=augmentation[0],
            vflip=augmentation[1], rot90=augmentation[2])
