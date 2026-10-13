"""Stored-LMS RR observations with dataset-global, not invented scene, lineage.

This is the explicit bGlobal path. Legacy scene-aware adapters remain separate.
No source images, priors, scene IDs, crop origins or interpolation are invented.
FR observations may be inspected as raw evaluation data; method batches remain
blocked until an explicit FR prior/conditioning policy is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import copy
import hashlib
import json

import h5py
import numpy as np
import torch

from .calibration import canonical_sha256
from .data import require_sha256
from .geometry import transform_prior
from .global_prior import GlobalPriorAsset


SENSORS = {'WV3': (8, 2047.), 'QB': (4, 2047.), 'GF2': (4, 1023.)}
BAND_ORDER = {'WV3': ('Coastal', 'B', 'G', 'Y', 'R', 'RE', 'NIR1', 'NIR2'),
              'QB': ('B', 'G', 'R', 'NIR'), 'GF2': ('B', 'G', 'R', 'NIR')}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


@dataclass(frozen=True)
class GlobalBatchLineage:
    split: str
    resolution: str
    source_manifest_sha256: str
    prior_mapping_sha256: str
    sample_ids: tuple
    grid_name: str
    geometry_evidence: str
    sensor: str
    train_source_sha256: str
    canonical_prior_dy_dx: tuple
    # One (horizontal flip, vertical flip, CCW quarter turns) per sample.
    augmentations: tuple
    prior_scope: str = 'dataset_global'

    def __post_init__(self):
        if self.split not in ('train', 'validation', 'test', 'inference'):
            raise ValueError('Explicit split is required')
        if self.resolution not in ('RR', 'FR') or self.grid_name != 'PAN_' + self.resolution:
            raise ValueError('Explicit current PAN grid is required')
        if self.sensor not in SENSORS or self.prior_scope != 'dataset_global':
            raise ValueError('Single-sensor dataset_global lineage required')
        for key in ('source_manifest_sha256', 'prior_mapping_sha256', 'train_source_sha256'):
            require_sha256(getattr(self, key), key)
        if not isinstance(self.geometry_evidence, str) or not self.geometry_evidence.strip():
            raise ValueError('Stored/online geometry convention must be recorded')
        if (not self.sample_ids or len(self.sample_ids) != len(self.augmentations)
                or any(not isinstance(v, str) or not v for v in self.sample_ids)):
            raise ValueError('Sample IDs and augmentation records must agree')
        prior = np.asarray(self.canonical_prior_dy_dx, dtype=np.float64)
        if prior.shape != (2,) or not np.isfinite(prior).all() or not prior.any():
            raise ValueError('Finite nonzero canonical prior required; no fallback')
        for aug in self.augmentations:
            if (len(aug) != 3 or type(aug[0]) is not bool or type(aug[1]) is not bool
                    or type(aug[2]) is not int or aug[2] not in range(4)):
                raise ValueError('Explicit H/V/rot90 augmentation tuple required')
        object.__setattr__(self, 'sample_ids', tuple(self.sample_ids))
        object.__setattr__(self, 'augmentations', tuple(tuple(a) for a in self.augmentations))
        object.__setattr__(self, 'canonical_prior_dy_dx', tuple(float(v) for v in prior))

    def require_training(self):
        if self.split != 'train' or self.resolution != 'RR':
            raise ValueError('Only RR train observations may enter bGlobal training')


@dataclass(frozen=True)
class GlobalObservationBatch:
    pan: torch.Tensor
    ms: torch.Tensor
    native_lms: torch.Tensor
    prior_dy_dx: torch.Tensor
    lineage: GlobalBatchLineage
    gt: torch.Tensor | None = None
    lpan: torch.Tensor | None = None

    def validate(self, bands, *, training=False, require_gt=False):
        if not isinstance(self.lineage, GlobalBatchLineage):
            raise TypeError('Dataset-global lineage required, not legacy scene lineage')
        if self.lineage.resolution == 'FR':
            raise ValueError('FR bGlobal prior policy unresolved; no implicit RR/4x prior')
        p, m, lms = self.pan, self.ms, self.native_lms
        if p.ndim != 4 or p.shape[0] == 0 or p.shape[1] != 1:
            raise ValueError('Nonempty N1HW PAN required')
        n, _, h, w = p.shape
        if min(h, w) < 8 or h % 4 or w % 4:
            raise ValueError('PAN sides >=8 and divisible by four required')
        if (bands != SENSORS[self.lineage.sensor][0]
                or m.shape != (n, bands, h//4, w//4)
                or lms.shape != (n, bands, h, w)):
            raise ValueError('Provided LMS and MS shape/band order mismatch')
        if self.prior_dy_dx.shape != (n, 2) or len(self.lineage.sample_ids) != n:
            raise ValueError('One transformed dataset prior per observation required')
        if require_gt and self.gt is None:
            raise ValueError('RR supervision requires GT')
        if self.gt is not None and self.gt.shape != lms.shape:
            raise ValueError('GT shape mismatch')
        if self.lpan is not None and self.lpan.shape != (n, 1, h//4, w//4):
            raise ValueError('LPAN cache shape mismatch')
        for t in (p, m, lms, self.prior_dy_dx, self.gt, self.lpan):
            if t is not None and (not t.is_floating_point() or t.device != p.device
                                  or not bool(torch.isfinite(t).all())):
                raise ValueError('Finite floating observations on one device required')
        if self.prior_dy_dx.requires_grad:
            raise ValueError('Dataset prior must be frozen, not learnable')
        b = self.prior_dy_dx.new_tensor(self.lineage.canonical_prior_dy_dx)
        expected = torch.stack([transform_prior(b, hflip=a[0], vflip=a[1], rot90=a[2])
                                for a in self.lineage.augmentations])
        if not torch.equal(expected, self.prior_dy_dx):
            raise ValueError('Prior differs from frozen canonical asset + recorded augmentation')
        if training:
            self.lineage.require_training()
        return self


def augment_global_batch(batch, *, hflip, vflip, rot90):
    """Apply exactly one recorded online transform, with no prior recomputation."""
    batch.validate(batch.ms.shape[1])
    if any(a != (False, False, 0) for a in batch.lineage.augmentations):
        raise ValueError('Online augmentation is applied once, before mixing batches')
    prior = transform_prior(batch.prior_dy_dx, hflip=hflip, vflip=vflip, rot90=rot90)
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
        geometry_evidence=batch.lineage.geometry_evidence + f'; online H={hflip},V={vflip},CCW={rot90%4}')
    return replace(batch, pan=change(batch.pan), ms=change(batch.ms), native_lms=change(batch.native_lms),
                   gt=change(batch.gt), lpan=change(batch.lpan), prior_dy_dx=prior,
                   lineage=lineage).validate(batch.ms.shape[1])


class GlobalRRDataSources:
    """Pin the existing sensor source registry, not the old scene source manifest.

    H5 stores DN; model tensors use existing [-1,1] scaling. Stored LMS is read
    verbatim apart from this scaling, including finite interpolation ringing.
    QB uses the user-selected original LMS with separately pinned msfix LRMS.
    Hashes attest bytes, not unknown geographic disjointness or offline flips.
    """
    def __init__(self, catalog_path, *, root, sensor, prior_asset, lms_overrides=None):
        if not isinstance(prior_asset, GlobalPriorAsset) or prior_asset.sensor != sensor:
            raise ValueError('A validated sensor-specific frozen global prior asset is required')
        self.root, self.sensor = Path(root).resolve(), sensor
        self.bands, self.peak = SENSORS[sensor]
        self.asset = prior_asset
        self.catalog_path = Path(catalog_path).resolve(strict=True)
        before = stamp(self.catalog_path)
        data = json.loads(self.catalog_path.read_text())
        self.catalog_sha256 = file_sha(self.catalog_path)
        if stamp(self.catalog_path) != before:
            raise ValueError('Source registry changed while reading')
        if data.get('schema') != 'ABLR2_SENSOR_SOURCES_v1':
            raise ValueError('Explicit existing sensor source registry required')
        self.entry = copy.deepcopy(data['sensors'][sensor])
        if tuple(self.entry['band_order']) != BAND_ORDER[sensor]:
            raise ValueError('Stored band order differs from declared sensor convention')
        if tuple(self.asset.manifest['normalization_and_band_handling']['band_order']) != BAND_ORDER[sensor]:
            raise ValueError('Prior asset band order differs from the training source')
        self.lms_overrides = copy.deepcopy(lms_overrides or {})
        if sensor == 'QB':
            if (set(self.lms_overrides) != {'train', 'val'}
                    or any(not r.get('reason') for r in self.lms_overrides.values())):
                raise ValueError('QB requires explicit original-LMS train/val source overrides')
        elif self.lms_overrides:
            raise ValueError('No unrequested sensor source override is allowed')
        self.train_sha = self.lms_overrides.get('train', self.entry['splits']['train'])['sha256']
        if self.train_sha != self.asset.train_source_sha256:
            raise ValueError('Prior was not estimated from this exact train file')
        self._files = {}
        self._verify('train')
        if 'train' in self.lms_overrides:
            self._verify('train', lms=True)
        prior_path = self._verify('train', lms='train' in self.lms_overrides)
        with h5py.File(prior_path, 'r') as f:
            n = self.asset.manifest['train_rows']
            if (f['pan'].shape != (n, 1, 64, 64)
                    or f['lms'].shape != (n, self.bands, 64, 64)):
                raise ValueError('Frozen prior row count/grid differs from actual training PAN/LMS')

    def _verify(self, split, *, lms=False):
        if split not in ('train', 'val', 'rr', 'fr'):
            raise ValueError('Only registered train/val/rr/fr splits are allowed')
        ref = self.lms_overrides[split] if lms else self.entry['splits'][split]
        path = (self.root / ref['path']).resolve(strict=True)
        require_sha256(ref['sha256'], split + ' file SHA')
        now = stamp(path)
        identity = (split, lms)
        if identity in self._files:
            if self._files[identity] != (path, now):
                raise ValueError('Pinned H5 changed after verification')
        else:
            if file_sha(path) != ref['sha256'] or stamp(path) != now:
                raise ValueError('Source H5 SHA mismatch/changed during verification')
            self._files[identity] = (path, now)
        return path

    def manifest_sha(self, split):
        return canonical_sha256({'schema': 'PANDA_BGLOBAL_RR_SOURCE_V1',
            'sensor': self.sensor, 'registry_sha256': self.catalog_sha256,
            'split': split, 'source': self.entry['splits'][split],
            'band_order': self.entry['band_order'], 'radiometry': '2*stored_DN/peak-1',
            'aligner_reference': 'STORED_LMS_NO_RESAMPLING',
            'lms_override': self.lms_overrides.get(split),
            'reconstruction_ms_source': 'unchanged_registered_source',
            'scene_mapping_used': False, 'offline_augmentation': 'unknown_stored_coordinates',
            'geographic_split_disjointness': 'not_proven_by_this_adapter',
            'source_provenance': self.entry.get('source_provenance', {})})

    def training_sources(self):
        from .global_rr import RRTrainingSources
        return RRTrainingSources(rr_manifest_sha256=self.manifest_sha('train'),
            prior_mapping_sha256=self.asset.fingerprint(), sensor=self.sensor,
            train_source_sha256=self.train_sha)

    def raw(self, split, rows):
        path = self._verify(split)
        if not rows or any(type(i) is not int or i < 0 for i in rows):
            raise ValueError('Nonempty explicit H5 row indices required')
        expected_side = {'train': 64, 'val': 64, 'rr': 256, 'fr': 512}[split]
        keys = ('pan', 'ms', 'lms') + (() if split == 'fr' else ('gt',))
        with h5py.File(path, 'r') as f:
            n = f['pan'].shape[0]
            if (split in ('rr', 'fr') and n != 20) or max(rows) >= n:
                raise ValueError('Invalid row or official evaluation is not all20')
            expected = {'pan': (n, 1, expected_side, expected_side),
                        'ms': (n, self.bands, expected_side//4, expected_side//4),
                        'lms': (n, self.bands, expected_side, expected_side),
                        'gt': (n, self.bands, expected_side, expected_side)}
            result = {}
            for key in keys:
                d = f[key]
                if (not isinstance(d, h5py.Dataset) or d.is_virtual or d.external
                        or Path(d.file.filename).resolve() != path
                        or d.shape != expected[key]):
                    raise ValueError('Independent stored H5 arrays on the declared grid required')
                a = np.stack([d[i] for i in rows])
                if not np.isfinite(a).all():
                    raise ValueError('Nonfinite source observation')
                if key in ('pan', 'gt') and (a.min() < 0 or a.max() > self.peak):
                    raise ValueError('PAN/GT outside nominal sensor DN range')
                result[key] = a
        self._verify(split)
        if split in self.lms_overrides:
            lms_path = self._verify(split, lms=True)
            with h5py.File(lms_path, 'r') as f:
                # Same H5 row order is checked against actual PAN and GT values,
                # not guessed from file names or sensor IDs.
                for key in ('pan', 'gt', 'lms'):
                    d = f[key]
                    if (not isinstance(d, h5py.Dataset) or d.is_virtual or d.external
                            or Path(d.file.filename).resolve() != lms_path
                            or d.shape != expected[key]):
                        raise ValueError('Original LMS source shape/storage mismatch')
                    a = np.stack([d[i] for i in rows])
                    if not np.isfinite(a).all():
                        raise ValueError('Nonfinite original LMS source')
                    if key in ('pan', 'gt') and not np.array_equal(a, result[key]):
                        raise ValueError('Original LMS and active observations have different PAN/GT rows')
                    if key == 'lms':
                        result['lms'] = a
            self._verify(split, lms=True)
        return result

    def batch(self, split, rows, *, device='cpu', augmentation=(False, False, 0)):
        if split == 'fr':
            raise ValueError('FR bGlobal prior policy unresolved; no implicit RR/4x prior')
        arrays = self.raw(split, rows)
        tensor = {k: torch.as_tensor(v.astype(np.float32), device=device) * (2./self.peak) - 1.
                  for k, v in arrays.items()}
        prior = torch.tensor(self.asset.vector, dtype=torch.float32, device=device).expand(len(rows), 2).clone()
        lineage = GlobalBatchLineage(
            split={'train': 'train', 'val': 'validation', 'rr': 'test'}[split], resolution='RR',
            source_manifest_sha256=self.manifest_sha(split), prior_mapping_sha256=self.asset.fingerprint(),
            sample_ids=tuple(f'{self.sensor}:{split}:{i}' for i in rows), grid_name='PAN_RR',
            geometry_evidence='Stored H5 coordinates; unknown offline augmentation; no fabricated scene IDs',
            sensor=self.sensor, train_source_sha256=self.train_sha,
            canonical_prior_dy_dx=self.asset.vector, augmentations=((False, False, 0),) * len(rows))
        batch = GlobalObservationBatch(tensor['pan'], tensor['ms'], tensor['lms'], prior,
                                       lineage, tensor.get('gt'))
        return augment_global_batch(batch, hflip=augmentation[0], vflip=augmentation[1],
                                    rot90=augmentation[2])
