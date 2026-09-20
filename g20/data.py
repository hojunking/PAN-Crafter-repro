"""Explicit QB/GF2 source contracts and immutable native-PAN LP caches.

The FH12 Gaussian implementation is reused unchanged. No path, sensor, band
order, normalization, or source-provenance fallback is permitted.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from fh12.data import (RECIPE, AUGMENTATION, canonical_sha, sha256_file,
                       check_deadline, write_immutable_json, _create_lp_cache)
from g20.plan import sensor_spec

SPLITS = ('train', 'val', 'rr', 'fr')
PROVENANCE_FIELDS = ('band_order_evidence', 'mtf_evidence', 'units',
                     'no_data_policy', 'saturation_policy', 'split_correspondence_evidence')


def canonical_band_indices(order):
    aliases = {'b': 'blue', 'g': 'green', 'r': 'red', 'nir': 'nir',
               'blue': 'blue', 'green': 'green', 'red': 'red', 'near_infrared': 'nir'}
    names = [aliases.get(str(x).lower()) for x in (order or ())]
    if len(names) != 4 or set(names) != {'blue', 'green', 'red', 'nir'}:
        raise ValueError('Explicit source band order must contain blue, green, red, NIR exactly once')
    return tuple(names.index(b) for b in ('blue', 'green', 'red', 'nir'))


def rgb_indices(spec):
    """C4 visualization indices derived from the bound source order."""
    blue, green, red, _nir = canonical_band_indices(spec.band_order)
    return red, green, blue


def inventory_sources(root, sensor):
    """Read-only metadata inventory; existence is never a P0 provenance PASS."""
    spec = sensor_spec(sensor)
    base = Path(root) / 'data/PanCollection' / spec.sensor
    name = spec.sensor.lower()
    paths = dict(train=base / f'train_{name}{"_msfix" if name == "qb" else ""}.h5',
                 val=base / f'valid_{name}{"_msfix" if name == "qb" else ""}.h5',
                 rr=base / f'reduced_examples_h5/test_{name}_multiExm1.h5',
                 fr=base / f'full_examples_mat20/test_{name}_OrigScale_mat20.h5')
    rows = {}
    for split, path in paths.items():
        row = dict(candidate_path=str(path), resolved_path=str(path.resolve()), exists=path.is_file())
        if path.is_file():
            with h5py.File(path, 'r') as src:
                row.update(shapes={k: list(v.shape) for k, v in src.items()},
                           dtypes={k: str(v.dtype) for k, v in src.items()},
                           attributes={k: str(v) for k, v in src.attrs.items()})
        rows[split] = row
    return dict(sensor=spec.sensor, status='TO_VERIFY', splits=rows,
                blockers=['Resolved file SHA, source identity, band order, MTF and source policies must be explicitly bound'])


def _scan_source(path, split, spec, deadline=None):
    keys = ['ms', 'lms', 'pan'] + ([] if split == 'fr' else ['gt'])
    size = {'train': 64, 'val': 64, 'rr': 256, 'fr': 512}[split]
    with h5py.File(path, 'r') as src:
        if not set(keys) <= set(src):
            raise ValueError(f'{split}: missing native data keys')
        n = len(src['pan'])
        expected = dict(pan=(n, 1, size, size), ms=(n, 4, size // 4, size // 4),
                        lms=(n, 4, size, size), gt=(n, 4, size, size))
        if not n or any(src[k].shape != expected[k] for k in keys):
            raise ValueError(f'{split}: C4/native PAN/ratio4 geometry mismatch')
        if split in ('rr', 'fr') and n != 20:
            raise ValueError(f'{split}: all 20 native scenes are required')
        stats = {}
        for key in keys:
            lo, hi, zeros, saturation = np.inf, -np.inf, 0, 0
            for start in range(0, n, 64):
                check_deadline(deadline)
                x = np.asarray(src[key][start:start + 64])
                if not np.isfinite(x).all():
                    raise ValueError(f'{split}/{key}: nonfinite native source')
                lo, hi = min(lo, float(x.min())), max(hi, float(x.max()))
                zeros += int(np.count_nonzero(x == 0))
                saturation += int(np.count_nonzero(x == spec.max_dn))
            # Interpolated/filtered MS may ring; preserve and report those values.
            if key in ('gt', 'pan') and (lo < 0 or hi > spec.max_dn):
                raise ValueError(f'{split}/{key}: source is not declared {spec.max_dn}DN')
            stats[key] = dict(shape=list(src[key].shape), dtype=str(src[key].dtype),
                              minimum=lo, maximum=hi, zero_count=zeros, saturation_count=saturation)
        return dict(count=n, shapes={k: list(src[k].shape) for k in keys}, statistics=stats)


def verify_lp_cache(source_path, cache_path, source_sha, deadline=None):
    """Verify every sample against its actual native PAN, not its filename/attrs."""
    from tools.repair_lpan import make_lpan
    digest, order = hashlib.sha256(), []
    with h5py.File(source_path, 'r') as src, h5py.File(cache_path, 'r') as cache:
        pan = src['pan']
        n, _, h, w = pan.shape
        if (cache.attrs.get('source_sha256') != source_sha or cache.attrs.get('recipe_sha256') != canonical_sha(RECIPE)
                or cache['lpan'].shape != (n, 1, h // 4, w // 4) or cache['lpan'].dtype != np.dtype('float32')):
            raise ValueError('LP cache source/recipe/shape/dtype identity mismatch')
        for start in range(0, n, 64):
            check_deadline(deadline)
            native = np.asarray(pan[start:start + 64], dtype=np.float64)
            actual = np.asarray(cache['lpan'][start:start + 64])
            expected = make_lpan(native).astype(np.float32)
            if not np.isfinite(actual).all() or not np.array_equal(actual, expected):
                raise ValueError(f'Native PAN/LP correspondence failed at chunk {start}')
            digest.update(np.asarray(actual, dtype='<f4').tobytes())
            order.extend(hashlib.sha256(x.astype('<f4').tobytes()).hexdigest() for x in native)
        order_sha = canonical_sha(order)
        if cache.attrs.get('sample_order_sha256') != order_sha:
            raise ValueError('LP source sample-order identity mismatch')
    if sha256_file(source_path) != source_sha:
        raise ValueError('Native source changed during LP verification')
    return dict(sample_order_sha256=order_sha, lpan_canonical_sha256=digest.hexdigest())


def verify_qb_msfix(raw, fixed, *, raw_sha256, fixed_sha256, band_order,
                    deadline=None, wald=None):
    """Prove whole-source GT/PAN equality and every reconstructed MS/LMS patch."""
    from tools.metrics.eval_fr import load_dlpan, genmtf_matlab, GNYQ_TABLE
    from tools.repair_qb_ms import mtf_down
    raw, fixed = Path(raw).resolve(), Path(fixed).resolve()
    if sha256_file(raw) != raw_sha256 or sha256_file(fixed) != fixed_sha256:
        raise ValueError('QB raw/msfix source SHA mismatch')
    if wald is None:
        wald = load_dlpan(os.environ.get('PANCRAFTER_DLPAN', str(Path(__file__).resolve().parents[2] / 'DLPan-Toolbox')))
    order = canonical_band_indices(band_order)
    inverse = np.argsort(order)
    kernel = genmtf_matlab(GNYQ_TABLE['QB'], 4, 41)
    digests = {k: hashlib.sha256() for k in ('gt', 'pan')}
    with h5py.File(raw, 'r') as src, h5py.File(fixed, 'r') as dst:
        if any(src[k].shape != dst[k].shape or src[k].dtype != dst[k].dtype for k in ('gt', 'pan', 'ms', 'lms')):
            raise ValueError('QB raw/msfix source shape/dtype differs')
        for i in range(len(src['pan'])):
            check_deadline(deadline)
            for key in ('gt', 'pan'):
                a, b = src[key][i], dst[key][i]
                if not np.array_equal(a, b):
                    raise ValueError(f'QB msfix changed {key} at source sample {i}')
                digests[key].update(a.tobytes())
            ms = mtf_down(np.asarray(src['gt'][i], dtype=np.float64)[list(order)], kernel)[inverse]
            # Match the original recipe: interpolation consumes float64 ms before disk cast.
            lms = wald.interp23tap(ms.transpose(1, 2, 0), 4).transpose(2, 0, 1)
            for key, expected in (('ms', ms), ('lms', lms)):
                if not np.allclose(dst[key][i], expected.astype(dst[key].dtype), rtol=0, atol=1e-8):
                    raise ValueError(f'QB {key} MTF/phase/interpolation recipe mismatch at sample {i}')
    if sha256_file(raw) != raw_sha256 or sha256_file(fixed) != fixed_sha256:
        raise ValueError('QB source changed during audit')
    return dict(status='PASS', raw_sha256=raw_sha256, msfix_sha256=fixed_sha256,
                gt_unchanged=True, pan_unchanged=True,
                unchanged_array_sha256={k: d.hexdigest() for k, d in digests.items()},
                recipe='genMTF(QB), replicate, [2::4,2::4], interp23tap',
                interp23tap_source_sha256=sha256_file(wald.__file__) if getattr(wald, '__file__', None) else None,
                samples_verified=i + 1, test_inputs_regenerated=False)


def prepare_data(root, server, spec, deadline_utc=None):
    """Prepare bound data only; missing provenance raises before publishing assets."""
    root = Path(root).resolve()
    if server not in {'s1', 's2', 's3', 's4', 's5'}:
        raise ValueError('Unknown G20 server')
    if spec.sensor != 'GF2':
        raise ValueError('Bound sensor differs from registered server assignment')
    canonical_band_indices(spec.band_order)
    proof = dict(spec.source_provenance)
    missing = [key for key in PROVENANCE_FIELDS if not proof.get(key)]
    if missing or proof.get('units') != 'DN':
        raise ValueError(f'BLOCKED_INTEGRITY: source provenance incomplete: {missing}; units must be DN')
    bindings = {s: spec.split(s) for s in SPLITS}
    checks, splits = {}, {}
    # Validate every source before creating any cache.
    for split, binding in bindings.items():
        source = Path(binding.path).resolve(strict=True)
        if str(source) != binding.path or sha256_file(source) != binding.sha256 or not binding.source_identity:
            raise ValueError(f'{split}: unbound resolved path/source SHA/identity')
        checks[split] = _scan_source(source, split, spec, deadline_utc)
    if spec.sensor == 'QB':
        for split in ('train', 'val'):
            for key in (f'raw_{split}_path', f'raw_{split}_sha256', 'msfix_recipe'):
                if not proof.get(key):
                    raise ValueError(f'BLOCKED_INTEGRITY: QB provenance missing {key}')
            checks[split]['msfix_audit'] = verify_qb_msfix(
                proof[f'raw_{split}_path'], bindings[split].path,
                raw_sha256=proof[f'raw_{split}_sha256'], fixed_sha256=bindings[split].sha256,
                band_order=spec.band_order, deadline=deadline_utc)
    assets = root / 'work_dir/_g20' / server / spec.sensor
    for split, binding in bindings.items():
        cache = assets / 'lpan' / f'{split}_{binding.sha256[:20]}_{canonical_sha(RECIPE)[:16]}.h5'
        _create_lp_cache(Path(binding.path), cache, binding.sha256, deadline_utc)
        lp_identity = verify_lp_cache(binding.path, cache, binding.sha256, deadline_utc)
        splits[split] = dict(dataroot=binding.path, sha256=binding.sha256,
                             source_identity=binding.source_identity, lpan_path=str(cache),
                             lpan_sha256=sha256_file(cache), **lp_identity, **checks[split])
    import cv2
    manifest = dict(schema='G20_DATA_v1', server=server, sensor=spec.sensor,
                    num_bands=4, max_pixel=spec.max_dn, band_order=list(spec.band_order),
                    mtf_sensor=spec.mtf_sensor, source_provenance=proof, splits=splits,
                    opencv_version=cv2.__version__,
                    recipe=RECIPE, augmentation=AUGMENTATION,
                    augmentation_sha256=canonical_sha(AUGMENTATION))
    write_immutable_json(assets / 'dataset_manifest.json', manifest)
    return manifest


class G20Dataset(Dataset):
    def __init__(self, dataroot, lpan_path, *, spec, split, augment=False):
        canonical_band_indices(spec.band_order)
        self.spec, self.split, self.augment = spec, split, bool(augment)
        self.raw_h5_path = self.dataroot = str(Path(dataroot).resolve(strict=True))
        self.lp_path = str(Path(lpan_path).resolve(strict=True))
        self.max_pixel, self.bands = float(spec.max_dn), 4
        with h5py.File(self.raw_h5_path, 'r') as src:
            self.has_gt = 'gt' in src
            self.arrays = {k: src[k][:].astype(np.float32) for k in ['lms', 'ms', 'pan'] + (['gt'] if self.has_gt else [])}
        with h5py.File(self.lp_path, 'r') as src:
            if src.attrs.get('source_sha256') != sha256_file(self.raw_h5_path) or src.attrs.get('recipe_sha256') != canonical_sha(RECIPE):
                raise ValueError('LP native source/recipe identity mismatch')
            self.arrays['lpan'] = src['lpan'][:]
        self.base_count = len(self.arrays['pan'])
        n, c, h, w = self.arrays['pan'].shape
        if (c != 1 or self.arrays['ms'].shape != (n, 4, h // 4, w // 4)
                or self.arrays['lms'].shape != (n, 4, h, w)
                or self.arrays['lpan'].shape != (n, 1, h // 4, w // 4)
                or self.arrays['lpan'].dtype != np.dtype('float32')
                or (self.has_gt and self.arrays['gt'].shape != (n, 4, h, w))):
            raise ValueError('C4 native source/LP geometry mismatch')
        if not all(np.isfinite(x).all() for x in self.arrays.values()):
            raise ValueError('Nonfinite source/LP cache')
        self.source_hashes = dict(source=sha256_file(self.raw_h5_path), lp=sha256_file(self.lp_path))

    def __len__(self):
        return self.base_count * (4 if self.augment else 1)

    def get_view(self, index, rot=0, augment=False):
        index, rot = int(index), int(rot)
        if not 0 <= index < self.base_count or rot not in (0, 1, 2, 3):
            raise IndexError((index, rot))
        if not augment and rot:
            raise ValueError('Nonzero rotation requires explicit augmentation')
        keys = (['gt'] if self.has_gt else []) + ['lms', 'ms', 'lpan', 'pan']
        values = []
        for key in keys:
            x = self.arrays[key][index]
            if augment:
                x = np.rot90(x[:, ::-1, ::-1], rot, axes=(1, 2))
            values.append(torch.from_numpy(np.array(x, dtype=np.float32, copy=True)).mul_(2 / self.max_pixel).sub_(1))
        return (*values, torch.tensor([index, rot if augment else 0, int(augment), int(augment)]))

    def base(self, index):
        return self.get_view(index)

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            return self.get_view(*index, augment=True)
        return self.get_view(index // 4, index % 4, augment=True) if self.augment else self.base(index)


def build_dataset(manifest, split, root=None, server=None, augment=False):
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    if manifest.get('schema') != 'G20_DATA_v1' or manifest.get('num_bands') != 4:
        raise ValueError('Explicit G20 C4 manifest required')
    spec = sensor_spec(manifest['sensor'])
    if manifest.get('max_pixel') != spec.max_dn or manifest.get('mtf_sensor') != spec.mtf_sensor:
        raise ValueError('Sensor DN/MTF mismatch; fallback is forbidden')
    spec = replace(spec, band_order=tuple(manifest['band_order']))
    if manifest.get('recipe') != RECIPE or manifest.get('augmentation_sha256') != canonical_sha(AUGMENTATION):
        raise ValueError('LP/augmentation contract changed')
    item = manifest['splits'][split]
    source, lp = Path(root or '.') / item['dataroot'], Path(root or '.') / item['lpan_path']
    if sha256_file(source) != item['sha256'] or sha256_file(lp) != item['lpan_sha256']:
        raise ValueError('Resolved source/LP SHA mismatch')
    if not item.get('source_identity'):
        raise ValueError('Source sample identity is unbound')
    dataset = G20Dataset(source, lp, spec=spec, split=split, augment=augment)
    dataset.contract_hash = canonical_sha(manifest)
    return dataset
