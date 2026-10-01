"""Read-only benchmark bindings; all generated LP artifacts belong to this project."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
import torch

from .common import canonical_sha, file_sha, atomic_json
from .frontend import SENSOR_PROFILES, LP_RECIPE, AUGMENTATION, make_lpan, normalize_dn, fixed_view

SPLITS = ('train', 'val', 'rr', 'fr')
COUNTS = {'WV3': {'train': 9714, 'val': 1080}, 'GF2': {'train': 19809, 'val': 2201},
          'QB': {'train': 17139, 'val': 1905}}
PROVENANCE_FIELDS = ('band_order_evidence', 'mtf_evidence', 'units', 'no_data_policy',
                     'saturation_policy', 'split_correspondence_evidence')


def frozen_catalog_path():
    return Path(__file__).resolve().parents[2] / 'vendor_reference/model_source/ablr2/sensor_sources.json'


def _under(child, parent):
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def scan_source(path, sensor, split, *, expected_count=None, check_cancel=None):
    """Full arrays, not a sample scan; zero/ringing remain untouched."""
    profile = SENSOR_PROFILES[sensor]
    size = {'train': 64, 'val': 64, 'rr': 256, 'fr': 512}[split]
    bands, maximum = profile['bands'], profile['max_dn']
    keys = ['pan', 'ms', 'lms'] + ([] if split == 'fr' else ['gt'])
    with h5py.File(path, 'r') as source:
        if not set(keys) <= set(source):
            raise ValueError('BLOCKED_DATA_GEOMETRY: missing arrays')
        n = len(source['pan'])
        expected = {'pan': (n, 1, size, size), 'ms': (n, bands, size // 4, size // 4),
                    'lms': (n, bands, size, size), 'gt': (n, bands, size, size)}
        required = expected_count if expected_count is not None else (20 if split in ('rr', 'fr') else COUNTS[sensor][split])
        if n != required or any(source[k].shape != expected[k] for k in keys):
            raise ValueError(f'BLOCKED_DATA_GEOMETRY: {sensor}/{split} count={n}, expected={required}')
        statistics = {}
        # Include every tensor key, even an optional FR GT: never silently ignore nonfinite data.
        for key, dataset in source.items():
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim == 0:
                raise ValueError('BLOCKED_DATA_GEOMETRY: unregistered non-array H5 object')
            if dataset.shape[0] != n or not np.issubdtype(dataset.dtype, np.number):
                raise ValueError('BLOCKED_DATA_GEOMETRY: unregistered tensor count/dtype')
            low, high, zeros, saturated, nonfinite = np.inf, -np.inf, 0, 0, 0
            for start in range(0, n, 32):
                if check_cancel is not None:
                    check_cancel()
                array = dataset[start:start + 32]
                nonfinite += int(np.count_nonzero(~np.isfinite(array)))
                if nonfinite:
                    raise ValueError(f'BLOCKED_DATA_NONFINITE: {sensor}/{split}/{key}')
                low, high = min(low, float(array.min())), max(high, float(array.max()))
                zeros += int(np.count_nonzero(array == 0))
                saturated += int(np.count_nonzero(array == maximum))
            if key in ('pan', 'gt') and (low < 0 or high > maximum):
                raise ValueError('BLOCKED_DATA_RANGE: PAN/GT exceeds declared DN')
            statistics[key] = dict(shape=list(dataset.shape), dtype=str(dataset.dtype), minimum=low,
                                   maximum=high, zero_count=zeros, saturation_count=saturated,
                                   nonfinite_count=nonfinite, ringing_outside_nominal=low < 0 or high > maximum)
    return dict(count=n, shapes={k: list(v['shape']) for k, v in statistics.items()}, statistics=statistics,
                full_scan=True, qb_val_count_policy='declared1905_exact_or_block' if sensor == 'QB' and split == 'val' else None)


def verify_qb_msfix(raw, fixed, raw_sha, fixed_sha, *, check_cancel=None):
    """Frozen full source GT/PAN + MTF/interp proof, never regenerate the source."""
    import cv2
    from .metrics.eval_fr import genmtf_matlab, GNYQ_TABLE
    from .metrics.interp23 import interp23tap
    if file_sha(raw) != raw_sha or file_sha(fixed) != fixed_sha:
        raise ValueError('BLOCKED_DATA_IDENTITY: QB raw/msfix SHA')
    kernel = genmtf_matlab(GNYQ_TABLE['QB'], 4, 41)
    digests = {key: hashlib.sha256() for key in ('gt', 'pan')}
    with h5py.File(raw, 'r') as source, h5py.File(fixed, 'r') as target:
        if any(source[k].shape != target[k].shape or source[k].dtype != target[k].dtype for k in ('gt', 'pan', 'ms', 'lms')):
            raise ValueError('BLOCKED_QB_MSFIX: raw/fixed shape/dtype differs')
        for index in range(len(source['pan'])):
            if check_cancel is not None:
                check_cancel()
            for key, digest in digests.items():
                old, new = source[key][index], target[key][index]
                if not np.array_equal(old, new):
                    raise ValueError(f'BLOCKED_QB_MSFIX: changed {key} sample{index}')
                digest.update(old.tobytes())
            gt = np.asarray(source['gt'][index], dtype=np.float64)
            ms = np.stack([cv2.filter2D(gt[b], -1, kernel[:, :, b], borderType=cv2.BORDER_REPLICATE)[2::4, 2::4] for b in range(4)])
            lms = interp23tap(ms.transpose(1, 2, 0), 4).transpose(2, 0, 1)
            for key, expected in (('ms', ms), ('lms', lms)):
                if not np.allclose(target[key][index], expected.astype(target[key].dtype), rtol=0, atol=1e-8):
                    raise ValueError(f'BLOCKED_QB_MSFIX: reconstructed {key} sample{index}')
        count = len(source['pan'])
    if file_sha(raw) != raw_sha or file_sha(fixed) != fixed_sha:
        raise ValueError('BLOCKED_DATA_IDENTITY: QB changed during audit')
    return dict(status='PASS', raw_sha256=raw_sha, msfix_sha256=fixed_sha, gt_unchanged=True,
                pan_unchanged=True, samples_verified=count,
                unchanged_array_sha256={k: d.hexdigest() for k, d in digests.items()},
                recipe='genMTF(QB), replicate, [2::4,2::4], interp23tap', test_inputs_regenerated=False)


def ensure_lp_cache(source_path, cache_path, source_sha, *, allowed_root=None, check_cancel=None):
    cache_path, source_path = Path(cache_path), Path(source_path)
    if check_cancel is not None:
        check_cancel()
    if allowed_root is not None and not _under(cache_path, allowed_root):
        raise ValueError('BLOCKED_PATH: cache symlink escapes campaign cache root')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() == cache_path.resolve():
        raise ValueError('Cache must not overwrite source')
    recipe_hash = canonical_sha(LP_RECIPE)
    if not cache_path.exists():
        fd, temporary = tempfile.mkstemp(prefix='.lpan-', suffix='.h5', dir=cache_path.parent)
        os.close(fd)
        try:
            with h5py.File(source_path, 'r') as source, h5py.File(temporary, 'w') as target:
                n, _, h, w = source['pan'].shape
                dataset = target.create_dataset('lpan', shape=(n, 1, h // 4, w // 4), dtype='float32')
                order = []
                for start in range(0, n, 32):
                    if check_cancel is not None:
                        check_cancel()
                    pan = source['pan'][start:start + 32].astype(np.float64)
                    dataset[start:start + 32] = make_lpan(pan)
                    order.extend(hashlib.sha256(x.astype('<f4').tobytes()).hexdigest() for x in pan)
                target.attrs['source_sha256'] = source_sha
                target.attrs['recipe_sha256'] = recipe_hash
                target.attrs['sample_order_sha256'] = canonical_sha(order)
                target.flush()
            with open(temporary, 'rb') as stream:
                os.fsync(stream.fileno())
            if file_sha(source_path) != source_sha:
                raise ValueError('BLOCKED_DATA_IDENTITY: source changed during LP creation')
            os.chmod(temporary, 0o444)
            os.link(temporary, cache_path)  # exclusive immutable publication
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    digest, order = hashlib.sha256(), []
    with h5py.File(source_path, 'r') as source, h5py.File(cache_path, 'r') as cache:
        n, _, h, w = source['pan'].shape
        if (cache.attrs.get('source_sha256') != source_sha or cache.attrs.get('recipe_sha256') != recipe_hash or
            cache['lpan'].shape != (n, 1, h // 4, w // 4) or cache['lpan'].dtype != np.dtype('float32')):
            raise ValueError('BLOCKED_LP_IDENTITY: native source/recipe/shape')
        for start in range(0, n, 32):
            if check_cancel is not None:
                check_cancel()
            pan = source['pan'][start:start + 32].astype(np.float64)
            actual = cache['lpan'][start:start + 32]
            expected = make_lpan(pan)
            if not np.array_equal(actual, expected) or not np.isfinite(actual).all():
                raise ValueError('BLOCKED_LP_IDENTITY: full native PAN correspondence')
            digest.update(actual.astype('<f4').tobytes())
            order.extend(hashlib.sha256(x.astype('<f4').tobytes()).hexdigest() for x in pan)
        if cache.attrs.get('sample_order_sha256') != canonical_sha(order):
            raise ValueError('BLOCKED_LP_IDENTITY: sample order')
    if file_sha(source_path) != source_sha:
        raise ValueError('BLOCKED_DATA_IDENTITY: source changed during LP verification')
    return dict(lpan_path=str(cache_path.resolve()), lpan_sha256=file_sha(cache_path),
                lpan_canonical_sha256=digest.hexdigest(), sample_order_sha256=canonical_sha(order), full_lp_verified=True)


def prepare_catalog(catalog_path, data_root, cache_root, full_scan=True, check_cancel=None):
    if not full_scan:
        raise ValueError('Full data scan is mandatory; no sampling shortcut')
    catalog = json.loads(Path(catalog_path).read_text())
    frozen = json.loads(frozen_catalog_path().read_text())
    if catalog != frozen:
        raise ValueError('BLOCKED_DATA_IDENTITY: catalog differs from fixed source commit')
    data_root, cache_root = Path(data_root).resolve(), Path(cache_root).resolve()
    project = Path(__file__).resolve().parents[2]
    if not _under(cache_root, project / 'work_dir') or _under(cache_root, data_root) or _under(data_root, cache_root):
        raise ValueError('BLOCKED_PATH: LP cache must be in independent project work_dir, outside source tree')
    sources, scans = {}, {}
    # All identity/geometry checks precede any LP creation.
    for sensor, entry in catalog['sensors'].items():
        if entry['band_order'] != SENSOR_PROFILES[sensor]['band_order'] or any(not entry['source_provenance'].get(k) for k in PROVENANCE_FIELDS):
            raise ValueError('BLOCKED_DATA_IDENTITY: source band/provenance missing')
        for split in SPLITS:
            if check_cancel is not None:
                check_cancel()
            binding = entry['splits'][split]
            path = (data_root / binding['path']).resolve(strict=True)
            if _under(path, cache_root) or file_sha(path) != binding['sha256']:
                raise ValueError(f'BLOCKED_DATA_IDENTITY: {sensor}/{split}')
            sources[sensor, split] = path
            scans[sensor, split] = scan_source(path, sensor, split, check_cancel=check_cancel)
    proof = catalog['sensors']['QB']['source_provenance']
    for split in ('train', 'val'):
        if check_cancel is not None:
            check_cancel()
        raw = (data_root / proof['raw_' + split + '_path']).resolve(strict=True)
        scans['QB', split]['msfix_audit'] = verify_qb_msfix(raw, sources['QB', split], proof['raw_' + split + '_sha256'], catalog['sensors']['QB']['splits'][split]['sha256'], check_cancel=check_cancel)
    required = sum(scan['count'] * scan['shapes']['ms'][-1] * scan['shapes']['ms'][-2] * 4 for scan in scans.values())
    cache_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache_root).free < max(20 * 1024 ** 3, required * 2):
        raise ValueError('PAUSED_DISK: insufficient LP cache reserve')
    result = dict(schema='PANDEP_DATA_v1', catalog_sha256=file_sha(catalog_path),
                  recipe=LP_RECIPE, augmentation=AUGMENTATION, sensors={},
                  original_data_read_only=True, source_geo_test_independence_verified=False)
    for sensor, entry in catalog['sensors'].items():
        profile = dict(SENSOR_PROFILES[sensor], source_provenance=entry['source_provenance'], splits={})
        for split in SPLITS:
            if check_cancel is not None:
                check_cancel()
            binding = entry['splits'][split]
            path = sources[sensor, split]
            cache = cache_root / sensor / f'{split}_{binding["sha256"]}_{canonical_sha(LP_RECIPE)}.h5'
            lp = ensure_lp_cache(path, cache, binding['sha256'], allowed_root=cache_root, check_cancel=check_cancel)
            profile['splits'][split] = dict(dataroot=str(path), sha256=binding['sha256'],
                source_identity=binding['source_identity'], **scans[sensor, split], **lp)
        result['sensors'][sensor] = profile
    result['dataset_sha256'] = canonical_sha(result)
    return result


class SensorDataset:
    """Per-fetch H5 reads keep three large corpora out of RAM; sources always mode r."""
    def __init__(self, manifest, sensor, split, work_root=None):
        if isinstance(manifest, (str, Path)):
            manifest = json.loads(Path(manifest).read_text())
        self.sensor, self.split = sensor, split
        profile = manifest['sensors'][sensor]
        self.binding = profile['splits'][split]
        self.source_path = Path(self.binding['dataroot']).resolve(strict=True)
        self.lp_path = Path(self.binding['lpan_path']).resolve(strict=True)
        self.max_dn = SENSOR_PROFILES[sensor]['max_dn']
        if profile['bands'] != SENSOR_PROFILES[sensor]['bands'] or profile['max_dn'] != self.max_dn:
            raise ValueError('BLOCKED_DATA_IDENTITY: incorrect sensor profile')
        if file_sha(self.source_path) != self.binding['sha256'] or file_sha(self.lp_path) != self.binding['lpan_sha256']:
            raise ValueError('BLOCKED_DATA_IDENTITY: source/LP file changed')
        self.count = int(self.binding['count'])
        self.sample_ids = list(range(self.count))
        self._stats = self._signatures()

    def _signatures(self):
        return [(p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) for p in (self.source_path, self.lp_path)]

    def __len__(self):
        return self.count

    def raw(self, sample_ids):
        if self._signatures() != self._stats:
            raise ValueError('BLOCKED_DATA_IDENTITY: source/LP changed during run')
        if any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)) for x in sample_ids):
            raise IndexError('Sample IDs must be integers')
        ids = [int(x) for x in sample_ids]
        if not ids or any(x < 0 or x >= self.count for x in ids):
            raise IndexError(ids)
        with h5py.File(self.source_path, 'r') as source, h5py.File(self.lp_path, 'r') as cache:
            keys = ['pan', 'ms', 'lms'] + (['gt'] if self.split != 'fr' else [])
            result = {key: np.stack([source[key][i] for i in ids]) for key in keys}
            result['lpan'] = np.stack([cache['lpan'][i] for i in ids])
        return result

    def fetch(self, sample_ids, rotations=None):
        arrays = self.raw(sample_ids)
        if rotations is not None:
            if self.split != 'train' or len(rotations) != len(sample_ids):
                raise ValueError('Augmentation is restricted to explicit training views')
            arrays = {key: np.stack([fixed_view(a, r) for a, r in zip(value, rotations)]) for key, value in arrays.items()}
        return {key: torch.from_numpy(normalize_dn(value, self.sensor).copy()) for key, value in arrays.items()}

    def batch(self, sample_ids, rotations=None, device=None):
        result = self.fetch(sample_ids, rotations)
        return {key: value.to(device) for key, value in result.items()} if device is not None else result

    def __getitem__(self, index):
        return {key: value[0] for key, value in self.fetch([index]).items()}
