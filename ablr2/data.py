"""Verified WV3/QB native data; all new assets are restricted to one ABLR2 lane."""
from pathlib import Path
import h5py
import numpy as np

from fh12.data import (RECIPE, AUGMENTATION, canonical_sha, _create_lp_cache, FH12Dataset)
from qg40.data import verify_lp_cache, verify_qb_msfix, PROVENANCE_FIELDS
from ablr2.common import CAMPAIGN_ID, camp, read, read_json, sha256, immutable_json, atomic_json, check_deadline
from ablr2.plan import SensorSpec, verify_lane

SPLITS = ('train', 'val', 'rr', 'fr')


def source_catalog(root, server):
    sensor = verify_lane(server)
    catalog = read_json(Path(root) / 'ablr2/sensor_sources.json')
    if catalog.get('schema') != 'ABLR2_SENSOR_SOURCES_v1':
        raise ValueError('Unregistered source catalog')
    entry = catalog['sensors'][sensor]
    spec = SensorSpec(sensor, tuple(entry['band_order']))
    from ablr2.evaluation import canonical_band_indices
    if canonical_band_indices(spec) != tuple(range(spec.num_bands)):
        raise ValueError('Only native channel ordering is registered')
    proof = entry['source_provenance']
    if proof.get('units') != 'DN' or any(not proof.get(k) for k in PROVENANCE_FIELDS):
        raise ValueError('Source provenance is incomplete')
    return spec, entry


def scan_source(path, split, spec, deadline=None):
    size = dict(train=64, val=64, rr=256, fr=512)[split]
    keys = ['pan', 'ms', 'lms'] + ([] if split == 'fr' else ['gt'])
    with h5py.File(path, 'r') as source:
        if not set(keys) <= set(source): raise ValueError('Missing H5 native arrays')
        n, c = len(source['pan']), spec.num_bands
        expected = dict(pan=(n,1,size,size), ms=(n,c,size//4,size//4),
                        lms=(n,c,size,size), gt=(n,c,size,size))
        if n <= 0 or any(source[k].shape != expected[k] for k in keys):
            raise ValueError('Native H5 sensor geometry mismatch: ' + split)
        required = dict(train=spec.nominal_train_n, val=spec.nominal_val_n, rr=20, fr=20)[split]
        if required is not None and n != required:
            raise ValueError('Unexpected source sample count: ' + split)
        stats = {}
        for key in keys:
            lo, hi, zeros, saturated = np.inf, -np.inf, 0, 0
            for start in range(0, n, 64):
                check_deadline(deadline)
                x = source[key][start:start+64]
                if not np.isfinite(x).all(): raise ValueError('Nonfinite H5 array')
                lo, hi = min(lo, float(x.min())), max(hi, float(x.max()))
                zeros += int(np.count_nonzero(x == 0))
                saturated += int(np.count_nonzero(x == spec.max_dn))
            if key in ('pan','gt') and (lo < 0 or hi > spec.max_dn):
                raise ValueError('Native PAN/GT range differs from declared DN2047')
            stats[key] = dict(minimum=lo, maximum=hi, zero_count=zeros, saturation_count=saturated)
        return dict(count=n, shapes={k:list(source[k].shape) for k in keys}, statistics=stats)


def validate_manifest(manifest, server=None):
    server = server or manifest['server']
    sensor = verify_lane(server, manifest.get('sensor'))
    spec = SensorSpec(sensor, tuple(manifest.get('band_order', [])))
    from ablr2.evaluation import canonical_band_indices
    if (manifest.get('schema') != 'ABLR2_DATA_v1' or manifest.get('server') != server
            or manifest.get('num_bands') != spec.num_bands or manifest.get('max_pixel') != 2047
            or manifest.get('mtf_sensor') != sensor or set(manifest.get('splits', {})) != set(SPLITS)
            or canonical_band_indices(spec) != tuple(range(spec.num_bands))):
        raise ValueError('ABLR2 sensor/source manifest mismatch')
    if (manifest.get('recipe') != RECIPE or manifest.get('augmentation') != AUGMENTATION
            or manifest.get('augmentation_sha256') != canonical_sha(AUGMENTATION)):
        raise ValueError('LP/augmentation recipe changed')
    for split, expected in dict(train=spec.nominal_train_n, val=spec.nominal_val_n, rr=20, fr=20).items():
        n = manifest['splits'][split].get('count', 0)
        if n <= 0 or (expected is not None and n != expected):
            raise ValueError('Incorrect source count: ' + split)
    return manifest


def prepare_data(root, server, deadline=None, manifest_path=None):
    root = Path(root).resolve()
    folder = camp(root, server)
    target = folder / 'dataset_manifest.json'
    spec, entry = source_catalog(root, server)
    # No discovery/fallback to alternate or repaired evaluation datasets.
    bindings = entry['splits']
    proof = entry['source_provenance']
    sources = {}
    for split in SPLITS:
        binding = bindings[split]
        path = (root / binding['path']).resolve(strict=True)
        if sha256(path) != binding['sha256']:
            raise ValueError('Pinned native source changed: ' + split)
        sources[split] = path
    if spec.sensor == 'QB':
        for split in ('train','val'):
            if sha256(root/proof[f'raw_{split}_path']) != proof[f'raw_{split}_sha256']:
                raise ValueError('Pinned raw QB source changed: '+split)
    verifier=dict(source_catalog_sha256=sha256(root/'ablr2/sensor_sources.json'),
        verifier_sha256={name:sha256(root/name) for name in ('ablr2/data.py','qg40/data.py',
            'fh12/data.py','tools/repair_lpan.py','tools/repair_qb_ms.py')})
    def receipt_for(data):
        return dict(schema='ABLR2_DATA_VERIFICATION_v1',campaign_id=CAMPAIGN_ID,server=server,
            dataset_manifest_sha256=canonical_sha(data),complete=True,full_lp_verified=True,
            raw_qb_msfix_verified=spec.sensor=='QB',previous_campaign_written=False,**verifier)
    if target.is_file() or manifest_path:
        external=(root/manifest_path).resolve() if manifest_path else None
        data = validate_manifest(read_json(target if target.is_file() else external), server)
        if external and read_json(external) != data:
            raise ValueError('Explicit manifest differs from immutable lane manifest')
        for split, item in data['splits'].items():
            if ((root/item['dataroot']).resolve() != sources[split] or item['sha256'] != bindings[split]['sha256']
                    or item['source_identity'] != bindings[split]['source_identity']
                    or sha256(root/item['lpan_path']) != item['lpan_sha256']):
                raise ValueError('Source/LP binding mismatch: ' + split)
        if data['source_provenance'] != entry['source_provenance']:
            raise ValueError('Source proof changed')
        # External/self-described manifests are not audit receipts. Only this
        # lane's previously issued matching proof may skip the full scans.
        own_receipt=read(folder/'data_verification.json')
        if not target.is_file() or own_receipt != receipt_for(data):
            for split,path in sources.items():
                scanned=scan_source(path,split,spec,deadline)
                identity=verify_lp_cache(path,root/data['splits'][split]['lpan_path'],bindings[split]['sha256'],deadline)
                if spec.sensor=='QB' and split in ('train','val'):
                    scanned['msfix_audit']=verify_qb_msfix(root/proof[f'raw_{split}_path'],path,
                        raw_sha256=proof[f'raw_{split}_sha256'],fixed_sha256=bindings[split]['sha256'],
                        band_order=spec.band_order,deadline=deadline)
                if any(data['splits'][split].get(k)!=v for k,v in dict(scanned,**identity).items()):
                    raise ValueError('Imported manifest does not match the full native/LP audit: '+split)
            atomic_json(folder/'data_verification.json',receipt_for(data))
        immutable_json(target, data)
        return target
    scans = {s: scan_source(p, s, spec, deadline) for s,p in sources.items()}
    if spec.sensor == 'QB':
        for split in ('train','val'):
            scans[split]['msfix_audit'] = verify_qb_msfix(root / proof[f'raw_{split}_path'], sources[split],
                raw_sha256=proof[f'raw_{split}_sha256'], fixed_sha256=bindings[split]['sha256'],
                band_order=spec.band_order, deadline=deadline)
    result = {}
    for split, path in sources.items():
        binding = bindings[split]
        cache = folder / 'lpan' / f'{split}_{binding["sha256"][:20]}_{canonical_sha(RECIPE)[:16]}.h5'
        _create_lp_cache(path, cache, binding['sha256'], deadline)
        identity = verify_lp_cache(path, cache, binding['sha256'], deadline)
        result[split] = dict(dataroot=str(path), sha256=binding['sha256'],
            source_identity=binding['source_identity'], lpan_path=str(cache), lpan_sha256=sha256(cache),
            **scans[split], **identity)
    import cv2
    data = dict(schema='ABLR2_DATA_v1', server=server, sensor=spec.sensor, num_bands=spec.num_bands,
        max_pixel=spec.max_dn, mtf_sensor=spec.sensor, band_order=list(spec.band_order),
        source_provenance=proof, recipe=RECIPE, augmentation=AUGMENTATION,
        augmentation_sha256=canonical_sha(AUGMENTATION), opencv_version=cv2.__version__, splits=result)
    validate_manifest(data, server)
    immutable_json(target, data)
    atomic_json(folder / 'data_verification.json',receipt_for(data))
    return target


class ABLR2Dataset(FH12Dataset):
    def __init__(self, source, lp, *, spec, split, augment=False):
        super().__init__(source, lp, max_pixel=spec.max_dn, augment=False)
        self.spec, self.split, self.augment = spec, split, bool(augment)
        self.bands, self.base_count = spec.num_bands, len(self.arrays['pan'])
        n, _, h, w = self.arrays['pan'].shape
        expected = dict(ms=(n,self.bands,h//4,w//4), lms=(n,self.bands,h,w), gt=(n,self.bands,h,w))
        if any(self.arrays[k].shape != expected[k] for k in expected if k in self.arrays):
            raise ValueError('Dataset sensor geometry mismatch')
        if not all(np.isfinite(x).all() for x in self.arrays.values()):
            raise ValueError('Nonfinite dataset')
        self.source_hashes = dict(source=sha256(source), lp=sha256(lp))

    def __len__(self): return self.base_count * (4 if self.augment else 1)

    def get_view(self, index, rot=0, augment=False):
        if not 0 <= int(index) < self.base_count: raise IndexError(index)
        return super().get_view(index, rot, augment)

    def __getitem__(self, index):
        if isinstance(index, (tuple,list)): return self.get_view(*index, augment=True)
        return self.get_view(index//4,index%4,augment=True) if self.augment else self.base(index)


def build_dataset(manifest, split, root=None, server=None, augment=False):
    if isinstance(manifest, (str,Path)): manifest = read_json(manifest)
    validate_manifest(manifest, server)
    item = manifest['splits'][split]
    source, lp = Path(root or '.') / item['dataroot'], Path(root or '.') / item['lpan_path']
    if sha256(source) != item['sha256'] or sha256(lp) != item['lpan_sha256']:
        raise ValueError('Source/LP immutable bytes changed')
    dataset = ABLR2Dataset(source, lp, spec=SensorSpec(manifest['sensor'], manifest['band_order']),
                          split=split, augment=augment)
    dataset.contract_hash = canonical_sha(manifest)
    return dataset
