"""Verify/reuse existing GF2 bytes read-only; new LP assets stay in _l100."""
from pathlib import Path

from fh12.data import RECIPE, AUGMENTATION, canonical_sha, _create_lp_cache
from g20.data import (SPLITS, PROVENANCE_FIELDS, canonical_band_indices,
                      _scan_source, verify_lp_cache, build_dataset)
from l100.common import (atomic_json, camp, check_deadline, immutable_json, object_sha,
                         read_json, sha256)


def validate_manifest(manifest, server):
    if (manifest.get('schema') != 'G20_DATA_v1' or manifest.get('server') != server
            or manifest.get('sensor') != 'GF2' or manifest.get('num_bands') != 4
            or manifest.get('max_pixel') != 1023 or manifest.get('mtf_sensor') != 'GF2'):
        raise ValueError('Local GF2 C4 / DN1023 / owner identity mismatch')
    if canonical_band_indices(manifest.get('band_order')) != (0, 1, 2, 3):
        raise ValueError('LOCAL-T requires native B,G,R,NIR ordering')
    if (manifest.get('recipe') != RECIPE or manifest.get('augmentation') != AUGMENTATION
            or manifest.get('augmentation_sha256') != canonical_sha(AUGMENTATION)):
        raise ValueError('Native LP / fixed augmentation contract changed')
    proof = manifest.get('source_provenance', {})
    if proof.get('units') != 'DN' or any(not proof.get(k) for k in PROVENANCE_FIELDS):
        raise ValueError('GF2 source provenance is not fully bound')
    if set(manifest.get('splits', {})) != set(SPLITS):
        raise ValueError('Exactly train/val/rr/fr are required')
    for split, expected in dict(train=19809, val=2201, rr=20, fr=20).items():
        if manifest.get('splits', {}).get(split, {}).get('count') != expected:
            raise ValueError(f'{split}: expected {expected} GF2 source samples')
    return manifest


def validate_source_bindings(manifest, spec):
    """Self-consistent files are not enough: bind the pinned official catalog."""
    if spec.sensor != 'GF2' or spec.num_bands != 4 or spec.max_dn != 1023:
        raise ValueError('Bundled source catalog is not the GF2 C4/DN1023 contract')
    for split in SPLITS:
        expected, actual = spec.split(split), manifest['splits'][split]
        if (Path(actual.get('dataroot', '')).resolve() != Path(expected.path).resolve()
                or actual.get('sha256') != expected.sha256
                or actual.get('source_identity') != expected.source_identity):
            raise ValueError('Existing source differs from the pinned GF2 catalog: ' + split)
    return manifest


def prepare_data(root, server, deadline=None, manifest_path=None):
    from qg40.bootstrap import default_sensor_spec
    import cv2
    root = Path(root).resolve()
    folder = camp(root, server)
    target = folder / 'dataset_manifest.json'
    candidates = [target] if target.is_file() else []
    if manifest_path:
        explicit = Path(manifest_path)
        if not explicit.is_absolute():
            explicit = root / explicit
        if not explicit.is_file():
            raise ValueError('Explicit local data manifest does not exist: ' + str(explicit))
        if target.is_file() and read_json(target) != read_json(explicit):
            raise ValueError('Explicit data manifest differs from the already-bound local manifest')
        candidates.append(explicit)
    candidates += [root / f'work_dir/_g20/{server}/dataset_manifest.json',
                   root / f'work_dir/_g20/{server}/GF2/dataset_manifest.json']
    source = next((p for p in candidates if p.is_file()), None)
    catalog = default_sensor_spec(root, 'GF2')
    if source:
        data = validate_manifest(read_json(source), server)
        validate_source_bindings(data, catalog)
        from dataclasses import replace
        from l100.plan import sensor_spec
        spec = replace(sensor_spec(), band_order=tuple(data['band_order']))
        for split, item in data['splits'].items():
            check_deadline(deadline)
            if sha256(item['dataroot']) != item['sha256'] or sha256(item['lpan_path']) != item['lpan_sha256']:
                raise ValueError('Existing GF2 source/LP bytes changed: ' + split)
            scanned = _scan_source(item['dataroot'], split, spec, deadline)
            if scanned['count'] != item['count']:
                raise ValueError('GF2 source sample count changed')
            actual = verify_lp_cache(item['dataroot'], item['lpan_path'], item['sha256'], deadline)
            if any(item.get(k) != v for k, v in actual.items()):
                raise ValueError('Existing GF2 LP correspondence changed')
    else:
        spec = catalog
        proof = dict(spec.source_provenance)
        if any(not proof.get(k) for k in PROVENANCE_FIELDS) or proof.get('units') != 'DN':
            raise ValueError('Bound GF2 source evidence missing')
        splits = {}
        for split in SPLITS:
            check_deadline(deadline)
            binding = spec.split(split)
            native = Path(binding.path).resolve(strict=True)
            if sha256(native) != binding.sha256 or not binding.source_identity:
                raise ValueError('GF2 source binding changed: ' + split)
            scan = _scan_source(native, split, spec, deadline)
            cache = folder / 'GF2/lpan' / f'{split}_{binding.sha256[:20]}_{canonical_sha(RECIPE)[:16]}.h5'
            _create_lp_cache(native, cache, binding.sha256, deadline)
            identity = verify_lp_cache(native, cache, binding.sha256, deadline)
            splits[split] = dict(dataroot=str(native), sha256=binding.sha256,
                source_identity=binding.source_identity, lpan_path=str(cache),
                lpan_sha256=sha256(cache), **identity, **scan)
        data = dict(schema='G20_DATA_v1', server=server, sensor='GF2', num_bands=4,
            max_pixel=spec.max_dn, band_order=list(spec.band_order), mtf_sensor=spec.mtf_sensor,
            source_provenance=proof, splits=splits, opencv_version=cv2.__version__,
            recipe=RECIPE, augmentation=AUGMENTATION, augmentation_sha256=canonical_sha(AUGMENTATION))
        validate_manifest(data, server)
    immutable_json(target, data)
    atomic_json(folder / 'data_verification.json', dict(complete=True,
        dataset_manifest_sha256=object_sha(data), reused_from=str(source) if source else None,
        all_pixels_lp_verified=True, previous_campaign_written=False))
    return target
