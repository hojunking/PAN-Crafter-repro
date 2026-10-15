"""Read-only discovery of observation assets; geometry is not provenance.

Follow explicit local roots (including their directory symlinks), but visit
each physical directory/file only once. Read HDF5 headers and small attributes,
never imagery or metadata datasets. No entry is admitted for training here.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import h5py
import numpy as np


EXTENSIONS = {'.h5', '.hdf5', '.hdf'}
MAX_ATTRIBUTES = 64
MAX_ATTRIBUTE_ELEMENTS = 32
MAX_ATTRIBUTE_BYTES = 4096
MAX_STRING = 512
MAX_OBJECTS = 1024


def _plain(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    if isinstance(value, str):
        return value if len(value) <= MAX_STRING else value[:MAX_STRING] + ' [TRUNCATED]'
    if isinstance(value, np.ndarray):
        return _plain(value.item()) if value.ndim == 0 else [_plain(v) for v in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_numeric_attribute': str(value)}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _plain(str(value))


def _attributes(obj):
    result = {}
    names = sorted(obj.attrs.keys())
    for name in names[:MAX_ATTRIBUTES]:
        try:
            aid = obj.attrs.get_id(name)
            shape, dtype = list(aid.shape), str(aid.dtype)
            count = math.prod(shape) if shape else 1
            size = aid.get_storage_size()
            if count > MAX_ATTRIBUTE_ELEMENTS or size > MAX_ATTRIBUTE_BYTES:
                result[name] = {'omitted_large_attribute': True, 'shape': shape,
                                'dtype': dtype, 'storage_bytes': int(size)}
            else:
                result[name] = _plain(obj.attrs[name])
        except Exception as exc:
            result[name] = {'attribute_error': type(exc).__name__ + ': ' + str(exc)[:MAX_STRING]}
    if len(names) > MAX_ATTRIBUTES:
        result['_inventory_omitted_attribute_count'] = len(names) - MAX_ATTRIBUTES
    return result


def _pair_geometry(datasets, pan_key, ms_key):
    if pan_key not in datasets or ms_key not in datasets:
        return False
    p, m = datasets[pan_key]['shape'], datasets[ms_key]['shape']
    return (len(p) == len(m) == 4 and p[0] == m[0] and p[0] > 0
            and p[1] == 1 and m[1] > 0 and min(m[-2:]) > 0
            and p[-2] == 4 * m[-2] and p[-1] == 4 * m[-1])


def _geometry(datasets):
    if _pair_geometry(datasets, 'native/pan', 'native/ms'):
        return 'NATIVE_SCENE_CANDIDATE'
    if _pair_geometry(datasets, 'pan', 'ms'):
        p = datasets['pan']['shape']
        gt = datasets.get('gt', {}).get('shape', [])
        if p[-2:] == [64, 64] and len(gt) == 4 and gt[0] == p[0] and gt[-2:] == p[-2:]:
            return 'RR_PATCH_ONLY'
        return 'RATIO4_OBSERVATIONS_UNKNOWN_ROLE'
    return 'UNKNOWN_GEOMETRY'


def _lineage_fields(attributes, objects):
    matches = {'scene_fields': [], 'acquisition_fields': [], 'crop_fields': [],
               'provenance_containers': []}

    def check(location, key):
        k = key.lower()
        if any(s in k for s in ('source_scene', 'scene_id', 'parent_scene')):
            matches['scene_fields'].append(location)
        if any(s in k for s in ('acquisition', 'product_id', 'catalog_id')):
            matches['acquisition_fields'].append(location)
        if any(s in k for s in ('crop', 'patch_index', 'origin', 'offset', 'coordinate')):
            matches['crop_fields'].append(location)
        if any(s in k for s in ('provenance', 'manifest', 'source_metadata', 'geometry', 'split')):
            matches['provenance_containers'].append(location)

    for key in attributes:
        check('@' + key, key)
    for path, meta in objects.items():
        check(path, path.rsplit('/', 1)[-1])
        for key in meta['attributes']:
            check(path + '@' + key, key)
    return {**{k: sorted(set(v)) for k, v in matches.items()},
            'verified_source_identity': False,
            'note': 'Field presence only; metadata dataset values and underlying lineage are not verified.'}


def inspect_h5(path):
    """Return JSON-ready metadata without reading any HDF5 dataset values."""
    path = Path(path)
    result = {'path': str(path), 'resolved_path': str(path.resolve()),
              'size_bytes': path.stat().st_size, 'full_file_sha256_verified': False,
              'admission_granted': False}
    objects = {}
    with h5py.File(path, 'r') as source:
        result['attributes'] = _attributes(source)

        def visit(name, obj):
            if len(objects) >= MAX_OBJECTS:
                result['object_inventory_truncated'] = True
                return 'stop'
            record = {'kind': 'dataset' if isinstance(obj, h5py.Dataset) else 'group',
                      'attributes': _attributes(obj)}
            if isinstance(obj, h5py.Dataset):
                record.update(shape=list(obj.shape) if obj.shape is not None else [],
                              dtype=str(obj.dtype))
            objects[name] = record
            return None

        source.visititems(visit)
    result['datasets'] = {k: v for k, v in objects.items() if v['kind'] == 'dataset'}
    result['groups'] = {k: v for k, v in objects.items() if v['kind'] == 'group'}
    result['lineage_metadata'] = _lineage_fields(result['attributes'], objects)
    result['geometry_class'] = _geometry(result['datasets'])
    result['classification'] = result['geometry_class']
    return result


def _named_role(paths):
    # Filename evidence can disqualify a training candidate, never establish it.
    names = [str(p).lower() for p in paths]
    tokens = ('test_', '/test/', 'full_examples', 'reduced_examples', 'benchmark', '/eval/')
    if any(any(t in name for t in tokens) for name in names):
        return 'EVAL_NOT_TRAIN'
    if any(any(t in name for t in ('valid_', '/valid/', '/validation/')) for name in names):
        return 'VALIDATION_NOT_TRAIN'
    return 'UNKNOWN_ROLE'


def inventory(roots):
    """Inspect only explicit roots; return errors and ambiguous roles honestly."""
    roots = sorted({str(Path(p).expanduser().absolute()) for p in roots})
    if not roots:
        raise ValueError('At least one explicit inventory root is required')
    errors, directories, files = [], set(), {}
    pending = list(reversed(roots))
    while pending:
        path = Path(pending.pop())
        try:
            stat = path.stat()
            key = (stat.st_dev, stat.st_ino)
            if path.is_dir():
                if key in directories:
                    continue
                directories.add(key)
                pending.extend(reversed([str(p) for p in sorted(path.iterdir(), key=lambda p: p.name)]))
            elif path.is_file() and path.suffix.lower() in EXTENSIONS:
                files.setdefault(key, []).append(str(path))
        except (OSError, RuntimeError) as exc:
            errors.append({'path': str(path), 'error': type(exc).__name__ + ': ' + str(exc)[:MAX_STRING]})
    entries = []
    for paths in sorted(files.values(), key=lambda x: min(x)):
        paths = sorted(set(paths))
        try:
            entry = inspect_h5(paths[0])
            entry['discovered_paths'] = paths
            entry['role_from_path_only'] = _named_role(paths + [entry['resolved_path']])
            if entry['role_from_path_only'] != 'UNKNOWN_ROLE':
                entry['classification'] = entry['role_from_path_only']
            entries.append(entry)
        except Exception as exc:
            errors.append({'path': paths[0], 'error': type(exc).__name__ + ': ' + str(exc)[:MAX_STRING]})
    return {'schema': 'TEACHER48_PREPARATION_INVENTORY_V1', 'roots': roots,
            'admission_granted': False, 'dataset_values_read': False,
            'full_file_sha256_verified': False, 'files': entries, 'errors': errors,
            'summary': {'file_count': len(entries), 'error_count': len(errors),
                        'classifications': dict(sorted(Counter(e['classification'] for e in entries).items()))},
            'limitations': ['Shapes do not establish native acquisition, source-scene identity, or training role.',
                           'Metadata datasets are listed but not read; large attributes and object lists are capped.',
                           'No exact/near-duplicate, crop correspondence, geographic separation, or prior audit performed.',
                           'Repeated physical directories/files are deduplicated, including symlink cycles.']}
