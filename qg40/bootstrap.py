"""Read-only binding of the bundled, evidence-backed source catalog.

This is only a binding bootstrap: source hashes and declared provenance are
checked, but no HDF5 scan, P0 certification, cache creation, or clock is run.
Declared paths are checkout-relative; their symlink targets may be shared data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qg40.data import PROVENANCE_FIELDS, SPLITS, canonical_band_indices
from qg40.plan import sensor_spec


def _declared_file(root, value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: a declared relative source path is required")
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f"{label}: source paths must be relative without '..'")
    try:
        path = (root / relative).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label}: missing or inaccessible declared source: {value}") from exc
    if not path.is_file():
        raise ValueError(f"{label}: declared source is not a file: {value}")
    return str(path)


def _checked_sha(value, label):
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in '0123456789abcdef' for c in value)):
        raise ValueError(f"{label}: a lowercase SHA256 is required")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def default_sensor_spec(root, sensor):
    """Bind one catalog sensor and verify all declared source/raw file hashes.

    Explicit local configuration takes precedence at the caller. This function
    neither discovers missing paths nor infers band order or provenance.
    """
    spec = sensor_spec(sensor)
    root = Path(root)
    catalog_path = root / 'qg40' / 'sensor_sources.json'
    try:
        catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read QG40 source catalog: {catalog_path}") from exc
    if not isinstance(catalog, dict) or catalog.get('schema') != 'QG40_SENSOR_SOURCES_v1':
        raise ValueError('Invalid QG40 source catalog schema')
    sensors = catalog.get('sensors')
    if not isinstance(sensors, dict) or not isinstance(sensors.get(sensor), dict):
        raise ValueError(f"QG40 source catalog has no declared {sensor} sensor")
    entry = sensors[sensor]
    if entry.get('sensor', sensor) != sensor:
        raise ValueError(f"QG40 source catalog sensor mismatch for {sensor}")
    order = entry.get('band_order')
    if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
        raise ValueError(f"{sensor}: explicit source band_order list is required")
    canonical_band_indices(order)
    sources = entry.get('splits')
    if not isinstance(sources, dict) or set(sources) != set(SPLITS):
        raise ValueError(f"{sensor}: explicitly declare train, val, rr, and fr sources")
    splits = {}
    for split in SPLITS:
        source = sources[split]
        label = f'{sensor}/{split}'
        if not isinstance(source, dict) or set(source) != {'path', 'sha256', 'source_identity'}:
            raise ValueError(f"{label}: path, sha256, and source_identity are required")
        identity = source['source_identity']
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"{label}: a nonempty source_identity is required")
        splits[split] = dict(path=_declared_file(root, source['path'], label),
                             sha256=_checked_sha(source['sha256'], label),
                             source_identity=identity)
    proof = entry.get('source_provenance')
    if not isinstance(proof, dict):
        raise ValueError(f"{sensor}: source_provenance is required")
    proof = dict(proof)
    missing = [key for key in PROVENANCE_FIELDS if not proof.get(key)]
    if missing:
        raise ValueError(f"{sensor}: missing source provenance: {', '.join(missing)}")
    if proof['units'] != 'DN':
        raise ValueError(f"{sensor}: source provenance units must be DN")
    if sensor == 'QB':
        if not proof.get('msfix_recipe'):
            raise ValueError('QB: source provenance requires msfix_recipe')
        for split in ('train', 'val'):
            key = f'raw_{split}'
            label = f'QB/{key}'
            path = _declared_file(root, proof.get(f'{key}_path'), label)
            expected = _checked_sha(proof.get(f'{key}_sha256'), label)
            try:
                actual = _sha(path)
            except OSError as exc:
                raise ValueError(f'{label}: cannot read declared raw source: {path}') from exc
            if actual != expected:
                raise ValueError(f'{label}: missing or changed raw source SHA256: {path}')
            proof[f'{key}_path'] = path
    try:
        return spec.bind(band_order=order, splits=splits,
                         source_provenance=proof, verify_files=True)
    except OSError as exc:
        raise ValueError(f'{sensor}: cannot read a declared source file') from exc
