"""Read-only lineage-preserving lane adapter for historical QG40 GF2 data.

The old manifest is evidence and is never rewritten. A new lane manifest may
change only the schema tag and resolve existing paths; it does not regenerate
LP, move samples, rename bands, change the native recipe, or conceal provenance.
"""
from copy import deepcopy
from pathlib import Path

from fh12.data import canonical_sha, sha256_file
from l100.data import validate_manifest, validate_source_bindings
from l100.references import data_signature as l100_signature
from qg40.references import data_signature as qg40_signature
from qg40.bootstrap import default_sensor_spec


def native_method_signature(manifest):
    """Both legacy signature conventions, deliberately independent of schema."""
    return dict(local_t=l100_signature(manifest), legacy_qg40=qg40_signature(manifest))


def bind_lane_data(original, server, root):
    """Return ``(new_manifest, explicit_provenance)`` without any writes.

    ``root`` is the local repository/source binding root, not a guessed location
    for absent assets. A missing or changed source/cache fails closed. Existing
    G20 manifests are validated the same way, without silently altering metadata.
    """
    if server not in ('s3', 's4', 's5'):
        raise ValueError('GFB20 lane data cannot bind protected s1/s2')
    if not isinstance(original, dict) or original.get('schema') not in ('G20_DATA_v1', 'QG40_DATA_v1'):
        raise ValueError('Only original G20 or historical QG40 native data is supported')
    if original['schema'] == 'QG40_DATA_v1' and server != 's3':
        raise ValueError('Historical QG40 lane wrapping is only registered for s3 P3OLD')
    root = Path(root).resolve()
    before = canonical_sha(original)
    lane = deepcopy(original)
    lane['schema'] = 'G20_DATA_v1'
    # Validate the full native method/count/provenance contract before resolving
    # paths or reading bytes. This does not treat a renamed schema as evidence.
    validate_manifest(lane, server)
    paths = []
    for split, item in lane['splits'].items():
        for key, hash_key in (('dataroot', 'sha256'), ('lpan_path', 'lpan_sha256')):
            previous = item[key]
            source = Path(previous)
            if not source.is_absolute():
                source = root / source
                item[key] = str(source.resolve(strict=True))
                paths.append(dict(split=split, field=key, original=previous, resolved=item[key]))
            if sha256_file(source) != item[hash_key]:
                raise ValueError(f'Original native data/LP bytes changed: {split}/{key}')
    catalog = default_sensor_spec(root, 'GF2')
    validate_source_bindings(lane, catalog)
    signature = native_method_signature(original)
    if native_method_signature(lane) != signature or canonical_sha(original) != before:
        raise ValueError('Lane adapter changed the original native method/provenance')
    provenance = dict(schema='GFB20_NATIVE_LANE_WRAPPER_v1', server=server,
        original_schema=original['schema'], original_manifest_sha256=before,
        lane_schema=lane['schema'], lane_manifest_sha256=canonical_sha(lane),
        native_method_sha256=canonical_sha(signature), original_manifest_unchanged=True,
        original_assets_written=False, source_and_lp_bytes_verified=True,
        official_source_catalog_verified=True, path_resolutions=paths,
        transformation=('SCHEMA_ONLY_PLUS_EXPLICIT_PATH_RESOLUTION' if original['schema'] != lane['schema']
                        else 'UNCHANGED_SCHEMA_EXPLICIT_PATH_RESOLUTION_ONLY'))
    return lane, provenance
