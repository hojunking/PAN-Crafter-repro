"""Immutable sensor/Teacher-specific references and portable parity-checked import."""
from pathlib import Path
import io
import json
import tarfile

import numpy as np
import torch
import yaml

from qg40.common import (ROOT, camp, atomic_json, immutable_json, load_checkpoint_model,
                        check_deadline, numerical_compatibility, object_sha, read_json, sha256,
                        source_identity, utcnow)

FILE_KEYS = {
    'teacher_checkpoint': 'teacher_checkpoint_sha256',
    'teacher_training_state': 'teacher_training_state_sha256',
    'teacher_config': 'teacher_config_file_sha256',
    'teacher_checkpoint_identity': 'teacher_checkpoint_identity_sha256',
    'calibration_path': 'calibration_sha256', 'q_cache_path': 'q_cache_sha256',
    'dataset_manifest_path': 'dataset_manifest_sha256',
    'q_cache_parity_path': 'q_cache_parity_sha256',
}


def reference_path(reference_id, server, root=ROOT):
    from qg40.plan import teacher_for
    teacher_for(reference_id)  # Reject arbitrary path-like aliases.
    return camp(root, server) / 'references' / reference_id / 'reference_manifest.json'


def data_signature(data):
    """Separate immutable tensor/recipe identity from relocatable local paths."""
    return dict(sensor=data['sensor'], num_bands=data['num_bands'], mtf_sensor=data['mtf_sensor'],
                max_pixel=data['max_pixel'], band_order=data['band_order'],
                recipe=data['recipe'], augmentation=data['augmentation'],
                source_provenance={k: v for k, v in data.get('source_provenance', {}).items() if not k.endswith('_path')},
                splits={split: {key: value for key, value in entry.items()
                    if key in ('count', 'sha256', 'sample_order_sha256', 'shapes', 'source_identity')} | {
                    'lpan_content': entry.get('lpan_canonical_sha256', entry['lpan_sha256'])}
                    for split, entry in data['splits'].items()})


def _origin(path):
    doc = read_json(path)
    if doc.get('schema') == 'QG40_REFERENCE_BRIDGE_v1':
        if not doc.get('complete') or not doc.get('parity', {}).get('pass'):
            raise ValueError('Reference import parity has not passed')
        return doc['origin_reference'], doc['resolved_artifacts'], doc
    return doc, {key: doc[key] for key in FILE_KEYS}, None


def validate_reference(reference_id, server, root=ROOT, *, manifest_path=None,
                       dataset_manifest=None):
    from qg40.plan import teacher_for, sensor_spec
    from fh12.data import AUGMENTATION, RECIPE
    case = teacher_for(reference_id)
    path = Path(manifest_path) if manifest_path else reference_path(reference_id, server, root)
    manifest, paths, bridge = _origin(path)
    if (manifest.get('schema') != 'QG40_REFERENCE_v1' or manifest.get('teacher_update') != 50000
            or manifest.get('reference_id') != reference_id or manifest.get('sensor') != case.sensor
            or manifest.get('teacher_run_id') != case.run_id or manifest.get('teacher_layout') != 'P0'):
        raise ValueError('Reference sensor/Teacher/exact50K identity mismatch')
    current = source_identity(root)
    numerical_compatibility(manifest['source_identity'], current)
    if bridge and bridge['consumer_source_identity'] != current:
        raise ValueError('Local import execution identity changed; do not rewrite original provenance')
    for key, hash_key in FILE_KEYS.items():
        if sha256(paths[key]) != manifest[hash_key]:
            raise ValueError('Reference bytes changed: ' + key)
    cfg = yaml.safe_load(Path(paths['teacher_config']).read_text())
    if object_sha(cfg) != manifest['teacher_config_sha256']:
        raise ValueError('Teacher resolved config changed')
    f = cfg['qg40']
    from qg40.training import validate_config
    if validate_config(cfg) != case:
        raise ValueError('Reference resolved recipe differs from registered Teacher')
    if (f['role'] != 'T' or f['sensor'] != case.sensor or f['reference_id'] != reference_id
            or cfg['seed'] != case.teacher_seed or f['profile'] != case.profile
            or manifest.get('num_bands') != 4 or manifest.get('max_pixel') != sensor_spec(case.sensor).max_dn):
        raise ValueError('Teacher configuration contradicts reference')
    identity = read_json(paths['teacher_checkpoint_identity'])
    for key, expected in {'update': 50000, 'model_sha256': manifest['teacher_checkpoint_sha256'],
                          'config_sha256': manifest['teacher_config_sha256'],
                          'source_identity': manifest['source_identity']}.items():
        if identity.get(key) != expected:
            raise ValueError('Teacher checkpoint identity differs: ' + key)
    if identity.get('training_state_sha256') != manifest['teacher_training_state_sha256']:
        raise ValueError('Teacher exact50K fullstate identity differs')
    state = torch.load(paths['teacher_training_state'], map_location='cpu', weights_only=False)
    from qg40.model import state_hash
    if (state.get('full_state') is not True or state.get('update') != 50000
            or state.get('model_sha256') != manifest['teacher_checkpoint_sha256']
            or state.get('config_sha256') != manifest['teacher_config_sha256']
            or state.get('source_identity') != manifest['source_identity']
            or state_hash(state['model_state']) != identity.get('state_hash')):
        raise ValueError('Teacher fullstate payload is not the declared exact50K state')
    cal = read_json(paths['calibration_path'])
    if cal.get('schema') != 'QG40_CALIBRATION_v1' or cal.get('synthetic_test') is not False:
        raise ValueError('Production reference needs actual nonsynthetic calibration')
    for key in ('tau_R', 'q_ref', 'calibration_indices_sha256', 'q_shape',
                'teacher_checkpoint_sha256', 'source_identity'):
        if cal.get(key) != manifest.get(key):
            raise ValueError('Calibration/reference mismatch: ' + key)
    if manifest.get('LP_recipe') != RECIPE or manifest.get('augmentation') != AUGMENTATION:
        raise ValueError('Reference LP recipe/augmentation changed')
    source_data = read_json(paths['dataset_manifest_path'])
    if state.get('data_sha256') != object_sha(source_data) or identity.get('data_sha256') != object_sha(source_data):
        raise ValueError('Teacher fullstate/calibration data provenance mismatch')
    del state
    local_path = dataset_manifest or camp(root, server) / 'dataset_manifest.json'
    local_data = read_json(local_path) if isinstance(local_path, (str, Path)) else local_path
    if data_signature(local_data) != data_signature(source_data):
        raise ValueError('Local sensor/split/LP/band source differs from Teacher calibration')
    # H5/LP validation is performed by the sensor reader as well; no old WV3 fallback.
    item = local_data['splits']['train']
    if sha256(item['dataroot']) != item['sha256'] or sha256(item['lpan_path']) != item['lpan_sha256']:
        raise ValueError('Local training H5/LP changed after binding')
    with np.load(paths['q_cache_path'], allow_pickle=False) as cache:
        q, indices = cache['q'].copy(), cache['calibration_indices'].copy()
    if (q.shape != (item['count'], 4) or list(q.shape) != manifest['q_shape']
            or not np.isfinite(q).all() or (q < 0).any()
            or indices.shape != (3072,) or len(np.unique(indices)) != 3072
            or indices.min() < 0 or indices.max() >= len(q)
            or object_sha(indices.tolist()) != manifest['calibration_indices_sha256']):
        raise ValueError('Full train/four-view q cache or calibration3072 identity invalid')
    # Calibration implementation owns the exact deterministic selection ordering.
    from qg40.calibration import select_calibration_indices
    expected_indices = select_calibration_indices(len(q))
    if not np.array_equal(indices, expected_indices):
        raise ValueError('Sensor Teacher calibration does not use canonical shared base IDs')
    if (not np.isfinite(manifest['tau_R']) or manifest['tau_R'] < 1e-6
            or not np.isfinite(manifest['q_ref']) or manifest['q_ref'] <= 0
            or float(np.median(q[indices])) != manifest['q_ref']):
        raise ValueError('Measured tau/q_ref invalid; no numerical repair permitted')
    from qg40.reference_parity import identity_for, validate_receipt
    parity_identity = dict(reference_identity=identity_for(manifest),
                           data_identity=object_sha(data_signature(local_data)), q=q)
    validate_receipt(read_json(paths['q_cache_parity_path']), **parity_identity)
    if bridge:
        validate_receipt(bridge.get('q_cache_online_parity', {}), **parity_identity)
    return manifest, q, cfg, paths


def _model(cfg, paths, device):
    from qg40.model import build_model
    from safetensors.torch import load_file
    f, args = cfg['qg40'], cfg['model_args']
    model, _ = build_model('P0', args['hidden_size'], args['depth'], cfg['seed'],
                          role='T', num_bands=f['num_bands'])
    model.load_state_dict(load_file(str(paths['teacher_checkpoint']), device='cpu'), strict=True)
    return model.to(device).eval().requires_grad_(False)


def _online_parity(model, manifest, q, local_data, device, deadline):
    from qg40.data import build_dataset
    from qg40.reference_parity import identity_for, verify_q_cache
    dataset = build_dataset(local_data, 'train')
    return verify_q_cache(model, dataset, q, reference_identity=identity_for(manifest),
                          data_identity=object_sha(data_signature(local_data)),
                          device=device, deadline=deadline)


def load_reference(reference_id, server, root=ROOT, device='cuda', *, manifest_path=None,
                   dataset_manifest=None, deadline=None):
    check_deadline(deadline)
    manifest, q, cfg, paths = validate_reference(reference_id, server, root,
                        manifest_path=manifest_path, dataset_manifest=dataset_manifest)
    model = _model(cfg, paths, device)
    data = dataset_manifest or camp(root, server) / 'dataset_manifest.json'
    data = read_json(data) if isinstance(data, (str, Path)) else data
    receipt = _online_parity(model, manifest, q, data, device, deadline)
    model.q_cache_parity_receipt = receipt
    atomic_json(camp(root, server) / 'diagnostics' / (reference_id + '_q_cache_online_load.json'), receipt)
    return model, manifest, q


@torch.no_grad()
def _probe(model, dataset, device):
    batch = [dataset.base(i) for i in (0, 1)]
    values = [torch.stack([row[j] for row in batch]).to(device) for j in range(5)]
    _, _, ms, lp, pan = values
    out = model(pan, ms, lp)
    return {key: tensor.detach().cpu().numpy() for key, tensor in
            {'pan': pan, 'ms': ms, 'lp': lp, 'output': out['y'], 'delta': out['delta']}.items()}


def export_reference(reference_id, server, root=ROOT, device='cpu', *, deadline=None):
    from qg40.data import build_dataset
    from qg40.plan import teacher_for
    check_deadline(deadline)
    if teacher_for(reference_id).server_id != server:
        raise ValueError('Only the registered Teacher owner may export this reference')
    manifest, _, cfg, paths = validate_reference(reference_id, server, root)
    check_deadline(deadline)
    dataset = build_dataset(read_json(camp(root, server) / 'dataset_manifest.json'), 'train')
    values = _probe(_model(cfg, paths, device), dataset, device)
    check_deadline(deadline)
    buffer = io.BytesIO()
    np.savez(buffer, **values)
    entries = {'parity_probe.npz': buffer.getvalue(),
               'origin_manifest.json': reference_path(reference_id, server, root).read_bytes()}
    mapping = {}
    for key, source in paths.items():
        name = 'assets/' + key + Path(source).suffix
        entries[name] = Path(source).read_bytes()
        mapping[key] = name
    import hashlib
    index = dict(schema='QG40_REFERENCE_EXPORT_v1', reference_id=reference_id,
                 sensor=manifest['sensor'], mapping=mapping,
                 hashes={key: hashlib.sha256(raw).hexdigest() for key, raw in entries.items()},
                 origin_manifest_sha256=sha256(reference_path(reference_id, server, root)),
                 probe_device=device, probe_scope='train base samples 0,1; no test selection')
    entries['index.json'] = (json.dumps(index, indent=2) + '\n').encode()
    path = camp(root, server) / 'outgoing' / (reference_id + '.tar.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with tarfile.open(path) as tf:
            if json.load(tf.extractfile('index.json')) != index:
                raise ValueError('Existing reference export differs; refusing overwrite')
        return path
    temporary = path.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as tf:
        for name, raw in entries.items():
            check_deadline(deadline)
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(raw), 0o644
            tf.addfile(info, io.BytesIO(raw))
    check_deadline(deadline)
    temporary.replace(path)
    return path


def import_reference(archive, reference_id, server, root=ROOT, device='cpu', *, deadline=None):
    import hashlib
    from qg40.plan import cases_for
    check_deadline(deadline)
    allowed = {c.reference_id for c in cases_for(server) if c.role == 'S'}
    if reference_id not in allowed:
        raise ValueError('Reference not registered for this server')
    target = reference_path(reference_id, server, root)
    if target.exists():
        validate_reference(reference_id, server, root)
        return target
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        if any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):
            raise ValueError('Unsafe reference archive path/type')
        if len({m.name for m in members}) != len(members):
            raise ValueError('Duplicate reference archive entries')
        index = json.load(tf.extractfile('index.json'))
        if index.get('schema') != 'QG40_REFERENCE_EXPORT_v1' or index.get('reference_id') != reference_id:
            raise ValueError('Wrong reference package')
        if set(m.name for m in members) != set(index['hashes']) | {'index.json'}:
            raise ValueError('Unlisted reference package payload')
        if (set(index.get('mapping', {})) != set(FILE_KEYS)
                or any(name not in index['hashes'] or not name.startswith('assets/')
                       for name in index['mapping'].values())):
            raise ValueError('Unsafe or incomplete reference asset mapping')
        contents = {name: tf.extractfile(name).read() for name in index['hashes']}
        for name, raw in contents.items():
            if hashlib.sha256(raw).hexdigest() != index['hashes'][name]:
                raise ValueError('Reference transfer checksum mismatch: ' + name)
    origin = json.loads(contents['origin_manifest.json'])
    if hashlib.sha256(contents['origin_manifest.json']).hexdigest() != index['origin_manifest_sha256']:
        raise ValueError('Reference original manifest changed in transit')
    destination = target.parent / 'imported_bytes'
    # Refuse stale filesystem redirects, including dangling links, before writing.
    for candidate in (destination, *destination.parents):
        if candidate.is_symlink():
            raise ValueError('Symlink in reference import destination')
    destination.mkdir(parents=True, exist_ok=True)
    for name, raw in contents.items():
        check_deadline(deadline)
        path = destination / name
        for candidate in (path, *path.parents):
            if candidate.is_symlink():
                raise ValueError('Symlink in reference import destination')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('Existing imported reference bytes differ')
        if not path.exists():
            path.write_bytes(raw)
    paths = {key: str(destination / name) for key, name in index['mapping'].items()}
    cfg = yaml.safe_load(Path(paths['teacher_config']).read_text())
    numerical_compatibility(origin['source_identity'], source_identity(root))
    check_deadline(deadline)
    model = _model(cfg, paths, device)
    with np.load(io.BytesIO(contents['parity_probe.npz']), allow_pickle=False) as probe:
        with torch.no_grad():
            out = model(*[torch.from_numpy(probe[key]).to(device) for key in ('pan', 'ms', 'lp')])
        differences = {}
        for key, model_key in (('output', 'y'), ('delta', 'delta')):
            actual, expected = out[model_key].cpu().numpy(), probe[key]
            differences[key] = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, atol=3e-6, rtol=2e-5):
                raise ValueError('Reference import numerical parity failed: ' + key)
    check_deadline(deadline)
    with np.load(paths['q_cache_path'], allow_pickle=False) as cache:
        local_q = cache['q'].copy()
    online_parity = _online_parity(model, origin, local_q,
        read_json(camp(root, server) / 'dataset_manifest.json'), device, deadline)
    bridge = dict(schema='QG40_REFERENCE_BRIDGE_v1', complete=True, reference_id=reference_id,
                  imported_at_utc=utcnow(), origin_reference=origin, resolved_artifacts=paths,
                  consumer_source_identity=source_identity(root),
                  origin_manifest_sha256=index['origin_manifest_sha256'],
                  parity=dict(pass_=True, **differences, atol=3e-6, rtol=2e-5),
                  q_cache_online_parity=online_parity)
    bridge['parity']['pass'] = bridge['parity'].pop('pass_')
    # Validate complete bridge before publishing it as consumable.
    pending = target.with_name('pending_reference.json')
    atomic_json(pending, bridge)
    validate_reference(reference_id, server, root, manifest_path=pending)
    check_deadline(deadline)
    immutable_json(target, bridge)
    return target
