"""G20 immutable reference registry and measured cross-campaign readback.

R0 retains its QG40 provenance. A G20 bridge is a new receipt, never an edit to
the old manifest or a HEAD/hash waiver. R1/R2 remain distinct local origins.
"""
from pathlib import Path
import hashlib
import io
import json
import tarfile

import h5py
import numpy as np
import torch
import yaml

from fh12.common import ROOT, check_deadline, object_sha, read_json, sha256, utcnow
from fh12.data import AUGMENTATION, RECIPE
from qg40.references import FILE_KEYS, _probe
from qg40.reference_parity import identity_for, verify_q_cache, validate_receipt, ATOL, RTOL

R0_PINS = dict(teacher_checkpoint_sha256='0fb973376f9f00947d295620d96062b621145607aee311e28c889caefb656dab',
    calibration_id='45cc75ebdc29867bd904d69ecefd3f2a3de1a80daf7145e5a6227350754c5da1',
    q_cache_sha256='ddbb74a2a97661098a10db7bd9d17f5b7f7c5cb92b65ce39dd35751ed6b6d8ac',
    tau_R=0.005695626139640808, q_ref=0.4086490869522095,
    source_content_sha256='0cf7bf2af1a81036a115b3b9a181c4cc0d3f292e7a7927383f25c5819dee9d8c')
R0_SPLIT_PINS = {
    'train': ('243a0bc8a4cc0a2740ed24409f9e87afe531d87e9d57524c6bcbec2b5f937621', 'af43c2aa96c26d92874e61052647e4cf3909e8fcb680a9c584d48cfe20e3fc3b'),
    'val': ('ae15b19ccc6ece799ebfeb3fb374b1a333238226397441e3d523b0195dff4f43', 'aad58b4fd6435b695e1d00adc100c5a8d79450ea6e2d69ad33089e2fa7d9feae'),
    'rr': ('709a9a53b2e0f29d3dcd5c6ca4410c2913b62cc03c104fed6c049911dbe1c8ea', '6535574fe8efb3dbfb72c1938ca2557fb868fd9f32af9598fb7ba367267d9489'),
    'fr': ('e52d151f262a74f96e59f03124f86a27d80841073ce60376856ae679020d00dc', 'f17335ea6c433c39081ad623a22a355a5f8b469e148c77d574e283c2f3165167')}
REFERENCE_SPECS = {
    'R0': ('GF2_TA', 's3', 91001, 1e-4),
    'R1': ('GF2_G20_S1_TB_C100', 's1', 91002, 1e-4),
    'R2': ('GF2_G20_S2_TB_C100', 's2', 91002, 1e-4),
    'R3': ('GF2_G20_S2_TB_C030', 's2', 91002, 3e-5),
    'R4': ('GF2_G20_S2_TB_C300', 's2', 91002, 3e-4)}
CORE_PATHS = ('fh12/model.py', 'fh12/data.py', 'fh12/calibration.py',
    'qg40/model.py', 'qg40/data.py', 'qg40/calibration.py', 'qg40/reference_parity.py',
    'model/pancrafter_paper.py', 'model/pancrafter.py', 'model/swin.py', 'pa/aligner.py', 'pa/warp.py')
FORWARD_CONTRACT = dict(layout='P0', num_bands=4, width=112, depth=[1, 2, 3],
    attention=False, mode_modulation=False, norm='ln', margin=4,
    warp='fp32/bicubic/border/align_corners=False', unclamped_dy_dx=True)


def reference_path(reference_id, server, root=ROOT):
    from g20.common import camp
    if reference_id not in REFERENCE_SPECS:
        raise ValueError('Unknown G20 reference ID')
    return camp(root, server) / 'references' / reference_id / 'reference_manifest.json'


def _data(value, root, server):
    from g20.common import camp
    value = value or camp(root, server) / 'dataset_manifest.json'
    return read_json(value) if isinstance(value, (str, Path)) else value


def canonical_h5_identity(path, keys, deadline=None):
    """Exact ordered numeric tensor identity, independent of H5 serialization.

    No normalization/rounding occurs: all source values are represented as
    little-endian float64 (exact for the declared C4 DN/float32 LP inputs).
    """
    result = {}
    with h5py.File(path, 'r') as source:
        for key in sorted(keys):
            item = source[key]
            digest = hashlib.sha256()
            digest.update(json.dumps(list(item.shape)).encode())
            for start in range(0, len(item), 32):
                check_deadline(deadline)
                values = np.asarray(item[start:start + 32], dtype='<f8')
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite data in canonical identity')
                digest.update(values.tobytes(order='C'))
            result[key] = dict(shape=list(item.shape), sha256=digest.hexdigest())
    return dict(schema='G20_ORDERED_TENSOR_IDENTITY_v1', tensors=result,
                content_sha256=object_sha(result), normalization='none', sample_order='unchanged axis0')


def data_signature(data):
    """Path-independent identity; equality of differing files needs a receipt."""
    return {key: data.get(key) for key in ('sensor', 'num_bands', 'max_pixel', 'mtf_sensor',
        'band_order', 'recipe', 'augmentation')} | {'splits': {
        name: {key: item.get(key) for key in ('count', 'sha256', 'lpan_sha256',
            'lpan_canonical_sha256', 'sample_order_sha256', 'shapes', 'source_identity')}
        for name, item in data['splits'].items()}}


def verify_data_equivalence(origin, local, *, r0=False, deadline=None,
                            origin_tensor_receipt=None, origin_authenticated=False):
    """Authenticated callers have verified the original manifest/checkpoint chain.

    Only those callers may reuse the old manifest-bound full-LP float32 hash.
    Raw MS/GT equality always requires complete canonical tensor evidence.
    """
    if origin_tensor_receipt is not None:
        expected_binding = {split: {key: value[key] for key in ('sha256', 'lpan_sha256')}
                            for split, value in origin['splits'].items()}
        if (origin_tensor_receipt.get('schema') != 'G20_ORIGIN_TENSORS_v1'
                or origin_tensor_receipt.get('origin_data_sha256') != object_sha(origin)
                or origin_tensor_receipt.get('origin_files') != expected_binding):
            raise ValueError('Canonical tensor receipt is not bound to original data manifest')
    for data in (origin, local):
        if (data.get('sensor') != 'GF2' or data.get('num_bands') != 4
                or data.get('max_pixel') != 1023 or data.get('mtf_sensor') != 'GF2'
                or data.get('recipe') != RECIPE or data.get('augmentation') != AUGMENTATION):
            raise ValueError('G20 requires explicit unchanged GF2/C4/DN1023/LP/view identity')
    for key in ('band_order',):
        if origin.get(key) != local.get(key):
            raise ValueError('Teacher/local band order differs')
    if set(origin['splits']) != {'train', 'val', 'rr', 'fr'} or set(local['splits']) != set(origin['splits']):
        raise ValueError('All four GF2 data splits are required')
    results = {}
    for split, old in origin['splits'].items():
        check_deadline(deadline)
        new = local['splits'][split]
        if r0 and (old['sha256'], old['lpan_sha256']) != R0_SPLIT_PINS[split]:
            raise ValueError('R0 original data/LP differs from Appendix B: ' + split)
        if (old['count'] != new['count'] or old.get('shapes') != new.get('shapes')
                or old.get('sample_order_sha256') != new.get('sample_order_sha256')):
            raise ValueError('Teacher/local sample geometry/order differs: ' + split)
        for path_key, hash_key in (('dataroot', 'sha256'), ('lpan_path', 'lpan_sha256')):
            if sha256(new[path_key]) != new[hash_key]:
                raise ValueError('Local data bytes changed: ' + split + '/' + path_key)
        row = {}
        for label, path_key, hash_key, keys in (
            ('data', 'dataroot', 'sha256', ['ms', 'lms', 'pan'] + ([] if split == 'fr' else ['gt'])),
            ('lp', 'lpan_path', 'lpan_sha256', ['lpan'])):
            if old[hash_key] == new[hash_key]:
                row[label] = dict(method='file_sha256', sha256=old[hash_key])
                continue
            if label == 'lp' and origin_authenticated and old.get('lpan_canonical_sha256'):
                digest = hashlib.sha256()
                with h5py.File(new[path_key], 'r') as cache:
                    if cache['lpan'].dtype != np.dtype('float32'):
                        raise ValueError('Native LP must retain float32 cache dtype')
                    for start in range(0, len(cache['lpan']), 32):
                        check_deadline(deadline)
                        values = np.asarray(cache['lpan'][start:start + 32], dtype='<f4')
                        if not np.isfinite(values).all():
                            raise ValueError('Nonfinite LP canonical proof')
                        digest.update(values.tobytes())
                if digest.hexdigest() != old['lpan_canonical_sha256']:
                    raise ValueError('Original manifest-bound LP canonical identity differs')
                row[label] = dict(method='authenticated_origin_LP_float32_sha256',
                    origin_sha256=old[hash_key], local_sha256=new[hash_key],
                    canonical_sha256=digest.hexdigest(), sample_order_sha256=old['sample_order_sha256'])
                continue
            measured = canonical_h5_identity(new[path_key], keys, deadline)
            original_path = Path(old[path_key])
            if original_path.is_file() and sha256(original_path) == old[hash_key]:
                expected = canonical_h5_identity(original_path, keys, deadline)
            else:
                expected = (origin_tensor_receipt or {}).get('splits', {}).get(split, {}).get(label)
                if expected is None:
                    raise ValueError('Reserialized source requires original pinned tensor proof: ' + split + '/' + label)
            if measured != expected:
                raise ValueError('Teacher/local canonical ordered tensors differ: ' + split + '/' + label)
            row[label] = dict(method='canonical_ordered_tensors', origin_sha256=old[hash_key],
                             local_sha256=new[hash_key], identity=measured)
        results[split] = row
    return dict(schema='G20_DATA_EQUALITY_v1', passed=True, splits=results,
                origin_identity=object_sha(data_signature(origin)), local_identity=object_sha(data_signature(local)))


def _origin(path):
    doc = read_json(path)
    if doc.get('schema') == 'G20_REFERENCE_BRIDGE_v1':
        if not doc.get('complete') or not doc.get('output_parity', {}).get('passed'):
            raise ValueError('G20 reference readback is incomplete')
        return doc['origin_reference'], doc['resolved_artifacts'], doc
    if doc.get('schema') == 'QG40_REFERENCE_BRIDGE_v1':
        if not doc.get('complete') or not doc.get('parity', {}).get('pass'):
            raise ValueError('QG40 import is incomplete')
        return doc['origin_reference'], doc['resolved_artifacts'], None
    return doc, {key: doc[key] for key in FILE_KEYS}, None


def _check_origin(reference_id, manifest, cfg, source_data, root):
    alias, owner, seed, coefficient = REFERENCE_SPECS[reference_id]
    field = cfg.get('g20', cfg.get('qg40', {}))
    if (manifest.get('teacher_update') != 50000 or manifest.get('sensor') != 'GF2'
            or manifest.get('num_bands') != 4 or manifest.get('max_pixel') != 1023
            or manifest.get('teacher_layout') != 'P0' or cfg.get('seed') != seed
            or field.get('role') != 'T' or field.get('input_layout') != 'P0'
            or cfg.get('model_args', {}).get('hidden_size') != 112
            or list(cfg.get('model_args', {}).get('depth', [])) != [1, 2, 3]):
        raise ValueError('Reference is not the registered exact50K GF2 W112D123/P0 Teacher')
    if reference_id == 'R0':
        if manifest.get('reference_id') != 'GF2_TA' or manifest.get('server') != 's3':
            raise ValueError('R0 must retain the original s3 GF2_TA identity')
        for key in ('teacher_checkpoint_sha256', 'q_cache_sha256', 'tau_R', 'q_ref'):
            if manifest.get(key) != R0_PINS[key]:
                raise ValueError('R0 source snapshot pin differs: ' + key)
        # QG40's publisher did not store calibration_id: its uploader used the
        # complete manifest object hash. A pasted matching ID cannot authenticate
        # altered paths, receipts, source maps, or calibration metadata.
        if object_sha(manifest) != R0_PINS['calibration_id']:
            raise ValueError('R0 original calibration ID differs')
        src = manifest['source_identity']
        if (src.get('numeric_method_revision') != 'QG40_SYNC_FREQ_C4_v1'
                or src.get('content_sha256') != R0_PINS['source_content_sha256']
                or object_sha(src.get('files', {})) != src['content_sha256']):
            raise ValueError('R0 numerical source bundle pin differs')
        for name in CORE_PATHS:
            if src.get('files', {}).get(name) != sha256(Path(root) / name):
                raise ValueError('R0 numerical dependency changed; no blind compatibility waiver: ' + name)
    else:
        from g20.common import source_identity
        from g20.plan import teacher_for
        from g20.training import validate_config
        case = teacher_for(reference_id)
        if (validate_config(cfg) != case or manifest.get('reference_id') != reference_id
                or manifest.get('teacher_run_id') != case.run_id or manifest.get('server') != owner):
            raise ValueError('R1-R4 local Teacher/config/origin identity mismatch')
        if manifest.get('teacher_alias') != alias or manifest.get('consistency_weight') != coefficient:
            raise ValueError('Reference alias/consistency coefficient differs')
        current = source_identity(root)
        for key in ('files', 'content_sha256', 'numeric_method_revision'):
            if manifest['source_identity'].get(key) != current.get(key):
                raise ValueError('New G20 reference numerical source differs: ' + key)
    if manifest.get('LP_recipe') != RECIPE or manifest.get('augmentation') != AUGMENTATION:
        raise ValueError('Reference LP/view recipe differs')


def _validate_payload(reference_id, origin, paths, root):
    for key, hash_key in FILE_KEYS.items():
        if sha256(paths[key]) != origin[hash_key]:
            raise ValueError('Reference artifact bytes changed: ' + key)
    cfg = yaml.safe_load(Path(paths['teacher_config']).read_text())
    if object_sha(cfg) != origin['teacher_config_sha256']:
        raise ValueError('Teacher config object differs')
    source_data = read_json(paths['dataset_manifest_path'])
    _check_origin(reference_id, origin, cfg, source_data, root)
    identity = read_json(paths['teacher_checkpoint_identity'])
    for key, expected in {'update': 50000, 'model_sha256': origin['teacher_checkpoint_sha256'],
        'config_sha256': origin['teacher_config_sha256'], 'source_identity': origin['source_identity'],
        'training_state_sha256': origin['teacher_training_state_sha256'],
        'data_sha256': object_sha(source_data)}.items():
        if identity.get(key) != expected:
            raise ValueError('Exact50K checkpoint identity differs: ' + key)
    state = torch.load(paths['teacher_training_state'], map_location='cpu', weights_only=False)
    from qg40.model import state_hash
    from safetensors.torch import load_file
    if (state.get('full_state') is not True or state.get('update') != 50000
            or state.get('source_identity') != origin['source_identity']
            or state.get('config_sha256') != origin['teacher_config_sha256']
            or state.get('model_sha256') != origin['teacher_checkpoint_sha256']
            or state.get('data_sha256') != object_sha(source_data)
            or state_hash(state['model_state']) != identity.get('state_hash')):
        raise ValueError('Exact50K full-state payload differs')
    if state_hash(load_file(str(paths['teacher_checkpoint']), device='cpu')) != identity.get('state_hash'):
        raise ValueError('Exact50K weight tensors differ from full-state identity')
    if origin['schema'] == 'G20_REFERENCE_v1':
        shapes = {key: list(value.shape) for key, value in state['model_state'].items()}
        if (shapes != origin.get('architecture_state_shapes')
                or object_sha(shapes) != origin.get('architecture_state_shapes_sha256')
                or origin.get('architecture_forward') != FORWARD_CONTRACT):
            raise ValueError('Reference architecture state signature differs')
    del state
    cal = read_json(paths['calibration_path'])
    if cal.get('schema') not in {'QG40_CALIBRATION_v1', 'G20_CALIBRATION_v1'} or cal.get('synthetic_test') is not False:
        raise ValueError('Actual nonsynthetic calibration is required')
    for key in ('tau_R', 'q_ref', 'q_shape', 'calibration_indices_sha256', 'teacher_checkpoint_sha256', 'source_identity'):
        if cal.get(key) != origin.get(key):
            raise ValueError('Calibration/reference mismatch: ' + key)
    with np.load(paths['q_cache_path'], allow_pickle=False) as cache:
        q, indices = cache['q'].copy(), cache['calibration_indices'].copy()
    from qg40.calibration import select_calibration_indices
    if (q.shape != (source_data['splits']['train']['count'], 4) or list(q.shape) != origin['q_shape']
            or not np.isfinite(q).all() or (q < 0).any()
            or not np.array_equal(indices, select_calibration_indices(len(q)))
            or object_sha(indices.tolist()) != origin['calibration_indices_sha256']
            or not np.isfinite(origin['tau_R']) or origin['tau_R'] < 1e-6
            or not np.isfinite(origin['q_ref']) or origin['q_ref'] <= 0
            or float(np.median(q[indices])) != origin['q_ref']):
        raise ValueError('Train3072/full train x four-view calibration invalid')
    # Original receipt uses the original package's data-signature algorithm.
    from qg40.references import data_signature as legacy_signature
    signature = legacy_signature if origin['schema'] == 'QG40_REFERENCE_v1' else data_signature
    validate_receipt(read_json(paths['q_cache_parity_path']), reference_identity=identity_for(origin),
                     data_identity=object_sha(signature(source_data)), q=q)
    return q, cfg, source_data


def validate_reference(reference_id, server, root=ROOT, *, manifest_path=None, dataset_manifest=None):
    from g20.common import source_identity
    if reference_id not in REFERENCE_SPECS:
        raise ValueError('Unknown reference ID')
    origin, paths, bridge = _origin(manifest_path or reference_path(reference_id, server, root))
    q, cfg, source_data = _validate_payload(reference_id, origin, paths, root)
    local = _data(dataset_manifest, root, server)
    if bridge is None:
        if reference_id == 'R0':
            raise ValueError('R0 must complete explicit G20 measured readback first')
        if origin['source_identity'] != source_identity(root):
            raise ValueError('Local Teacher source changed before readback')
        verify_data_equivalence(source_data, local, origin_authenticated=True)
    else:
        if (bridge.get('reference_id') != reference_id
                or bridge.get('consumer_source_identity') != source_identity(root)
                or bridge.get('origin_manifest_sha256') != object_sha(origin)):
            raise ValueError('Reference bridge release/origin/alias differs')
        receipt = verify_data_equivalence(source_data, local, r0=reference_id == 'R0',
                         origin_tensor_receipt=bridge.get('origin_tensor_identity'), origin_authenticated=True)
        if bridge.get('data_equality') != receipt:
            raise ValueError('Reference bridge data proof changed')
        validate_receipt(bridge['q_cache_online_parity'], reference_identity=identity_for(origin),
                         data_identity=object_sha(data_signature(local)), q=q)
        parity = bridge['output_parity']
        if (parity.get('atol') != ATOL or parity.get('rtol') != RTOL
                or set(parity.get('max_abs_error', {})) != {'output', 'delta'}
                or any(not np.isfinite(v) or v < 0 for v in parity['max_abs_error'].values())):
            raise ValueError('Output/c parity receipt invalid')
        if sha256(bridge['output_probe_path']) != bridge['output_probe_sha256']:
            raise ValueError('Reference output/c probe changed')
    result = dict(origin, g20_reference_id=reference_id,
                  calibration_id=origin.get('calibration_id', object_sha(origin)))
    return result, q, cfg, paths


def _model(cfg, paths, device):
    from qg40.model import build_model
    from safetensors.torch import load_file
    model, _ = build_model('P0', 112, (1, 2, 3), cfg['seed'], role='T', num_bands=4)
    model.load_state_dict(load_file(str(paths['teacher_checkpoint']), device='cpu'), strict=True)
    return model.to(device).eval().requires_grad_(False)


def _online(model, origin, q, local, device, deadline):
    from g20.data import build_dataset
    dataset = build_dataset(local, 'train')
    receipt = verify_q_cache(model, dataset, q, reference_identity=identity_for(origin),
        data_identity=object_sha(data_signature(local)), device=device, deadline=deadline)
    return dataset, receipt


def _check_probe(model, probe, device):
    with torch.no_grad():
        out = model(*[torch.from_numpy(probe[key]).to(device) for key in ('pan', 'ms', 'lp')])
    differences = {}
    for key, model_key in (('output', 'y'), ('delta', 'delta')):
        actual, expected = out[model_key].detach().cpu().numpy(), probe[key]
        if actual.shape != expected.shape or not np.isfinite(expected).all() or not np.allclose(actual, expected, atol=ATOL, rtol=RTOL):
            raise ValueError('Reference output/c numerical parity failed: ' + key)
        differences[key] = float(np.max(np.abs(actual - expected)))
    return dict(passed=True, max_abs_error=differences, atol=ATOL, rtol=RTOL,
                scope='train base samples 0,1; output and native c, no test selection')


def _publish_bridge(origin, paths, reference_id, server, root, local, device, deadline,
                    *, probe=None, origin_tensors=None):
    from g20.common import immutable_json, source_identity
    q, cfg, source_data = _validate_payload(reference_id, origin, paths, root)
    equality = verify_data_equivalence(source_data, local, r0=reference_id == 'R0',
                deadline=deadline, origin_tensor_receipt=origin_tensors, origin_authenticated=True)
    model = _model(cfg, paths, device)
    dataset, parity = _online(model, origin, q, local, device, deadline)
    check_deadline(deadline)
    if probe is None:
        # In-process local readback compares the unchanged original model path
        # to the newly constructed consumer. It is not an old-GPU replay claim.
        from qg40.references import _model as legacy_model
        probe = _probe(legacy_model(cfg, paths, device), dataset, device) if reference_id == 'R0' else _probe(model, dataset, device)
        probe_origin = 'local immutable numerical-core readback; not historical-runtime replay'
    else:
        probe_origin = 'original checksummed portable probe'
    output = _check_probe(model, probe, device)
    output['probe_origin'] = probe_origin
    from fh12.calibration import _write_npz_immutable
    probe_path = reference_path(reference_id, server, root).parent / 'output_probe.npz'
    _write_npz_immutable(probe_path, probe)
    bridge = dict(schema='G20_REFERENCE_BRIDGE_v1', complete=True, reference_id=reference_id,
        imported_at_utc=utcnow(), origin_reference=origin, resolved_artifacts=paths,
        consumer_source_identity=source_identity(root), origin_manifest_sha256=object_sha(origin),
        data_equality=equality, origin_tensor_identity=origin_tensors,
        output_parity=output, output_probe_path=str(probe_path), output_probe_sha256=sha256(probe_path),
        q_cache_online_parity=parity)
    path = reference_path(reference_id, server, root)
    if path.exists():
        validate_reference(reference_id, server, root, dataset_manifest=local)
        return path
    check_deadline(deadline)
    immutable_json(path, bridge)
    validate_reference(reference_id, server, root, dataset_manifest=local)
    return path


def adopt_reference(manifest_path, reference_id, server, root=ROOT, device='cpu', *, dataset_manifest=None, deadline=None):
    """Explicit CLI action; preserve the original local assets read-only."""
    check_deadline(deadline)
    origin, paths, _ = _origin(manifest_path)
    return _publish_bridge(origin, paths, reference_id, server, root,
                           _data(dataset_manifest, root, server), device, deadline)


def load_reference(reference_id, server, root=ROOT, device='cuda', *, manifest_path=None, dataset_manifest=None, deadline=None):
    check_deadline(deadline)
    manifest, q, cfg, paths = validate_reference(reference_id, server, root,
                            manifest_path=manifest_path, dataset_manifest=dataset_manifest)
    model = _model(cfg, paths, device)
    _, receipt = _online(model, manifest, q, _data(dataset_manifest, root, server), device, deadline)
    _, _, bridge = _origin(manifest_path or reference_path(reference_id, server, root))
    if bridge:
        with np.load(bridge['output_probe_path'], allow_pickle=False) as probe:
            _check_probe(model, probe, device)
    model.q_cache_parity_receipt = receipt
    return model, manifest, q


def inventory_references(root=ROOT):
    """Metadata-only discovery. No GPU, no adoption, no file writes."""
    rows = []
    for campaign in ('_qg40', '_g20'):
        for path in sorted((Path(root) / 'work_dir' / campaign).glob('s*/references/*/reference_manifest.json')):
            try:
                origin, paths, _ = _origin(path)
                rows.append(dict(path=str(path), reference_id=origin.get('reference_id'),
                    server=origin.get('server'), teacher_checkpoint_sha256=origin.get('teacher_checkpoint_sha256'),
                    calibration_id=origin.get('calibration_id', object_sha(origin)),
                    artifacts_present=all(Path(p).is_file() for p in paths.values()),
                    status='DISCOVERED_NOT_VALIDATED'))
            except (OSError, ValueError, KeyError) as error:
                rows.append(dict(path=str(path), status='INVALID_METADATA', reason=str(error)))
    return rows


def _tensor_receipt(data, deadline, *, origin=None):
    result = {}
    for split, item in data['splits'].items():
        for key, hash_key in (('dataroot', 'sha256'), ('lpan_path', 'lpan_sha256')):
            if sha256(item[key]) != item[hash_key]:
                raise ValueError('Export source changed during tensor proof')
        result[split] = dict(data=canonical_h5_identity(item['dataroot'],
            ['ms', 'lms', 'pan'] + ([] if split == 'fr' else ['gt']), deadline),
            lp=canonical_h5_identity(item['lpan_path'], ['lpan'], deadline))
    origin = origin or data
    return dict(schema='G20_ORIGIN_TENSORS_v1', origin_data_sha256=object_sha(origin),
        origin_files={split: {key: value[key] for key in ('sha256', 'lpan_sha256')}
                      for split, value in origin['splits'].items()}, splits=result)


def export_reference(reference_id, server, root=ROOT, device='cpu', *, deadline=None, output=None):
    from g20.common import camp
    from g20.data import build_dataset
    check_deadline(deadline)
    manifest, _, cfg, paths = validate_reference(reference_id, server, root)
    origin, _, _ = _origin(reference_path(reference_id, server, root))
    local = _data(None, root, server)
    probe = _probe(_model(cfg, paths, device), build_dataset(local, 'train'), device)
    buffer = io.BytesIO(); np.savez(buffer, **probe)
    entries = {'parity_probe.npz': buffer.getvalue(),
        'origin_manifest.json': (json.dumps(origin, sort_keys=True, allow_nan=False) + '\n').encode(),
        'origin_tensor_identity.json': (json.dumps(_tensor_receipt(local, deadline,
            origin=read_json(paths['dataset_manifest_path'])), sort_keys=True) + '\n').encode()}
    mapping = {}
    for key, path in paths.items():
        name = 'assets/' + key + Path(path).suffix
        entries[name] = Path(path).read_bytes(); mapping[key] = name
    index = dict(schema='G20_REFERENCE_EXPORT_v1', reference_id=reference_id, mapping=mapping,
        hashes={key: hashlib.sha256(raw).hexdigest() for key, raw in entries.items()})
    entries['index.json'] = (json.dumps(index, sort_keys=True) + '\n').encode()
    path = Path(output) if output else camp(root, server) / 'outgoing' / (reference_id + '.tar.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with tarfile.open(path) as archive:
            if json.load(archive.extractfile('index.json')) != index:
                raise ValueError('Existing reference export differs')
            if any(hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected
                   for name, expected in index['hashes'].items()):
                raise ValueError('Existing reference export payload changed')
        return path
    temporary = path.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for name, raw in entries.items():
            check_deadline(deadline)
            info = tarfile.TarInfo(name); info.size = len(raw); info.mode = 0o644
            archive.addfile(info, io.BytesIO(raw))
    check_deadline(deadline); temporary.replace(path)
    return path


def import_reference(archive, reference_id, server, root=ROOT, device='cpu', *, deadline=None):
    check_deadline(deadline)
    target = reference_path(reference_id, server, root)
    if target.exists():
        validate_reference(reference_id, server, root)
        return target
    with tarfile.open(archive) as package:
        members = package.getmembers()
        if (len({m.name for m in members}) != len(members)
                or any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members)):
            raise ValueError('Unsafe/duplicate reference archive entry')
        index = json.load(package.extractfile('index.json'))
        if index.get('schema') not in {'G20_REFERENCE_EXPORT_v1', 'QG40_REFERENCE_EXPORT_v1'}:
            raise ValueError('Unknown reference archive schema')
        expected_id = 'GF2_TA' if reference_id == 'R0' and index['schema'].startswith('QG40_') else reference_id
        if index.get('reference_id') != expected_id:
            raise ValueError('Wrong reference archive alias; R1/R2 cannot be merged')
        if set(m.name for m in members) != set(index['hashes']) | {'index.json'}:
            raise ValueError('Unlisted archive payload')
        if set(index.get('mapping', {})) != set(FILE_KEYS) or any(
                name not in index['hashes'] or not name.startswith('assets/') for name in index['mapping'].values()):
            raise ValueError('Invalid archive asset mapping')
        contents = {name: package.extractfile(name).read() for name in index['hashes']}
        for name, raw in contents.items():
            if hashlib.sha256(raw).hexdigest() != index['hashes'][name]:
                raise ValueError('Reference transfer checksum mismatch: ' + name)
    origin = json.loads(contents['origin_manifest.json'])
    destination = target.parent / 'imported_bytes'
    for name, raw in contents.items():
        check_deadline(deadline)
        path = destination / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Symlink in reference import destination')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('Existing imported reference bytes differ')
        if not path.exists():
            path.write_bytes(raw)
    paths = {key: str(destination / name) for key, name in index['mapping'].items()}
    with np.load(io.BytesIO(contents['parity_probe.npz']), allow_pickle=False) as values:
        probe = {key: values[key].copy() for key in values.files}
    tensors = json.loads(contents['origin_tensor_identity.json']) if 'origin_tensor_identity.json' in contents else None
    return _publish_bridge(origin, paths, reference_id, server, root, _data(None, root, server),
                           device, deadline, probe=probe, origin_tensors=tensors)
