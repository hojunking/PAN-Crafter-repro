"""Read-only binding of the exact original Student and native Teacher assets.

No native recalibration, guessed reference, cross-server Teacher, endpoint
replacement, or old-manifest rewrite is performed here. A binding is evidence,
not permission to waive a mismatched source, runtime, data, or checkpoint.
"""
from pathlib import Path
import math

import numpy as np
import torch
from safetensors.torch import load_file

from fh12.common import ROOT, check_deadline, object_sha, read_json, resolved_path, sha256
from fh12.data import RECIPE, AUGMENTATION
from g20.model import build_model, state_hash
from qg40.references import FILE_KEYS
from gfb20.common import read_config

BUNDLE = 'research_log/PANDA_GF2_B20_S345_20H_Bundle_2026-09-22'
SCHEMA = 'GFB20_PARENT_BINDING_v1'
REFERENCE_SCHEMA = 'GFB20_NATIVE_REFERENCE_BINDING_v1'
# Campaign orchestration may change; these imported numerical implementations may not.
NUMERICAL_PATHS = ('fh12/model.py', 'fh12/data.py', 'fh12/calibration.py',
    'qg40/model.py', 'qg40/data.py', 'qg40/calibration.py', 'qg40/reference_parity.py',
    'model/pancrafter_paper.py', 'model/pancrafter.py', 'model/swin.py',
    'pa/aligner.py', 'pa/warp.py')
RUNTIME_KEYS = ('torch', 'numpy', 'scipy', 'skimage', 'cuda', 'cudnn',
                'tf32_matmul', 'tf32_cudnn')


def registry(root=ROOT):
    return read_json(Path(root) / BUNDLE / 'GFB20_ParentAssets.json')


def _path(value, root):
    if not value:
        raise FileNotFoundError('Original asset path is unbound (null)')
    path = resolved_path(value, root).resolve()
    if not path.is_file():
        raise FileNotFoundError('Original asset not found: ' + str(path))
    return path


def _field(cfg):
    found = [cfg[k] for k in ('l100', 'qg40', 'g20', 'gfb20') if k in cfg]
    if len(found) != 1:
        raise ValueError('Exactly one original campaign config is required')
    return found[0]


def _architecture_config(cfg, role):
    args = cfg.get('model_args', {})
    expected = dict(hidden_size=112 if role == 'T' else 104,
        depth=[1, 2, 3] if role == 'T' else [1, 2, 2], out_channels=4,
        attn_locations=[], mode_modulation=False, norm='ln', dropout=0.)
    if (any(args.get(k) != v for k, v in expected.items())
            or cfg.get('num_bands') != 4 or cfg.get('max_pixel') != 1023
            or cfg.get('mixed_precision') != 'no' or cfg.get('batch_size') != 48):
        raise ValueError('Original backbone/task/bands/precision contract differs')


def source_compatibility(origin, root=ROOT):
    """Compare shared imported numerical sources and actual runtime, not HEAD."""
    from gfb20.common import source_identity
    current = source_identity(root)
    if (not isinstance(origin, dict) or not origin.get('files')
            or object_sha(origin['files']) != origin.get('content_sha256')):
        raise ValueError('Original numerical source manifest is incomplete')
    changed = [name for name in NUMERICAL_PATHS
               if origin['files'].get(name) != sha256(Path(root) / name)]
    changed += [key for key in RUNTIME_KEYS if origin.get(key) != current.get(key)]
    # Newer originals also bind these flags; older QG40 did not record them.
    changed += [key for key in ('cudnn_benchmark', 'cudnn_deterministic',
        'deterministic_algorithms') if key in origin and origin[key] != current.get(key)]
    if changed:
        raise ValueError('Original numerical/runtime mismatch: ' + ', '.join(changed))
    return dict(shared_files={name: origin['files'][name] for name in NUMERICAL_PATHS},
                runtime={key: current.get(key) for key in RUNTIME_KEYS}, passed=True)


def _signed(value):
    return dict(value, binding_sha256=object_sha(value))


def _verify_signature(value, schema):
    if value.get('schema') != schema or value.get('binding_sha256') != object_sha(
            {k: v for k, v in value.items() if k != 'binding_sha256'}):
        raise ValueError('Asset binding schema/content checksum differs')


def discover_bindings(server, root=ROOT):
    """Discover only registered run IDs and their actual original resolved configs.

    A directory name is never accepted as proof; bind_parent performs full proof.
    Missing/ambiguous discovery stays unbound and needs an explicit input map.
    """
    if server not in ('s3', 's4', 's5'):
        raise ValueError('GFB20 is restricted to s3/s4/s5')
    root = Path(root)
    result = {}
    for key, spec in registry(root)['parents'].items():
        if spec['owner_server'] != server:
            continue
        candidates = []
        for base in (root / 'work_dir', root / 'work_dir/l100', root / 'work_dir/g20',
                     root / 'work_dir/qg40'):
            run = base / spec['source_run_id']
            cfg = run / 'meta/config.resolved.yaml'
            if cfg.is_file():
                candidates.append(dict(parent_config=str(cfg.resolve()),
                    parent_checkpoint=str((run / 'candidates' / str(spec['parent_step']) / 'model.safetensors').resolve()),
                    parent_identity=str((run / 'candidates' / str(spec['parent_step']) / 'identity.json').resolve()),
                    origin_root=str(root.resolve())))
        unique = {row['parent_config']: row for row in candidates}
        result[key] = next(iter(unique.values())) if len(unique) == 1 else None
    return result


def _finite_state(path, identity):
    if sha256(path) != identity.get('model_sha256'):
        raise ValueError('Checkpoint bytes differ from original identity')
    state = load_file(str(path), device='cpu')
    if not state or state_hash(state) != identity.get('state_hash') or not all(
            torch.isfinite(t).all().item() for t in state.values()):
        raise ValueError('Checkpoint tensor identity/nonfinite weights')
    if not any(k.startswith('backbone.') for k in state) or not any(k.startswith('aligner.') for k in state):
        raise ValueError('Both original U and A weights are mandatory')
    return state


def _fullstate(path, identity, tensors):
    if identity.get('full_state') is not True or sha256(path) != identity.get('training_state_sha256'):
        raise ValueError('Exact original endpoint fullstate is missing or changed')
    value = torch.load(path, map_location='cpu', weights_only=False)
    keys = ('update', 'config_sha256', 'data_sha256', 'source_identity', 'reference_sha256', 'model_sha256')
    if (value.get('full_state') is not True or value.get('precision') != 'fp32'
            or any(value.get(k) != identity.get(k) for k in keys)
            or value.get('scheduler', {}).get('last_epoch') != identity['update']
            or state_hash(value['model_state']) != state_hash(tensors)):
        raise ValueError('Exact original fullstate progress/model/provenance differs')
    return value


def _original_progress(state, cfg, count):
    from fh12.training import BatchStream
    from qg40.exposure import validate_counts
    step = state['update']
    if cfg.get('num_iter') != step:
        raise ValueError('Original checkpoint is not its exact declared training horizon')
    stream = BatchStream(count, cfg['batch_size'], cfg['seed'])
    stream.load_state_dict(state['sampler'])
    if stream.epoch * stream.count + stream.cursor != step:
        raise ValueError('Original endpoint sampler progress differs')
    validate_counts(state.get('exposure_counts'), count, step, cfg['batch_size'])
    groups = state.get('optimizer', {}).get('param_groups', [])
    rates = state.get('scheduler', {}).get('_last_lr')
    if ([g.get('name') for g in groups] != ['U', 'A'] or rates != [0., 0.]
            or any(g.get('lr') != 0. for g in groups)):
        raise ValueError('Original endpoint optimizer does not end its cosine at zero')


def _data_check(data, root):
    if (data.get('sensor') != 'GF2' or data.get('num_bands') != 4
            or data.get('max_pixel') != 1023 or data.get('mtf_sensor') != 'GF2'
            or data.get('recipe') != RECIPE or data.get('augmentation') != AUGMENTATION
            or set(data.get('splits', {})) != {'train', 'val', 'rr', 'fr'}
            or data['splits']['rr']['count'] != 20 or data['splits']['fr']['count'] != 20):
        raise ValueError('Original GF2 native data/LP/evaluation contract differs')
    for split, item in data['splits'].items():
        for key, digest in (('dataroot', 'sha256'), ('lpan_path', 'lpan_sha256')):
            if sha256(_path(item[key], root)) != item[digest]:
                raise ValueError('Actual original data/LP bytes changed: ' + split + '/' + key)


def _reference_binding(path, reference_key, server, spec, root, origin_root, local_data=None):
    from qg40.calibration import select_calibration_indices
    from qg40.reference_parity import identity_for, validate_receipt
    doc = read_json(path)
    if doc.get('schema') in ('QG40_REFERENCE_BRIDGE_v1', 'G20_REFERENCE_BRIDGE_v1'):
        # Importing the old G20 controller/reference module would also import
        # its runtime exceptions. Read only a genuinely passed bridge, and
        # re-authenticate every original artifact and native online q below.
        passed = (doc.get('parity', {}).get('pass') if doc['schema'] == 'QG40_REFERENCE_BRIDGE_v1'
                  else doc.get('output_parity', {}).get('passed'))
        if doc.get('complete') is not True or passed is not True:
            raise ValueError('Original reference bridge lacks passed measured parity')
        origin, locations = doc['origin_reference'], doc['resolved_artifacts']
    else:
        origin, locations = doc, {key: doc[key] for key in FILE_KEYS}
    # Preserve the original object exactly: no path rewrites to spoof its SHA.
    if (origin.get('schema') not in ('L100_LOCAL_REFERENCE_v2', 'QG40_REFERENCE_v1')
            or origin.get('server') != server or origin.get('sensor') != 'GF2'
            or origin.get('num_bands') != 4 or origin.get('max_pixel') != 1023
            or origin.get('teacher_layout') != 'P0'
            or origin.get('teacher_update') != spec['teacher_updates']
            or ('teacher_seed' in origin and origin['teacher_seed'] != spec['teacher_seed'])
            or origin.get('LP_recipe') != RECIPE or origin.get('augmentation') != AUGMENTATION):
        raise ValueError('Original local Teacher/reference identity differs')
    if spec.get('teacher_run_id') and origin.get('teacher_run_id') != spec['teacher_run_id']:
        raise ValueError('Original Teacher run was substituted')
    expected_id = reference_key if origin['schema'] == 'L100_LOCAL_REFERENCE_v2' else spec.get('teacher_alias', 'GF2_TA')
    if origin.get('reference_id') != expected_id:
        raise ValueError('Original reference ID was substituted')
    if origin['schema'] == 'L100_LOCAL_REFERENCE_v2' and any(
            origin.get(key) != server for key in ('owner_server', 'producer_server')):
        raise ValueError('Original local reference was produced on another server')
    if spec.get('published_teacher_sha256') and origin.get('teacher_checkpoint_sha256') != spec['published_teacher_sha256']:
        raise ValueError('Published Teacher checkpoint SHA differs')
    proof = source_compatibility(origin['source_identity'], root)
    paths = {key: _path(locations[key], origin_root) for key in FILE_KEYS}
    for key, hash_key in FILE_KEYS.items():
        if sha256(paths[key]) != origin.get(hash_key):
            raise ValueError('Original reference artifact SHA differs: ' + key)
    cfg = read_config(paths['teacher_config'])
    _architecture_config(cfg, 'T')
    field, args = _field(cfg), cfg['model_args']
    if (object_sha(cfg) != origin.get('teacher_config_sha256') or cfg.get('seed') != spec['teacher_seed']
            or field.get('role') != 'T' or field.get('sensor') != 'GF2'
            or field.get('input_layout') != 'P0' or args.get('hidden_size') != 112
            or args.get('depth') != [1, 2, 3]
            or Path(cfg['work_dir']).name != origin.get('teacher_run_id')):
        raise ValueError('Original Teacher config/architecture differs')
    identity = read_json(paths['teacher_checkpoint_identity'])
    if any(identity.get(key) != expected for key, expected in dict(update=spec['teacher_updates'],
            role='T', sensor='GF2', num_bands=4, input_layout='P0',
            model_sha256=origin['teacher_checkpoint_sha256'],
            config_sha256=object_sha(cfg), source_identity=origin['source_identity']).items()):
        raise ValueError('Original Teacher endpoint identity differs')
    state = _finite_state(paths['teacher_checkpoint'], identity)
    full = _fullstate(paths['teacher_training_state'], identity, state)
    data = read_json(paths['dataset_manifest_path'])
    if identity.get('data_sha256') != object_sha(data):
        raise ValueError('Original Teacher data identity differs')
    _original_progress(full, cfg, data['splits']['train']['count'])
    _data_check(data, origin_root)
    from l100.references import data_signature as shared_signature
    if local_data is not None and shared_signature(local_data) != shared_signature(data):
        raise ValueError('Local data differs from native original reference')
    if local_data is not None:
        _data_check(local_data, root)
    calibration = read_json(paths['calibration_path'])
    if (calibration.get('synthetic_test') is not False or calibration.get('schema') not in
            ('L100_CALIBRATION_v2', 'QG40_CALIBRATION_v1')):
        raise ValueError('Original production calibration is required')
    for key in ('tau_R', 'q_ref', 'q_shape', 'teacher_checkpoint_sha256',
                'source_identity', 'calibration_indices_sha256'):
        if calibration.get(key) != origin.get(key):
            raise ValueError('Original calibration manifest differs: ' + key)
    with np.load(paths['q_cache_path'], allow_pickle=False) as cache:
        q, indices = cache['q'].copy(), cache['calibration_indices'].copy()
    if (q.dtype != np.float32 or q.shape != (data['splits']['train']['count'], 4)
            or list(q.shape) != origin['q_shape'] or not np.isfinite(q).all() or (q < 0).any()
            or not np.array_equal(indices, select_calibration_indices(len(q)))
            or object_sha(indices.tolist()) != origin['calibration_indices_sha256']
            or not math.isfinite(origin['tau_R']) or origin['tau_R'] < 1e-6
            or not math.isfinite(origin['q_ref']) or origin['q_ref'] <= 0
            or float(np.median(q[indices])) != origin['q_ref']):
        raise ValueError('Native train3072/four-view q/scales do not match original')
    if origin['schema'] == 'QG40_REFERENCE_v1':
        from qg40.references import data_signature
    else:
        from l100.references import data_signature
    validate_receipt(read_json(paths['q_cache_parity_path']), reference_identity=identity_for(origin),
        data_identity=object_sha(data_signature(data)), q=q)
    result = dict(schema=REFERENCE_SCHEMA, native_reference_id=reference_key,
        server=server, sensor='GF2', native_manifest=origin,
        original_manifest_path=str(path), original_manifest_sha256=sha256(path),
        native_manifest_sha256=object_sha(origin), origin_root=str(origin_root),
        resolved_artifacts={key: str(value) for key, value in paths.items()},
        data=data, data_sha256=object_sha(data), compatibility=proof,
        teacher_config=cfg, teacher_checkpoint_identity=identity)
    return _signed(result)


def bind_parent(parent_id, server, binding, root=ROOT, *, local_dataset_manifest=None):
    """Read and authenticate original endpoint bytes; caller persists the receipt."""
    doc = registry(root)
    spec = doc['parents'].get(parent_id)
    if spec is None or spec['owner_server'] != server:
        raise ValueError('Unregistered or nonlocal Student parent')
    if not binding:
        raise FileNotFoundError('Parent binding is missing: ' + parent_id)
    origin_root = Path(binding.get('origin_root', root)).resolve()
    cfg_path = _path(binding.get('parent_config'), origin_root)
    model_path = _path(binding.get('parent_checkpoint'), origin_root)
    identity_path = _path(binding.get('parent_identity'), origin_root)
    cfg = read_config(cfg_path)
    _architecture_config(cfg, 'S')
    field, args = _field(cfg), cfg['model_args']
    if (Path(cfg['work_dir']).name != spec['source_run_id'] or cfg.get('seed') != spec['parent_seed']
            or field.get('role') != 'S' or field.get('sensor') != 'GF2'
            or field.get('input_layout') != 'PLH' or args.get('hidden_size') != 104
            or args.get('depth') != [1, 2, 2]):
        raise ValueError('Wrong parent run/seed/Student architecture')
    identity = read_json(identity_path)
    expected = dict(update=spec['parent_step'], role='S', sensor='GF2', num_bands=4,
                    input_layout='PLH', config_sha256=object_sha(cfg))
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError('Parent is not the exact registered endpoint/config')
    if spec.get('published_model_sha256') and sha256(model_path) != spec['published_model_sha256']:
        raise ValueError('Parent bytes differ from published endpoint SHA')
    source_compatibility(identity['source_identity'], root)
    tensors = _finite_state(model_path, identity)
    full_path = _path(binding.get('parent_fullstate', model_path.parent / 'training_state.pt'), origin_root)
    full = _fullstate(full_path, identity, tensors)
    data_path = _path(field.get('dataset_manifest'), origin_root)
    data = read_json(data_path)
    if object_sha(data) != identity.get('data_sha256'):
        raise ValueError('Original Student dataset manifest differs')
    _original_progress(full, cfg, data['splits']['train']['count'])
    _data_check(data, origin_root)
    ref_path = _path(binding.get('reference_manifest') or field.get('reference_manifest'), origin_root)
    configured_ref = _path(field.get('reference_manifest'), origin_root)
    if sha256(ref_path) != sha256(configured_ref):
        raise ValueError('Explicit reference does not match original Student configuration')
    local_data = (read_json(_path(local_dataset_manifest, root)) if isinstance(local_dataset_manifest, (str, Path))
                  else local_dataset_manifest)
    native = _reference_binding(ref_path, spec['reference_key'], server,
        doc['references'][spec['reference_key']], root, origin_root, local_data or data)
    if identity.get('reference_sha256') != native['native_manifest_sha256']:
        raise ValueError('Student did not train with this exact native reference')
    model, _ = build_model('PLH', 104, [1, 2, 2], spec['parent_seed'], role='S', num_bands=4,
        teacher_aligner_state={k[len('aligner.'):]: v for k, v in tensors.items() if k.startswith('aligner.')})
    model.load_state_dict(tensors, strict=True)
    result = dict(schema=SCHEMA, parent_id=parent_id, server=server, sensor='GF2',
        parent_run_id=spec['source_run_id'], parent_step=spec['parent_step'], parent_seed=spec['parent_seed'],
        parent_checkpoint=str(model_path), parent_model_sha256=sha256(model_path),
        parent_config=str(cfg_path), parent_config_sha256=sha256(cfg_path),
        parent_identity=str(identity_path), parent_identity_sha256=sha256(identity_path),
        parent_fullstate=str(full_path), parent_fullstate_sha256=sha256(full_path),
        parent_dataset_manifest=str(data_path), parent_dataset_manifest_sha256=sha256(data_path),
        parent_tensor_hashes=dict(full=state_hash(tensors), U=state_hash(model.backbone.state_dict()),
                                  A=state_hash(model.aligner.state_dict())),
        parent_compute={key: full.get(key) for key in ('training_seconds', 'evaluation_seconds', 'io_seconds')},
        native_reference=native, original_source_identity=identity['source_identity'],
        original_binding={k: str(v) for k, v in binding.items()}, origin_root=str(origin_root))
    return _signed(result)


def verify_parent(value, root=ROOT, *, local_dataset_manifest=None):
    value = read_json(value) if isinstance(value, (str, Path)) else value
    _verify_signature(value, SCHEMA)
    if value.get('parent_kind') == 'GFB20_FRESH_TRUNK':
        checked = bind_fresh_parent(value['parent_id'], value['server'], value['parent_config'], root)
    else:
        checked = bind_parent(value['parent_id'], value['server'], value['original_binding'], root,
                              local_dataset_manifest=local_dataset_manifest)
    if checked != value:
        raise ValueError('Original parent/reference binding changed after registration')
    return value


def bind_fresh_parent(parent_id, server, config_path, root=ROOT):
    """Bind only A11/A14's exact100K trunk, never an intermediate best model."""
    from gfb20.common import read_config, source_identity
    from gfb20.plan import case_for, validate_config
    ids = {'P3NEW1': 'A11', 'P3NEW2': 'A14'}
    if parent_id not in ids or server != 's3':
        raise ValueError('Only the two registered s3 fresh100K trunks can become new parents')
    cfg_path = _path(config_path, root)
    cfg = read_config(cfg_path)
    case = validate_config(cfg, require_bound=True)
    if case != case_for(ids[parent_id]) or case.updates != 100000 or case.is_ft:
        raise ValueError('Fresh parent config is not the declared A11/A14 trunk')
    run = resolved_path(cfg['work_dir'], root)
    status = read_json(run / 'meta/training_status.json')
    if status.get('actual_updates') != 100000 or status.get('training_complete') is not True:
        raise ValueError('Fresh parent trunk has not completed all100K updates')
    folder = run / 'candidates/100000'
    model_path, identity_path = folder / 'model.safetensors', folder / 'identity.json'
    identity = read_json(identity_path)
    expected = dict(update=100000, role='S', sensor='GF2', num_bands=4,
                    input_layout='PLH', config_sha256=object_sha(cfg), source_identity=source_identity(root))
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError('Fresh trunk exact endpoint identity differs')
    tensors = _finite_state(model_path, identity)
    full_path = folder / 'training_state.pt'
    full = _fullstate(full_path, identity, tensors)
    data_path = _path(cfg['gfb20']['dataset_manifest'], root)
    data = read_json(data_path)
    if identity.get('data_sha256') != object_sha(data):
        raise ValueError('Fresh trunk data identity differs')
    from gfb20.postrun import verify_endpoint_progress
    verify_endpoint_progress(cfg, full, data)
    _data_check(data, root)
    native = verify_reference(_path(cfg['gfb20']['reference_manifest'], root), root,
                              local_dataset_manifest=data)
    if native['native_reference_id'] != 'R3_100' or native['server'] != server:
        raise ValueError('Fresh trunk reference is not local R3_100')
    # Native reference identity is recorded by the trainer in the full run's
    # immutable start manifest; reference_sha also contains measured parity.
    start = read_json(run / 'meta/training_start_manifest.json')
    if start.get('reference_sha256') != identity.get('reference_sha256'):
        raise ValueError('Fresh trunk reference changed within its own run')
    model, _ = build_model('PLH', 104, [1, 2, 2], case.seed, role='S', num_bands=4,
        teacher_aligner_state={k[len('aligner.'):]: v for k, v in tensors.items() if k.startswith('aligner.')})
    model.load_state_dict(tensors, strict=True)
    return _signed(dict(schema=SCHEMA, parent_kind='GFB20_FRESH_TRUNK', parent_id=parent_id,
        server=server, sensor='GF2', parent_run_id=case.run_id, parent_step=100000, parent_seed=case.seed,
        parent_checkpoint=str(model_path), parent_model_sha256=sha256(model_path),
        parent_config=str(cfg_path), parent_config_sha256=sha256(cfg_path),
        parent_identity=str(identity_path), parent_identity_sha256=sha256(identity_path),
        parent_fullstate=str(full_path), parent_fullstate_sha256=sha256(full_path),
        parent_dataset_manifest=str(data_path), parent_dataset_manifest_sha256=sha256(data_path),
        parent_tensor_hashes=dict(full=state_hash(tensors), U=state_hash(model.backbone.state_dict()),
                                  A=state_hash(model.aligner.state_dict())),
        parent_compute={k: full.get(k) for k in ('training_seconds', 'evaluation_seconds', 'io_seconds')},
        native_reference=native, original_source_identity=identity['source_identity'],
        origin_root=str(Path(root).resolve()), shared_trunk_cost_counted_once=True))


def verify_reference(value, root=ROOT, *, local_dataset_manifest=None):
    value = read_json(value) if isinstance(value, (str, Path)) else value
    if value.get('schema') == SCHEMA:
        value = value['native_reference']
    _verify_signature(value, REFERENCE_SCHEMA)
    local = read_json(_path(local_dataset_manifest, root)) if isinstance(local_dataset_manifest, (str, Path)) else local_dataset_manifest
    spec = registry(root)['references'][value['native_reference_id']]
    checked = _reference_binding(_path(value['original_manifest_path'], root),
        value['native_reference_id'], value['server'], spec, root, value['origin_root'], local)
    if checked != value:
        raise ValueError('Native reference binding changed after registration')
    return value


def load_parent(value, device='cpu', root=ROOT, *, local_dataset_manifest=None):
    evidence = verify_parent(value, root, local_dataset_manifest=local_dataset_manifest)
    state = load_file(evidence['parent_checkpoint'], device='cpu')
    model, _ = build_model('PLH', 104, [1, 2, 2], evidence['parent_seed'], role='S', num_bands=4,
        teacher_aligner_state={k[len('aligner.'):]: v for k, v in state.items() if k.startswith('aligner.')})
    # This is the parent's own A, never a new Teacher clone.
    model.load_state_dict(state, strict=True)
    return model.to(device), evidence


def load_reference(value, device='cpu', root=ROOT, *, local_dataset_manifest=None, deadline=None):
    check_deadline(deadline)
    evidence = verify_reference(value, root, local_dataset_manifest=local_dataset_manifest)
    origin, paths = evidence['native_manifest'], evidence['resolved_artifacts']
    # Original QG40 manifests bind the seed through the signed Teacher config,
    # rather than an extra top-level teacher_seed field.
    seed = evidence['teacher_config']['seed']
    model, _ = build_model('P0', 112, [1, 2, 3], seed, role='T', num_bands=4)
    model.load_state_dict(load_file(paths['teacher_checkpoint'], device='cpu'), strict=True)
    model.to(device).eval().requires_grad_(False)
    with np.load(paths['q_cache_path'], allow_pickle=False) as cache:
        q = cache['q'].copy()
    data = evidence['data']
    from qg40.reference_parity import identity_for, verify_q_cache
    if data.get('schema') == 'QG40_DATA_v1':
        from qg40.data import build_dataset
        from qg40.references import data_signature
    elif data.get('schema') == 'G20_DATA_v1':
        from g20.data import build_dataset
        from l100.references import data_signature
    else:
        raise ValueError('Unknown original native data schema; no silent conversion')
    parity = verify_q_cache(model, build_dataset(data, 'train', root=evidence['origin_root']), q,
        reference_identity=identity_for(origin), data_identity=object_sha(data_signature(data)),
        device=device, deadline=deadline)
    info = dict(origin, **paths)
    info.update(native_reference_id=evidence['native_reference_id'], native_manifest=origin,
        native_binding=evidence, native_binding_sha256=evidence['binding_sha256'],
        data=data, dataset_manifest_path=paths['dataset_manifest_path'],
        train_sha256=data['splits']['train']['sha256'], train_lpan_sha256=data['splits']['train']['lpan_sha256'],
        teacher_sha=origin['teacher_checkpoint_sha256'], teacher_seed=seed, native_online_parity=parity)
    return model, q, info


def load_training_assets(cfg, root=ROOT, device='cpu', deadline=None):
    field = cfg['gfb20']
    parent = parent_model = None
    if field.get('parent_manifest'):
        parent_model, parent = load_parent(_path(field['parent_manifest'], root), device, root,
            local_dataset_manifest=field.get('dataset_manifest'))
        native = parent['native_reference']
    else:
        native = read_json(_path(field.get('reference_manifest'), root))
    teacher, q, reference = load_reference(native, device, root,
        local_dataset_manifest=field.get('dataset_manifest'), deadline=deadline)
    if (reference['native_binding']['server'] != field['server']
            or reference['native_reference_id'] != field['reference_key']):
        raise ValueError('Training case is not bound to its registered local reference')
    data = read_json(_path(field['dataset_manifest'], root))
    # An online CUDA measurement may vary within the fixed accepted tolerance.
    # It is audit evidence, not a new cache/reference identity on every resume.
    parity = reference.pop('native_online_parity')
    return dict(parent_model=parent_model, parent=parent, teacher=teacher, q=q,
                reference=reference, data=data, native_online_parity=parity)
