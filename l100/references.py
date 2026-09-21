"""Endpoint-bound LOCAL-T references; no cross-server import or legacy fallback."""
from pathlib import Path
import math

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from fh12.data import RECIPE, AUGMENTATION
from g20.model import build_model, state_hash
from l100.common import ROOT, camp, object_sha, read_json, resolved_path, sha256, source_identity, check_deadline
from l100.plan import CAMPAIGN_ID, REFERENCES, teacher_for, validate_config
from qg40.calibration import select_calibration_indices
from qg40.reference_parity import identity_for, verify_q_cache, validate_receipt

FILE_KEYS = {'teacher_checkpoint': 'teacher_checkpoint_sha256',
             'teacher_training_state': 'teacher_training_state_sha256',
             'teacher_config': 'teacher_config_file_sha256',
             'teacher_checkpoint_identity': 'teacher_checkpoint_identity_sha256',
             'calibration_path': 'calibration_sha256', 'q_cache_path': 'q_cache_sha256',
             'dataset_manifest_path': 'dataset_manifest_sha256',
             'q_cache_parity_path': 'q_cache_parity_sha256'}
FORWARD_CONTRACT = dict(layout='P0', num_bands=4, width=112, depth=[1, 2, 3],
    attention=False, mode_modulation=False, norm='ln', margin=4,
    warp='fp32/bicubic/border/align_corners=False', unclamped_dy_dx=True)


def data_signature(data):
    """The existing path-independent GF2 tensor/recipe identity, not a bridge."""
    return {key: data.get(key) for key in ('sensor', 'num_bands', 'max_pixel', 'mtf_sensor',
        'band_order', 'recipe', 'augmentation')} | {'splits': {
        name: {key: item.get(key) for key in ('count', 'sha256', 'lpan_sha256',
            'lpan_canonical_sha256', 'sample_order_sha256', 'shapes', 'source_identity')}
        for name, item in data['splits'].items()}}


def reference_path(reference_id, server, root=ROOT):
    if reference_id not in REFERENCES or REFERENCES[reference_id].server_id != server:
        raise ValueError('Only this server\'s registered local Teacher reference is permitted')
    return camp(root, server) / 'references' / reference_id / 'reference_manifest.json'


def _validate_endpoint_progress(state, cfg, case, count):
    """Direct calibration cannot bypass exact sampler/optimizer endpoint checks."""
    from fh12.training import BatchStream
    from l100.training import cosine_factor, aligner_factor
    stream = BatchStream(count, cfg['batch_size'], case.seed)
    stream.load_state_dict(state['sampler'])
    if stream.epoch * stream.count + stream.cursor != case.updates:
        raise ValueError('Teacher endpoint sampler differs from completed updates')
    factor = cosine_factor(case.updates, cfg['num_warmup'], case.updates)
    expected = [cfg['learning_rate'] * factor,
                cfg['l100']['aligner_lr'] * factor * aligner_factor(case.updates, case.profile)]
    groups = state.get('optimizer', {}).get('param_groups', [])
    learning_rates = state.get('scheduler', {}).get('_last_lr')
    if ([group.get('name') for group in groups] != ['U', 'A']
            or not isinstance(learning_rates, list) or len(learning_rates) != 2
            or any(not math.isclose(float(group['lr']), wanted, rel_tol=1e-10, abs_tol=1e-14)
                   or not math.isclose(float(actual), wanted, rel_tol=1e-10, abs_tol=1e-14)
                   for group, actual, wanted in zip(groups, learning_rates, expected))):
        raise ValueError('Teacher endpoint optimizer LR differs from fresh-horizon scheduler')


def teacher_endpoint(teacher_run, root, server):
    """Validate the exact declared endpoint without requiring its 50 RR/FR reports."""
    root, run = Path(root).resolve(), Path(teacher_run)
    if not run.is_absolute():
        run = root / (run if len(run.parts) > 1 else Path('work_dir') / run)
    cfg_path = run / 'meta/config.resolved.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    case = validate_config(cfg, require_bound=True)
    if case.role != 'T' or case.server_id != server or run.name != case.run_id or teacher_for(case.reference_id) != case:
        raise ValueError('Endpoint must be the registered local Teacher')
    if resolved_path(cfg['work_dir'], root).resolve() != run.resolve():
        raise ValueError('Teacher endpoint run/config path mismatch')
    folder = run / 'candidates' / str(case.updates)
    identity = read_json(folder / 'identity.json')
    data_path = resolved_path(cfg['l100']['dataset_manifest'], root)
    data = read_json(data_path)
    release = source_identity(root)
    expected = dict(update=case.updates, full_state=True, role='T', sensor='GF2', num_bands=4,
        input_layout='P0', config_sha256=object_sha(cfg), data_sha256=object_sha(data),
        source_identity=release, reference_sha256=None)
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError('Teacher endpoint identity/horizon/source differs')
    weights, state_path = folder / 'model.safetensors', folder / 'training_state.pt'
    if sha256(weights) != identity.get('model_sha256') or sha256(state_path) != identity.get('training_state_sha256'):
        raise ValueError('Teacher endpoint checksum mismatch')
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    checks = {k: v for k, v in expected.items() if k not in ('role', 'sensor', 'num_bands', 'input_layout')}
    if (any(state.get(k) != v for k, v in checks.items())
            or state.get('precision') != 'fp32' or state.get('scheduler', {}).get('last_epoch') != case.updates
            or state.get('model_sha256') != identity['model_sha256']
            or state_hash(state['model_state']) != identity.get('state_hash')):
        raise ValueError('Teacher endpoint full-state/scheduler provenance differs')
    tensors = load_file(str(weights), device='cpu')
    if state_hash(tensors) != identity['state_hash'] or not all(bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise ValueError('Teacher endpoint weights are nonfinite or differ from full-state')
    from qg40.exposure import validate_counts
    validate_counts(state.get('exposure_counts'), data['splits']['train']['count'], case.updates, cfg['batch_size'])
    _validate_endpoint_progress(state, cfg, case, data['splits']['train']['count'])
    del state
    model, _ = build_model('P0', 112, (1, 2, 3), case.seed, role='T', num_bands=4)
    model.load_state_dict(tensors, strict=True)
    model.eval().requires_grad_(False)
    return case, cfg, data, identity, model, dict(config=cfg_path, data=data_path, candidate=folder)


def validate_reference(reference_id, server, root=ROOT, *, manifest_path=None, dataset_manifest=None):
    canonical = reference_path(reference_id, server, root)
    path = Path(manifest_path) if manifest_path else canonical
    if not path.is_absolute():
        path = Path(root) / path
    manifest = read_json(path)
    ref, case = REFERENCES[reference_id], teacher_for(reference_id)
    required = dict(schema='L100_LOCAL_REFERENCE_v2', campaign_id=CAMPAIGN_ID,
        reference_id=reference_id, owner_server=server, producer_server=server, server=server,
        teacher_run_id=case.run_id, teacher_seed=ref.teacher_seed, teacher_update=ref.updates,
        teacher_layout='P0', sensor='GF2', num_bands=4, max_pixel=1023,
        calibration_id=ref.calibration_id, calibration_n=3072, calibration_batch_size=16,
        LP_recipe=RECIPE, augmentation=AUGMENTATION, augmentation_sha256=object_sha(AUGMENTATION),
        architecture_forward=FORWARD_CONTRACT, source_identity=source_identity(root))
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError('Local reference owner/endpoint/configuration/source mismatch')
    paths = {key: resolved_path(manifest[key], root) for key in FILE_KEYS}
    for key, hash_key in FILE_KEYS.items():
        if sha256(paths[key]) != manifest.get(hash_key):
            raise ValueError('Local reference artifact checksum mismatch: ' + key)
    cfg = yaml.safe_load(paths['teacher_config'].read_text())
    if validate_config(cfg, require_bound=True) != case or object_sha(cfg) != manifest.get('teacher_config_sha256'):
        raise ValueError('Reference Teacher config/registered case mismatch')
    run = paths['teacher_config'].parent.parent
    endpoint_case, _cfg, data, identity, model, checked = teacher_endpoint(run, root, server)
    if (endpoint_case != case or paths['teacher_checkpoint'].resolve() != (checked['candidate'] / 'model.safetensors').resolve()
            or paths['teacher_training_state'].resolve() != (checked['candidate'] / 'training_state.pt').resolve()
            or paths['teacher_checkpoint_identity'].resolve() != (checked['candidate'] / 'identity.json').resolve()
            or paths['dataset_manifest_path'].resolve() != checked['data'].resolve()
            or identity['model_sha256'] != manifest['teacher_checkpoint_sha256']):
        raise ValueError('Reference artifacts are not this local endpoint')
    shapes = {key: list(value.shape) for key, value in model.state_dict().items()}
    if manifest.get('architecture_state_shapes') != shapes or manifest.get('architecture_state_shapes_sha256') != object_sha(shapes):
        raise ValueError('Reference architecture signature mismatch')
    del model
    local = dataset_manifest or camp(root, server) / 'dataset_manifest.json'
    local = read_json(resolved_path(local, root)) if isinstance(local, (str, Path)) else local
    if (data_signature(local) != data_signature(data) or manifest.get('data_sha256') != object_sha(data)
            or data.get('sensor') != 'GF2' or data.get('num_bands') != 4 or data.get('max_pixel') != 1023):
        raise ValueError('Reference does not match the exact local GF2 data')
    for item in local['splits'].values():
        for name, key in (('dataroot', 'sha256'), ('lpan_path', 'lpan_sha256')):
            if sha256(resolved_path(item[name], root)) != item[key]:
                raise ValueError('Actual local source/LP bytes differ')
    calibration = read_json(paths['calibration_path'])
    if (calibration.get('schema') != 'L100_CALIBRATION_v2' or calibration.get('synthetic_test') is not False
            or calibration.get('batch_size') != 16):
        raise ValueError('Actual nonsynthetic LOCAL-T calibration required')
    for key in ('tau_R', 'q_ref', 'q_shape', 'teacher_run_id', 'teacher_update', 'teacher_seed',
                'teacher_checkpoint_sha256', 'source_identity', 'calibration_indices_sha256'):
        if calibration.get(key) != manifest.get(key):
            raise ValueError('Calibration/reference identity mismatch: ' + key)
    with np.load(paths['q_cache_path'], allow_pickle=False) as cache:
        q, indices = cache['q'].copy(), cache['calibration_indices'].copy()
    if (q.dtype != np.float32 or q.shape != (data['splits']['train']['count'], 4)
            or list(q.shape) != manifest.get('q_shape') or not np.isfinite(q).all() or (q < 0).any()
            or not np.array_equal(indices, select_calibration_indices(len(q)))
            or object_sha(indices.tolist()) != manifest.get('calibration_indices_sha256')
            or not np.isfinite(manifest['tau_R']) or manifest['tau_R'] < 1e-6
            or not np.isfinite(manifest['q_ref']) or manifest['q_ref'] <= 0
            or float(np.median(q[indices])) != manifest['q_ref']):
        raise ValueError('Train3072/full-train four-view calibration invalid')
    validate_receipt(read_json(paths['q_cache_parity_path']), reference_identity=identity_for(manifest),
                     data_identity=object_sha(data_signature(data)), q=q)
    return manifest, q, cfg, paths


def load_reference(reference_id, server, root=ROOT, device='cuda', *, manifest_path=None,
                   dataset_manifest=None, deadline=None):
    check_deadline(deadline)
    manifest, q, cfg, paths = validate_reference(reference_id, server, root,
        manifest_path=manifest_path, dataset_manifest=dataset_manifest)
    model, _ = build_model('P0', 112, (1, 2, 3), cfg['seed'], role='T', num_bands=4)
    model.load_state_dict(load_file(str(paths['teacher_checkpoint']), device='cpu'), strict=True)
    model.to(device).eval().requires_grad_(False)
    from g20.data import build_dataset
    data = dataset_manifest or read_json(paths['dataset_manifest_path'])
    if isinstance(data, (str, Path)):
        data = read_json(resolved_path(data, root))
    verify_q_cache(model, build_dataset(data, 'train', root=root), q,
        reference_identity=identity_for(manifest), data_identity=object_sha(data_signature(data)),
        device=device, deadline=deadline)
    return model, manifest, q
