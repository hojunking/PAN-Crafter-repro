"""Sensor-local exact50K Teacher references; no cross-lane or legacy fallback."""
from pathlib import Path
import math

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from fh12.data import RECIPE, AUGMENTATION
from ablr2.model import build_model, state_hash
from ablr2.reference_parity import identity_for, verify_q_cache, validate_receipt
from qg40.calibration import select_calibration_indices


def data_signature(data):
    return {key: data.get(key) for key in ('sensor','num_bands','max_pixel','mtf_sensor',
        'band_order','recipe','augmentation')} | {'splits': {
        name: {key: item.get(key) for key in ('count','sha256','lpan_sha256',
            'lpan_canonical_sha256','sample_order_sha256','shapes','source_identity')}
        for name,item in data['splits'].items()}}


def reference_path(reference_id, server, root=None):
    from ablr2.common import ROOT, camp
    if not reference_id or '/' in reference_id or '\\' in reference_id or '..' in reference_id:
        raise ValueError('Registered plain local reference identifier required')
    return camp(root or ROOT, server)/'references'/reference_id/'reference_manifest.json'


def teacher_endpoint(teacher_run, root, server):
    from ablr2.common import read_json, read_config, resolved_path, sha256, object_sha, source_identity
    from ablr2.plan import validate_config
    run = resolved_path(teacher_run, root)
    cfg_path = run/'meta/config.resolved.yaml'
    cfg = read_config(cfg_path)
    case = validate_config(cfg, require_bound=True)
    field = cfg['ablr2']
    if case.role != 'T' or case.server_id != server or run.name != case.run_id:
        raise ValueError('Reference must be this lane registered Teacher')
    if resolved_path(cfg['work_dir'],root).resolve() != run.resolve():
        raise ValueError('Teacher work_dir identity differs')
    folder = run/'candidates/50000'
    identity = read_json(folder/'identity.json')
    data_path = resolved_path(field['dataset_manifest'],root)
    data = read_json(data_path)
    expected = dict(update=50000,full_state=True,role='T',sensor=case.sensor,
        num_bands=cfg['num_bands'],input_layout='P0',config_sha256=object_sha(cfg),
        data_sha256=object_sha(data),source_identity=source_identity(root),reference_sha256=None)
    if any(identity.get(k) != v for k,v in expected.items()):
        raise ValueError('Teacher endpoint/source/data identity differs')
    if (identity.get('model_sha256') != sha256(folder/'model.safetensors')
            or identity.get('training_state_sha256') != sha256(folder/'training_state.pt')):
        raise ValueError('Teacher exact endpoint checksum differs')
    state = torch.load(folder/'training_state.pt',map_location='cpu',weights_only=False)
    if (state.get('full_state') is not True or state.get('update') != 50000
            or state.get('precision') != 'fp32' or state.get('scheduler',{}).get('last_epoch') != 50000
            or any(state.get(k) != expected[k] for k in ('config_sha256','data_sha256','source_identity','reference_sha256'))
            or state.get('model_sha256') != identity['model_sha256']
            or state_hash(state['model_state']) != identity.get('state_hash')):
        raise ValueError('Teacher fullstate progress/provenance differs')
    from fh12.training import BatchStream
    from qg40.exposure import validate_counts
    stream = BatchStream(data['splits']['train']['count'],cfg['batch_size'],cfg['seed'])
    stream.load_state_dict(state['sampler'])
    if stream.epoch*stream.count+stream.cursor != 50000:
        raise ValueError('Teacher sampler endpoint differs')
    validate_counts(state.get('exposure_counts'),data['splits']['train']['count'],50000,cfg['batch_size'])
    groups=state['optimizer']['param_groups']
    if ([group.get('name') for group in groups] != ['U','A']
            or any(not math.isclose(float(group['lr']),0.,abs_tol=1e-14) for group in groups)
            or state['scheduler'].get('_last_lr') != [0.,0.]):
        raise ValueError('Teacher exact50K optimizer/scheduler mismatch')
    tensors=load_file(str(folder/'model.safetensors'),device='cpu')
    if state_hash(tensors) != identity['state_hash'] or not all(bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise ValueError('Teacher safetensors differ or are nonfinite')
    model,_=build_model(bands=cfg['num_bands'],seed=cfg['seed'],role='T')
    model.load_state_dict(tensors,strict=True)
    model.eval().requires_grad_(False)
    return case,cfg,data,identity,model,dict(config=cfg_path,data=data_path,candidate=folder)


def validate_reference(reference_id,server,root=None,*,manifest_path=None,dataset_manifest=None):
    from ablr2.common import ROOT, CAMPAIGN_ID, object_sha,read_json,resolved_path,sha256,source_identity
    root=Path(root or ROOT).resolve()
    path=resolved_path(manifest_path,root) if manifest_path else reference_path(reference_id,server,root)
    manifest=read_json(path)
    expected=dict(schema='ABLR2_REFERENCE_v1',campaign_id=CAMPAIGN_ID,reference_id=reference_id,
        owner_server=server,producer_server=server,teacher_update=50000,teacher_layout='P0',
        calibration_n=3072,calibration_batch_size=16,LP_recipe=RECIPE,augmentation=AUGMENTATION,
        source_identity=source_identity(root))
    if any(manifest.get(k) != v for k,v in expected.items()):
        raise ValueError('Local reference owner/source/endpoint mismatch')
    names={'teacher_checkpoint':'teacher_checkpoint_sha256','teacher_training_state':'teacher_training_state_sha256',
        'teacher_config':'teacher_config_file_sha256','teacher_checkpoint_identity':'teacher_checkpoint_identity_sha256',
        'calibration_path':'calibration_sha256','q_cache_path':'q_cache_sha256',
        'dataset_manifest_path':'dataset_manifest_sha256','q_cache_parity_path':'q_cache_parity_sha256'}
    paths={key:resolved_path(manifest[key],root) for key in names}
    for key,digest in names.items():
        if sha256(paths[key]) != manifest[digest]:
            raise ValueError('Reference artifact checksum changed: '+key)
    case,cfg,data,identity,model,checked=teacher_endpoint(paths['teacher_config'].parent.parent,root,server)
    if (case.reference_id != reference_id or manifest['teacher_run_id'] != case.run_id
            or manifest['teacher_seed'] != cfg['seed'] or manifest['sensor'] != case.sensor
            or manifest['num_bands'] != cfg['num_bands'] or manifest['max_pixel'] != cfg['max_pixel']
            or manifest['teacher_kind'] != cfg['ablr2']['teacher_kind']
            or manifest['consistency_weight'] != cfg['ablr2']['consistency_weight']
            or manifest['teacher_config_sha256'] != object_sha(cfg)
            or paths['teacher_checkpoint'].resolve() != (checked['candidate']/'model.safetensors').resolve()
            or paths['teacher_training_state'].resolve() != (checked['candidate']/'training_state.pt').resolve()
            or paths['teacher_checkpoint_identity'].resolve() != (checked['candidate']/'identity.json').resolve()
            or paths['dataset_manifest_path'].resolve() != checked['data'].resolve()):
        raise ValueError('Reference not bound to this exact Teacher')
    del model
    local = dataset_manifest if dataset_manifest is not None else data
    if isinstance(local,(str,Path)):
        local=read_json(resolved_path(local,root))
    if data_signature(local) != data_signature(data) or manifest['data_sha256'] != object_sha(data):
        raise ValueError('Teacher/Student data identities differ')
    for item in local['splits'].values():
        for name,digest in (('dataroot','sha256'),('lpan_path','lpan_sha256')):
            if sha256(resolved_path(item[name],root)) != item[digest]:
                raise ValueError('Actual data/LP changed')
    cal=read_json(paths['calibration_path'])
    if cal.get('schema') != 'ABLR2_CALIBRATION_v1' or cal.get('synthetic_test') is not False or cal.get('batch_size') != 16:
        raise ValueError('Production calibration required')
    for key in ('tau_R','q_ref','q_shape','s_bar','s_bar_population','calibration_indices_sha256',
                'teacher_run_id','teacher_checkpoint_sha256','source_identity'):
        if cal.get(key) != manifest.get(key):
            raise ValueError('Calibration/reference mismatch: '+key)
    with np.load(paths['q_cache_path'],allow_pickle=False) as cache:
        q,indices=cache['q'].copy(),cache['calibration_indices'].copy()
    if (q.dtype != np.float32 or q.shape != (data['splits']['train']['count'],4)
            or list(q.shape) != manifest['q_shape'] or not np.isfinite(q).all() or (q<0).any()
            or not np.array_equal(indices,select_calibration_indices(len(q)))
            or object_sha(indices.tolist()) != manifest['calibration_indices_sha256']
            or not math.isfinite(manifest['tau_R']) or manifest['tau_R'] < 1e-6
            or not math.isfinite(manifest['q_ref']) or manifest['q_ref'] <= 0
            or float(np.median(q[indices])) != manifest['q_ref']):
        raise ValueError('Train3072/four-view q calibration invalid')
    actual_mean=float(np.mean(manifest['q_ref']/(manifest['q_ref']+q.astype(np.float64))))
    if (manifest['s_bar'] != actual_mean or manifest['s_bar_population'] != dict(base_count=len(q),
            views=[0,1,2,3],n_values=4*len(q),q_cache_sha256=manifest['q_cache_sha256'])
            or manifest['s_bar_sha256'] != object_sha(dict(value=manifest['s_bar'],population=manifest['s_bar_population']))):
        raise ValueError('C15 mean is not the measured full-train-view reliability')
    validate_receipt(read_json(paths['q_cache_parity_path']),reference_identity=identity_for(manifest),
                     data_identity=object_sha(data_signature(data)),q=q)
    return manifest,q,cfg,paths


def load_reference(reference_id,server,root=None,device='cuda',*,manifest_path=None,dataset_manifest=None,deadline=None):
    from ablr2.common import ROOT,check_deadline,read_json,object_sha
    from ablr2.data import build_dataset
    root=root or ROOT
    check_deadline(deadline)
    manifest,q,cfg,paths=validate_reference(reference_id,server,root,manifest_path=manifest_path,dataset_manifest=dataset_manifest)
    model,_=build_model(bands=cfg['num_bands'],seed=cfg['seed'],role='T')
    model.load_state_dict(load_file(str(paths['teacher_checkpoint']),device='cpu'),strict=True)
    model.to(device).eval().requires_grad_(False)
    data=dataset_manifest if dataset_manifest is not None else read_json(paths['dataset_manifest_path'])
    if isinstance(data,(str,Path)):
        data=read_json(data)
    verify_q_cache(model,build_dataset(data,'train',root=root),q,
        reference_identity=identity_for(manifest),data_identity=object_sha(data_signature(data)),device=device,deadline=deadline)
    return model,manifest,q


def load_endpoint_only(teacher_run,server,root,device='cpu'):
    """C03/C04 endpoint-only path: never opens calibration, tau, q or q cache."""
    case,cfg,data,identity,model,paths=teacher_endpoint(teacher_run,root,server)
    model.to(device).eval().requires_grad_(False)
    return model,dict(schema='ABLR2_ENDPOINT_REFERENCE_v1',reference_id=case.reference_id,
        teacher_run_id=case.run_id,teacher_seed=cfg['seed'],teacher_update=50000,
        teacher_kind=cfg['ablr2']['teacher_kind'],sensor=case.sensor,owner_server=server,
        teacher_checkpoint_sha256=identity['model_sha256'],source_identity=identity['source_identity'],
        data_sha256=identity['data_sha256'])
