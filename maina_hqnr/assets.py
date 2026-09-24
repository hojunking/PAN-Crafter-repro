"""Exact original F1 assets with explicit path-only relocation; never regenerate."""
from copy import deepcopy
import os
from pathlib import Path

from maina_hqnr.common import (ROOT, camp, immutable_json, object_sha,
    read_json, read_config, sha256, verify_server)
from maina_hqnr.plan import (CAMPAIGN_ID, TEACHER_RUN, TEACHER_SHA, BRIDGE_SHA,
    CALIBRATION_SHA, Q_CACHE_SHA, DATA_SHA, EVALUATOR_SHA, INITIAL_A_SHA, Q_REF, TAU_R,
    verify_sources)

ORIGINAL_ROOT = Path('/home/knuvi/Desktop/song/PAN-Crafter')
BRIDGE_PATH = 'work_dir/_fh20r1/s1/imported_refs/F1/bridge_manifest.json'
COUNTS = dict(train=9714, val=1080, rr=20, fr=20)
SIDES = dict(train=64, val=64, rr=256, fr=512)
LP_SHAS = {
    'train':'53ea13dd0a0de8c4f69cbbcf6fa6f7d9b70eb0362af22ff4d55f16f978a255f3',
    'val':'5410316e91f801ceb7684174c7675ec1f501aaddcc6cb2b3888735d50856ad0e',
    'rr':'2b81b4c5bf27137a0030840304533d4bc40fb16da586414627123348fe098371',
    'fr':'83e969b1f25234f3908ca2dfeb2fc725fa5eb0fca401aa279b6b9093e927db41'}
BAND_ORDER = ['Coastal','Blue','Green','Yellow','Red','RedEdge','NIR1','NIR2']
ORIGINAL_WALD_SHA = 'c7d076395eec052dd4ed391e6f51e5fa79661a3fc283816a63ccdb100ff5bdbf'


def _path(value, root):
    path = Path(value)
    if '..' in path.parts:
        raise ValueError('Asset path must not contain parent traversal')
    if not path.is_absolute(): return Path(root)/path
    # Only the documented repository prefix is portable by default. External
    # source paths require explicit overrides on hosts where they do not exist.
    try: return Path(root)/path.relative_to(ORIGINAL_ROOT)
    except ValueError: return path


def _checked_file(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('PAUSED_ASSET_MISSING: '+str(path))
    if not isinstance(expected,str) or len(expected)!=64 or sha256(path)!=expected:
        raise ValueError('PAUSED_ASSET_MISMATCH: '+str(path))
    return str(path.resolve())


def _validate_overrides(overrides):
    if not isinstance(overrides,dict) or set(overrides)-{'bridge_path','artifacts','data'}:
        raise ValueError('Overrides may relocate only bridge_path, artifacts and data')
    if not isinstance(overrides.get('artifacts',{}),dict) or not isinstance(overrides.get('data',{}),dict):
        raise ValueError('Path overrides must be dictionaries')
    if set(overrides.get('data',{}))-set(COUNTS):
        raise ValueError('Unknown data split')
    for split,fields in overrides.get('data',{}).items():
        if not isinstance(fields,dict) or set(fields)-{'dataroot','lpan_path'}:
            raise ValueError('Data override changes content instead of paths: '+split)
    return overrides


def _validate_data(canonical, lp_manifest, root, overrides):
    import h5py
    from fh12.data import RECIPE, AUGMENTATION
    if (canonical.get('schema')!='FH12_DATA_v1' or canonical.get('sensor')!='WV3'
            or canonical.get('max_pixel')!=2047 or canonical.get('server')!='s1'
            or canonical.get('recipe')!=RECIPE or canonical.get('augmentation')!=AUGMENTATION
            or lp_manifest.get('recipe')!=RECIPE):
        raise ValueError('Canonical WV3/FH12 frequency or augmentation protocol changed')
    resolved = deepcopy(canonical)
    for split,count in COUNTS.items():
        item = canonical['splits'][split]; side=SIDES[split]
        if item.get('count')!=count or item.get('lpan_sha256')!=LP_SHAS[split]:
            raise ValueError('Canonical split count/LP identity differs: '+split)
        paths = overrides.get(split,{})
        native = _checked_file(_path(paths.get('dataroot',item['dataroot']),root),item['sha256'])
        lp = _checked_file(_path(paths.get('lpan_path',item['lpan_path']),root),LP_SHAS[split])
        with h5py.File(native,'r') as data:
            expected = dict(pan=(count,1,side,side),ms=(count,8,side//4,side//4),lms=(count,8,side,side))
            if split!='fr': expected['gt']=(count,8,side,side)
            if set(data)!=set(expected) or any(tuple(data[k].shape)!=shape for k,shape in expected.items()):
                raise ValueError('Native WV3 full-population NCHW geometry differs: '+split)
        with h5py.File(lp,'r') as data:
            if (set(data)!={'lpan'} or data['lpan'].shape!=(count,1,side//4,side//4)
                    or data.attrs.get('source_sha256')!=item['sha256']
                    or data.attrs.get('sample_order_sha256')!=item['sample_order_sha256']
                    or data.attrs.get('recipe_sha256')!=object_sha(RECIPE)):
                raise ValueError('LP source/phase/sample-order differs: '+split)
        if any(lp_manifest['splits'][split][k]!=item[k]
               for k in ('sha256','lpan_sha256','sample_order_sha256')):
            raise ValueError('Canonical LP and data manifest disagree: '+split)
        resolved['splits'][split].update(dataroot=native,lpan_path=lp)
    return resolved


def _q_values(origin, path):
    import numpy as np
    with np.load(path,allow_pickle=False) as cache:
        q=cache['q'].copy(); indices=cache['calibration_indices'].copy()
    if q.shape!=(9714,4) or not np.isfinite(q).all() or (q<0).any():
        raise ValueError('F1 q-cache must cover all 9714 patches and four native train views')
    if (indices.shape!=(3072,) or indices.dtype.kind not in 'iu'
            or len(np.unique(indices))!=3072 or indices.min()<0 or indices.max()>=9714
            or object_sha(indices.tolist())!=origin['calibration_indices_sha256']):
        raise ValueError('F1 calibration indices changed')
    if float(np.median(q[indices]))!=Q_REF or origin.get('q_ref')!=Q_REF or origin.get('tau_R')!=TAU_R:
        raise ValueError('F1 q_ref/tau_R differs; recalibration is prohibited')
    return q


def verify_assets(server, root=ROOT, overrides=None, persist=True):
    """Hash canonical bytes, then build a separate, immutable path-only binding.

    This function never edits assets or invokes teacher_for(server). A full F1
    portable copy can live anywhere by listing its explicit local artifact paths.
    """
    verify_server(server); root=Path(root).resolve()
    overrides=_validate_overrides(deepcopy(overrides or {}))
    verify_sources(root)
    dlpan=Path(os.environ.get('PANCRAFTER_DLPAN',str(root.parent/'DLPan-Toolbox')))
    wald_path=_checked_file(dlpan/'01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py',
                           ORIGINAL_WALD_SHA)
    bridge_path=_checked_file(_path(overrides.get('bridge_path',BRIDGE_PATH),root),BRIDGE_SHA)
    bridge=read_json(bridge_path)
    if (bridge.get('schema')!='FH20R1_REFERENCE_BRIDGE_v1' or bridge.get('status')!='PASS'
            or bridge.get('complete') is not True or bridge.get('alias')!='F1' or bridge.get('server')!='s1'
            or bridge.get('teacher_checkpoint_sha256')!=TEACHER_SHA
            or bridge.get('consumer_source_identity',{}).get('content_sha256')!=EVALUATOR_SHA
            or bridge.get('parity',{}).get('status')!='PASS'):
        raise ValueError('Original F1 bridge identity/parity differs')
    expected=bridge['resolved_artifact_sha256']
    if set(overrides.get('artifacts',{}))-set(expected):
        raise ValueError('Unknown F1 artifact relocation key')
    resolved={key:_checked_file(_path(overrides.get('artifacts',{}).get(key,
        bridge['resolved_artifacts'][key]),root),digest) for key,digest in expected.items()}
    origin=read_json(resolved['origin_manifest'])
    if (origin!=bridge['origin_reference'] or origin.get('schema')!='FH12_REFERENCE_v1'
            or origin.get('teacher_run_id')!=TEACHER_RUN or origin.get('teacher_layout')!='P0'
            or origin.get('teacher_update')!=50000 or origin.get('server')!='s1'
            or origin.get('teacher_checkpoint_sha256')!=TEACHER_SHA
            or origin.get('calibration_sha256')!=CALIBRATION_SHA
            or origin.get('q_cache_sha256')!=Q_CACHE_SHA
            or origin.get('q_ref')!=Q_REF or origin.get('tau_R')!=TAU_R):
        raise ValueError('F1 reference/calibration/q identity differs')
    pairs=dict(teacher_checkpoint='teacher_checkpoint_sha256',teacher_training_state='teacher_training_state_sha256',
        teacher_config='teacher_config_file_sha256',teacher_checkpoint_identity='teacher_checkpoint_identity_sha256',
        calibration_path='calibration_sha256',q_cache_path='q_cache_sha256',
        dataset_manifest_path='dataset_manifest_sha256',lpan_manifest_path='lpan_manifest_sha256')
    if any(origin[hash_key]!=expected[key] for key,hash_key in pairs.items()):
        raise ValueError('Bridge file SHA and original manifest SHA disagree')
    cfg=read_config(resolved['teacher_config']); ident=read_json(resolved['teacher_checkpoint_identity'])
    if (cfg!=bridge['origin_config'] or object_sha(cfg)!=origin['teacher_config_sha256']
            or cfg['seed']!=71001 or cfg['fh12']['role']!='T' or cfg['fh12']['input_layout']!='P0'
            or cfg['model_args']['hidden_size']!=112 or cfg['model_args']['depth']!=[1,2,3]
            or ident.get('update')!=50000 or ident.get('model_sha256')!=TEACHER_SHA
            or ident.get('config_sha256')!=origin['teacher_config_sha256']
            or ident.get('training_state_sha256')!=origin['teacher_training_state_sha256']
            or ident.get('source_identity')!=origin['source_identity']):
        raise ValueError('F1 exact50K checkpoint/config provenance differs')
    calibration=read_json(resolved['calibration_path'])
    for key in ('source_identity','tau_R','q_ref','teacher_layout','teacher_update',
                'teacher_checkpoint_sha256','calibration_indices_sha256','q_shape'):
        if calibration[key]!=origin[key]: raise ValueError('Calibration origin differs: '+key)
    _q_values(origin,resolved['q_cache_path'])
    canonical=read_json(resolved['dataset_manifest_path'])
    if object_sha(canonical)!=DATA_SHA:
        raise ValueError('Canonical data JSON object hash differs (not a file-SHA comparison)')
    data=_validate_data(canonical,read_json(resolved['lpan_manifest_path']),root,overrides.get('data',{}))
    for key,ref_key in (('sha256','train_sha256'),('lpan_sha256','train_lpan_sha256'),
                        ('sample_order_sha256','train_sample_order_sha256')):
        if canonical['splits']['train'][key]!=origin[ref_key]:
            raise ValueError('F1 calibration and train-view dataset differ')
    from safetensors.torch import load_file
    from fh12.model import state_hash
    state=load_file(resolved['teacher_checkpoint'],device='cpu')
    aligner={k[len('aligner.'):]:v for k,v in state.items() if k.startswith('aligner.')}
    if not aligner or state_hash(aligner)!=INITIAL_A_SHA:
        raise ValueError('F1 initial A tensor-report hash differs')
    evaluation=dict(rr_n_scenes=20,fr_n_scenes=20,reference='RAW_ORIGINAL_PAN',
        paper_identity_status='PAPERSET_IDENTITY_UNVERIFIED',
        warning='Native H5 hashes/order verified; paper-scene correspondence not independently reverified',
        band_order=BAND_ORDER,scene_order='original H5 index 0..19; no filtering',
        original_evaluator_sha=EVALUATOR_SHA)
    result=dict(schema='MAINA_HQNR_ASSET_BINDING_v1',campaign_id=CAMPAIGN_ID,server=server,root=str(root),
        overrides=overrides,canonical_bridge_path=bridge_path,canonical_bridge_sha256=BRIDGE_SHA,
        canonical_data_manifest_path=resolved['dataset_manifest_path'],canonical_data_sha256=DATA_SHA,
        original_evaluator_sha256=EVALUATOR_SHA,original_reference_server='s1',
        original_external_mtf=dict(path=wald_path,sha256=ORIGINAL_WALD_SHA,
            provenance='Verified local original evaluator dependency; separate from historical 42-file manifest'),
        teacher=dict(checkpoint=resolved['teacher_checkpoint'],config=resolved['teacher_config'],
            identity=resolved['teacher_checkpoint_identity'],sha256=TEACHER_SHA,alias='F1',
            step=50000,layout='P0',initial_A_sha256=INITIAL_A_SHA),
        resolved_artifacts=resolved,resolved_artifact_sha256=deepcopy(expected),
        origin_reference=origin,bridge=bridge,dataset_manifest=data,
        canonical_dataset_manifest=canonical,q_ref=Q_REF,tau_R=TAU_R,evaluation=evaluation,
        data={split:dict(path=item['dataroot'],file_sha256=item['sha256'],count=item['count'],
            lpan_path=item['lpan_path'],lpan_sha256=item['lpan_sha256']) for split,item in data['splits'].items()})
    if persist: immutable_json(camp(root,server)/'runtime_bindings.json',result)
    return result


def validate_bindings(bindings, root=ROOT, server=None, rehash=True, **kwargs):
    bindings=read_json(bindings) if isinstance(bindings,(str,Path)) else bindings
    server=server or bindings.get('server'); verify_server(server)
    if (bindings.get('schema')!='MAINA_HQNR_ASSET_BINDING_v1' or bindings.get('campaign_id')!=CAMPAIGN_ID
            or bindings.get('server')!=server or bindings.get('canonical_bridge_sha256')!=BRIDGE_SHA
            or bindings.get('canonical_data_sha256')!=DATA_SHA or bindings.get('q_ref')!=Q_REF
            or bindings.get('tau_R')!=TAU_R or bindings.get('teacher',{}).get('sha256')!=TEACHER_SHA):
        raise ValueError('Invalid fixed F1 runtime binding')
    if rehash and verify_assets(server,root,bindings.get('overrides'),persist=False)!=bindings:
        raise ValueError('Runtime assets changed after immutable binding')
    return bindings


def load_training_reference(bindings, device='cuda'):
    bindings=validate_bindings(bindings,root=bindings.get('root',ROOT),rehash=False)
    paths=bindings['resolved_artifacts']
    for key in ('teacher_checkpoint','q_cache_path','calibration_path','teacher_config'):
        _checked_file(paths[key],bindings['resolved_artifact_sha256'][key])
    _checked_file(bindings['canonical_bridge_path'],BRIDGE_SHA)
    from fh20r1.references import _load_frozen
    teacher=_load_frozen(read_config(paths['teacher_config']),paths['teacher_checkpoint'],device)
    q=_q_values(bindings['origin_reference'],paths['q_cache_path'])
    return teacher,deepcopy(bindings['origin_reference']),q,deepcopy(bindings['bridge'])


def load_evaluation_datasets(bindings, root=None):
    from fh12.data import build_dataset
    bindings=validate_bindings(bindings,root=root or bindings.get('root',ROOT),rehash=False)
    return {split:build_dataset(bindings['dataset_manifest'],split,augment=False) for split in ('rr','fr')}


# Familiar name for deployment callers, without historical server-dependent loaders.
def prepare_bindings(root=ROOT, server='s4', overrides=None):
    return verify_assets(server,root,overrides,persist=True)
