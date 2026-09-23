"""Strict, local G23 T0/cue/data binding. No replacement or regenerated assets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
BASE_CONFIG='config/PAKD50_QRC24_S4_G23_W104_D121_WV3_T0_S1234_FRESH50_v2.yaml'
BASE_BLOB='070d5c934117d8577d699ae1be11d5e8d080b688'
TEACHER_SHA='16b5cf78614be121122d3cb28c2e361b9d557b63e974e7083503ef23bdc37b32'
Q0=.3276133416220546
TAU0=.012463942170143127
CAMPAIGN_ID='PANDA_G23_SENS_WV3_S45_20260923_v1'
BAND_ORDER=['Coastal','Blue','Green','Yellow','Red','RedEdge','NIR1','NIR2']
SPLIT_KEYS=dict(train='train_feeder_args',val='val_feeder_args',rr='test_reduced_feeder_args',fr='test_full_feeder_args')
COUNTS=dict(train=9714,val=1080,rr=20,fr=20)
# Measured read-only from the exact reference config on 2026-09-23. Binding
# another path is allowed only for these same original bytes, never a new set.
NATIVE_FILE_SHAS={
    'train':'f9e7ec381390ff36ddffd0714df7af07a3c5f456078b219a25f3e18372228f82',
    'train_lp':'3ce697e7d57b04a02175b2f393fb762c897ed47c1a1e11e7f1a27018be5a5628',
    'val':'ba883e57408d1f3c2263cc2818a9fbbf2dbe8e62554c531288f9e8bf9f59dc87',
    'val_lp':'a639d60f2f7b80c8c37aa565b5a0b8f29e0274cf16094d6a6aab5e9e70434a6d',
    'rr':'00db0d62f63693410935208e287aea21a736d11840d92eedd2d093c99be5314a',
    'rr_lp':'b71b4e383ef6f074b74be9152df1093120da9ba1f36a4ce7549e0906d7792550',
    'fr':'8fe579a59594ddf42bce19743a5b65361feab22cbd0c8f433d3173f29b87a4d6',
    'fr_lp':'c7e7cd94244d311246290a0b4ed88e3592561a61fc67a902e202d4572f981f60'}
CUE_NPZ_SHA='b83cc5d1a25e8766d1813c9d19d956efc24e0f152553e53c4894fa18dcdf563e'
CALIBRATION_SHA='dc76a3b02e16fa3d1b9fc8d91b24bb0b420db118d56af5d416e540f82f195ea8'


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha256(path,stopcheck=None):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):
            _stop(stopcheck);digest.update(part)
    return digest.hexdigest()


def _stop(stopcheck):
    if stopcheck and stopcheck():raise InterruptedError('Safe asset verification pause')


def _read(path):
    with Path(path).open() as stream:return json.load(stream)


def _path(value,root):
    path=Path(value)
    if not path.is_absolute():return Path(root)/path
    # The reference config itself records this exact original repository root.
    prefix=Path('/home/knuvi/Desktop/song/PAN-Crafter')
    try:return Path(root)/path.relative_to(prefix)
    except ValueError:return path


def reference_config(root=ROOT):
    path=Path(root)/BASE_CONFIG;payload=path.read_bytes()
    if hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()!=BASE_BLOB:
        raise ValueError('BLOCKED_ASSET_MISMATCH: exact G23 reference config blob changed')
    cfg=yaml.safe_load(payload)
    fa=cfg['train_feeder_args']
    if (cfg['feeder']!='feeders.feeder.PanFeeder' or cfg['num_bands']!=8 or cfg['max_pixel']!=2047.
            or any(fa.get(k)!=v for k,v in dict(crop=False,hflip=True,vflip=True,rot=True,return_meta=True).items())):
        raise ValueError('Reference legacy feeder contract changed')
    return cfg


def canonical_calibration(root=ROOT):
    """Prefer the distributable original pack; legacy path is byte-identical only."""
    for relative in ('assets/pakd50/calibration_resolved.json','work_dir/_pakd50/calibration_resolved.json'):
        path=Path(root)/relative
        if path.is_file():
            if sha256(path)!=CALIBRATION_SHA:
                raise ValueError('BLOCKED_ASSET_MISMATCH: original T0 calibration bytes changed')
            return path
    raise FileNotFoundError('BLOCKED_ASSET_MISSING: fixed T0 calibration manifest')


def h5_identity(path,split,*,lp=False,stopcheck=None):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError('BLOCKED_ASSET_MISSING: '+str(path))
    arrays={};count=None;combined=hashlib.sha256()
    with h5py.File(path,'r') as h5:
        expected={'lpan'} if lp else ({'lms','ms','pan'}|({'gt'} if split!='fr' else set()))
        if set(h5)!=expected:raise ValueError('Native H5 keys differ: '+str(path))
        size=64 if split in ('train','val') else 256 if split=='rr' else 512
        for name in sorted(expected):
            ds=h5[name];n=ds.shape[0] if ds.ndim==4 else None
            channels=1 if name in ('pan','lpan') else 8
            spatial=size//4 if name in ('ms','lpan') else size
            if ds.shape!=(n,channels,spatial,spatial) or ds.dtype.kind not in 'fiu':
                raise ValueError('Native WV3 NCHW ratio4 geometry/dtype differs: '+name)
            if count is not None and n!=count:raise ValueError('Native array scene counts differ')
            count=n;digest=hashlib.sha256();minimum=float('inf');maximum=float('-inf')
            for start in range(0,n,32):
                _stop(stopcheck);value=np.ascontiguousarray(ds[start:start+32])
                if not np.isfinite(value).all():raise ValueError('Nonfinite native asset '+name)
                minimum=min(minimum,float(value.min()));maximum=max(maximum,float(value.max()))
                digest.update(value.tobytes())
            # Bicubic LMS and legacy LP may overshoot the sensor range. Never clip them.
            if name in ('gt','ms','pan') and (minimum<0 or maximum>2047):
                raise ValueError('WV3 native DN units/range differ: '+name)
            arrays[name]=dict(shape=list(ds.shape),dtype=str(ds.dtype),sha256=digest.hexdigest(),
                minimum=minimum,maximum=maximum)
            combined.update(name.encode());combined.update(str(ds.dtype).encode())
            combined.update(json.dumps(list(ds.shape)).encode());combined.update(bytes.fromhex(digest.hexdigest()))
    if count!=COUNTS[split]:raise ValueError('Reference-config native population changed: '+split)
    return dict(path=str(path.resolve()),file_sha256=sha256(path,stopcheck),array_content_sha256=combined.hexdigest(),
        count=count,arrays=arrays,layout='NCHW',units='native DN',max_pixel=2047,
        band_order=['PAN'] if lp else BAND_ORDER)


def _teacher(run,stopcheck=None):
    from safetensors.torch import load_file
    from kdv.teacher_assets import extract_aligner_sd,tensors_sha
    run=Path(run);weight=run/'best_hqnr/model.safetensors';pack=_read(run/'teacher_assets.json')
    if sha256(weight,stopcheck)!=TEACHER_SHA or pack.get('checkpoint_sha256')!=TEACHER_SHA:
        raise ValueError('BLOCKED_ASSET_MISMATCH: fixed T0 checkpoint differs')
    for name,digest in pack['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe T0 package path')
        if sha256(run/relative,stopcheck)!=digest:raise ValueError('T0 package provenance changed: '+name)
    tag=_read(run/'best_hqnr_meta.json');cfg=yaml.safe_load((run/'meta/config.yaml').read_text())
    if (tag.get('step')!=24240 or pack.get('selected_update')!=24240 or pack.get('aligner_view_margin')!=4
            or cfg['model_args']['hidden_size']!=112 or cfg['model_args']['depth']!=[1,2,3]):
        raise ValueError('T0 exact step/architecture/margin differs')
    state=load_file(str(weight));aligner=extract_aligner_sd(state)
    if tensors_sha(state)!=pack['tensors_sha256_16'] or tensors_sha(aligner)!=pack['aligner_tensors_sha256_16']:
        raise ValueError('T0 tensor identity differs')
    h=hashlib.sha256()
    for name in sorted(aligner):h.update(name.encode());h.update(aligner[name].numpy().tobytes())
    return dict(path=str(weight.resolve()),run_path=str(run.resolve()),checkpoint_sha256_expected=TEACHER_SHA,
        checkpoint_sha256=TEACHER_SHA,aligner_component_sha256=h.hexdigest(),
        aligner_state_hash=tensors_sha(aligner),tensors_sha256_16=tensors_sha(state),step=24240,view_margin_hr=4,
        config_path=str((run/'meta/config.yaml').resolve()),config_sha256=sha256(run/'meta/config.yaml'),
        package_path=str((run/'teacher_assets.json').resolve()),package_sha256=sha256(run/'teacher_assets.json'))


def cue_identity(json_path,npz_path,teacher,train,train_lp,stopcheck=None):
    from kdv.edge_gate import check_asset,NORMALIZATION
    man=_read(json_path)
    if sha256(npz_path,stopcheck)!=man['npz_sha256']:raise ValueError('Cue NPZ SHA differs')
    with np.load(npz_path,allow_pickle=False) as packed:z={k:packed[k] for k in packed.files}
    bad=check_asset(man,z)
    if bad:raise ValueError('Cue internal contract differs: '+'; '.join(bad))
    idx=z['index'].astype(np.int64);rot=z['rot'].astype(np.int64);q=z['q']
    n=train['count'];pairs=idx*4+rot
    if (q.dtype!=np.float64 or not np.isfinite(q).all() or np.any(q<0)
            or not np.array_equal(np.sort(pairs),np.arange(n*4))):
        raise ValueError('Cue raw q nonnegative/finite/dtype or full index/actual rotation coverage differs')
    expected_teacher=dict(file_sha256=TEACHER_SHA,aligner_state_hash=teacher['aligner_state_hash'],
        aligner_view_margin=4,selected_step=24240,tensors_sha256_16=teacher['tensors_sha256_16'])
    if any(man['teacher'].get(k)!=v for k,v in expected_teacher.items()):raise ValueError('Cue belongs to another Teacher/A/step')
    if (man['dataset']['train_sha256']!=train['file_sha256']
            or man['dataset']['train_pan_sha256']!=train_lp['file_sha256']
            or man['dataset']['normalization']!=NORMALIZATION or man['dataset']['n_train']!=n
            or man['theta_q']!=Q0 or np.median(q[z['calib_mask'].astype(bool)])!=Q0
            or man['calibration']['n_base']!=3072 or man['calibration']['seed']!=1234
            or any(man['feeder_contract'].get(k)!=v for k,v in dict(hflip=True,vflip=True,rot=True,crop=False).items())):
        raise ValueError('Cue native data/LP/calibration/legacy feeder provenance differs')
    raw_sha=hashlib.sha256(np.ascontiguousarray(q,dtype=np.float64).tobytes()).hexdigest()
    return dict(json_path=str(Path(json_path).resolve()),json_sha256=sha256(json_path),npz_path=str(Path(npz_path).resolve()),
        npz_sha256=sha256(npz_path),raw_q_sha256=raw_sha,asset_id=man['asset_id'],teacher_identity_verified=True,
        n_views=len(q),q_ref_expected=Q0,calibration_index_sha256_16=man['calibration']['index_sha256_16'])


def calibration_identity(path,teacher,cue,train):
    doc=_read(path);src=doc.get('tau_source',{});patch=doc.get('patch_manifest',{})
    if (doc.get('tau_R')!=TAU0 or doc.get('seed')!=1234 or doc.get('max_train_patches')!=3072
            or src.get('file_sha256')!=TEACHER_SHA or src.get('selected_update')!=24240
            or src.get('tensors_sha256_16')!=teacher['tensors_sha256_16']
            or patch.get('dataset_len')!=train['count'] or patch.get('seed')!=1234
            or patch.get('n_patches')!=3072 or patch.get('augmentation')!='none'
            or patch.get('dataroot')!='/home/knuvi/Desktop/song/PAN-Crafter/data/PanCollection/WV3/train_wv3.h5'
            or doc.get('geometric_augmentation') is not False or src.get('tag')!='best_hqnr'
            or doc.get('patch_manifest_sha256')!=hashlib.sha256(json.dumps(patch,sort_keys=True,default=str).encode()).hexdigest()
            or patch.get('index_sha256_16')!=cue['calibration_index_sha256_16']):
        raise ValueError('BLOCKED_ASSET_MISMATCH: tau calibration is not fixed T0/native train3072')
    return dict(source_manifest_path=str(Path(path).resolve()),source_manifest_sha256=sha256(path),
        q_ref_expected=Q0,tau_R_expected=TAU0,calibration_seed=1234,n_patches=3072,
        q_source='cue train3072 x four actual rotation states median',tau_source='T0 native unaugmented train3072',
        q_index_sha256_16=cue['calibration_index_sha256_16'])


def resolve_bindings(root=ROOT,server='s4',overrides=None,stopcheck=None):
    """Read-only builder. Overrides relocate exact assets, never replace identity."""
    if server not in ('s4','s5'):raise ValueError('G23 SENS is s4/s5 only')
    root=Path(root);cfg=reference_config(root);overrides=deepcopy(overrides or {})
    allowed={'data','teacher','cue','calibration'}
    if set(overrides)-allowed:raise ValueError('Unknown asset binding override')
    for key,fields in dict(data=set(NATIVE_FILE_SHAS),teacher={'run_path'},cue={'json_path','npz_path'},
                           calibration={'source_manifest_path'}).items():
        if set(overrides.get(key,{}))-fields:raise ValueError('Unknown '+key+' binding override')
    data={}
    for split,key in SPLIT_KEYS.items():
        default=_path(cfg[key]['dataroot'],root)
        for label,path in ((split,default),(split+'_lp',Path(str(default).replace('.h5','_pan.h5')))):
            override=overrides.get('data',{}).get(label,{})
            if set(override)-{'path','file_sha256'}:raise ValueError('Unsupported data override fields')
            resolved=Path(override.get('path',path))
            if not resolved.is_absolute():resolved=root/resolved
            data[label]=h5_identity(resolved,split,lp=label.endswith('_lp'),stopcheck=stopcheck)
            if data[label]['file_sha256']!=NATIVE_FILE_SHAS[label]:
                raise ValueError('BLOCKED_ASSET_MISMATCH: original reference '+label+' bytes differ')
            if resolved.resolve()!=path.resolve() and not override.get('file_sha256'):
                raise ValueError('Relocated data requires explicit expected SHA; no guessed replacement')
            if override.get('file_sha256') and data[label]['file_sha256']!=override['file_sha256']:
                raise ValueError('Explicit data SHA mismatch')
    teacher_run=overrides.get('teacher',{}).get('run_path',_path(cfg['kdv']['teacher']['run'],root))
    teacher=_teacher(_path(teacher_run,root),stopcheck)
    cue_json=_path(overrides.get('cue',{}).get('json_path',cfg['kdv']['qrecon']['asset']),root)
    cue_npz=_path(overrides.get('cue',{}).get('npz_path',_read(cue_json)['npz']),root)
    cue=cue_identity(cue_json,cue_npz,teacher,data['train'],data['train_lp'],stopcheck)
    if cue['npz_sha256']!=CUE_NPZ_SHA:raise ValueError('Original frozen raw-q cue bytes differ')
    selected_cal=overrides.get('calibration',{}).get('source_manifest_path')
    cal=_path(selected_cal,root) if selected_cal else canonical_calibration(root)
    calibration=calibration_identity(cal,teacher,cue,data['train'])
    if calibration['source_manifest_sha256']!=CALIBRATION_SHA:
        raise ValueError('Original fixed T0 calibration provenance bytes differ')
    scenes={}
    for split in ('rr','fr'):
        spec=data[split];ids=[f'{split}:native_h5_index:{i:03d}' for i in range(spec['count'])]
        scenes[split]=dict(schema='G23SENS_NATIVE_SCENES_v1',split=split,sensor='WV3',scene_ids=ids,
            count=spec['count'],path=spec['path'],file_sha256=spec['file_sha256'],band_order=BAND_ORDER,
            identity_validation=dict(status='PAPERSET_IDENTITY_UNVERIFIED',
                reason='Reference-config native population retained; paper correspondence not independently reverified'),
            source_scene_order='original H5 row order, no filtering',full_population=True)
    return dict(schema='G23SENS_ASSET_BINDING_v1',campaign_id=CAMPAIGN_ID,server=server,
        base_config_path=str((root/BASE_CONFIG).resolve()),base_config_git_blob_sha1=BASE_BLOB,
        base_config_sha256=sha256(root/BASE_CONFIG),teacher=teacher,cue=cue,calibration=calibration,data=data,
        evaluation=dict(rr_manifest=scenes['rr'],fr_manifest=scenes['fr'],rr_n_scenes=data['rr']['count'],
            fr_n_scenes=data['fr']['count'],paper_identity_status='PAPERSET_IDENTITY_UNVERIFIED'),
        feeder=dict(class_path='feeders.feeder.PanFeeder',source_sha256=sha256(root/'feeders/feeder.py'),
            crop=False,hflip=True,vflip=True,rot=True,return_meta=True,hflip_probability=1.,vflip_probability=1.,
            rotation='Python random.randint(0,3)',normalization='float64 array -> float32 then 2*x/2047-1',
            workers=4,shuffle=True,drop_last=True),overrides=overrides)


def prepare_bindings(root=ROOT,server='s4',overrides=None,stopcheck=None):
    from g23sens.common import camp,immutable_json,read
    path=camp(root,server)/'runtime_bindings.json'
    old=read(path)
    if old:
        validate_bindings(old,root,server,rehash=True,stopcheck=stopcheck)
        if overrides is not None and old.get('overrides')!=overrides:raise ValueError('Locked asset overrides differ')
        return old
    value=resolve_bindings(root,server,overrides,stopcheck);immutable_json(path,value);return value


def validate_bindings(bindings,root=ROOT,server=None,rehash=True,stopcheck=None):
    if isinstance(bindings,(str,Path)):bindings=_read(bindings)
    server=server or bindings.get('server')
    if (server not in ('s4','s5') or bindings.get('server')!=server or bindings.get('schema')!='G23SENS_ASSET_BINDING_v1'
            or bindings.get('campaign_id')!=CAMPAIGN_ID):raise ValueError('Asset binding identity mismatch')
    reference_config(root)
    if (any(bindings.get('data',{}).get(key,{}).get('file_sha256')!=value for key,value in NATIVE_FILE_SHAS.items())
            or bindings.get('teacher',{}).get('checkpoint_sha256')!=TEACHER_SHA
            or bindings.get('cue',{}).get('npz_sha256')!=CUE_NPZ_SHA
            or bindings.get('calibration',{}).get('tau_R_expected')!=TAU0
            or bindings.get('calibration',{}).get('q_ref_expected')!=Q0
            or bindings.get('calibration',{}).get('source_manifest_sha256')!=CALIBRATION_SHA):
        raise ValueError('Binding does not describe the original fixed G23 assets')
    if rehash:
        current=resolve_bindings(root,server,bindings.get('overrides'),stopcheck)
        if current!=bindings:raise ValueError('BLOCKED_ASSET_MISMATCH: locked native assets/provenance changed')
    return bindings


def load_training_assets(bindings,root=ROOT,device='cpu'):
    """No model initialization or global RNG changes; trainer owns frozen Teacher."""
    bindings=validate_bindings(bindings,root,rehash=False)
    cue=bindings['cue'];teacher=bindings['teacher']
    if sha256(teacher['path'])!=TEACHER_SHA or sha256(cue['npz_path'])!=cue['npz_sha256']:
        raise ValueError('Teacher/cue changed after admission')
    with np.load(cue['npz_path'],allow_pickle=False) as z:
        q=np.empty((bindings['data']['train']['count'],4),dtype=np.float64)
        q[z['index'].astype(np.int64),z['rot'].astype(np.int64)]=z['q']
    return dict(bindings=bindings,teacher=teacher,teacher_run_path=teacher['run_path'],
        teacher_config=yaml.safe_load(Path(teacher['config_path']).read_text()),
        checkpoint_path=teacher['path'],raw_q=torch.from_numpy(q).to(device),cue=cue,calibration=bindings['calibration'])
