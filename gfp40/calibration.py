"""Immutable Teacher-specific gamma q/error shards and exact family medians.

Each slice is computed once, at batch16, for ALL train IDs x ROT4. Family
probabilities only affect pooled calibration order statistics, never q itself.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from fh12.calibration import AXIS16, RADII, axis16_q, _write_npz_immutable
from fh12.data import canonical_sha, sha256_file, check_deadline, write_immutable_json
from qg40.calibration import select_calibration_indices
from qg40.reference_parity import ATOL, RTOL, isolated_teacher, parity_indices
from qg40.model import state_hash
from gfp40.augmentation import (VIEW_RECIPE, gamma_key, prepare_augmentation,
                                verify_augmentation_cache)
from gfp40.data import GFP40Dataset

BATCH = 16


def weighted_median_arrays(arrays, tokens):
    """Exact nonnegative float order statistic, without token expansion.

Search the monotone IEEE754 bit order using exact integer population counts.
Even populations average the two middle observations in the input dtype,
identical to numpy.median(explicitly_repeated_float32_population).
"""
    values = [np.asarray(x) for x in arrays]
    if (not values or len(values) != len(tokens) or any(int(w) != w or w <= 0 for w in tokens)
            or any(x.size == 0 or not np.isfinite(x).all() or (x < 0).any() for x in values)):
        raise ValueError('Exact weighted median needs finite nonnegative values and positive integer tokens')
    dtype = np.result_type(*[x.dtype for x in values])
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        dtype = np.dtype('float64')
    uint = np.uint32 if dtype == np.dtype('float32') else np.uint64
    sorted_values = [np.sort(x.astype(dtype, copy=False).reshape(-1)) for x in values]
    # Normalize negative zero before exploiting nonnegative floating-point order.
    for x in sorted_values: x[x == 0] = 0
    lower = min(int(x[0:1].view(uint)[0]) for x in sorted_values)
    upper = max(int(x[-1:].view(uint)[0]) for x in sorted_values)
    total = sum(x.size * int(w) for x,w in zip(sorted_values, tokens))
    def kth(rank):
        lo, hi = lower, upper
        while lo < hi:
            mid = (lo+hi)//2
            threshold = np.array([mid], dtype=uint).view(dtype)[0]
            population = sum(int(np.searchsorted(x, threshold, side='right')) * int(w)
                             for x,w in zip(sorted_values, tokens))
            if population > rank: hi = mid
            else: lo = mid+1
        return np.array([lo], dtype=uint).view(dtype)[0]
    return float(np.mean(np.array([kth((total-1)//2), kth(total//2)], dtype=dtype)))


def weighted_median_tokens(values, gamma_axis, tokens=(1, 2, 1)):
    values = np.asarray(values)
    if values.shape[gamma_axis] != len(tokens):
        raise ValueError('Gamma axis and token multiplicities differ')
    return weighted_median_arrays([np.take(values, i, axis=gamma_axis) for i in range(len(tokens))], tokens)


def verify_native_slice(measured, original):
    measured, original = np.asarray(measured), np.asarray(original)
    if (measured.shape != original.shape or measured.ndim != 2 or measured.shape[1] != 4
            or not np.isfinite(measured).all() or not np.isfinite(original).all()
            or (measured < 0).any() or (original < 0).any()):
        raise ValueError('Native q shape/numerical mismatch')
    difference = np.abs(measured.astype(np.float64)-original.astype(np.float64))
    if not np.allclose(measured, original, atol=ATOL, rtol=RTOL):
        raise ValueError(f'Native gamma1 AXIS16 parity failed: {difference.max()}')
    return dict(passed=True, population=len(original), views=4, batch_size=BATCH,
        atol=ATOL, rtol=RTOL, max_abs_error=float(difference.max()),
        mean_abs_error=float(difference.mean()))


def _digest(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def runtime_identity(device):
    value = dict(torch=torch.__version__, numpy=np.__version__, device=str(device),
        batch_size=BATCH, dtype='float32', amp=False,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic)
    if str(device).startswith('cuda'):
        value.update(gpu_name=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                     cudnn=torch.backends.cudnn.version())
    return value


def _rows(dataset, ids, gamma_id, device, view=None):
    rows = [dataset.get_view(int(i), 0 if view is None else view,
        augment=view is not None, gamma_id=gamma_id) for i in ids]
    return tuple(torch.stack([r[k] for r in rows]).to(device) for k in range(5))


def _validate_population(model, dataset, indices, native_q, synthetic_test=False):
    indices = np.asarray(indices)
    n = dataset.base_count
    if (dataset.split != 'train' or not dataset.has_gt or model.bands != 4
            or len(dataset) != n or indices.ndim != 1 or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= n):
        raise ValueError('Calibration requires distinct valid train-only base IDs')
    if not synthetic_test and (n != 19809 or not np.array_equal(indices, select_calibration_indices(n))):
        raise ValueError('Calibration requires GF2 train19809, seed1234 train3072')
    if (np.asarray(native_q).shape != (n, 4) or not np.isfinite(native_q).all()
            or (np.asarray(native_q) < 0).any()):
        raise ValueError('Original full native q cache is required')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Calibration requires FP32 Teacher')
    return indices.astype(np.int64)


def compute_gamma_shard(model, dataset, indices, native_q, gamma_id, output_dir, *,
        device='cuda', deadline_utc=None, identity=None, synthetic_test=False,
        reuse_q=None, reuse_provenance=None):
    """Resumable one-gamma whole-train q plus calibration pixel error store."""
    indices = _validate_population(model, dataset, indices, native_q, synthetic_test)
    gamma = dataset.gammas[gamma_id]; n = dataset.base_count
    teacher = state_hash(model.state_dict())
    contract = dict(binding=identity, teacher_state_sha256=teacher, gamma=gamma,
        indices=indices.tolist(), native_q_sha256=_digest(native_q),
        runtime=runtime_identity(device), recipe=VIEW_RECIPE, count=n,
        numerical_source_sha256={name: sha256_file(Path(__file__).parent/name)
                                for name in ('augmentation.py', 'data.py', 'calibration.py')},
        synthetic_test=bool(synthetic_test))
    folder = Path(output_dir).resolve()/ 'q_shards'/gamma_key(gamma)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder/'manifest.json'; path = folder/'progress.h5'; final = folder/'shard.h5'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('contract') != contract: raise ValueError('Existing q shard identity changed')
        _load_shard(manifest, synthetic_test=synthetic_test)
        return manifest_path
    height,width = dataset.base(0)[0].shape[-2:]
    if not final.exists():
        with h5py.File(path, 'a') as store:
            if 'identity_sha256' not in store.attrs:
                store.attrs['identity_sha256'] = canonical_sha(contract)
                for key, shape in dict(q=(n,4), online_q=(n,4), per_radius_q=(n,4,4),
                    delta=(n,4,2), errors=(len(indices),height,width)).items():
                    store.create_dataset(key, shape, dtype='float32', fillvalue=np.nan)
                store.create_dataset('q_checksums', ((n+15)//16,4), dtype='S64')
                store.create_dataset('e_checksums', ((len(indices)+15)//16,), dtype='S64'); store.flush()
            if store.attrs['identity_sha256'] != canonical_sha(contract):
                raise ValueError('Cannot resume q shard under another source/Teacher/runtime')
            with isolated_teacher(model):
                for chunk,start in enumerate(range(0,len(indices),BATCH)):
                    check_deadline(deadline_utc); stop=min(start+BATCH,len(indices))
                    previous=store['e_checksums'][chunk].decode()
                    if previous:
                        if previous != _digest(store['errors'][start:stop]):
                            raise ValueError('Completed Teacher error chunk changed')
                        continue
                    gt,_,ms,lp,pan = _rows(dataset,indices[start:stop],gamma_id,device)
                    errors=(model(pan,ms,lp)['y'].float()-gt.float()).abs().mean(1).cpu().numpy()
                    if not np.isfinite(errors).all(): raise ValueError('Nonfinite Teacher errors')
                    store['errors'][start:stop]=errors; store.flush()
                    store['e_checksums'][chunk]=_digest(errors); store.flush()
                for view in range(4):
                    for chunk,start in enumerate(range(0,n,BATCH)):
                        check_deadline(deadline_utc); stop=min(start+BATCH,n)
                        previous=store['q_checksums'][chunk,view].decode()
                        if previous:
                            parts=[store[k][start:stop,view] for k in ('online_q','per_radius_q','delta')]
                            if previous != canonical_sha([_digest(x) for x in parts]):
                                raise ValueError('Completed q shard chunk changed')
                            continue
                        if reuse_q is None:
                            _,_,ms,_,pan=_rows(dataset,np.arange(start,stop),gamma_id,device,view)
                            parts=[v.detach().cpu().numpy() for v in axis16_q(model,pan,ms)]
                        else:
                            parts=[np.asarray(reuse_q[k])[start:stop,view]
                                   for k in ('online_q','per_radius_q','delta')]
                        if any(not np.isfinite(x).all() for x in parts) or (parts[0]<0).any():
                            raise ValueError('Nonfinite imported/measured q')
                        for k,val in zip(('online_q','per_radius_q','delta'),parts):
                            store[k][start:stop,view]=val
                        store.flush();store['q_checksums'][chunk,view]=canonical_sha([_digest(x) for x in parts]);store.flush()
            measured=store['online_q'][:]
            parity=verify_native_slice(measured,native_q) if gamma == 1. else None
            store['q'][:]=native_q if gamma == 1. else measured
            store.attrs['native_parity_json']=json.dumps(parity,sort_keys=True); store.flush()
        if state_hash(model.state_dict()) != teacher: raise ValueError('Teacher changed during calibration')
        check_deadline(deadline_utc); os.chmod(path,0o444);os.link(path,final)
    with h5py.File(final,'r') as store:
        parity=json.loads(store.attrs['native_parity_json'])
    manifest=dict(schema='GFP40_Q_SHARD_v1',complete=True,contract=contract,
        cache_path=str(final),cache_sha256=sha256_file(final),q_shape=[n,4],
        errors_shape=[len(indices),height,width],native_slice_parity=parity,
        reuse_provenance=reuse_provenance)
    _load_shard(manifest,synthetic_test=synthetic_test)
    write_immutable_json(manifest_path,manifest)
    return manifest_path


def _load_shard(manifest, *, synthetic_test=False):
    if (manifest.get('schema') != 'GFP40_Q_SHARD_v1' or not manifest.get('complete')
            or bool(manifest['contract']['synthetic_test']) != bool(synthetic_test)
            or sha256_file(manifest['cache_path']) != manifest['cache_sha256']):
        raise ValueError('q shard schema/completion/SHA mismatch')
    with h5py.File(manifest['cache_path'],'r') as src:
        if src.attrs.get('identity_sha256')!=canonical_sha(manifest['contract']):
            raise ValueError('q shard file/manifest contract differs')
        arrays={k:src[k][:] for k in ('q','online_q','per_radius_q','delta','errors')}
    n=manifest['contract']['count']; nc=len(manifest['contract']['indices'])
    if (arrays['q'].shape!=(n,4) or arrays['online_q'].shape!=(n,4)
            or arrays['per_radius_q'].shape!=(n,4,4) or arrays['delta'].shape!=(n,4,2)
            or arrays['errors'].shape!=(nc,64,64)
            or list(arrays['q'].shape)!=manifest['q_shape'] or list(arrays['errors'].shape)!=manifest['errors_shape']
            or any(x.dtype!=np.float32 or not np.isfinite(x).all() for x in arrays.values())
            or (arrays['q']<0).any() or (arrays['online_q']<0).any() or (arrays['errors']<0).any()):
        raise ValueError('q/error shard incomplete or nonfinite')
    if manifest['contract']['gamma']!=1. and not np.array_equal(arrays['q'],arrays['online_q']):
        raise ValueError('Non-native q differs from measured/imported values')
    return arrays


def _native_paths(binding):
    original=binding.get('native_manifest',binding); result={}
    for key in ('teacher_checkpoint','q_cache_path'):
        path=Path(binding.get(key,original[key]))
        if not path.is_absolute(): raise ValueError('Needs binder-resolved original artifact paths')
        result[key]=str(path.resolve(strict=True))
    return result


def _native_inputs(binding,dataset,*,production=True):
    reference=binding.get('native_manifest',binding);paths=_native_paths(binding);n=dataset.base_count
    if production and (n!=19809 or reference.get('sensor')!='GF2'
            or reference.get('server') not in ('s3','s4','s5') or reference.get('teacher_update')!=100000):
        raise ValueError('Family reference requires local GF2 native100K Teacher')
    for pk,sk in (('teacher_checkpoint','teacher_checkpoint_sha256'),('q_cache_path','q_cache_sha256')):
        if sha256_file(paths[pk])!=reference[sk]: raise ValueError('Original reference bytes changed')
    if (reference.get('train_sha256')!=sha256_file(dataset.raw_h5_path)
            or reference.get('train_lpan_sha256')!=sha256_file(dataset.lp_path)):
        raise ValueError('Reference train/LP identity mismatch')
    with np.load(paths['q_cache_path'],allow_pickle=False) as src:
        indices,q=src['calibration_indices'].copy(),src['q'].copy()
    if (not np.array_equal(indices,select_calibration_indices(n))
            or canonical_sha(indices.tolist())!=reference['calibration_indices_sha256']
            or q.dtype!=np.float32 or q.shape!=(n,4) or not np.isfinite(q).all() or (q<0).any()
            or float(np.median(q[indices]))!=reference['q_ref'] or reference['q_ref']<=0
            or not np.isfinite(reference['tau_R']) or reference['tau_R']<1e-6):
        raise ValueError('Original native calibration/q invalid')
    return reference,indices,q


def _validate_b20_reuse(path, source_root, model, dataset, reference, native_q, device,
                        deadline_utc=None):
    """No import/alias of old campaign; verify frozen bytes and real probe values."""
    if source_root is None: raise ValueError('B20 reuse requires its actual frozen source archive')
    root=Path(source_root).resolve(strict=True);path=Path(path).resolve(strict=True)
    old=json.loads(path.read_text())
    if (old.get('schema')!='GFB20_MIXED_REFERENCE_v1' or old.get('server')!='s5'
            or old.get('native_reference')!=reference or old.get('runtime')!=runtime_identity(device)):
        raise ValueError('B20 Teacher/reference/runtime mismatch; cannot reuse bank')
    for pk,sk in (('q_cache_path','q_cache_sha256'),('calibration_path','calibration_sha256'),
                  ('augmentation_manifest_path','augmentation_manifest_sha256')):
        if sha256_file(old[pk])!=old[sk]: raise ValueError('B20 bank bytes changed')
    cal=json.loads(Path(old['calibration_path']).read_text())
    for name,expected in cal.get('numerical_source_sha256',{}).items():
        if sha256_file(root/'gfb20'/name)!=expected: raise ValueError('B20 numerical archive mismatch')
    if set(cal.get('numerical_source_sha256',{}))!={'augmentation.py','data.py','calibration.py'}:
        raise ValueError('B20 numerical source provenance incomplete')
    current_root=Path(__file__).resolve().parents[1]
    files=old.get('source_identity',{}).get('files',{})
    required=('g20/model.py','qg40/model.py','pa/warp.py','fh12/calibration.py','fh12/data.py',
              'g20/data.py','tools/repair_lpan.py')
    for name in required:
        if not files.get(name) or sha256_file(root/name)!=files[name] or sha256_file(current_root/name)!=files[name]:
            raise ValueError('B20 core numerical source differs: '+name)
    aug=json.loads(Path(old['augmentation_manifest_path']).read_text())
    if (aug.get('schema')!='GFB20_AUGMENTATION_v1' or not aug.get('complete')
            or not aug.get('native_bitwise') or aug['source']['source_sha256']!=reference['train_sha256']
            or aug['source']['native_lp_sha256']!=reference['train_lpan_sha256']
            or sha256_file(aug['cache_path'])!=aug['cache_sha256']):
        raise ValueError('B20 PAN/LP bank provenance mismatch')
    old_recipe=aug['source']['recipe']
    if any(old_recipe.get(k)!=v for k,v in VIEW_RECIPE.items() if k!='schema'):
        raise ValueError('B20 augmentation numerical recipe differs')
    with np.load(old['q_cache_path'],allow_pickle=False) as src:
        values={k:src[k].copy() for k in ('q','native_online_q','per_radius_q','native_delta','calibration_indices')}
    q=values['q'];n=dataset.base_count
    if (q.dtype!=np.float32 or q.shape!=(n,4,3) or not np.isfinite(q).all() or (q<0).any()
            or values['per_radius_q'].shape!=(n,4,3,4) or values['native_delta'].shape!=(n,4,3,2)
            or not np.isfinite(values['per_radius_q']).all() or not np.isfinite(values['native_delta']).all()
            or not np.array_equal(q[:,:,1],native_q)
            or not np.array_equal(values['calibration_indices'],select_calibration_indices(n))
            or weighted_median_tokens(q[values['calibration_indices']],2)!=old['q_ref']):
        raise ValueError('B20 full q bank/calibration invalid')
    parity=verify_native_slice(values['native_online_q'],native_q)
    if any(old['native_slice_parity'].get(k)!=v for k,v in parity.items()):
        raise ValueError('B20 full native parity evidence differs')
    # Current source/runtime is checked above; independently remeasure fixed16
    # IDs for every gamma/view, while retaining old full-native recomputation.
    ids=parity_indices(n);checks=[]
    with isolated_teacher(model):
        for gamma_id,gamma in enumerate(dataset.gammas):
            if gamma not in (.75,1.,1.25): continue
            old_id=(.75,1.,1.25).index(gamma)
            for view in range(4):
                check_deadline(deadline_utc)
                _,_,ms,_,pan=_rows(dataset,ids,gamma_id,device,view)
                online=axis16_q(model,pan,ms)[0].cpu().numpy()
                if not np.allclose(online,q[ids,view,old_id],atol=ATOL,rtol=RTOL):
                    raise ValueError('B20 imported augmented q failed current online parity')
                checks.append(dict(gamma=gamma,view=view,online_sha256=_digest(online),
                    max_abs=float(np.max(np.abs(online-q[ids,view,old_id])))))
    slices={gamma:dict(online_q=values['native_online_q'] if gamma==1. else q[:,:,index],
        per_radius_q=values['per_radius_q'][:,:,index],delta=values['native_delta'][:,:,index])
        for index,gamma in enumerate((.75,1.,1.25))}
    provenance=dict(status='REUSED_EXTERNAL',original_reference=str(path),
        original_reference_sha256=sha256_file(path),original_q_sha256=old['q_cache_sha256'],
        original_source_content_sha256=old['source_identity']['content_sha256'],
        actual_archive=str(root),native_full_parity=parity,current_parity=checks,
        historical_cost_not_recharged=True,error_population='recomputed_current_3072_batch16')
    return slices,provenance


def prepare_family_reference(model,dataset,native_reference,output_dir,*,family='G025',
        device='cuda',deadline_utc=None,source_identity=None,b20_reference=None,b20_source_root=None):
    from safetensors.torch import load_file
    from gfp40.plan import family_for
    spec=family_for(family);reference,indices,native_q=_native_inputs(native_reference,dataset)
    paths=_native_paths(native_reference)
    if state_hash(model.state_dict())!=state_hash(load_file(paths['teacher_checkpoint'],device='cpu')):
        raise ValueError('Family calibration Teacher differs from original native reference')
    folder=Path(output_dir).resolve();folder.mkdir(parents=True,exist_ok=True)
    destination=folder/('family_'+family+'.json')
    if destination.exists():
        load_family_reference(destination,native_reference,dataset,device=device)
        return destination
    old_augmentation=None
    if b20_reference is not None:
        old=json.loads(Path(b20_reference).read_text())
        old_augmentation=old['augmentation_manifest_path']
        if sha256_file(old_augmentation)!=old['augmentation_manifest_sha256']:
            raise ValueError('B20 augmentation manifest changed before reuse')
    augmentation=prepare_augmentation(dataset,folder,spec['gammas'],deadline_utc,
                                     reuse_manifest=old_augmentation)
    adapter=GFP40Dataset(dataset,augmentation,arm='MIX',family=family)
    reuse,provenance=({},None)
    if b20_reference is not None:
        reuse,provenance=_validate_b20_reuse(b20_reference,b20_source_root,model,adapter,
            reference,native_q,device,deadline_utc)
    # Binding deliberately excludes family and changing bank membership.
    identity=dict(native_reference_sha256=canonical_sha(reference),source_identity=source_identity,
        view_version=canonical_sha(VIEW_RECIPE))
    shards=[]
    for gid,gamma in enumerate(spec['gammas']):
        shard_path=compute_gamma_shard(model,adapter,indices,native_q,gid,folder,
            device=device,deadline_utc=deadline_utc,identity=identity,
            reuse_q=reuse.get(gamma),reuse_provenance=provenance if gamma in reuse else None)
        shards.append(dict(path=str(shard_path),sha256=sha256_file(shard_path),
                           **json.loads(shard_path.read_text())))
    arrays=[_load_shard(s) for s in shards]
    q=np.stack([a['q'] for a in arrays],axis=2)
    tau_raw=weighted_median_arrays([a['errors'] for a in arrays],spec['tokens'])
    qref=weighted_median_tokens(q[indices],2,spec['tokens'])
    if not np.isfinite(qref) or qref<=0: raise ValueError('Invalid family qref; no epsilon fallback')
    # Actual subset parity with explicit integer repetition is a publication gate.
    subset=np.stack([a['errors'][:2,:4,:4] for a in arrays],axis=1)
    expanded=np.repeat(subset,spec['tokens'],axis=1)
    if weighted_median_tokens(subset,1,spec['tokens'])!=float(np.median(expanded)):
        raise ValueError('Weighted order-statistic differs from explicit token expansion')
    q_path=folder/('family_'+family+'_q.npz')
    _write_npz_immutable(q_path,dict(q=q,calibration_indices=indices))
    native_gid=list(spec['gammas']).index(1.)
    calibration=dict(schema='GFP40_FAMILY_CALIBRATION_v1',family=family,gammas=spec['gammas'],
        tokens=spec['tokens'],probabilities=spec['probabilities'],tau_R=max(tau_raw,1e-6),
        tau_raw_pooled_median=tau_raw,tau_floor_used=tau_raw<1e-6,q_ref=qref,
        calibration_n=3072,calibration_seed=1234,calibration_indices_sha256=canonical_sha(indices.tolist()),
        q_shape=list(q.shape),runtime=runtime_identity(device),AXIS16=[list(v) for v in AXIS16],
        radii=list(RADII),weighted_median_rule='exact integer token multiplicity; even middle average',
        weighted_median_actual_subset_expansion_verified=True,
        error_population=int(arrays[0]['errors'].size*sum(spec['tokens'])),
        q_ref_population=int(len(indices)*4*sum(spec['tokens'])),
        native_slice_parity=shards[native_gid]['native_slice_parity'])
    cal_path=folder/('family_'+family+'_calibration.json');write_immutable_json(cal_path,calibration)
    aug=json.loads(augmentation.read_text())
    manifest=dict(schema='GFP40_FAMILY_REFERENCE_v1',sensor='GF2',server=reference['server'],
        native_reference_id=reference['reference_id'],native_reference=reference,native_artifact_paths=paths,
        family=family,gammas=spec['gammas'],tokens=spec['tokens'],probabilities=spec['probabilities'],
        native_gamma_id=native_gid,tau_R=calibration['tau_R'],q_ref=qref,
        native_tau_R=reference['tau_R'],native_q_ref=reference['q_ref'],
        teacher_checkpoint_sha256=reference['teacher_checkpoint_sha256'],
        q_cache_path=str(q_path),q_cache_sha256=sha256_file(q_path),q_shape=list(q.shape),
        calibration_path=str(cal_path),calibration_sha256=sha256_file(cal_path),
        augmentation_manifest_path=str(augmentation),augmentation_manifest_sha256=sha256_file(augmentation),
        gamma_bank_sha256=aug['gamma_bank_sha256'],q_shards=shards,
        calibration_indices_sha256=calibration['calibration_indices_sha256'],
        native_slice_parity=calibration['native_slice_parity'],runtime=calibration['runtime'],
        source_identity=source_identity,
        calibration_id=canonical_sha(dict(teacher=reference['teacher_checkpoint_sha256'],
            data=reference['train_sha256'],gamma_bank=aug['gamma_bank_sha256'],
            view_version=canonical_sha(VIEW_RECIPE),tokens=spec['tokens'])))
    _native_inputs(native_reference,dataset);check_deadline(deadline_utc)
    write_immutable_json(destination,manifest)
    load_family_reference(destination,native_reference,dataset,device=device)
    return destination


def load_family_reference(path,native_reference=None,dataset=None,*,device=None):
    from gfp40.plan import family_for
    manifest=json.loads(Path(path).read_text())
    if (manifest.get('schema')!='GFP40_FAMILY_REFERENCE_v1' or manifest.get('sensor')!='GF2'
            or manifest.get('server') not in ('s3','s4','s5')):
        raise ValueError('Not a local GF2 P40 family reference')
    spec=family_for(manifest['family'])
    if any(manifest.get(k)!=spec[k] for k in ('gammas','tokens','probabilities')):
        raise ValueError('Family definition changed')
    if manifest.get('native_gamma_id')!=list(spec['gammas']).index(1.):
        raise ValueError('Family native gamma index differs')
    reference=manifest['native_reference'];paths=manifest['native_artifact_paths']
    if native_reference is not None and (native_reference.get('native_manifest',native_reference)!=reference
                                       or _native_paths(native_reference)!=paths):
        raise ValueError('Family points at another native Teacher/reference')
    if device is not None and manifest['runtime']!=runtime_identity(device):
        raise ValueError('Family reference runtime differs from consumer')
    for pk,sk in (('q_cache_path','q_cache_sha256'),('calibration_path','calibration_sha256'),
                  ('augmentation_manifest_path','augmentation_manifest_sha256')):
        if sha256_file(manifest[pk])!=manifest[sk]: raise ValueError('Family artifact SHA changed: '+pk)
    for pk,sk in (('teacher_checkpoint','teacher_checkpoint_sha256'),('q_cache_path','q_cache_sha256')):
        if sha256_file(paths[pk])!=reference[sk]: raise ValueError('Native artifact changed')
    from safetensors.torch import load_file
    teacher_state=state_hash(load_file(paths['teacher_checkpoint'],device='cpu'))
    with np.load(paths['q_cache_path'],allow_pickle=False) as src: native_q=src['q'].copy()
    with np.load(manifest['q_cache_path'],allow_pickle=False) as src:
        q,indices=src['q'].copy(),src['calibration_indices'].copy()
    if (q.shape!=(19809,4,len(spec['gammas'])) or q.dtype!=np.float32
            or list(q.shape)!=manifest['q_shape'] or not np.isfinite(q).all() or (q<0).any()
            or not np.array_equal(indices,select_calibration_indices(19809))
            or canonical_sha(indices.tolist())!=manifest['calibration_indices_sha256']
            or not np.array_equal(q[:,:,manifest['native_gamma_id']],native_q)
            or weighted_median_tokens(q[indices],2,spec['tokens'])!=manifest['q_ref']
            or manifest['q_ref']<=0 or not np.isfinite(manifest['tau_R']) or manifest['tau_R']<1e-6):
        raise ValueError('Family full-population q/scale/calibration mismatch')
    current_sources={name:sha256_file(Path(__file__).parent/name)
                     for name in ('augmentation.py','data.py','calibration.py')}
    errors=[]
    if len(manifest['q_shards'])!=len(spec['gammas']):raise ValueError('Missing/extra family q shard')
    for gid,shard in enumerate(manifest['q_shards']):
        if (sha256_file(shard['path'])!=shard['sha256']
                or json.loads(Path(shard['path']).read_text()) != {k:v for k,v in shard.items() if k not in ('path','sha256')}
                or shard['contract']['gamma']!=spec['gammas'][gid]
                or shard['contract']['numerical_source_sha256']!=current_sources
                or shard['contract']['runtime']!=manifest['runtime']
                or shard['contract']['teacher_state_sha256']!=teacher_state
                or shard['contract']['native_q_sha256']!=_digest(native_q)
                or shard['contract']['indices']!=indices.tolist()
                or shard['contract']['binding']['native_reference_sha256']!=canonical_sha(reference)
                or shard['contract']['binding']['source_identity']!=manifest['source_identity']):
            raise ValueError('Gamma q shard provenance differs')
        arrays=_load_shard(shard)
        if not np.array_equal(arrays['q'],q[:,:,gid]): raise ValueError('q bank/slice mismatch')
        errors.append(arrays['errors'])
        if spec['gammas'][gid]==1. and verify_native_slice(arrays['online_q'],native_q)!=manifest['native_slice_parity']:
            raise ValueError('Full native parity evidence mismatch')
    if max(weighted_median_arrays(errors,spec['tokens']),1e-6)!=manifest['tau_R']:
        raise ValueError('Family exact pooled pixel median differs')
    calibration=json.loads(Path(manifest['calibration_path']).read_text())
    for key in ('family','gammas','tokens','probabilities','tau_R','q_ref','q_shape','runtime',
                'calibration_indices_sha256','native_slice_parity'):
        if calibration.get(key)!=manifest[key]: raise ValueError('Family calibration disagreement: '+key)
    if (calibration.get('calibration_n')!=3072
            or not calibration.get('weighted_median_actual_subset_expansion_verified')):
        raise ValueError('Incomplete family calibration')
    if dataset is not None:
        _native_inputs(dict(native_manifest=reference,**paths),dataset)
        aug=verify_augmentation_cache(manifest['augmentation_manifest_path'],dataset)
        if aug['gamma_bank_sha256']!=manifest['gamma_bank_sha256']: raise ValueError('Gamma bank SHA differs')
    expected_id=canonical_sha(dict(teacher=reference['teacher_checkpoint_sha256'],
        data=reference['train_sha256'],gamma_bank=manifest['gamma_bank_sha256'],
        view_version=canonical_sha(VIEW_RECIPE),tokens=spec['tokens']))
    if manifest['calibration_id']!=expected_id:raise ValueError('Family calibration identity differs')
    return manifest,q
