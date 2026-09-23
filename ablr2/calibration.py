"""Fresh per-sweep C4/C8 train3072 calibration and full-train C15 mean."""
from pathlib import Path
import time
import numpy as np
import torch

from fh12.calibration import _write_npz_immutable
from fh12.data import RECIPE, AUGMENTATION
from qg40.calibration import select_calibration_indices, compute_calibration as _compute


def compute_calibration(model,dataset,indices,device='cuda',batch_size=16,deadline_utc=None,*,synthetic_test=False):
    if not synthetic_test and batch_size != 16:
        raise ValueError('Production ABLR2 calibration uses batch16')
    result,arrays=_compute(model,dataset,indices,device=device,batch_size=batch_size,
        deadline_utc=deadline_utc,synthetic_test=synthetic_test)
    result.update(schema='ABLR2_CALIBRATION_v1',batch_size=batch_size,
        s_bar=float(np.mean(result['q_ref']/(result['q_ref']+arrays['q'].astype(np.float64)))))
    return result,arrays


def calibrate(teacher_run,root=None,server=None,device='cuda',deadline_utc=None,batch_size=16,*,deadline=None):
    from ablr2.common import (ROOT,campaign_id,apply_runtime_policy,check_deadline,immutable_json,
        object_sha,resolved_path,sha256,source_identity,run_dir,read_json,check_runtime,assert_compatible_source)
    from ablr2.references import teacher_endpoint,reference_path,validate_reference,data_signature
    from ablr2.reference_parity import verify_q_cache,identity_for
    from ablr2.data import build_dataset
    from ablr2.training import runtime_context
    root=Path(root or ROOT).resolve()
    started=time.monotonic()
    deadline_utc=deadline if deadline is not None else deadline_utc
    if batch_size != 16: raise ValueError('Production calibration batch16 is fixed')
    apply_runtime_policy(root)
    path=Path(teacher_run)
    path=resolved_path(path,root) if len(path.parts)>1 else run_dir(str(path),root=root)
    case,cfg,data,identity,model,paths=teacher_endpoint(path,root,server)
    _,deadline_utc=runtime_context(cfg,root,deadline_utc,resume=True)
    check_runtime(cfg,root,50000)
    check_deadline(deadline_utc)
    destination=reference_path(case.reference_id,server,root)
    if destination.exists():
        validate_reference(case.reference_id,server,root,dataset_manifest=data)
        return destination
    model.to(device).eval().requires_grad_(False)
    dataset=build_dataset(data,'train',root=root)
    class StopAwareDataset:
        """Dataset-boundary stop hook without changing the shared numeric core."""
        def __len__(self): return len(dataset)
        def __getattr__(self,name): return getattr(dataset,name)
        def base(self,index):
            check_runtime(cfg,root,50000)
            return dataset.base(index)
        def get_view(self,*args,**kwargs):
            check_runtime(cfg,root,50000)
            return dataset.get_view(*args,**kwargs)
    checked_dataset=StopAwareDataset()
    indices=select_calibration_indices(dataset.base_count)
    calibration,arrays=compute_calibration(model,checked_dataset,indices,device=device,batch_size=batch_size,deadline_utc=deadline_utc)
    check_deadline(deadline_utc)
    candidate,cfg_path,data_path=paths['candidate'],paths['config'],paths['data']
    assert_compatible_source(identity['source_identity'],source_identity(root),root,server)
    if (sha256(candidate/'model.safetensors') != identity['model_sha256']
            or sha256(candidate/'training_state.pt') != identity['training_state_sha256']):
        raise ValueError('Teacher/source changed during calibration')
    for item in data['splits'].values():
        for key,digest in (('dataroot','sha256'),('lpan_path','lpan_sha256')):
            if sha256(resolved_path(item[key],root)) != item[digest]:
                raise ValueError('Data/LP changed during calibration')
    q_path,cal_path=destination.parent/'q_cache.npz',destination.parent/'calibration.json'
    _write_npz_immutable(q_path,arrays)
    teacher_identity=dict(teacher_run_id=case.run_id,teacher_seed=case.seed,teacher_update=50000,
        teacher_kind=case.teacher_kind,teacher_checkpoint=str(candidate/'model.safetensors'),
        teacher_checkpoint_sha256=identity['model_sha256'],teacher_training_state=str(candidate/'training_state.pt'),
        teacher_training_state_sha256=identity['training_state_sha256'],teacher_config=str(cfg_path),
        teacher_config_sha256=object_sha(cfg),teacher_config_file_sha256=sha256(cfg_path),
        teacher_checkpoint_identity=str(candidate/'identity.json'),
        teacher_checkpoint_identity_sha256=sha256(candidate/'identity.json'),source_identity=identity['source_identity'],
        teacher_layout='P0',owner_server=server,producer_server=server,server=server,
        consistency_weight=case.lambda_con)
    if identity['source_identity'] != source_identity(root):
        # Keep the immutable Teacher provenance intact while naming the actual
        # bridge-authorized runtime that executed this newly measured cache.
        teacher_identity['calibration_execution_source_identity']=source_identity(root)
    calibration.update(teacher_identity)
    calibration['calibration_seconds']=time.monotonic()-started
    calibration['s_bar_population']=dict(base_count=len(dataset),views=[0,1,2,3],n_values=4*len(dataset),q_cache_sha256=sha256(q_path))
    if cal_path.exists():
        old=read_json(cal_path)
        comparable=lambda value:{k:v for k,v in value.items() if k!='calibration_seconds'}
        if comparable(old)!=comparable(calibration):
            raise ValueError('Resumed calibration differs from the immutable numerical result')
        calibration=old
    else:
        immutable_json(cal_path,calibration)
    manifest=dict(schema='ABLR2_REFERENCE_v1',campaign_id=campaign_id(server),reference_id=case.reference_id,
        **teacher_identity,sensor=case.sensor,num_bands=case.num_bands,max_pixel=cfg['max_pixel'],
        tau_R=calibration['tau_R'],q_ref=calibration['q_ref'],s_bar=calibration['s_bar'],
        s_bar_population=calibration['s_bar_population'],s_bar_sha256=object_sha(dict(
            value=calibration['s_bar'],population=calibration['s_bar_population'])),
        calibration_n=3072,calibration_batch_size=16,calibration_path=str(cal_path),calibration_sha256=sha256(cal_path),
        q_cache_path=str(q_path),q_cache_sha256=sha256(q_path),q_shape=calibration['q_shape'],
        dataset_manifest_path=str(data_path),dataset_manifest_sha256=sha256(data_path),data_sha256=object_sha(data),
        LP_recipe=RECIPE,augmentation=AUGMENTATION,augmentation_sha256=object_sha(AUGMENTATION),
        calibration_indices_sha256=calibration['calibration_indices_sha256'],
        runtime=dict(torch=torch.__version__,numpy=np.__version__,device=str(device)))
    parity=verify_q_cache(model,checked_dataset,arrays['q'],reference_identity=identity_for(manifest),
        data_identity=object_sha(data_signature(data)),device=device,deadline=deadline_utc)
    parity_path=destination.parent/'q_cache_parity.json'
    immutable_json(parity_path,parity)
    manifest.update(q_cache_parity_path=str(parity_path),q_cache_parity_sha256=sha256(parity_path))
    manifest['calibration_seconds']=time.monotonic()-started
    check_deadline(deadline_utc)
    immutable_json(destination,manifest)
    validate_reference(case.reference_id,server,root,dataset_manifest=data)
    return destination
