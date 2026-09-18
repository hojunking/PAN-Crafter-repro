"""Same-checkpoint FH12 selections, official report and measured RR256 cost."""
from pathlib import Path
import copy
import math
import time
import numpy as np
import torch
import yaml

from fh12.common import ROOT, atomic_json, check_deadline, load_checkpoint_model, object_sha, read_json, sha256, source_identity, utcnow
from fh12.plan import GRID_STEPS, case_for
from fh12.evaluation import RR_KEYS, FR_KEYS


def select_records(records,threshold=.9585,expected_steps=GRID_STEPS):
    by_step={int(r['update']):r for r in records}
    if len(by_step)!=len(records) or sorted(by_step)!=list(expected_steps):
        raise ValueError('FH12 selection requires every unique candidate on the fixed update grid')
    for r in records:
        if not r['rr'].get('official_complete') or not all(math.isfinite(float(r['rr'][k])) for k in RR_KEYS):
            raise ValueError('FH12 official RR missing/nonfinite; never rank proxy/incomplete candidates')
        if not all(math.isfinite(float(r['fr'][k])) for k in FR_KEYS) or not math.isfinite(float(r['val_ergas'])):
            raise ValueError('FH12 FR/validation grid incomplete')
        if r['checkpoint_identity']['update']!=r['update']:
            raise ValueError('FH12 candidate/metric checkpoint mismatch')
    raw=max(records,key=lambda r:(r['fr']['hqnr'],-r['update']))
    eligible=[r for r in records if r['fr']['hqnr']>=threshold]
    # FH12 explicitly uses E, SCC, PSNR, then earlier step. Do not inherit
    # M20's extra SAM/Q8/SSIM tie-breaks or epsilon/tie bands.
    target=min(eligible,key=lambda r:(r['rr']['ergas'],-r['rr']['scc'],-r['rr']['psnr'],r['update'])) if eligible else None
    val=min(records,key=lambda r:(r['val_ergas'],r['update']))
    emin=min(records,key=lambda r:(r['rr']['ergas'],r['update']))
    return dict(raw_max=raw,target=target,exact50k=by_step[50000],rr_val_selected=val,e_min_diag=emin,
                n_eligible=len(eligible),target_status='official' if eligible else 'no_eligible')


def report_selection(label,record,context,**extra):
    if record is None:
        return dict(context,selection_id=label,official_complete=True,selection=None,**extra)
    return dict(context,selection_id=label,official_complete=True,step=record['update'],
                checkpoint_sha256=record['checkpoint_identity']['model_sha256'],
                checkpoint_identity=record['checkpoint_identity'],rr=record['rr'],fr=record['fr'],
                val_ergas=record['val_ergas'],eval_mode='A_ON',precision='fp32',**extra)


def profile_model(model,dataset,device,deadline=None):
    """Whole A+frontend+U runtime; FLOPs report includes explicit coverage, never hidden zeros."""
    check_deadline(deadline)
    sample=dataset[0]; _,_lms,ms,lp,pan,_meta=sample
    args=tuple(x.unsqueeze(0).to(device) for x in (pan,ms,lp))
    model.eval(); dev=torch.device(device)
    conv_macs=[0]
    handles=[]
    def hook(layer,inputs,output):
        if isinstance(layer,torch.nn.Linear):
            conv_macs[0]+=output.numel()*layer.in_features
        elif isinstance(layer,torch.nn.ConvTranspose2d):
            conv_macs[0]+=inputs[0].numel()*(layer.out_channels//layer.groups)*math.prod(layer.kernel_size)
        else:
            conv_macs[0]+=output.numel()*(layer.in_channels//layer.groups)*math.prod(layer.kernel_size)
    for layer in model.modules():
        if isinstance(layer,(torch.nn.Conv2d,torch.nn.ConvTranspose2d,torch.nn.Linear)):
            handles.append(layer.register_forward_hook(hook))
    with torch.no_grad(): model(*args)
    for handle in handles: handle.remove()
    # Existing Sheet Cost/FLOPs(G) is THOP's MAC scale, not 2*MAC. Keep
    # that scale, adding the functional frontend that THOP cannot observe.
    from thop import profile as thop_profile
    # THOP leaves total_ops/total_params buffers on unrecognized composite
    # modules. Never let profiling mutate the checkpoint's inference module.
    scratch=copy.deepcopy(model)
    with torch.no_grad(): thop_macs,_=thop_profile(scratch,inputs=args,verbose=False)
    del scratch
    h,w=pan.shape[-2:]; bicubic_outputs=(8+1+2)*h*w
    frontend_flops=32*bicubic_outputs+h*w+(8*h*w)
    with torch.no_grad():
        for _ in range(3): model(*args)
        if dev.type=='cuda':
            torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
        start=time.perf_counter()
        for _ in range(10):
            check_deadline(deadline); model(*args)
        if dev.type=='cuda': torch.cuda.synchronize(dev)
        milliseconds=(time.perf_counter()-start)*1000/10
    return dict(params_m=sum(p.numel() for p in model.parameters())/1e6,
                flops_g=(thop_macs+frontend_flops/2)/1e9,
                backbone_aligner_thop_macs_g=thop_macs/1e9,
                frontend_mac_equivalent_g=frontend_flops/2/1e9,
                arithmetic_flops_estimate_g=(2*conv_macs[0]+frontend_flops)/1e9,
                flops_convention='THOP MAC scale (legacy Cost convention) + frontend MAC-equivalent (32-op/output bicubic + HP/base arithmetic)/2; functional norm/activation/grid-coordinate arithmetic may be omitted',
                flops_is_estimate=True,frontend_included=True,profile_shape=[1,1,256,256],
                infer_ms=milliseconds,mem_mb=(torch.cuda.max_memory_allocated(dev)/2**20 if dev.type=='cuda' else None),
                device=str(dev),torch=torch.__version__,precision='fp32',LP_cache_generation='offline; calibration/setup time, not inference',
                scope='entire A + synchronized PAN/LP/HP frontend + U')


def process(run,device='cuda',deadline=None,upload=False,upload_only=False,root=ROOT):
    root=Path(root); case=case_for(run); wd=root/'work_dir'/run
    status_path=wd/'official/postrun_status.json'
    if upload_only:
        status=read_json(status_path)
        if not status.get('official_complete'): raise ValueError('Cannot upload unfinished FH12 evaluation')
        return _upload(run,status,status_path,root) if upload else 0
    check_deadline(deadline)
    cfg=yaml.safe_load((wd/'meta/config.resolved.yaml').read_text())
    grid=read_json(wd/'official/raw_grid.json'); training=read_json(wd/'meta/training_status.json')
    if training.get('actual_updates')!=50000 or not training.get('training_complete') or not grid.get('complete'):
        raise ValueError('FH12 official postrun requires exact50K and a complete raw grid')
    if grid['config_sha256']!=object_sha(cfg) or grid['source_identity']!=source_identity(root):
        raise ValueError('FH12 evaluation provenance differs from the current source/config')
    for record in grid['records']:
        check_deadline(deadline)
        directory=wd/'candidates'/str(record['update'])
        ident=read_json(directory/'identity.json')
        if ident!=record['checkpoint_identity'] or ident['model_sha256']!=sha256(directory/'model.safetensors'):
            raise ValueError('FH12 candidate file/grid provenance mismatch')
    selected=select_records(grid['records'])
    context=dict(run_id=run,campaign_id=cfg['fh12']['campaign_id'],role=case.role,
                 config_sha256=object_sha(cfg),data_sha256=grid['data_sha256'],source_identity=grid['source_identity'])
    for key,label in [('raw_max','RAW_MAX'),('target','TARGET'),('exact50k','EXACT50K'),('rr_val_selected','RR_VAL_SELECTED'),('e_min_diag','E_MIN_DIAG50')]:
        extra={}
        if key=='target':
            t=selected['target']; extra=dict(n_eligible=selected['n_eligible'],n_evaluated=len(grid['records']),
                 target_status=selected['target_status'],hqnr_threshold=.9585,
                 selector_order=['ergas','-scc','-psnr','step'],joint_pass=bool(t and t['rr']['ergas']<2.040),
                 test_aware=True)
        if key=='e_min_diag': extra=dict(n_evaluated=50,n_candidates=50,development_oracle=True,independent_test=False)
        filename='target_selection' if key=='target' else key
        atomic_json(wd/'official'/f'{filename}.json',report_selection(label,selected[key],context,**extra))
    cost_path=wd/'official/profile.json'
    if cost_path.exists():
        cost=read_json(cost_path)
        if cost.get('config_sha256')!=object_sha(cfg) or cost.get('source_identity')!=grid['source_identity']:
            raise ValueError('FH12 cached profile belongs to a different config/source')
    else:
        from fh12.data import build_dataset
        data=read_json(root/'work_dir/_fh12'/case.server_id/'dataset_manifest.json')
        dataset=build_dataset(data,'rr',root=root)
        model,_=load_checkpoint_model(cfg,wd/'candidates'/str(selected['raw_max']['update']),device,
                                      expected_source=grid['source_identity'])
        cost=profile_model(model,dataset,device,deadline)
        cost.update(config_sha256=object_sha(cfg),source_identity=grid['source_identity'])
        atomic_json(cost_path,cost)
        del model
    status=dict(status='EVAL_COMPLETE_UPLOAD_PENDING',official_complete=True,sheet_uploaded=False,
                config_sha256=object_sha(cfg),source_identity=grid['source_identity'],
                actual_updates=50000,completed_at_utc=utcnow(),target_status=selected['target_status'])
    atomic_json(status_path,status)
    return _upload(run,status,status_path,root) if upload else 0


def _upload(run,status,status_path,root):
    try:
        from fh12.upload import upload_run
        receipt=upload_run(run,root=root)
        status.update(status='COMPLETE',sheet_uploaded=True,upload_receipt=receipt)
        status.pop('upload_error',None)
    except Exception as exc:
        # Valid results survive a network/auth failure; no trainer or GPU rerun.
        status.update(status='EVAL_COMPLETE_UPLOAD_PENDING',sheet_uploaded=False,
                      upload_error=f'{type(exc).__name__}: {exc}')
    atomic_json(status_path,status)
    return 0
