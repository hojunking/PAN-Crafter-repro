"""Isolated ABLR2 fresh50K trainer, exact resume, and update-safe lease stops."""
from __future__ import annotations
import math
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
import time

import numpy as np
import torch
import yaml
from safetensors.torch import save_file,load_file
from torch.utils.data import DataLoader

from fh12.training import BatchStream,atomic_torch,restore_rng,save_resume_bundle,tensor_cpu_tree
from ablr2.model import build_model,state_hash
from ablr2.losses import student_losses,teacher_loss,routed_student_backward
from ablr2.plan import GRID_STEPS,validate_config as _validate_config
from qg40.exposure import record_batch,exposure_report,validate_counts


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def cosine_factor(completed_updates,warmup=100,total=50000):
    if total != 50000 or warmup != 100 or not 0 <= completed_updates <= total:
        raise ValueError('ABLR2 only permits fresh cosine50K/warmup100')
    if completed_updates < warmup: return completed_updates/warmup
    return max(0.,.5*(1.+math.cos(math.pi*(completed_updates-warmup)/(total-warmup))))


def aligner_factor(completed_updates,schedule='BASE'):
    if schedule not in ('BASE','BASE_TIMES_1over3_AFTER_24240'):
        raise ValueError('Unregistered A schedule')
    return 1/3 if schedule != 'BASE' and completed_updates >= 24240 else 1.


def make_scheduler(optimizer,schedule='BASE',warmup=100,total=50000):
    names=[g.get('name') for g in optimizer.param_groups]
    if names not in (['U'],['U','A']):
        raise ValueError('Only explicit trainable U and optional A groups permitted')
    aligner_factor(0,schedule)
    functions=[lambda t:cosine_factor(t,warmup,total)]
    if 'A' in names: functions.append(lambda t:cosine_factor(t,warmup,total)*aligner_factor(t,schedule))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,functions)


def validate_config(cfg):
    return _validate_config(cfg,require_bound=True)


def runtime_context(cfg,root,deadline_arg=None,*,resume=False):
    from ablr2.common import runtime_context as context
    return context(cfg,root,deadline_arg=deadline_arg,resume=resume)


def make_optimizer(model,cfg):
    groups=[dict(params=[p for p in model.backbone.parameters() if p.requires_grad],lr=cfg['learning_rate'],name='U')]
    a=[p for p in model.aligner.parameters() if p.requires_grad]
    if a:
        if cfg['ablr2']['aligner_lr'] <= 0: raise ValueError('Trainable A requires positive registered LR')
        groups.append(dict(params=a,lr=cfg['ablr2']['aligner_lr'],name='A'))
    elif cfg['ablr2']['aligner_lr'] != 0:
        raise ValueError('Identity/frozen A cannot have an optimizer/decay group')
    return torch.optim.AdamW(groups,betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])


def _load_teacher(case,cfg,data,root,device,deadline):
    """No reference imports or reads occur in Teacher-free branches."""
    if not case.requires_teacher: return None,None,None
    from ablr2.common import run_dir,resolved_path,object_sha,sha256
    from ablr2.references import load_reference,load_endpoint_only
    field=cfg['ablr2']
    if case.requires_calibration:
        teacher,ref,q=load_reference(case.reference_id,case.server_id,root,device,
            manifest_path=field['reference_manifest'],dataset_manifest=data,deadline=deadline)
        for key,refkey in (('tau_R','tau_R'),('q_ref','q_ref'),('q_cache_sha256','q_cache_sha256'),
                           ('s_bar','s_bar'),('s_bar_sha256','s_bar_sha256')):
            if field[key] != ref[refkey]: raise ValueError('Resolved calibration binding differs: '+key)
    else:
        teacher,ref=load_endpoint_only(run_dir(case.teacher_run_id,root=root),case.server_id,root,device)
        q=None
    if (ref['teacher_run_id'] != case.teacher_run_id or ref['teacher_seed'] != case.teacher_seed
            or ref['reference_id'] != case.reference_id or ref['teacher_kind'] != case.teacher_kind
            or ref['sensor'] != case.sensor or ref['owner_server'] != case.server_id
            or ref['data_sha256'] != object_sha(data)
            or field['teacher_sha256'] != ref['teacher_checkpoint_sha256']
            or sha256(resolved_path(field['teacher_checkpoint'],root)) != field['teacher_sha256']):
        raise ValueError('Bound Teacher is not this exact sensor/seed/reference/data endpoint')
    return teacher,ref,q


def _clone_check(model,teacher,dataset,device):
    if set(map(id,model.parameters())) & set(map(id,teacher.parameters())):
        raise ValueError('Teacher and Student may not share parameter storage')
    if state_hash(model.aligner.state_dict()) != state_hash(teacher.aligner.state_dict()):
        raise ValueError('Student A is not the exact Teacher clone')
    modes=[(module,module.training) for net in (model,teacher) for module in net.modules()]
    saved=rng_state()
    try:
        model.eval();teacher.eval()
        row=dataset.base(0)
        pan,ms,lp=[row[i].unsqueeze(0).to(device) for i in (4,2,3)]
        with torch.no_grad():
            base=torch.nn.functional.interpolate(ms.float(),scale_factor=4,mode='bicubic',align_corners=False)
            a=model.predict_delta(pan,base);b=teacher.predict_delta(pan,base)
        if not torch.equal(a,b) or not bool(torch.isfinite(a).all()):
            raise ValueError('Initial Teacher/Student correction differs')
        return dict(passed=True,independent_clone=True,c_max_abs_error=0.)
    finally:
        for module,mode in modes: module.training=mode
        restore_rng(saved)


def train_run(config_path,device='cuda',resume=False,deadline_arg=None,root=None):
    from ablr2.common import (ROOT,CAMPAIGN_ID,apply_runtime_policy,source_identity,object_sha,
        read,read_json,read_config,resolved_path,atomic_json,immutable_json,sha256,utcnow,check_deadline)
    from ablr2.data import build_dataset
    from ablr2.common import check_runtime,RuntimePaused
    from ablr2.postrun import evaluate_candidate,record_candidate
    root=Path(root or ROOT).resolve()
    cfg=read_config(config_path);case=validate_config(cfg);field=cfg['ablr2']
    from ablr2.plan import case_for
    if case_for(case.run_id,root=root) != case:
        raise ValueError('Run was not registered in the immutable local task ledger')
    from ablr2.common import run_dir
    if resolved_path(cfg['work_dir'],root).resolve()!=run_dir(case.run_id,root=root).resolve():
        raise ValueError('Training work_dir must be the exact registered local lane path')
    from ablr2.controller import authorize_train
    authorize_train(root,case.server_id,config_path)
    policy=apply_runtime_policy(root)
    if field['runtime_policy_sha256'] != policy['sha256']: raise ValueError('Runtime flags SHA differs')
    lease,deadline=runtime_context(cfg,root,deadline_arg,resume=resume)
    check_runtime(cfg,root,0)
    dev=torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback')
    random.seed(case.seed);np.random.seed(case.seed);torch.manual_seed(case.seed)
    if dev.type == 'cuda': torch.cuda.manual_seed_all(case.seed)
    wd=resolved_path(cfg['work_dir'],root)
    data=read_json(resolved_path(field['dataset_manifest'],root))
    if data['sensor'] != case.sensor or data['num_bands'] != case.num_bands or data['max_pixel'] != 2047:
        raise ValueError('Cross-sensor or incorrect DN data')
    datasets={name:build_dataset(data,name,root=root) for name in ('train','val','rr','fr')}
    if len(datasets['train']) != datasets['train'].base_count:
        raise ValueError('Expected original base IDs, not expanded-view training population')
    teacher,reference,q=_load_teacher(case,cfg,data,root,str(dev),deadline)
    component=case.component
    model,init=build_model(bands=case.num_bands,seed=case.seed,role=case.role,component=component,
        teacher_aligner_state=teacher.aligner.state_dict() if component and component['teacher_A_clone_used'] else None)
    model.to(dev).train()
    if component and component['teacher_A_clone_used']:
        init['clone_P0']=_clone_check(model,teacher,datasets['train'],dev)
    # Clone-only case must not retain a prediction Teacher or read calibration.
    teacher_hash=state_hash(teacher.state_dict()) if teacher is not None else None
    if component and not component['teacher_predictions_used']:
        teacher=None;teacher_hash=None
    q=torch.as_tensor(q,dtype=torch.float32,device=dev) if q is not None else None
    context=dict(config_sha256=object_sha(cfg),data_sha256=object_sha(data),source_identity=source_identity(root),
                 reference_sha256=object_sha(reference) if reference is not None else None)
    if reference is not None:
        immutable_json(wd/'meta/consumed_reference.json',reference)
    optimizer=make_optimizer(model,cfg)
    scheduler=make_scheduler(optimizer,field['student_a_schedule'])
    stream=BatchStream(len(datasets['train']),cfg['batch_size'],case.seed)
    corruption=torch.Generator(device='cpu').manual_seed(field['corruption_seed'])
    counts=torch.zeros((len(datasets['train']),4),dtype=torch.int64)
    update=0;train_seconds=eval_seconds=io_seconds=0.;previous_peak=0
    records=read(wd/'official/raw_grid.json').get('records',[])
    if resume:
        last=read_json(wd/'last/identity.json');path=wd/'last/training_state.pt'
        if last.get('training_state_sha256') != sha256(path): raise ValueError('Resume fullstate checksum mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if (state.get('full_state') is not True or state.get('precision') != 'fp32'
                or any(state.get(k) != v or last.get(k) != v for k,v in context.items())
                or state_hash(state['model_state']) != last.get('state_hash')
                or read_config(wd/'meta/config.resolved.yaml') != cfg):
            raise ValueError('Resume config/source/data/reference/model identity mismatch')
        start=read_json(wd/'meta/training_start_manifest.json')
        if any(start.get(k) != v for k,v in context.items()) or start.get('run_id') != case.run_id:
            raise ValueError('Training-start identity differs at resume')
        model.load_state_dict(state['model_state'],strict=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        stream.load_state_dict(state['sampler']);corruption.set_state(state['corruption_rng']);update=state['update']
        if not 0 <= update <= 50000 or scheduler.last_epoch != update or stream.epoch*stream.count+stream.cursor != update:
            raise ValueError('Resume optimizer/scheduler/sampler endpoint mismatch')
        counts=validate_counts(state.get('exposure_counts'),len(datasets['train']),update,cfg['batch_size'])
        train_seconds,eval_seconds,io_seconds=state['training_seconds'],state['evaluation_seconds'],state['io_seconds']
        previous_peak=state.get('peak_memory_bytes') or 0
        if any(row['update']>update or row['update'] not in GRID_STEPS or
               any(row.get('checkpoint_identity',{}).get(k)!=v for k,v in context.items()) for row in records):
            raise ValueError('Evaluation grid is ahead of or differs from resume state')
        restore_rng(state['rng'])
    else:
        if records or (wd/'last').exists() or (wd/'meta/training_start_manifest.json').exists():
            raise ValueError('Existing run requires explicit exact resume')
        (wd/'meta').mkdir(parents=True,exist_ok=True)
        config_target=wd/'meta/config.resolved.yaml'
        if config_target.exists():
            if read_config(config_target)!=cfg:
                raise ValueError('Published immutable run configuration differs')
        else:
            immutable_json(config_target,cfg)  # JSON is valid YAML; never rewrite a pinned config.
        immutable_json(wd/'init_manifest.json',dict(init,**context))
        immutable_json(wd/'meta/training_start_manifest.json',dict(campaign_id=CAMPAIGN_ID,
            run_id=case.run_id,horizon_updates=50000,**context,started_at_utc=utcnow(),precision='fp32',
            runtime_policy=policy,initial_lease=lease,device=str(dev),
            sampler_hash=state_hash({'order':stream.order,'rotations':stream.rotations}),
            rng_roles=dict(data_order=case.seed+300000,augmentation=case.seed+400000,
                corruption=field['corruption_seed'],workers=case.seed+500000)))
    pause=[False]
    handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in handlers: signal.signal(sig,lambda *_:pause.__setitem__(0,True))
    fullsteps=set(field['fullstate_steps'])|{50000}
    engine=None

    def peak():
        return max(previous_peak,torch.cuda.max_memory_allocated(dev)) if dev.type=='cuda' else None

    def fullstate():
        return dict(full_state=True,update=update,model_state=tensor_cpu_tree(model.state_dict()),
            optimizer=tensor_cpu_tree(optimizer.state_dict()),scheduler=scheduler.state_dict(),scaler={},precision='fp32',
            rng=rng_state(),sampler=stream.state_dict(),corruption_rng=corruption.get_state(),
            exposure_counts=counts.clone(),**context,training_seconds=train_seconds,evaluation_seconds=eval_seconds,
            io_seconds=io_seconds,peak_memory_bytes=peak())

    def save_resume():
        nonlocal io_seconds
        started=time.monotonic()
        save_resume_bundle(wd,fullstate(),dict(update=update,full_state=True,state_hash=state_hash(model.state_dict()),**context))
        io_seconds+=time.monotonic()-started

    def save_candidate(folder,include_state):
        folder.parent.mkdir(parents=True,exist_ok=True)
        expected=dict(update=update,role=case.role,sensor=case.sensor,num_bands=case.num_bands,
            input_layout=case.input_layout,full_state=bool(include_state),state_hash=state_hash(model.state_dict()),**context)
        if folder.exists():
            identity=read_json(folder/'identity.json')
            if (any(identity.get(k)!=v for k,v in expected.items())
                    or identity.get('model_sha256')!=sha256(folder/'model.safetensors')
                    or identity.get('training_state_sha256')!=sha256(folder/'training_state.pt')):
                raise ValueError('Immutable candidate differs')
            return identity
        pending=Path(tempfile.mkdtemp(prefix=f'.{update}-',dir=folder.parent))
        try:
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(pending/'model.safetensors'))
            identity=dict(expected,model_sha256=sha256(pending/'model.safetensors'),saved_at_utc=utcnow())
            state=fullstate() if include_state else dict(full_state=False,update=update)
            state['model_sha256']=identity['model_sha256'];atomic_torch(pending/'training_state.pt',state)
            identity['training_state_sha256']=sha256(pending/'training_state.pt')
            atomic_json(pending/'identity.json',identity);os.replace(pending,folder)
            return identity
        finally:
            if pending.exists(): shutil.rmtree(pending)

    def status(name,reason=None):
        atomic_json(wd/'meta/training_status.json',dict(status=name,actual_updates=update,horizon_updates=50000,
            training_complete=update==50000,n_evaluated=len(records),training_seconds=train_seconds,
            evaluation_seconds=eval_seconds,io_seconds=io_seconds,peak_memory_bytes=peak(),
            pending_steps=[n for n in GRID_STEPS if n not in {r['update'] for r in records}],
            sample_exposure=exposure_report(counts,cfg['batch_size'],stream.epoch,stream.cursor),
            training_completed_at_utc=read(wd/'candidates/50000/identity.json').get('saved_at_utc'),
            reason=reason,updated_at_utc=utcnow()))

    def boundary():
        nonlocal eval_seconds,engine,records,io_seconds
        save_resume();started=time.monotonic()
        if update in fullsteps: save_candidate(wd/'restart_fullstates'/str(update),True)
        if update not in GRID_STEPS: return
        save_candidate(wd/'candidates'/str(update),update in fullsteps)
        io_seconds+=time.monotonic()-started
        if any(row['update']==update for row in records): return
        check_runtime(cfg,root,update);check_deadline(deadline)
        saved=rng_state();started=time.monotonic()
        try:
            if engine is None:
                from ablr2.evaluation import FRMetrics
                engine=FRMetrics(datasets['fr'])
            result=evaluate_candidate(cfg,wd/'candidates'/str(update),datasets,dev,
                engine=engine,deadline=deadline,root=root,model=model)
            record_candidate(cfg,result,root=root)
            records=read_json(wd/'official/raw_grid.json')['records']
            print(f'[ABLR2 {case.case_id} {update}/50000] HQNR={result["fr"]["hqnr"]:.8f} SCC={result["rr"]["scc"]:.8f} ERGAS={result["rr"]["ergas"]:.8f} valERGAS={result["val_ergas"]:.8f}',flush=True)
        finally:
            restore_rng(saved);eval_seconds+=time.monotonic()-started
        if teacher is not None and state_hash(teacher.state_dict())!=teacher_hash:
            raise ValueError('Frozen Teacher mutated')
        save_resume()

    try:
        status('RUNNING');boundary()
        while update < 50000:
            if pause[0]: raise RuntimePaused('Signal: safe update boundary')
            check_runtime(cfg,root,update)
            if stream.cursor==stream.count: stream.new_epoch()
            loader=DataLoader(datasets['train'],batch_sampler=stream.remaining_batches(),num_workers=cfg['num_worker'],
                pin_memory=dev.type=='cuda',generator=torch.Generator().manual_seed(case.seed+500000+stream.epoch))
            for gt,_lms,ms,lp,pan,meta in loader:
                if pause[0]: raise RuntimePaused('Signal: safe update boundary')
                check_runtime(cfg,root,update)
                started=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
                gt,ms,lp,pan=[x.to(dev,non_blocking=True) for x in (gt,ms,lp,pan)]
                out=model(pan,ms,lp)
                if not bool(torch.isfinite(out['y']).all()) or not bool(torch.isfinite(out['delta']).all()):
                    raise FloatingPointError('Nonfinite native output/correction')
                if case.role=='T':
                    losses=teacher_loss(model,out,gt,pan,ms,update,corruption,
                        consistency_weight=case.lambda_con,bands=case.num_bands)
                    if not bool(torch.isfinite(losses['total'])): raise FloatingPointError('Nonfinite Teacher loss')
                    losses['total'].backward()
                else:
                    teacher_out=None
                    if teacher is not None:
                        with torch.no_grad(): teacher_out=teacher(pan,ms,lp)
                    weights=None
                    if component['requires_q']:
                        weights=(reference['q_ref']/(reference['q_ref']+q[meta[:,0].to(dev),meta[:,1].to(dev)])).detach()
                    losses=student_losses(out,teacher_out,gt,component,
                        tau_R=reference['tau_R'] if component['requires_tau'] else None,
                        q_weights=weights,s_bar=reference['s_bar'] if 'TRAIN_MEAN' in (component['q_edge'],component['q_aligner']) else None,
                        bands=case.num_bands)
                    if not bool(torch.isfinite(losses['L_U'])&torch.isfinite(losses['L_A'])):
                        raise FloatingPointError('Nonfinite Student loss')
                    routed_student_backward(model,losses)
                gradients=[p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise FloatingPointError('Missing/nonfinite routed gradients')
                optimizer.step();scheduler.step();update+=1;stream.cursor+=1;record_batch(counts,meta)
                train_seconds+=time.monotonic()-started
                if update in GRID_STEPS or update in fullsteps: boundary();status('RUNNING')
                if update % cfg['log_iter']==0: print(f'[ABLR2 {case.case_id}] update={update}/50000',flush=True)
                if update==50000: break
        save_resume();status('OFFICIAL_EVAL_COMPLETE' if len(records)==50 else 'TRAIN_COMPLETE_EVAL_PENDING')
        return 0
    except (RuntimePaused,TimeoutError) as error:
        save_resume();status('PAUSED_SAFE' if update<50000 else 'TRAIN_COMPLETE_EVAL_PENDING',str(error));return 75
    except FloatingPointError as error:
        status('DIVERGED',str(error));raise
    except Exception as error:
        status('BLOCKED_NUMERICS',f'{type(error).__name__}: {error}');raise
    finally:
        for sig,handler in handlers.items(): signal.signal(sig,handler)


def recover_exact_endpoint(root,wd):
    """Reconcile a committed endpoint after a final status-write crash, CPU only."""
    from ablr2.common import read,read_json,read_config,object_sha,sha256,source_identity,atomic_json,utcnow
    wd=Path(wd);cfg=read_config(wd/'meta/config.resolved.yaml');case=validate_config(cfg)
    if read(wd/'meta/training_status.json').get('training_complete') is True: return False
    folder=wd/'candidates/50000'
    if not folder.exists(): return False
    identity=read_json(folder/'identity.json');state=torch.load(folder/'training_state.pt',map_location='cpu',weights_only=False)
    start=read_json(wd/'meta/training_start_manifest.json')
    data=read_json(Path(root)/cfg['ablr2']['dataset_manifest'])
    expected=dict(config_sha256=object_sha(cfg),data_sha256=object_sha(data),source_identity=source_identity(root),
        reference_sha256=start.get('reference_sha256'))
    if (identity.get('update')!=50000 or identity.get('full_state') is not True
            or identity.get('model_sha256')!=sha256(folder/'model.safetensors')
            or identity.get('training_state_sha256')!=sha256(folder/'training_state.pt')
            or state.get('full_state') is not True or state.get('update')!=50000
            or state.get('scheduler',{}).get('last_epoch')!=50000
            or state.get('model_sha256')!=identity['model_sha256']
            or state_hash(state['model_state'])!=identity.get('state_hash')
            or state_hash(load_file(str(folder/'model.safetensors'),device='cpu'))!=identity.get('state_hash')
            or any(identity.get(k)!=v or state.get(k)!=v or start.get(k)!=v for k,v in expected.items())):
        raise ValueError('Exact endpoint recovery provenance mismatch')
    stream=BatchStream(data['splits']['train']['count'],cfg['batch_size'],case.seed)
    stream.load_state_dict(state['sampler'])
    if stream.epoch*stream.count+stream.cursor!=50000: raise ValueError('Recovery sampler update differs')
    counts=validate_counts(state.get('exposure_counts'),data['splits']['train']['count'],50000,cfg['batch_size'])
    records=read(wd/'official/raw_grid.json').get('records',[])
    atomic_json(wd/'meta/training_status.json',dict(status='TRAIN_COMPLETE_EVAL_PENDING',training_complete=True,
        actual_updates=50000,horizon_updates=50000,n_evaluated=len(records),training_seconds=state['training_seconds'],
        evaluation_seconds=state['evaluation_seconds'],io_seconds=state['io_seconds'],peak_memory_bytes=state.get('peak_memory_bytes'),
        sample_exposure=exposure_report(counts,cfg['batch_size'],stream.epoch,stream.cursor),
        training_completed_at_utc=identity.get('saved_at_utc'),recovered_from_exact50k=True,updated_at_utc=utcnow()))
    return True
