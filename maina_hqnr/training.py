"""MAIN-A: original PLH/F1 numerics, coefficient-only OAT, retained HQNR grid.

No validation-ERGAS selector, G23 constructor, teacher_for(server), alternate
precision, alternate data augmentation or candidate pruning exists here.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time

import numpy as np
import torch
from safetensors.torch import save_file

from fh12.model import build_model, state_hash
from fh12.losses import student_losses as original_student_losses, routed_student_backward
from fh12.training import (rng_state, restore_rng, atomic_torch, tensor_cpu_tree,
                           save_resume_bundle)
from fh20r1.training import make_scheduler as original_make_scheduler
from maina_hqnr.data import BatchStream, build_dataset, completed_updates, stream_manifest, epoch_loader

GRID_STEPS = tuple(range(1010, 50000, 1010)) + (50000,)


def student_losses(out, teacher_out, gt, tau_R, q_weights,
                   alpha=1., beta=.1, lambda_E=.002):
    """Reuse original detached cues/edge/support; override coefficients once.

    BASE calls the unmodified original function and returns its exact graph.
    Other arms reuse its already-validated detached cues and per-sample edge,
    with the same per-band / spatial / batch reductions and epsilon.
    """
    alpha, beta, lambda_E = map(float, (alpha, beta, lambda_E))
    if not all(math.isfinite(v) and v >= 0 for v in (alpha,beta,lambda_E)):
        raise ValueError('Coefficients must be finite and nonnegative')
    original = original_student_losses(out, teacher_out, gt, tau_R, q_weights)
    if (alpha,beta,lambda_E) == (1.,.1,.002):
        return original
    with torch.autocast(device_type=out['y'].device.type, enabled=False):
        y, target = out['y'].float(), gt.float().detach()
        teacher = teacher_out['y'].float().detach()
        es = (y-target).abs().mean(dim=1,keepdim=True)
        soft_error = (y-teacher).abs().mean(dim=1,keepdim=True)
        hard_i = ((1.+alpha*original['difficulty'])*es).flatten(1).mean(1)
        soft_i = (beta*(1.-original['difficulty'])*original['advantage']*soft_error).flatten(1).mean(1)
        edge_i, weights = original['edge_i'], original['q_weights']
        edge_weighted = (lambda_E*weights*edge_i).mean()
        return dict(original, L_U=(hard_i+soft_i+lambda_E*weights*edge_i).mean(),
            L_A=(weights*hard_i).mean(), hard=hard_i.mean(), soft=soft_i.mean(),
            hard_i=hard_i, soft_i=soft_i, edge_weighted=edge_weighted)


def objectives(model, teacher, batch, q, q_ref, tau_R, case, device):
    gt, _lms, ms, lp, pan, meta = batch
    gt, ms, lp, pan = [v.to(device, non_blocking=True) for v in (gt,ms,lp,pan)]
    out = model(pan,ms,lp)
    with torch.no_grad():
        teacher_out = teacher(pan,ms,lp)
    raw_q = q[meta[:,0].to(device),meta[:,1].to(device)]
    weights = (float(q_ref)/(float(q_ref)+raw_q)).detach()
    losses = student_losses(out,teacher_out,gt,tau_R,weights,
        case['alpha'],case['beta'],case['lambda_E'])
    if not all(bool(torch.isfinite(v).all()) for v in
               (out['y'],out['delta'],teacher_out['y'],losses['L_U'],losses['L_A'])):
        raise FloatingPointError('Nonfinite Student/Teacher/objective')
    return out, losses


def make_optimizer(model,cfg):
    if (cfg['optimizer']!='AdamW' or cfg['learning_rate']!=1e-4 or
        cfg['weight_decay']!=.01 or tuple(cfg['betas'])!=(.9,.999) or
        cfg['eps']!=1e-8 or cfg['fh20r1']['aligner_lr']!=3e-6):
        raise ValueError('MAIN-A fixes original AdamW U/A optimizer')
    return torch.optim.AdamW([
        dict(params=model.backbone.parameters(),lr=1e-4,name='U'),
        dict(params=model.aligner.parameters(),lr=3e-6,name='A')],
        betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])


def make_scheduler(optimizer,cfg):
    if cfg['num_warmup']!=100 or cfg['num_iter']!=50000 or cfg['lr_scheduler']!='cosine':
        raise ValueError('MAIN-A fixes original warmup100/cosine50K')
    return original_make_scheduler(optimizer,'BASE',100,50000)


def fresh_model(cfg,teacher):
    model,init = build_model('PLH',104,[1,2,2],cfg['seed'],role='S',
                            teacher_aligner_state=teacher.aligner.state_dict())
    if (model.backbone.input.in_channels!=11 or model.aligner_margin!=4 or
        not init['extra_kernel_zero'] or not all(p.requires_grad for p in model.parameters()) or
        state_hash(model.aligner.state_dict())!=state_hash(teacher.aligner.state_dict())):
        raise ValueError('Fresh PLH/F1 Student initialization mismatch')
    if set(map(id,model.parameters())) & set(map(id,teacher.parameters())):
        raise ValueError('Teacher and Student must not share live parameters')
    init['hashes']['U']=state_hash(model.backbone.state_dict())
    return model,init


def verify_restored_state(state,context,stream,scheduler):
    if state.get('full_state') is not True or state.get('precision')!='fp32':
        raise ValueError('Weights-only/other-precision checkpoint cannot resume')
    if any(state.get(k)!=v for k,v in context.items()):
        raise ValueError('Resume source/config/bindings/init/stream identity mismatch')
    update=state.get('update')
    if (isinstance(update,bool) or not isinstance(update,int) or not 0<=update<=50000 or
        scheduler.last_epoch!=update or completed_updates(stream)!=update):
        raise ValueError('Resume optimizer/scheduler/sampler completed update mismatch')
    return update


def verify_optimizer_updates(optimizer,update):
    states=optimizer.state_dict()['state']
    expected=sum(len(group['params']) for group in optimizer.param_groups)
    if (update==0 and states) or (update>0 and (len(states)!=expected or
        any(float(item.get('step',-1))!=update for item in states.values()))):
        raise ValueError('AdamW moment/update counters differ from completed updates')


def _cycle_model(cfg,teacher,root):
    from maina_hqnr.common import cycle_dir,locked,immutable_json,read_json,sha256,object_sha
    case=cfg['maina_hqnr']['case']; directory=cycle_dir(root,case['server'],case['cycle'])
    model,init=fresh_model(cfg,teacher)
    # Constructor is deterministic keyed by seed/name; snapshot is nevertheless
    # persisted and loaded into each independent case, never from trained BASE.
    identity=dict(seed=case['seed'],policy=init['policy'],hashes=init['hashes'],
                  source_identity=cfg['maina_hqnr'].get('source_identity'))
    with locked(directory/'initialization.lock'):
        path=directory/'initial_snapshot.pt'
        if path.exists():
            saved=read_json(directory/'initial_snapshot.json')
            if any(saved.get(k)!=v for k,v in identity.items()) or saved['file_sha256']!=sha256(path):
                raise ValueError('Common cycle fresh U/A snapshot changed')
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state_hash(state)!=init['hashes']['full']:
                raise ValueError('Cycle snapshot differs from original seeded fresh model')
            model.load_state_dict(state,strict=True)
        else:
            atomic_torch(path,tensor_cpu_tree(model.state_dict()))
            immutable_json(directory/'initial_snapshot.json',dict(identity,file_sha256=sha256(path)))
    return model,init


@torch.no_grad()
def validation_metrics(model,dataset,device):
    """Unselected RR-val log only; full-frame ERGAS and SCC (not official RR)."""
    from torch.utils.data import DataLoader
    from tools.eval_dlpan import scc_dlpan
    from maina_hqnr.diagnostics import isolated_probe
    values=[]; correlations=[]
    with isolated_probe(model):
        model.eval()
        for gt,_lms,ms,lp,pan,_meta in DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0):
            y=model(pan.to(device),ms.to(device),lp.to(device))['y']
            y=(y.float().clamp(-1,1)+1)*1023.5
            target=(gt.to(device).float()+1)*1023.5
            mu=target.mean((2,3))
            per=25*torch.sqrt((((y-target)**2).mean((2,3))/mu.clamp_min(1e-12)**2).mean(1))
            values.extend(per.cpu().tolist())
            for a,b in zip(y.cpu().numpy(),target.cpu().numpy()):
                correlations.append(float(scc_dlpan(a.transpose(1,2,0),b.transpose(1,2,0))))
    if len(values)!=len(dataset) or not np.isfinite(values+correlations).all():
        raise FloatingPointError('Incomplete/nonfinite diagnostic validation')
    return dict(ergas=float(np.mean(values)),scc=float(np.mean(correlations)),
                hqnr=None,selection_authority=False,protocol='RR_VAL_FULLFRAME_NOT_OFFICIAL_RR',n_scenes=len(values))


def train_run(config_path,device='cuda',resume=False,root=None):
    from maina_hqnr.common import (ROOT,read_config,read,read_json,atomic_json,immutable_json,
        object_sha,sha256,utcnow,source_identity,apply_runtime_policy,cycle_dir,check_runtime,RuntimePaused)
    from maina_hqnr.plan import validate_config
    from maina_hqnr.assets import load_training_reference,validate_bindings
    from maina_hqnr.controller import authorize_train
    root=Path(root or ROOT).resolve();cfg=read_config(config_path)
    validate_config(cfg,root=root,require_bound=True)
    meta=cfg['maina_hqnr'];case=meta['case'];authorize_train(root,case,config_path)
    runtime=apply_runtime_policy(root);dev=torch.device(device)
    if dev.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('Production MAIN-A requires actual CUDA; no CPU fallback')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    wd=Path(cfg['work_dir']);wd=wd if wd.is_absolute() else root/wd
    wd.mkdir(parents=True,exist_ok=True)
    binding_path=Path(meta['bindings_path']);binding_path=binding_path if binding_path.is_absolute() else root/binding_path
    bindings=read_json(binding_path)
    validate_bindings(bindings,root,case['server'],rehash=True)
    if object_sha(bindings)!=meta['binding_sha256']:
        raise ValueError('Resolved binding differs from frozen config')
    teacher,reference,q,bridge=load_training_reference(bindings,device=dev)
    teacher.eval().requires_grad_(False)
    teacher_hash=state_hash(teacher.state_dict())
    model,init=_cycle_model(cfg,teacher,root);model.to(dev).train()
    datasets={s:build_dataset(bindings['dataset_manifest'],s,root=root) for s in ('train','val')}
    q=torch.as_tensor(q,dtype=torch.float32,device=dev)
    if q.shape!=(len(datasets['train']),4) or not bool(torch.isfinite(q).all()) or bool((q<0).any()):
        raise ValueError('Fixed F1 q cache must cover every train index/rotation')
    optimizer=make_optimizer(model,cfg);scheduler=make_scheduler(optimizer,cfg)
    stream=BatchStream(len(datasets['train']),48,cfg['seed'])
    sm=stream_manifest(len(datasets['train']),48,cfg['seed'])
    immutable_json(cycle_dir(root,case['server'],case['cycle'])/'stream_manifest.json',sm)
    immutable_json(wd/'stream_manifest.json',sm)
    context=dict(run_id=case['run_id'],attempt=meta.get('attempt',0),
        config_sha256=object_sha(cfg),source_identity=source_identity(root),bindings_sha256=object_sha(bindings),
        data_sha256=bindings.get('data_sha256',object_sha(bindings.get('canonical_dataset_manifest',bindings['dataset_manifest']))),
        reference_sha256=object_sha(bridge),teacher_sha256=bindings['teacher'].get('sha256',bindings['teacher'].get('checkpoint_sha256')),
        init_U_sha256=init['hashes']['U'],init_A_sha256=init['hashes']['A'],stream_sha256=object_sha(sm),
        candidate_grid_sha256=object_sha(list(GRID_STEPS)))
    immutable_json(wd/'meta/config.resolved.yaml',cfg)
    immutable_json(wd/'meta/consumed_bindings.json',bindings)
    immutable_json(wd/'init_manifest.json',dict(init,**context))
    immutable_json(wd/'meta/resolved_numerics.json',dict(alpha=case['alpha'],beta=case['beta'],lambda_E=case['lambda_E'],
        q_ref=reference['q_ref'],tau_R=reference['tau_R'],precision='fp32',U_peak_lr=1e-4,A_peak_lr=3e-6,
        gradient_routing='U <- H+betaK+lambdaE*wE; A <- wH only',offset_weight=0.,synthetic_shift_radius=0,
        source_criterion='fh12.losses.student_losses',schedule='COSINE_W100_BASE_v1'))
    update=0;training_seconds=evaluation_seconds=io_seconds=0.;safe_boundary=True
    records=read(wd/'val_records.json').get('records',[])
    if resume:
        identity=read_json(wd/'last/identity.json');path=wd/'last/training_state.pt'
        if identity.get('training_state_sha256')!=sha256(path):raise ValueError('Full-state checksum changed')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state_hash(state['model_state'])!=identity['state_hash']:raise ValueError('Full-state model differs')
        model.load_state_dict(state['model_state'],strict=True);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);stream.load_state_dict(state['sampler'])
        update=verify_restored_state(state,context,stream,scheduler)
        verify_optimizer_updates(optimizer,update)
        training_seconds=state['training_seconds'];evaluation_seconds=state['evaluation_seconds'];io_seconds=state['io_seconds']
        if any(row['update']>update for row in records):raise ValueError('Val ledger ahead of committed updates')
        restore_rng(state['rng'])
    elif (wd/'last').exists() or records or (wd/'meta/training_start_manifest.json').exists():
        raise ValueError('Existing attempt requires explicit exact resume')
    else:
        immutable_json(wd/'meta/training_start_manifest.json',dict(**context,started_at_utc=utcnow(),
            runtime_policy=runtime,device=str(dev),precision='fp32',horizon_updates=50000,recipe='PLH/W104D122/F1/HQNR_MAX50'))
    pause=[False];handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,lambda *_:pause.__setitem__(0,True))

    def fullstate():
        return dict(full_state=True,precision='fp32',update=update,model_state=tensor_cpu_tree(model.state_dict()),
            optimizer=tensor_cpu_tree(optimizer.state_dict()),scheduler=scheduler.state_dict(),rng=rng_state(),
            sampler=stream.state_dict(),scaler={},accumulation_state=None,training_seconds=training_seconds,
            evaluation_seconds=evaluation_seconds,io_seconds=io_seconds,**context)

    def save_resume():
        nonlocal io_seconds
        started=time.monotonic()
        save_resume_bundle(wd,fullstate(),dict(update=update,full_state=True,state_hash=state_hash(model.state_dict()),**context))
        io_seconds+=time.monotonic()-started

    def candidate():
        folder=wd/'candidates'/str(update)
        expected=dict(update=update,full_state=update==50000,precision='fp32',role='S',input_layout='PLH',
                      state_hash=state_hash(model.state_dict()),**context)
        if folder.exists():
            ident=read_json(folder/'identity.json')
            if (any(ident.get(k)!=v for k,v in expected.items()) or ident.get('model_sha256')!=sha256(folder/'model.safetensors') or
                (update==50000 and ident.get('training_state_sha256')!=sha256(folder/'training_state.pt'))):
                raise ValueError('Existing immutable A/U candidate differs')
            return ident
        folder.parent.mkdir(parents=True,exist_ok=True)
        pending=Path(tempfile.mkdtemp(prefix='.%d-'%update,dir=folder.parent))
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(pending/'model.safetensors'))
        ident=dict(expected,model_sha256=sha256(pending/'model.safetensors'),saved_at_utc=utcnow())
        if update==50000:
            atomic_torch(pending/'training_state.pt',fullstate());ident['training_state_sha256']=sha256(pending/'training_state.pt')
        atomic_json(pending/'identity.json',ident);os.replace(pending,folder)
        return ident

    def status(name,reason=None):
        item=dict(status=name,actual_updates=update,training_complete=update==50000,
            expected_candidates=50,saved_candidates=len(list((wd/'candidates').glob('*/identity.json'))),
            primary_selection='HQNR_MAX50',selection_pending=True,selected_step=None,
            training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,io_seconds=io_seconds,
            training_timing_scope='input_wait+forward+backward+optimizer+CUDA_sync; initialization excluded',
            safe_update_boundary=safe_boundary,last_valid_checkpoint_update=read(wd/'last/identity.json').get('update'),
            reason=reason,updated_at_utc=utcnow(),**context)
        atomic_json(wd/'meta/training_status.json',item)
        return item

    def boundary():
        nonlocal evaluation_seconds,records
        save_resume()
        if update in GRID_STEPS:
            candidate()
            entries=[read_json(wd/'candidates'/str(step)/'identity.json') for step in GRID_STEPS
                     if (wd/'candidates'/str(step)/'identity.json').exists()]
            if any(any(row.get(key)!=value for key,value in context.items()) for row in entries):
                raise ValueError('Candidate ledger contains different source/config/init/stream')
            atomic_json(wd/'candidate_ledger.json',dict(expected_steps=list(GRID_STEPS),
                records=[dict(update=row['update'],model_sha256=row['model_sha256'],state_hash=row['state_hash'])
                         for row in entries],retention='ALL50_NO_AUTOMATIC_PRUNING',**context))
            if not any(row['update']==update for row in records):
                started=time.monotonic();metric=validation_metrics(model,datasets['val'],dev)
                evaluation_seconds+=time.monotonic()-started
                records.append(dict(update=update,metrics=metric))
                atomic_json(wd/'val_records.json',dict(records=records,selection_authority=False,**context))
                print('[MAINA %s %d/50000] HQNR=N/A(pending FR20 postrun) SCC=%.8f ERGAS=%.8f [RR-val log; not selector]'%
                    (case['case_id'],update,metric['scc'],metric['ergas']),flush=True)
            if state_hash(teacher.state_dict())!=teacher_hash:raise ValueError('Frozen F1 mutated')
            save_resume()

    try:
        status('INITIALIZING' if update == 0 else 'RESUMING');boundary()
        first_session_update = True
        while update<50000:
            input_wait_started=time.monotonic()
            loader=epoch_loader(datasets['train'],stream,cfg['seed'],workers=4,pin_memory=True)
            for batch in loader:
                if pause[0]:raise RuntimePaused('Signal requested a safe update boundary')
                check_runtime(case,root,update)
                started=input_wait_started;model.train();optimizer.zero_grad(set_to_none=True)
                out,losses=objectives(model,teacher,batch,q,reference['q_ref'],reference['tau_R'],case,dev)
                routed_student_backward(model,losses)
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                    raise FloatingPointError('Nonfinite/missing routed gradient')
                safe_boundary=False
                optimizer.step();scheduler.step();torch.cuda.synchronize(dev)
                update+=1;stream.cursor+=1;safe_boundary=True
                training_seconds+=time.monotonic()-started
                if first_session_update:
                    from maina_hqnr.common import selected_gpu_uuid
                    if not (wd/'meta/training_start_receipt.json').exists():
                        immutable_json(wd/'meta/training_start_receipt.json',dict(actual_updates=update,pid=os.getpid(),
                            gpu_uuid=selected_gpu_uuid(),case=case,device=str(dev),started_at_utc=utcnow(),**context))
                    status('RUNNING')
                    first_session_update = False
                if update in GRID_STEPS or update%1000==0:boundary();status('RUNNING')
                if update%100==0:
                    print('[MAINA PLH/W104D122/F1/HQNR_MAX50 %s] update=%d/50000 L_U=%.8g L_A=%.8g'%
                          (case['case_id'],update,float(losses['L_U']),float(losses['L_A'])),flush=True)
                if update>=50000:break
                input_wait_started=time.monotonic()
        saved=sorted(int(p.name) for p in (wd/'candidates').iterdir() if p.is_dir() and p.name.isdigit())
        if saved!=list(GRID_STEPS):raise ValueError('All registered 50 A/U candidates must be retained')
        save_resume();done=status('TRAIN_COMPLETE_EVAL_PENDING')
        if not (wd/'meta/training_finished.json').exists():
            immutable_json(wd/'meta/training_finished.json',dict(done,completed_at_utc=utcnow()))
        return 0
    except (RuntimePaused,InterruptedError) as error:
        if safe_boundary:save_resume()
        status('PAUSED_SAFE',str(error));return 75
    except FloatingPointError as error:
        status('DIVERGED',str(error));raise
    except Exception as error:
        if safe_boundary:save_resume()
        status('TECHNICAL_FAILURE',type(error).__name__+': '+str(error));raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
