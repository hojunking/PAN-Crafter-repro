"""FH12-only native fresh Teacher / independently initialized Student training.

The old main.py/KDV loop is intentionally untouched. The input sampler has
explicit (sample, rotation) tuples, so worker prefetch cannot change exact resume.
"""
import csv
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import tempfile
import time

import numpy as np
import torch
import yaml
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler

from fh12.common import (ROOT, atomic_json, before_deadline, check_deadline,
                         load_checkpoint_model, object_sha, read_json, resolved_path,
                         sha256, source_identity, utcnow)
from fh12.plan import build_config, case_for, GRID_STEPS
from fh12.data import build_dataset
from fh12.model import build_model, state_hash
from fh12.losses import teacher_loss, student_losses, routed_student_backward
from fh12.evaluation import evaluate_model, FRMetrics


class BatchStream:
    """Independent data order / rotation RNGs; cursor advances only after an update."""
    def __init__(self,n,batch_size,seed):
        if n < batch_size or batch_size < 1:
            raise ValueError('Training set must contain at least one complete batch')
        self.n=int(n); self.batch_size=int(batch_size); self.epoch=-1; self.cursor=0
        self.data_rng=torch.Generator().manual_seed(int(seed)+300000)
        self.aug_rng=torch.Generator().manual_seed(int(seed)+400000)
        self.order=None; self.rotations=None
        self.new_epoch()

    @property
    def count(self):
        return self.n//self.batch_size

    def new_epoch(self):
        self.epoch+=1; self.cursor=0
        self.order=torch.randperm(self.n,generator=self.data_rng)[:self.count*self.batch_size]
        self.rotations=torch.randint(4,(len(self.order),),generator=self.aug_rng)

    def remaining_batches(self):
        return [[(int(self.order[j]),int(self.rotations[j]))
                 for j in range(i*self.batch_size,(i+1)*self.batch_size)]
                for i in range(self.cursor,self.count)]

    def state_dict(self):
        return dict(n=self.n,batch_size=self.batch_size,epoch=self.epoch,cursor=self.cursor,
                    order=self.order,rotations=self.rotations,data_rng=self.data_rng.get_state(),
                    aug_rng=self.aug_rng.get_state())

    def load_state_dict(self,value):
        if value['n']!=self.n or value['batch_size']!=self.batch_size:
            raise ValueError('FH12 resume changed data length or batch size')
        self.epoch=int(value['epoch']); self.cursor=int(value['cursor'])
        self.order=value['order'].clone(); self.rotations=value['rotations'].clone()
        if not 0<=self.cursor<=self.count or len(self.order)!=self.count*self.batch_size:
            raise ValueError('Invalid FH12 sampler cursor')
        if (self.order.dtype != torch.int64 or self.rotations.dtype != torch.int64
                or self.order.ndim != 1 or self.rotations.shape != self.order.shape
                or torch.unique(self.order).numel() != self.order.numel()
                or bool(((self.order < 0) | (self.order >= self.n)).any())
                or bool(((self.rotations < 0) | (self.rotations > 3)).any())):
            raise ValueError('Invalid FH12 sample permutation or rotation state')
        self.data_rng.set_state(value['data_rng']); self.aug_rng.set_state(value['aug_rng'])


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('Cannot exact-resume a CUDA run on a CPU')
        torch.cuda.set_rng_state_all(value['cuda'])


def atomic_torch(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    os.close(fd)
    try:
        torch.save(value,tmp)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def tensor_cpu_tree(x):
    if isinstance(x,torch.Tensor): return x.detach().cpu()
    if isinstance(x,dict): return {k:tensor_cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list): return [tensor_cpu_tree(v) for v in x]
    if isinstance(x,tuple): return tuple(tensor_cpu_tree(v) for v in x)
    return x


def save_resume_bundle(wd, state, identity):
    """Publish state and its checksum together, preserving the prior bundle on failure."""
    wd=Path(wd); wd.mkdir(parents=True,exist_ok=True)
    target=wd/'last'
    if target.exists() and not target.is_symlink():
        raise ValueError('FH12 last must be an atomic resume-bundle link, not a mutable directory')
    previous=target.resolve() if target.is_symlink() else None
    pending=Path(tempfile.mkdtemp(prefix='.resume-',dir=wd))
    link=wd/(pending.name+'.link')
    published=False
    try:
        atomic_torch(pending/'training_state.pt',state)
        atomic_json(pending/'identity.json',dict(identity,training_state_sha256=sha256(pending/'training_state.pt')))
        link.symlink_to(pending.name,target_is_directory=True)
        os.replace(link,target)
        published=True
    finally:
        if link.is_symlink(): link.unlink()
        if not published: shutil.rmtree(pending)
    # Only remove a superseded bundle created by this helper, never arbitrary
    # symlink targets or any experiment/user directory.
    if previous is not None and previous.parent==wd.resolve() and previous.name.startswith('.resume-'):
        shutil.rmtree(previous)


def validate_config(cfg):
    case=case_for(Path(cfg['work_dir']).name)
    if cfg!=build_config(case):
        raise ValueError('FH12 config differs from its registered immutable recipe; use a new revision')
    return case


def runtime_context(cfg,root,deadline_arg=None):
    window=read_json(resolved_path(cfg['fh12']['window_path'],root))
    deadline=window.get('deadline_utc')
    if not deadline or window.get('campaign_id')!=cfg['fh12']['campaign_id']:
        raise ValueError('FH12 requires its own immutable 12h window; use fh12_start.sh')
    if window.get('server_id',window.get('server'))!=cfg['fh12']['server_id']:
        raise ValueError('FH12 window belongs to another server')
    if deadline_arg and deadline_arg!=deadline:
        raise ValueError('Cannot reset or override the FH12 campaign deadline')
    return window,deadline


def train_run(config_path,device='cuda',resume=False,deadline_arg=None,root=ROOT):
    root=Path(root); cfg=yaml.safe_load(Path(config_path).read_text()); case=validate_config(cfg)
    window,deadline=runtime_context(cfg,root,deadline_arg); check_deadline(deadline)
    if window.get('device',device)!=device:
        raise ValueError('FH12 training device differs from the immutable campaign window')
    dev=torch.device(device)
    if dev.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; training will not silently fall back to CPU')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if dev.type=='cuda': torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    # The observed G23 recipe uses FP32, with the installed runtime's TF32 flags.
    if cfg.get('mixed_precision','no')!='no':
        raise ValueError('FH12 v1 preserves the observed FP32/no-AMP recipe')
    f=cfg['fh12']; wd=resolved_path(cfg['work_dir'],root)
    config_sha=object_sha(cfg); release=source_identity(root)
    data_path=root/'work_dir/_fh12'/f['server_id']/'dataset_manifest.json'
    data=read_json(data_path); data_sha=object_sha(data)
    datasets={name:build_dataset(data,name,root=root) for name in ('train','val','rr','fr')}
    for split,key in [('train','train_feeder_args'),('val','val_feeder_args'),('rr','test_reduced_feeder_args'),('fr','test_full_feeder_args')]:
        if sha256(resolved_path(cfg[key]['dataroot'],root)) != data['splits'][split]['sha256']:
            raise ValueError(f'FH12 {split} config/data manifest mismatch')
    teacher=None; reference=None; q=None; teacher_hash=None
    if case.role=='S':
        from fh12.calibration import load_reference
        reference,q=load_reference(resolved_path(f['reference_manifest'],root),
                                   expected_teacher_run=case.teacher_run_id,data_manifest=data)
        twd=root/'work_dir'/case.teacher_run_id
        tc=yaml.safe_load((twd/'meta/config.resolved.yaml').read_text())
        teacher,tid=load_checkpoint_model(tc,twd/'candidates/50000',dev,expected_source=release)
        if tid['update']!=50000 or tc['fh12']['input_layout']!=f['teacher_input_layout']:
            raise ValueError('FH12 Teacher reference is not the declared layout/exact50K')
        teacher.requires_grad_(False); teacher.eval(); teacher_hash=state_hash(teacher.state_dict())
        model,init=build_model(case.input_layout,case.width,list(case.depth),case.seed,role='S',
                               teacher_aligner_state=teacher.aligner.state_dict())
        if set(map(id,model.parameters())) & set(map(id,teacher.parameters())):
            raise ValueError('Teacher and Student unexpectedly share live parameters')
        q=torch.as_tensor(q,dtype=torch.float32,device=dev)
    else:
        model,init=build_model(case.input_layout,case.width,list(case.depth),case.seed,role='T')
    model=model.to(dev).train()
    optimizer=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':cfg['learning_rate'],'name':'U'},
                                {'params':model.aligner.parameters(),'lr':f['aligner_lr'],'name':'A'}],
                                betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
    scheduler=get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=cfg['num_warmup'],num_training_steps=cfg['num_iter'])
    scaler=torch.cuda.amp.GradScaler(enabled=False)
    stream=BatchStream(len(datasets['train']),cfg['batch_size'],cfg['seed'])
    corruption=torch.Generator().manual_seed(f['corruption_seed'])
    loader_rng=torch.Generator().manual_seed(cfg['seed']+500000)
    update=0; training_seconds=0.; evaluation_seconds=0.; records=[]
    state_path=wd/'last/training_state.pt'
    if resume:
        resume_identity=read_json(wd/'last/identity.json')
        if resume_identity.get('training_state_sha256') != sha256(state_path):
            raise ValueError('FH12 full-state resume checksum mismatch')
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        for key,want in [('config_sha256',config_sha),('data_sha256',data_sha),('source_identity',release),
                         ('reference_sha256',object_sha(reference) if reference else None)]:
            if state.get(key)!=want: raise ValueError(f'FH12 exact resume identity mismatch: {key}')
        if not state.get('full_state'): raise ValueError('Cannot resume weights-only candidate')
        model.load_state_dict(state['model_state'],strict=True)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        stream.load_state_dict(state['sampler']); corruption.set_state(state['corruption_rng']); loader_rng.set_state(state['loader_rng'])
        update=int(state['update']); training_seconds=state['training_seconds']; evaluation_seconds=state['evaluation_seconds']
        restore_rng(state['rng'])
        grid=wd/'official/raw_grid.json'
        records=read_json(grid)['records'] if grid.exists() else []
        if any(int(r['update'])>update for r in records):
            raise ValueError('RR/FR grid is ahead of the full-state resume checkpoint')
    elif state_path.exists() or (wd/'meta/training_start_manifest.json').exists():
        raise ValueError('FH12 run already exists; exact resume required, never auto-reinitialize')
    else:
        wd.mkdir(parents=True,exist_ok=True); (wd/'meta').mkdir(exist_ok=True)
        (wd/'meta/config.resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        atomic_json(wd/'init_manifest.json',dict(init,teacher_ref_sha=(reference or {}).get('teacher_checkpoint_sha256'),
                                               teacher_from_scratch=case.role=='T',pretrained_teacher_loads=0 if case.role=='T' else 1))
        atomic_json(wd/'meta/training_start_manifest.json',dict(campaign_id=f['campaign_id'],run_id=case.run_id,
                    config_sha256=config_sha,data_sha256=data_sha,source_identity=release,window=window,
                    reference_sha256=object_sha(reference) if reference else None,started_at_utc=utcnow(),
                    device=str(dev),precision='fp32',data_order='explicit_epoch_permutation_and_rot4',
                    optimizer=dict(type='AdamW',lr_U=cfg['learning_rate'],lr_A=f['aligner_lr'],betas=cfg['betas'],
                                   eps=cfg['eps'],weight_decay=cfg['weight_decay'])))
        atomic_json(wd/'diagnostics/loss_routing.json',dict(role=case.role,teacher_frozen=case.role=='S',
                     U_objective='rec+odd1e-4off(A-only)' if case.role=='T' else 'H+K+.002*q*E',
                     A_objective='rec+odd1e-4off' if case.role=='T' else 'q*H',LP_HP_delta_detached=False,
                     student_soft_to_A=False,student_edge_to_A=False,GT_used_in_forward=False))
    pause_requested=[False]
    old_signals={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in old_signals: signal.signal(sig,lambda *_:pause_requested.__setitem__(0,True))
    engine=None

    def full_state():
        return dict(full_state=True,update=update,model_state=tensor_cpu_tree(model.state_dict()),
                    optimizer=tensor_cpu_tree(optimizer.state_dict()),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                    rng=rng_state(),sampler=stream.state_dict(),corruption_rng=corruption.get_state(),loader_rng=loader_rng.get_state(),
                    config_sha256=config_sha,data_sha256=data_sha,source_identity=release,
                    reference_sha256=object_sha(reference) if reference else None,
                    training_seconds=training_seconds,evaluation_seconds=evaluation_seconds)

    def status(name):
        atomic_json(wd/'meta/training_status.json',dict(status=name,actual_updates=update,
                     training_complete=update==50000,training_seconds=training_seconds,
                     evaluation_seconds=evaluation_seconds,updated_at_utc=utcnow(),deadline_utc=deadline))

    def save_resume():
        save_resume_bundle(wd,full_state(),dict(update=update,config_sha256=config_sha,full_state=True))

    def save_candidate():
        dest=wd/'candidates'/str(update); dest.parent.mkdir(parents=True,exist_ok=True)
        path=dest/'model.safetensors'
        if dest.exists():
            identity=read_json(dest/'identity.json')
            if identity['config_sha256']!=config_sha or identity['state_hash']!=state_hash(model.state_dict()) or sha256(path)!=identity['model_sha256']:
                raise ValueError('Refusing to overwrite an inconsistent FH12 candidate')
            return identity
        pending=Path(tempfile.mkdtemp(prefix=f'.{update}-',dir=dest.parent))
        try:
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(pending/'model.safetensors'))
            identity=dict(update=update,config_sha256=config_sha,data_sha256=data_sha,source_identity=release,
                          input_layout=case.input_layout,role=case.role,state_hash=state_hash(model.state_dict()),
                          model_sha256=sha256(pending/'model.safetensors'),reference_sha256=object_sha(reference) if reference else None)
            snap=full_state() if update==50000 else dict(full_state=False,update=update,config_sha256=config_sha)
            snap['model_sha256']=identity['model_sha256']; atomic_torch(pending/'training_state.pt',snap)
            identity['training_state_sha256']=sha256(pending/'training_state.pt')
            atomic_json(pending/'identity.json',identity)
            os.replace(pending,dest)
        finally:
            if pending.exists(): shutil.rmtree(pending)
        with open(wd/'candidate_identity.jsonl','a') as out: out.write(json.dumps(identity)+'\n')
        return identity

    def evaluate_update():
        nonlocal engine,evaluation_seconds,records
        if update not in GRID_STEPS or any(r['update']==update for r in records): return
        identity=save_candidate(); check_deadline(deadline)
        if engine is None: engine=FRMetrics(datasets['fr'])
        # Monitoring has the official RR support/formulas. Q8 is also calculated,
        # avoiding a second GPU pass for selected candidates and E_MIN diagnostics.
        # DataLoader construction consumes Torch RNG even for sequential eval.
        # Interrupted/repeated evaluation must never advance the training RNG.
        training_rng=rng_state()
        try:
            result=evaluate_model(model,datasets,dev,engine,deadline,include_q=True,with_val=True)
        finally:
            restore_rng(training_rng)
        if teacher is not None and state_hash(teacher.state_dict())!=teacher_hash:
            raise ValueError('Frozen Teacher changed during Student training')
        evaluation_seconds+=result['seconds']
        record=dict(update=update,checkpoint_identity=identity,**result)
        records.append(record)
        atomic_json(wd/'official/raw_grid.json',dict(campaign_id=f['campaign_id'],run_id=case.run_id,
                     expected_steps=list(GRID_STEPS),records=records,complete=[r['update'] for r in records]==list(GRID_STEPS),
                     config_sha256=config_sha,data_sha256=data_sha,source_identity=release))
        atomic_json(wd/'official'/f'candidate_{update}.json',record)
        with open(wd/'checkpoint_metrics.csv','a',newline='') as stream_csv:
            fields=['step','hqnr_official','rr_scc_official','rr_ergas_official','val_ergas','d_lambda','d_s']
            writer=csv.DictWriter(stream_csv,fieldnames=fields)
            if stream_csv.tell()==0: writer.writeheader()
            writer.writerow(dict(step=update,hqnr_official=result['fr']['hqnr'],rr_scc_official=result['rr']['scc'],
                                 rr_ergas_official=result['rr']['ergas'],val_ergas=result['val_ergas'],
                                 d_lambda=result['fr']['d_lambda'],d_s=result['fr']['d_s']))
        with open(wd/'diagnostics/frequency_shift.csv','a',newline='') as shifts:
            fields=['step','fr_dy','fr_dx','fr_max_abs','rr_dy','rr_dx','rr_max_abs']
            writer=csv.DictWriter(shifts,fieldnames=fields)
            if shifts.tell()==0: writer.writeheader()
            s=result['shift']; writer.writerow(dict(step=update,fr_dy=s['fr_mean'][0],fr_dx=s['fr_mean'][1],
                fr_max_abs=s['fr_max_abs'],rr_dy=s['rr_mean'][0],rr_dx=s['rr_mean'][1],rr_max_abs=s['rr_max_abs']))
        atomic_json(wd/'meta/runtime_projection.json',dict(update=update,
                    training_seconds_per_update=training_seconds/max(1,update),
                    evaluator_checkpoint_seconds=evaluation_seconds/len(records),
                    training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,
                    projected_50k_seconds=training_seconds/max(1,update)*50000+evaluation_seconds/len(records)*50))
        print(f'[FH12 official step={update}] HQNR={result["fr"]["hqnr"]:.8f} SCC={result["rr"]["scc"]:.8f} ERGAS={result["rr"]["ergas"]:.8f} valERGAS={result["val_ergas"]:.8f}',flush=True)
        save_resume()

    try:
        status('RUNNING')
        # If interrupted after candidate save but before its metric commit,
        # complete that exact candidate before consuming another training batch.
        evaluate_update()
        while update<cfg['num_iter']:
            if pause_requested[0] or not before_deadline(deadline):
                save_resume(); status('PAUSED_DEADLINE' if not before_deadline(deadline) else 'PAUSED_SIGNAL'); return 75
            if stream.cursor==stream.count: stream.new_epoch()
            # Stable worker seed for an epoch, independent of how often the
            # iterator is reconstructed after a mid-epoch resume.
            epoch_worker_rng=torch.Generator().manual_seed(cfg['seed']+500000+stream.epoch)
            loader=DataLoader(datasets['train'],batch_sampler=stream.remaining_batches(),
                              num_workers=cfg['num_worker'],pin_memory=dev.type=='cuda',generator=epoch_worker_rng)
            for gt,_lms,ms,lp,pan,meta in loader:
                if pause_requested[0] or not before_deadline(deadline):
                    save_resume(); status('PAUSED_DEADLINE' if not before_deadline(deadline) else 'PAUSED_SIGNAL'); return 75
                started=time.monotonic(); model.train()
                gt,ms,lp,pan=[v.to(dev,non_blocking=True) for v in (gt,ms,lp,pan)]
                optimizer.zero_grad(set_to_none=True)
                out=model(pan,ms,lp)
                if case.role=='T':
                    losses=teacher_loss(model,out,gt,pan,ms,update,corruption)
                    if not torch.isfinite(losses['total']): raise FloatingPointError('Nonfinite Teacher loss')
                    losses['total'].backward(); shown=float(losses['total'].detach())
                else:
                    with torch.no_grad(): teacher_out=teacher(pan,ms,lp)
                    qm=q[meta[:,0].to(dev),meta[:,1].to(dev)]
                    weights=(float(reference['q_ref'])/(float(reference['q_ref'])+qm)).detach()
                    losses=student_losses(out,teacher_out,gt,float(reference['tau_R']),weights)
                    if not torch.isfinite(losses['L_U']) or not torch.isfinite(losses['L_A']):
                        raise FloatingPointError('Nonfinite Student loss')
                    routed_student_backward(model,losses); shown=float(losses['L_U'].detach())
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
                    raise FloatingPointError('Missing/nonfinite FH12 gradients')
                optimizer.step(); scheduler.step(); update+=1; stream.cursor+=1
                training_seconds+=time.monotonic()-started
                if update%cfg['log_iter']==0:
                    print(f'[FH12 {case.role} {case.input_layout}] update={update}/50000 loss={shown:.8g} lrU={optimizer.param_groups[0]["lr"]:.8g} lrA={optimizer.param_groups[1]["lr"]:.8g}',flush=True)
                if update in GRID_STEPS:
                    save_resume(); evaluate_update(); status('RUNNING')
                if update>=cfg['num_iter']: break
        save_resume(); status('TRAINING_COMPLETE'); return 0
    except TimeoutError:
        save_resume(); status('PAUSED_DEADLINE'); return 75
    except Exception:
        # Preserve the previous known-good resume. Do not publish bad gradients
        # or a partly applied update as a valid next-step state.
        status('FAILED'); raise
    finally:
        for sig,handler in old_signals.items(): signal.signal(sig,handler)
