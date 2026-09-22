"""Fresh50K MARs trainer; native validation only, never FR selection in training."""
from __future__ import annotations
from contextlib import contextmanager
import math
from pathlib import Path
import random
import signal
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pcrepro.common import (ROOT,apply_runtime_policy,atomic_json,camp,immutable_json,
    object_sha,read,read_json,read_config,resolved_path,source_identity,utcnow)
from pcrepro.model import build_model,parameter_counts,state_hash
from pcrepro.checkpoint import save_resume,load_resume,save_model,selection_checkpoints,tensor_cpu_tree
from pcrepro.plan import validate_config,GRID


def cosine_factor(completed,total=50000,warmup=100):
    if total!=50000 or warmup!=100 or isinstance(completed,bool) or int(completed)!=completed or not 0<=completed<=total:
        raise ValueError('Exact warmup100/cosine50K contract required')
    if completed<warmup:return completed/warmup
    return max(0.,.5*(1+math.cos(math.pi*(completed-warmup)/(total-warmup))))


def make_optimizer(model,cfg):
    return torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],
                            betas=tuple(cfg['betas']),eps=cfg['eps'])


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer,cosine_factor)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:torch.cuda.set_rng_state_all(value['cuda'])


@contextmanager
def preserve_runtime(model):
    state=rng_state();modes=[(m,m.training) for m in model.modules()]
    try:yield
    finally:
        for module,mode in modes:module.training=mode
        restore_rng(state)


def mars_loss(model,pan,ms,gt,pan_dn):
    """Duplicate the already augmented pair; sum each mode's own mean L1."""
    if pan_dn is None or gt.shape[:2]!=(len(pan),model.num_bands):
        raise ValueError('MARs requires same-view rawPAN and matching native GT')
    n=len(pan)
    modes=torch.cat((torch.ones(n,device=pan.device),torch.zeros(n,device=pan.device)))
    prediction=model(torch.cat((pan,pan)),torch.cat((ms,ms)),modes,
                     pan_dn=torch.cat((pan_dn,pan_dn)))
    ms_loss=(prediction[:n]-gt).abs().mean()
    pan_loss=(prediction[n:]-pan.expand(-1,model.num_bands,-1,-1)).abs().mean()
    return dict(loss=ms_loss+pan_loss,ms_loss=ms_loss,pan_loss=pan_loss,
                pair_count=n,mars_count=2*n)


def val_improved(previous,current):
    value=float(current['ergas'])
    if not math.isfinite(value):raise FloatingPointError('Nonfinite validation ERGAS')
    return previous is None or value<float(previous['ergas'])


def should_pause(control_path,signaled=False):
    return bool(signaled or read(control_path).get('command')=='STOP_NOW_SAFE')


def verify_metadata(stream,meta):
    expected=torch.as_tensor(stream.batch_at(stream.cursor),dtype=torch.int64)
    if meta.dtype!=torch.int64 or not torch.equal(meta.cpu(),expected):
        raise ValueError('Consumed geometry/source batch differs from exact run stream')


def verify_resume_environment(work_dir,runtime_policy,gpu_name):
    """Do not silently continue one run on a different accelerator or policy."""
    start=read_json(Path(work_dir)/'meta/training_start_manifest.json')
    if start.get('gpu')!=gpu_name or start.get('runtime_policy')!=runtime_policy:
        raise ValueError('Exact resume GPU/runtime policy differs from original run')
    return start


def train_run(config_path,root=ROOT,device='cuda',resume=False):
    from pcrepro.controller import authorize_train
    from pcrepro.data import NativeDataset,PairedBatchStream
    from pcrepro.evaluation import validation_metrics
    initialization_started=time.monotonic()
    root=Path(root);cfg=read_config(config_path);case=validate_config(cfg,require_bound=True)
    if case.stage!='TRAIN' or cfg['num_iter']!=50000:raise ValueError('WV2 has no training path')
    authorize_train(root,case.server,config_path)
    policy=apply_runtime_policy(root)
    field=cfg['pcrepro'];wd=resolved_path(cfg['work_dir'],root)
    source=source_identity(root)
    if field['source_identity']!=source:raise ValueError('Source changed before training')
    data=read_json(resolved_path(field['data_manifest'],root))
    if object_sha(data)!=field['data_sha']:raise ValueError('Data manifest changed before training')
    dev=torch.device(device)
    if dev.type!='cuda' or not torch.cuda.is_available():raise ValueError('Production training requires actual CUDA GPU')
    if resume:verify_resume_environment(wd,policy,torch.cuda.get_device_name(dev))
    random.seed(case.seed);np.random.seed(case.seed);torch.manual_seed(case.seed);torch.cuda.manual_seed_all(case.seed)
    datasets={split:NativeDataset(data,split,root=root) for split in ('train','val')}
    model=build_model(case.num_bands,seed=case.seed,max_pixel=case.max_dn).to(dev)
    optimizer=make_optimizer(model,cfg);scheduler=make_scheduler(optimizer)
    stream=PairedBatchStream(datasets['train'],cfg['batch_size'],case.seed)
    context=dict(campaign_id=field['campaign_id'],run_id=case.run_id,recipe_id=case.recipe_id,recipe_sha256=field['recipe_sha256'],
        source_identity=source,data_sha=field['data_sha'],config_sha=object_sha(cfg),
        model_seed=case.seed,num_bands=case.num_bands,max_pixel=case.max_dn)
    update=0;best=None;records=[]
    preprocessing=time.monotonic()-initialization_started
    costs=dict(training_seconds=0.,validation_seconds=0.,io_seconds=0.,preprocessing_seconds=preprocessing)
    consumed_prefix=[]
    if resume:
        state,identity=load_resume(wd,context)
        model.load_state_dict(state['model_state'],strict=True);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);stream.load_state_dict(state['sampler'])
        update=int(state['update']);best=state['best_validation'];records=state['validation_records']
        costs=state['costs'];costs['preprocessing_seconds']+=preprocessing
        consumed_prefix=state['consumed_prefix']
        if (not 0<=update<=50000 or scheduler.last_epoch!=update or stream.completed_updates!=update
                or state.get('precision')!='fp32' or any(state.get(k)!=v for k,v in context.items())):
            raise ValueError('Exact resume update/scheduler/stream/provenance mismatch')
        restore_rng(state['rng'])
    elif (wd/'meta/training_start_manifest.json').exists() or (wd/'resume/index.json').exists():
        raise ValueError('Started run may only resume its own fullstate')
    else:
        immutable_json(wd/'meta/config.resolved.yaml',cfg)
        immutable_json(wd/'meta/training_start_manifest.json',dict(context,started_at_utc=utcnow(),
            initialization='fresh_seed_no_checkpoint',initial_model_hash=state_hash(model.state_dict()),
            params=parameter_counts(model),runtime_policy=policy,gpu=torch.cuda.get_device_name(dev),
            optimizer_reset=True,updates=50000,test_during_training=False))
    control=camp(root,case.server)/'control.json'
    interrupted=[False];handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    for sig in handlers:signal.signal(sig,lambda *_:interrupted.__setitem__(0,True))

    def fullstate():
        return dict(context,full_state=True,update=update,precision='fp32',model_state=tensor_cpu_tree(model.state_dict()),
            optimizer=tensor_cpu_tree(optimizer.state_dict()),scheduler=scheduler.state_dict(),rng=rng_state(),
            sampler=stream.state_dict(),best_validation=best,validation_records=records,costs=dict(costs),
            consumed_prefix=consumed_prefix,peak_memory_bytes=torch.cuda.max_memory_allocated(dev))

    def snapshot():
        before=time.monotonic();save_resume(wd,fullstate(),dict(context,update=update));costs['io_seconds']+=time.monotonic()-before

    def status(name,**extra):
        atomic_json(wd/'meta/training_status.json',dict(context,status=name,actual_updates=update,
            training_complete=update==50000 and (wd/'checkpoints/exact_50000.json').is_file(),
            scheduler_end=scheduler.last_epoch,best_validation=best,
            validation_count=len(records),costs=dict(costs),updated_at_utc=utcnow(),**extra))

    def validate():
        nonlocal best
        if update not in GRID:return
        previous=next((r for r in records if r['update']==update),None)
        if previous is not None:return
        before=time.monotonic()
        with preserve_runtime(model):
            metrics=validation_metrics(model,datasets['val'],str(dev),
                stopcheck=lambda:should_pause(control,interrupted[0]))
        costs['validation_seconds']+=time.monotonic()-before
        record=dict(update=update,model_state_hash=state_hash(model.state_dict()),**metrics)
        if not all(math.isfinite(float(record[k])) for k in ('ergas','scc')):raise FloatingPointError('Nonfinite validation')
        if val_improved(best,record):
            before=time.monotonic()
            receipt=save_model(wd,'best_val',model,dict(context,update=update,val_ergas=record['ergas']))
            costs['io_seconds']+=time.monotonic()-before
            best=dict(record,model_sha256=receipt['model_sha256'])
        records.append(record)
        atomic_json(wd/'official/validation_grid.json',dict(context,records=records,
            expected_steps=list(GRID),complete=len(records)==len(GRID),test_metrics_used=False))
        print(f'[PCREPRO {case.dataset} {update}/50000] HQNR=NOT_MEASURED(TRAIN_VAL_ONLY) '
              f'SCC={record["scc"]:.8f} ERGAS={record["ergas"]:.8f}',flush=True)

    try:
        status('RUNNING');snapshot()
        # Reconcile an interrupted candidate validation before the next update.
        validate();snapshot()
        while update<50000:
            if should_pause(control,interrupted[0]):snapshot();status('PAUSED_SAFE');return 75
            if shutil.disk_usage(wd).free<20*1024**3:
                snapshot();status('PAUSED_DISK_LOW',minimum_free_bytes=20*1024**3);return 75
            if stream.cursor==stream.count:stream.new_epoch()
            loader=DataLoader(datasets['train'],batch_sampler=stream.remaining_batches(),num_workers=cfg['num_workers'],
                pin_memory=True,generator=torch.Generator().manual_seed(stream.worker_seed))
            waiting_for_batch=time.monotonic()
            for batch in loader:
                costs['io_seconds']+=time.monotonic()-waiting_for_batch
                if should_pause(control,interrupted[0]):snapshot();status('PAUSED_SAFE');return 75
                started=time.monotonic();verify_metadata(stream,batch['meta'])
                pan,ms,gt,pan_dn=[batch[k].to(dev,non_blocking=True) for k in ('pan','ms','gt','pan_dn')]
                model.train();optimizer.zero_grad(set_to_none=True)
                loss=mars_loss(model,pan,ms,gt,pan_dn)
                if not bool(torch.isfinite(loss['loss'])):raise FloatingPointError('Nonfinite MARs loss')
                loss['loss'].backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise FloatingPointError('Nonfinite MARs gradient')
                # Finish this optimizer update before honoring a mid-forward signal.
                optimizer.step();scheduler.step();update+=1;stream.advance()
                consumed_prefix.extend(batch['meta'].tolist()[:max(0,256-len(consumed_prefix))])
                torch.cuda.synchronize(dev)
                costs['training_seconds']+=time.monotonic()-started
                if update in GRID:
                    snapshot();validate();snapshot();status('RUNNING')
                    if shutil.disk_usage(wd).free<20*1024**3:
                        snapshot();status('PAUSED_DISK_LOW',minimum_free_bytes=20*1024**3);return 75
                elif update%100==0:
                    status('RUNNING')
                    print(f'[PCREPRO {case.run_id}] update={update}/50000 '
                          f'L_MS={float(loss["ms_loss"]):.8f} L_PAN={float(loss["pan_loss"]):.8f}',flush=True)
                if should_pause(control,interrupted[0]):snapshot();status('PAUSED_SAFE');return 75
                if update==50000:break
                waiting_for_batch=time.monotonic()
        if scheduler.last_epoch!=50000 or optimizer.param_groups[0]['lr']!=0. or len(records)!=50:
            raise ValueError('Exact50K scheduler/validation endpoint differs')
        before=time.monotonic()
        save_model(wd,'exact_50000',model,dict(context,update=50000),fullstate())
        costs['io_seconds']+=time.monotonic()-before
        snapshot();selection_checkpoints(wd)
        status('TRAIN_COMPLETE',completed_at_utc=utcnow())
        return 0
    except FloatingPointError:
        # Optimizer has not advanced on a nonfinite forward/backward.
        snapshot();status('FAILED_NUMERICAL');raise
    except torch.cuda.OutOfMemoryError:
        snapshot();status('FAILED_OOM');raise
    except InterruptedError:
        snapshot();status('PAUSED_SAFE');return 75
    except OSError:
        # Last atomic snapshot remains valid when storage itself is failing.
        status('FAILED_IO');raise
    except Exception:
        snapshot();status('BLOCKED_INTEGRITY');raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
