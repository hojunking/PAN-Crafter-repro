"""Serial sensor queues, independent failures, complete resume and local publication."""
from copy import deepcopy
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import torch
from directshift48.reporting import atomic_json,append_jsonl,utc_now
from panda_stage2_condq.training import configure_precision
from ta2.evaluation import evaluator_identity,file_sha
from . import plan,assets,training,evaluation
from .model import Stage1,Stage2


def initial_state(server):
    return {'schema':'PANDA_SENSOR_EQ40_QUEUE_V1','campaign_id':plan.CAMPAIGN_ID,'server':server,
        'queue_sha256':plan.digest(plan.queue(server)),'status':'READY','runs':[
            {'run_id':c['run_id'],'case':c['case_id'],'repeat':c['repeat'],'seed':c['seed'],
             'status':'QUEUED','step':0} for c in plan.queue(server)]}


def read_state(root,server):
    fresh=initial_state(server);p=Path(root)/'campaign_state.json'
    if not p.exists():return fresh
    old=json.loads(p.read_text())
    if any(old.get(k)!=fresh[k] for k in ('schema','campaign_id','server','queue_sha256')):
        raise ValueError('Saved queue differs from registered round-major sequence')
    if [r['run_id'] for r in old['runs']]!=[r['run_id'] for r in fresh['runs']]:
        raise ValueError('Saved queue order differs')
    for row,cfg in zip(old['runs'],plan.queue(server)):
        if row['status']=='COMPLETED' and row['step']!=cfg['horizon']:
            raise ValueError('Premature completed run')
    return old


def next_job(state):
    return next((r for r in state['runs'] if r['status'] not in ('COMPLETED','FAILED','BLOCKED_DONOR','EARLY_STOPPED')),None)


def block_failed_dependency(state,job,cfg):
    if cfg['stage']!='stage2':return False
    if cfg['server']=='s2':return False  # external Stage1 checked by assets.select
    parent=next(r for r in state['runs'] if r['run_id']==cfg['donor_run_id'])
    if parent['status']=='FAILED':
        job.update(status='BLOCKED_DONOR',reason='This repetition Stage1 failed; no donor substitution',
                   donor_run_id=parent['run_id'],updated_at=utc_now())
        return True
    if parent['status']!='COMPLETED' or parent['step']!=50000:
        raise ValueError('Stage2 requires its completed local RGB Stage1; no cross-repeat fallback')
    return False


def early_stop_evidence(cfg,directory,step):
    policy=cfg['early_stop']
    if not policy['enabled'] or step!=policy['at_step']:return None
    records=[json.loads((directory/f'evaluation_{s:07d}.json').read_text()) for s in policy['steps']]
    for record in records:evaluation.verify_record(record,cfg)
    values=[r['fr']['metrics']['HQNR'] for r in records]
    if all(v<policy['HQNR_below'] for v in values):
        return {'policy':policy,'HQNR':values,'step':step,'reason':'QB_HQNR_8K_9K_10K_ALL_BELOW_090'}
    return None


def latest_checkpoint(directory):
    candidates=[]
    # Evaluated models retain every A/T/S weight but omit redundant optimizer
    # states. Only complete 250-step generations can resume optimization.
    for p in sorted((directory/'resume').glob('*.pt')):
        sidecar=p.with_suffix('.json')
        if p.exists() and sidecar.exists():
            meta=json.loads(sidecar.read_text());candidates.append((meta['step'],p))
    return max(candidates,key=lambda x:x[0])[1] if candidates else None


def save_resume(directory,trainer,stream,pins):
    # A new generation plus atomic metadata avoids overwriting latest.pt before
    # its hash sidecar exists. Keep two complete generations; never prune models/.
    path=directory/'resume'/f'step_{trainer.updates:07d}_{time.time_ns()}.pt'
    meta=training.save(path,trainer,stream,pins,immutable=True)
    atomic_json(directory/'latest.json',meta)
    generations=sorted(path.parent.glob('*.json'),key=lambda p:p.name)
    for old in generations[:-2]:
        old.with_suffix('.pt').unlink(missing_ok=True);old.unlink()
    return meta


def rewind_trace(directory,step):
    """Retain only updates committed by the restored checkpoint; archive crash suffixes."""
    path=directory/'train.jsonl'
    if not path.exists():
        if step:raise ValueError('Resume checkpoint has no corresponding training trace')
        return
    prefix=[]
    with path.open('rb') as f:
        for expected in range(1,step+1):
            line=f.readline()
            if not line.endswith(b'\n') or json.loads(line).get('step')!=expected:
                raise ValueError('Training trace before resume boundary is incomplete')
            prefix.append(line)
        suffix=f.read()
    if suffix:
        os.link(path,directory/f'train_before_resume_{time.time_ns()}.jsonl')
        fd,tmp=tempfile.mkstemp(prefix='.trace-',dir=directory)
        try:
            with os.fdopen(fd,'wb') as f:
                f.writelines(prefix);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)


def publish_safely(root,server):
    from .publication import publish
    try:publish(root,server,apply=True)
    except Exception as exc:
        # Training evidence/weights persist; the next publication poll retries by fresh readback.
        atomic_json(root/'publication/last_error.json',{'at':utc_now(),'error':repr(exc)})
        print('Sheet publication pending:',repr(exc),flush=True)


def finish_evaluation(trainer,stream,directory,sources,pins,inventory):
    step=trainer.updates; evpath=directory/f'evaluation_{step:07d}.json'
    if evpath.exists():
        e=json.loads(evpath.read_text());evaluation.verify_record(e,trainer.cfg)
        if e['model_sha256']!=training.model_hash(trainer.model):
            raise ValueError('Already evaluated step belongs to another model state')
        return e
    record=evaluation.evaluate(trainer,sources,pins['evaluator'],
        on_scene=lambda split,n:print(f'Eval {step} {split} {n}/20',flush=True) if n%10==0 else None)
    if trainer.cfg['stage']=='stage1' and step in trainer.cfg['validation_steps']:
        val=evaluation.validation(trainer,sources,inventory)
        atomic_json(directory/f'validation_{step:07d}.json',val)
        record['validation']={k:v for k,v in val.items() if k not in ('per_sample','protocol')}
    elif trainer.cfg['stage']=='stage2':
        record['donor_selection']=deepcopy(trainer.model.selection_receipt)
        record['calibration_sha256']=trainer.model.stats.artifact_sha256
    checkpoint=directory/'models'/f'step_{step:07d}.pt'
    if checkpoint.exists():
        # An interrupted evaluation can leave a complete immutable checkpoint.
        sidecar=checkpoint.with_suffix('.json')
        if not sidecar.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if (saved.get('schema')!='PANDA_SENSOR_EQ40_EVALUATED_MODEL_V1'
                    or saved.get('config')!=trainer.cfg or saved.get('pins')!=pins
                    or saved.get('updates')!=step or saved.get('model_sha256')!=record['model_sha256']):
                raise ValueError('Interrupted immutable checkpoint is not this evaluated state')
            atomic_json(sidecar,{'path':str(checkpoint),'file_sha256':file_sha(checkpoint),
                'step':step,'model_sha256':record['model_sha256'],'config_sha256':trainer.cfg['config_sha256']})
        meta=json.loads(sidecar.read_text())
        if meta['model_sha256']!=record['model_sha256'] or file_sha(checkpoint)!=meta['file_sha256']:
            raise ValueError('Immutable checkpoint collision')
    else:meta=training.save(checkpoint,trainer,stream,pins,immutable=True,evaluated=True)
    record.update(checkpoint=meta,recorded_at=utc_now(),
        inference_parameters=sum(p.numel() for role in ('aligner','student' if trainer.cfg['stage']=='stage2' else 'teacher')
                                 for p in getattr(trainer.model,role).parameters()))
    evaluation.verify_record(record,trainer.cfg)
    atomic_json(evpath,record)
    if step in (trainer.cfg['primary_step'],trainer.cfg['horizon'],trainer.cfg.get('common_comparison_step')):
        atomic_json(directory/f'fixed_step_{step:07d}.json',{'config':trainer.cfg,'evaluation_path':str(evpath),
            'evaluation_sha256':file_sha(evpath),'checkpoint':meta,'rr':record['rr']['metrics'],
            'fr':record['fr']['metrics'],'fr_corrected_pan':record['fr_corrected_pan']['metrics'],'full_reference_case':'SENSOR_EQ40',
            'full_reference_repeats':2,'historical_full_reference_used':False})
    print(f"{trainer.cfg['experiment']} {step}: HQNR={record['fr']['metrics']['HQNR']:.6f} "
          f"ERGAS={record['rr']['metrics']['ERGAS']:.6f}",flush=True)
    return record


def run(server,*,device='cuda:0',publish=True):
    from .release import verify
    release=verify(environment=True)
    configure_precision()
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('Production training requires an explicit available CUDA device')
    root=plan.campaign_root(server);root.mkdir(parents=True,exist_ok=True)
    # Process-local duplicate writer protection, unrelated to donor sharing or server handoffs.
    with (root/'runner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This local campaign already has a writer')
        state=read_state(root,server)
        if next_job(state) is None:
            state['status']='FINISHED_WITH_FAILURES' if any(r['status'] in ('FAILED','BLOCKED_DONOR') for r in state['runs']) else 'COMPLETED'
            atomic_json(root/'campaign_state.json',state)
            if publish:publish_safely(root,server)
            return state
        sources,registered=assets.sources(server)
        pins={'source_release':release['source_sha256'],'evaluator':evaluator_identity(),
              'train_inventory':registered.inventory.manifest(),
              'split_manifests':{s:sources.manifest_sha(s) for s in ('train','val','rr','fr')}}
        admission=root/'admission.json'
        if admission.exists() and json.loads(admission.read_text())!=pins:
            raise ValueError('Admission code/data/environment changed; no silent resume')
        atomic_json(admission,pins)
        from .selection import validation_inventory,validate_inventory
        inventories={}
        for mode in sorted({c['reference_mode'] for c in plan.queue(server) if c['stage']=='stage1'}):
            inventory=validation_inventory(sources,reference_mode=mode)
            validate_inventory(inventory,sources,reference_mode=mode)
            invpath=root/f'validation_inventory_{mode}.json'
            if invpath.exists() and json.loads(invpath.read_text())!=inventory:
                raise ValueError('Validation inventory changed')
            atomic_json(invpath,inventory);inventories[mode]=inventory
        stop={'requested':False}
        def signal_stop(signum,frame):stop['requested']=True
        previous={s:signal.signal(s,signal_stop) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            if publish:
                # Missing credentials/header conflict is discovered before consuming training time.
                from .publication import publish as publish_once
                publish_once(root,server,apply=False)
            while (job:=next_job(state)) is not None:
                if stop['requested']:break
                cfg=plan.resolve(server,job['case'],job['repeat']);directory=root/'runs'/cfg['run_id']
                inventory=inventories.get(cfg['reference_mode'])
                if block_failed_dependency(state,job,cfg):
                    atomic_json(root/'campaign_state.json',state)
                    if publish:publish_safely(root,server)
                    continue
                directory.mkdir(parents=True,exist_ok=True)
                path=directory/'config.json'
                if path.exists() and json.loads(path.read_text())!=cfg:raise ValueError('Immutable run config changed')
                atomic_json(path,cfg)
                try:
                    if cfg['stage']=='stage1':
                        model=Stage1(cfg,registered.inventory.manifest()).to(device)
                    else:
                        donor,stats=assets.prepare_reference(cfg,registered,root,pins,device=device)
                        model=Stage2(cfg,donor,stats).to(device)
                        model.selection_receipt=deepcopy(donor.selection_receipt)
                        del donor,stats
                except Exception as exc:
                    if cfg['stage']=='stage2' and isinstance(exc,ValueError) and str(exc).startswith('NO_QUALIFIED_DONOR'):
                        job.update(status='BLOCKED_DONOR',error=str(exc),updated_at=utc_now())
                        atomic_json(directory/'failure.json',dict(job))
                        atomic_json(root/'campaign_state.json',state)
                        if publish:publish_safely(root,server)
                        continue
                    job.update(status='ERROR',error=repr(exc),updated_at=utc_now())
                    state['status']='ERROR';atomic_json(root/'campaign_state.json',state)
                    atomic_json(directory/'failure.json',dict(job))
                    raise
                trainer=training.Trainer(cfg,model);stream=training.stream_for(cfg,registered,device)
                resume=latest_checkpoint(directory)
                if resume is None and any((directory/'models').glob('step_*.pt')):
                    raise ValueError('Evaluated models exist but complete resume generations are missing')
                if resume:training.load(resume,trainer,stream,pins)
                rewind_trace(directory,trainer.updates)
                job.update(status='RUNNING',step=trainer.updates,updated_at=utc_now())
                state['status']='RUNNING';atomic_json(root/'campaign_state.json',state)
                if publish:publish_safely(root,server)
                try:
                    # Complete an interrupted boundary evaluation before advancing the optimizer.
                    if trainer.updates in cfg['evaluation_steps']:
                        finish_evaluation(trainer,stream,directory,sources,pins,inventory)
                    stopped_early=None
                    while trainer.updates<cfg['horizon']:
                        if stop['requested']:break
                        tick=time.monotonic();terms=trainer.step(stream.next_update())
                        append_jsonl(directory/'train.jsonl',dict(terms,seconds=time.monotonic()-tick,at=utc_now()))
                        if trainer.updates%100==0:
                            job.update(step=trainer.updates,updated_at=utc_now())
                            atomic_json(root/'campaign_state.json',state)
                            print(cfg['experiment'],trainer.updates,'loss',terms['loss'],flush=True)
                        if trainer.updates%250==0:save_resume(directory,trainer,stream,pins)
                        if trainer.updates in cfg['evaluation_steps']:
                            verify(environment=False)
                            finish_evaluation(trainer,stream,directory,sources,pins,inventory)
                            if publish:publish_safely(root,server)
                            stopped_early=early_stop_evidence(cfg,directory,trainer.updates)
                            if stopped_early:
                                atomic_json(directory/'early_stop.json',stopped_early)
                                break
                    save_resume(directory,trainer,stream,pins)
                    job.update(step=trainer.updates,status='EARLY_STOPPED' if stopped_early else 'COMPLETED' if trainer.updates==cfg['horizon'] else 'PAUSED',updated_at=utc_now())
                except Exception as exc:
                    independent_failure=(isinstance(exc,FloatingPointError) or 'SUPPORT_VIOLATION' in str(exc))
                    # Infrastructure/evidence errors remain retryable at this run after repair.
                    # Only an explicitly recorded numerical/support failure advances the queue.
                    job.update(status='FAILED' if independent_failure else 'ERROR',step=trainer.updates,
                               error=repr(exc),updated_at=utc_now())
                    atomic_json(directory/'failure.json',dict(job))
                    atomic_json(root/'campaign_state.json',state)
                    if publish:publish_safely(root,server)
                    if not independent_failure:
                        state['status']='ERROR';atomic_json(root/'campaign_state.json',state);raise
                atomic_json(root/'campaign_state.json',state)
                if publish:publish_safely(root,server)
                del trainer,stream,model;gc.collect();torch.cuda.empty_cache()
                if stop['requested']:break
            state['status']=('PAUSED' if next_job(state) is not None else
                'FINISHED_WITH_FAILURES' if any(r['status'] in ('FAILED','BLOCKED_DONOR') for r in state['runs']) else 'COMPLETED')
            atomic_json(root/'campaign_state.json',state)
            if publish:publish_safely(root,server)
            return state
        finally:
            for s,handler in previous.items():signal.signal(s,handler)
