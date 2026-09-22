"""Unlimited independent cycles: finite50K runs, safe cursor, durable reports."""
import os
from pathlib import Path
import subprocess
import sys
import time
from pcrepro.common import (ROOT,atomic_json,read,read_json,read_config,camp,run_dir,locked,
    immutable_json,object_sha,source_identity,apply_runtime_policy,utcnow,append_event,machine_lock)
from pcrepro.plan import (CAMPAIGN_ID,RECIPE_ID,RECIPE_SHA256,RECIPE,Case,DATASETS,SERVERS,
    case_for,cases_for,build_config,validate_config,verify_server,verify_sources)

CONTROLS=('STOP_NOW_SAFE','STOP_AFTER_CURRENT_RUN','STOP_AFTER_CYCLE','CONTINUE')
def _state(root,s):
    state=read(camp(root,s)/'status.json',dict(campaign_id=CAMPAIGN_ID,server=s,
        cycle=0,index=0,status='DEFINED_NOT_STARTED',runs={},blocked_datasets={},failure_counts={}))
    if (state.get('campaign_id')!=CAMPAIGN_ID or state.get('server')!=s
            or type(state.get('cycle')) is not int or state['cycle']<0
            or type(state.get('index')) is not int or not 0<=state['index']<=4):
        raise ValueError('Local cursor campaign/server/cycle/index differs')
    queue=cases_for(s,state['cycle']);active=state.get('active_run')
    if active and (state['index']==len(queue) or active!=queue[state['index']].run_id):
        raise ValueError('Local active run differs from deterministic cursor')
    return state
def _save(root,s,state):state['updated_at_utc']=utcnow();atomic_json(camp(root,s)/'status.json',state)
def status(root,s):return dict(state=_state(root,s),registration=read(camp(root,s)/'registration.json'),
    control=read(camp(root,s)/'control.json'),handoff=read(camp(root,s)/'handoff/transition.json'))
def control(root,s,command):
    verify_server(s)
    if command not in CONTROLS:raise ValueError('Unknown safe control')
    value=dict(command=command,at_utc=utcnow());atomic_json(camp(root,s)/'control.json',value)
    append_event(camp(root,s)/'control_ledger.jsonl','CONTROL',**value);return value
def register(root,s,bindings,upload=True):
    verify_server(s);verify_sources(root);apply_runtime_policy(root)
    path=Path(bindings).resolve(strict=True)
    value=dict(campaign_id=CAMPAIGN_ID,server=s,recipe_sha256=RECIPE_SHA256,
        source_identity=source_identity(root),bindings_path=str(path),bindings_sha256=object_sha(read_json(path)),upload=bool(upload))
    immutable_json(camp(root,s)/'registration.json',value);return value
def verify_registration(root,s):
    value=read_json(camp(root,s)/'registration.json');apply_runtime_policy(root)
    if (value.get('campaign_id')!=CAMPAIGN_ID or value.get('server')!=s
        or value.get('source_identity')!=source_identity(root) or value.get('recipe_sha256')!=RECIPE_SHA256
        or value.get('bindings_sha256')!=object_sha(read_json(value['bindings_path']))):
        raise ValueError('Registered source/data bindings changed; never mutate a live campaign')
    return value
def authorize_train(root,s,path):
    verify_registration(root,s);cfg=read_config(path);case=validate_config(cfg,require_bound=True)
    state=_state(root,s);ready=read_json(camp(root,s)/'preflight'/f'{case.dataset}.json')
    if (case.server!=s or state.get('active_run')!=case.run_id or state.get('status')!='TRAINING'
        or cases_for(s,state['cycle'])[state['index']]!=case or case.stage!='TRAIN'
        or not ready.get('complete') or ready['source_identity']!=source_identity(root)
        or ready['data_sha']!=object_sha(read_json(cfg['pcrepro']['data_manifest']))
        or not read(camp(root,s)/'handoff/transition.json').get('complete')):
        raise ValueError('Local controller has not authorized this exact run')
    return case
def build(root=ROOT,server=None,cycle=0):
    verify_sources(root);result=[]
    for s in (SERVERS if server is None else (server,)):
        rows=cases_for(s,cycle)
        path=Path(root)/'config/pcrepro'/f'{s}_C{cycle:06d}.json'
        immutable_json(path,dict(campaign_id=CAMPAIGN_ID,recipe_sha256=RECIPE_SHA256,cases=[c.to_dict() for c in rows],
            finite_export_only=True,campaign_max_cycles=None))
        result.extend(c.to_dict() for c in rows)
    return dict(cases=result,training_started=False,clock_created=False)

def _command(root,s,args,log):
    Path(log).parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    with Path(log).open('a') as output:
        process=subprocess.Popen([sys.executable,'-u','tools/pcrepro_runner.py',*args,'--server',s,'--root',str(root)],cwd=root,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=output,stderr=subprocess.STDOUT)
        signaled=False
        try:
            while True:
                try:return process.wait(timeout=5),time.monotonic()-started
                except subprocess.TimeoutExpired:
                    # Training/evaluation observe the same atomic local control;
                    # signals to controller itself are handled by the CLI.
                    if not signaled and read(camp(root,s)/'control.json').get('command')=='STOP_NOW_SAFE':
                        if process.poll() is None:process.terminate()
                        signaled=True
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=60)
                except subprocess.TimeoutExpired:raise RuntimeError('Own child still preserving; no forced kill or replacement')
            raise

def _status_row(root,c,state,status,reason='',**extra):
    wd=run_dir(c,root);old=read(wd/'meta/status.json')
    row=dict(old,campaign_id=CAMPAIGN_ID,run_id=c.run_id,recipe_id=RECIPE_ID,case=c.to_dict(),
        status=status,actual_updates=read(wd/'meta/training_status.json').get('actual_updates',0),
        reason=reason,source_identity=source_identity(root),
        started_at_utc=read(wd/'meta/training_start_manifest.json').get('started_at_utc',old.get('started_at_utc')),
        completed_at_utc=utcnow(),**extra)
    atomic_json(wd/'meta/status.json',row);state['runs'][c.run_id]=dict(status=status,reason=reason,**extra)
    _save(root,c.server,state)
    try:
        from pcrepro.upload import spool_run
        spool_run(c.run_id,root)
    except (ValueError,OSError,KeyError) as exc:
        atomic_json(wd/'official/upload_status.json',dict(status='UPLOAD_PENDING',reason=str(exc)))
    return row

def _config(root,c):
    data_path=camp(root,c.server)/'datasets'/f'{c.dataset}.json';data=read_json(data_path)
    cfg=build_config(c,str(data_path),object_sha(data),source_identity(root))
    return immutable_json(run_dir(c,root)/'meta/config.resolved.yaml',cfg)

def _case(root,s,state,c,binding):
    folder=camp(root,s);wd=run_dir(c,root)
    if c.dataset in state['blocked_datasets']:
        _status_row(root,c,state,'BLOCKED_DATASET',state['blocked_datasets'][c.dataset]);return 'TERMINAL'
    if c.source_run_id:
        tr=read(run_dir(c.source_run_id,root)/'meta/training_status.json')
        if not tr.get('training_complete') or tr.get('actual_updates')!=50000:
            _status_row(root,c,state,'BLOCKED_DEPENDENCY','Same-server same-cycle WV3 fresh50K failed or incomplete');return 'TERMINAL'
    state.update(status='PREFLIGHT',active_run=c.run_id);_save(root,s,state)
    code,seconds=_command(root,s,['preflight','--dataset',c.dataset,'--bindings',binding],folder/'logs'/f'{c.run_id}.preflight.log')
    if code==75:return 'PAUSE'
    if code:
        reason=read(folder/'preflight'/f'{c.dataset}.failure.json').get('reason','See preflight log')
        if 'INSUFFICIENT_DISK' in reason:return 'PAUSE_DISK'
        state['blocked_datasets'][c.dataset]=reason;_status_row(root,c,state,'BLOCKED_DATASET',reason);return 'TERMINAL'
    path=_config(root,c);attempt=state['runs'].setdefault(c.run_id,{})
    if c.stage=='TRAIN' and not read(wd/'meta/training_status.json').get('training_complete'):
        state.update(status='TRAINING',active_run=c.run_id);_save(root,s,state)
        while True:
            resume=(wd/'resume/index.json').is_file()
            code,elapsed=_command(root,s,['train','--config',str(path)]+(['--resume'] if resume else []),folder/'logs'/f'{c.run_id}.train.log')
            attempt['training_wall_seconds']=attempt.get('training_wall_seconds',0)+elapsed
            tr=read(wd/'meta/training_status.json')
            if tr.get('training_complete') and tr.get('actual_updates')==50000:break
            if code==75 or tr.get('status','').startswith('PAUSED'):return 'PAUSE'
            if code==74 and attempt.get('io_retries',0)<2 and (wd/'resume/index.json').is_file():
                attempt['io_retries']=attempt.get('io_retries',0)+1;_save(root,s,state);continue
            why=tr.get('status','FAILED_TRAINING')
            n=state['failure_counts'].get(c.dataset,{});count=n.get('count',0)+1 if n.get('reason')==why else 1
            state['failure_counts'][c.dataset]=dict(reason=why,count=count)
            if count>=2 or code in (1,74):state['blocked_datasets'][c.dataset]=why
            _status_row(root,c,state,why,'Training failed; recipe unchanged',training_complete=False);return 'TERMINAL'
    state.update(status='POSTRUN',active_run=c.run_id);_save(root,s,state)
    for attempt_no in range(3):
        code,_=_command(root,s,['postrun','--config',str(path)],folder/'logs'/f'{c.run_id}.postrun.log')
        if code!=74:break
    if code==75:return 'PAUSE'
    result=read(wd/'official/summary.json')
    if code or not result.get('complete'):
        _status_row(root,c,state,'EVAL_PENDING','Evaluation failed; checkpoints/cursor preserved',
            training_complete=c.stage=='TRAIN',postrun_pending=True)
        state.setdefault('pending_evaluations',{})[c.run_id]=dict(config=str(path),attempts=0)
    else:
        state['runs'][c.run_id]=dict(status=result['status'],complete=True,training_complete=c.stage=='TRAIN')
        from pcrepro.upload import spool_run
        try:spool_run(c.run_id,root)
        except (ValueError,OSError,KeyError) as exc:
            atomic_json(wd/'official/upload_status.json',dict(status='UPLOAD_PENDING',reason=str(exc)))
        state.setdefault('pending_evaluations',{}).pop(c.run_id,None)
    _save(root,s,state);return 'TERMINAL'

def retry_evaluations(root,s,state,limit=1):
    """One bounded old evaluation per cycle; no training or checkpoint reselection."""
    pending=state.setdefault('pending_evaluations',{})
    for run in list(pending)[:limit]:
        if read(camp(root,s)/'control.json').get('command')=='STOP_NOW_SAFE':return 75
        item=pending[run];c=case_for(run)
        if c.server!=s:raise ValueError('Cross-server evaluation debt')
        code,_=_command(root,s,['postrun','--config',item['config']],camp(root,s)/'logs'/f'{run}.postrun.log')
        item['attempts']+=1
        if code==75:_save(root,s,state);return 75
        if code==0 and read(run_dir(c,root)/'official/summary.json').get('complete'):
            _status_row(root,c,state,'COMPLETE','Deferred evaluation recovered')
            pending.pop(run)
        elif code not in (74,75):
            # A deterministic failure is not retried forever on every cycle.
            state.setdefault('blocked_evaluations',{})[run]=dict(item,reason='Postrun deterministic failure; explicit retry required')
            pending.pop(run)
        else:
            # Rotate temporary failures so one item cannot starve the others.
            pending.pop(run);pending[run]=item
        _save(root,s,state)
    return 0

def finish_cycle(root,s,state):
    """Idempotent recovery for a crash after the last case cursor was saved."""
    if state['index']!=len(cases_for(s,state['cycle'])):return False
    from pcrepro.reporting import cycle_report
    cycle_report(root,s,state['cycle'])
    state.update(completed_cycle=state['cycle'],cycle=state['cycle']+1,index=0,runs={},evaluation_retry_due=True)
    _save(root,s,state);return True

def run(root=ROOT,server='s3'):
    verify_server(server);root=Path(root).resolve();folder=camp(root,server);s=server
    with locked(machine_lock()),locked(folder/'runner.lock'):
        state=_state(root,s);registration=verify_registration(root,s)
        try:
            if read(folder/'control.json').get('command')=='STOP_NOW_SAFE':
                state['status']='PAUSED_SAFE';_save(root,s,state);return 75
            from pcrepro.handoff import begin,verify_stopped,gpu_processes,inventory
            release=read(folder/'runtime_release.json')
            begin(root,s,activated=True,origin_root=release.get('origin_root'))
            while not verify_stopped(root,s)['complete']:
                state['status']='WAIT_SAFE_LEGACY_EXIT';_save(root,s,state)
                if read(folder/'control.json').get('command')=='STOP_NOW_SAFE':return 75
                time.sleep(5)
            while True:
                finish_cycle(root,s,state)
                command=read(folder/'control.json').get('command','CONTINUE')
                if command=='STOP_NOW_SAFE' or command=='STOP_AFTER_CURRENT_RUN' and not state.get('active_run'):
                    state['status']='PAUSED_SAFE';_save(root,s,state);return 75
                if command=='STOP_AFTER_CYCLE' and state['index']==0 and state.get('completed_cycle') is not None:
                    state['status']='PAUSED_AFTER_CYCLE';_save(root,s,state);return 75
                if set(state['blocked_datasets'])>=set(DATASETS)-{'WV2'}:
                    state['status']='PAUSED_NO_RUNNABLE_DATASET';_save(root,s,state);return 75
                cycle=state['cycle'];queue=cases_for(s,cycle)
                immutable_json(folder/'cycles'/f'C{cycle:06d}.json',dict(cycle=cycle,cases=[c.to_dict() for c in queue],recipe_sha256=RECIPE_SHA256))
                c=queue[state['index']]
                from pcrepro.preflight import disk_guard
                disk=disk_guard(root)
                if not disk['allowed']:state.update(status='PAUSED_DISK',disk=disk);_save(root,s,state);return 75
                if gpu_processes()!=[] or any(r['script']!='pcrepro_runner.py' for r in inventory(root)):
                    state['status']='PAUSED_GPU_BUSY';_save(root,s,state);return 75
                if state.pop('evaluation_retry_due',False):
                    if retry_evaluations(root,s,state)==75:
                        state['status']='PAUSED_SAFE';_save(root,s,state);return 75
                outcome=_case(root,s,state,c,registration['bindings_path'])
                if outcome.startswith('PAUSE'):state['status']=outcome;_save(root,s,state);return 75
                state.update(active_run=None,index=state['index']+1);_save(root,s,state)
                if registration['upload']:
                    from pcrepro.upload import retry_pending
                    retry_pending(root,s,activated=True)
                if state['index']==len(queue):
                    finish_cycle(root,s,state)
        except (ValueError,OSError,RuntimeError) as exc:
            state.update(status='BLOCKED_INTEGRITY',reason=f'{type(exc).__name__}: {exc}');_save(root,s,state);raise

def start(root=ROOT,server='s3',bindings=None,foreground=False,upload=True):
    folder=camp(root,server)
    if bindings is None:bindings=read(folder/'registration.json').get('bindings_path')
    if not bindings:raise ValueError('Supply --bindings local_manifest.json on first start')
    register(root,server,bindings,upload)
    if foreground:return dict(exit_code=run(root,server))
    from pcrepro.deployment import process_start
    with locked(folder/'startup.lock'):
        try:
            with locked(folder/'runner.lock'):pass
        except BlockingIOError:return dict(status='ALREADY_RUNNING')
        old=read(folder/'runner.pid.json')
        if old.get('ticks') and process_start(old.get('pid'))==old['ticks']:return dict(status='ALREADY_RUNNING',pid=old['pid'])
        with (folder/'runner.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u','tools/pcrepro_runner.py','run','--server',server],cwd=root,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(folder/'runner.pid.json',dict(pid=child.pid,ticks=process_start(child.pid)))
    return dict(status='SUBMITTED',pid=child.pid,log=str(folder/'runner.log'),unlimited_cycles=True)
