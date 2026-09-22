"""Independent server-local P40 DAGs; no cross-server selection or transport."""
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import time
from gfp40.common import (ROOT,CAMPAIGN_ID,apply_runtime_policy,atomic_json,camp,immutable_json,locked,
    object_sha,read,read_json,read_config,run_dir,sha256,source_identity,utcnow,append_event)
from gfp40.plan import (SERVERS,PARENTS,SETUP_HOURS,INITIAL_FAMILIES,DIAGNOSTIC_HOURS,blocks_for,block_for,
    EXECUTION_POLICY,EXECUTION_POLICY_SHA256,
    build_config,case_for,cases_for,registry_sha256,registry_dict,validate_config,verify_lane,verify_sources,grid_steps,family_for)
from gfp40.policy import CampaignWindow,admission,estimate_block,trim_unstarted,utc

def window_path(root,server):return camp(root,server)/'campaign_window.json'
def write_window(root,t0,server):
    value=CampaignWindow(t0,server).to_dict()
    with locked(window_path(root,server).with_suffix('.lock')):immutable_json(window_path(root,server),value)
    return value
def local_window(root,server,initialize=False):
    path=window_path(root,server)
    if not path.is_file() and initialize:
        if (Path(root)/'work_dir/_gfp40/campaign_window.json').exists():
            raise ValueError('Obsolete shared-clock campaign exists; do not migrate a live run implicitly')
        with locked(path.with_suffix('.lock')):
            if not path.is_file():immutable_json(path,CampaignWindow(utcnow(),server).to_dict())
    result=CampaignWindow.from_dict(read_json(path))
    if result.server!=server:raise ValueError('Another server owns this local campaign clock')
    return result
def build(root=ROOT):
    verify_sources(root);folder=Path(root)/'config/gfp40'
    for s in SERVERS:
        for c in cases_for(s):immutable_json(folder/(c.run_id+'.yaml'),build_config(c))
        immutable_json(folder/(s+'_queue.json'),[b.to_dict() for b in blocks_for(s)])
    immutable_json(folder/'derived_registry.json',registry_dict())
    return dict(cases=83,primary=74,clock_created=False,training_started=False)
def _state(root,s):return read(camp(root,s)/'status.json',dict(campaign_id=CAMPAIGN_ID,server=s,
    status='DEFINED_NOT_LAUNCHED',runs={},blocks={},observations=[],family_manifests={}))
def _save(root,s,state):state['updated_at_utc']=utcnow();atomic_json(camp(root,s)/'status.json',state)
def control(root,s,command):
    if command not in ('STOP_NOW_SAFE','STOP_AFTER_BLOCK','CONTINUE'):raise ValueError('Unknown control')
    value=dict(command=command,at_utc=utcnow());atomic_json(camp(root,s)/'control.json',value)
    append_event(camp(root,s)/'control_ledger.jsonl','OPERATOR_CONTROL',**value);return value
def status(root,s):return dict(state=_state(root,s),window=read(window_path(root,s)),
    execution_policy=EXECUTION_POLICY,preflight=read(camp(root,s)/'preflight.json'))
def register_runtime(root,s):
    verify_lane(s);verify_sources(root);apply_runtime_policy(root)
    folder=camp(root,s)
    value=dict(campaign_id=CAMPAIGN_ID,server=s,source_identity=source_identity(root),
        registry_sha256=registry_sha256(),window=local_window(root,s).to_dict(),
        execution_policy_sha256=EXECUTION_POLICY_SHA256)
    immutable_json(folder/'runtime_binding.json',value);return value
def verify_registration(root,s):
    saved=read_json(camp(root,s)/'runtime_binding.json');apply_runtime_policy(root)
    current=dict(campaign_id=CAMPAIGN_ID,server=s,source_identity=source_identity(root),
        registry_sha256=registry_sha256(),window=local_window(root,s).to_dict(),
        execution_policy_sha256=EXECUTION_POLICY_SHA256)
    if current!=saved:raise ValueError('Registered P40 source/runtime/window changed')
    return current

def _command(root,s,args,log,deadline):
    """Poll clock/control, terminate only our own cooperative preparation child."""
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    if utc(utcnow())>=utc(deadline):return 75,0.
    start=time.monotonic();code=None
    with Path(log).open('a') as stream:
        child=subprocess.Popen([sys.executable,'-u','tools/gfp40_runner.py',*args,'--server',s,'--deadline',deadline],
            cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT)
        try:
            while code is None:
                try:code=child.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
                stop=read(camp(root,s)/'control.json').get('command')=='STOP_NOW_SAFE'
                expired=utc(utcnow())>=utc(deadline)+dt.timedelta(seconds=30)
                if code is None and (expired or stop and args[0]!='train'):
                    child.terminate()
                    try:child.wait(timeout=60)
                    except subprocess.TimeoutExpired:raise RuntimeError('Own child still preserving state; no forced kill/new work')
                    code=75
        except BaseException:
            # No second process may run on this GPU after an orchestration error.
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:pass
            raise
    return code,(time.monotonic()-start)/3600.

def prepare_family(root,s,family,deadline,bindings=None):
    from gfp40.assets import load_reference
    from gfp40.calibration import prepare_family_reference
    from g20.data import build_dataset
    folder=camp(root,s)
    teacher,q,ref=load_reference(folder/'references'/f'R{s[1]}_100.json','cuda',root,deadline=deadline)
    data=build_dataset(read_json(folder/'dataset_manifest.json'),'train',root=root)
    supplied=read_json(bindings) if bindings else {}
    old=supplied.get('b20_reference')
    if old is None:
        candidate=Path(root)/'work_dir/_gfb20'/s/'mixed/mixed_reference_manifest.json'
        if candidate.is_file():old=str(candidate)
    archive=read(folder/'b20_source_audit.json').get('archive_root')
    path=prepare_family_reference(teacher,data,ref,folder/'banks',family=family,device='cuda',
        deadline_utc=deadline,source_identity=source_identity(root),b20_reference=old,b20_source_root=archive)
    receipt=dict(family=family,manifest=str(path),manifest_sha256=sha256(path))
    immutable_json(folder/'families'/f'{family}.json',receipt)
    return receipt

def resolve_config(root,case):
    folder=camp(root,case.server)
    cfg=build_config(case);f=cfg['gfp40']
    f.update(dataset_manifest=str(folder/'dataset_manifest.json'),window_path=str(window_path(root,case.server)),
        runtime_policy_sha256=sha256(Path(root)/'gfp40/runtime_policy.json'),
        source_archive_sha256=read_json(folder/'b20_source_audit.json')['source_archive_sha256'])
    if case.is_ft:
        parentpath=folder/'parents'/f'{case.parent_id}.json'
        if case.parent_id not in PARENTS:
            from gfp40.assets import bind_fresh_parent
            parent=bind_fresh_parent(case.parent_id,case.server,run_dir(case.parent_id,root)/'meta/config.resolved.yaml',root)
            immutable_json(parentpath,parent)
        f['parent_manifest']=str(parentpath)
    refpath=folder/'references'/f'{case.reference}.json';ref=read_json(refpath)
    native,paths=ref['native_manifest'],ref['resolved_artifacts']
    f.update(reference_manifest=str(refpath),teacher_checkpoint=paths['teacher_checkpoint'],
        teacher_sha256=native['teacher_checkpoint_sha256'],tau_R=native['tau_R'],q_ref=native['q_ref'],
        q_cache=paths['q_cache_path'],q_cache_sha256=native['q_cache_sha256'])
    if case.arm!='NATIVE0':
        from gfp40.calibration import load_family_reference
        receipt=read_json(folder/'families'/f'{f["family"]}.json')
        if sha256(receipt['manifest'])!=receipt['manifest_sha256']:raise ValueError('Family manifest changed')
        mixed,_=load_family_reference(receipt['manifest'],native_reference=dict(native,**paths,native_manifest=native))
        f.update(mixed_calibration_manifest=receipt['manifest'],augmentation_manifest=mixed['augmentation_manifest_path'])
    data=read_json(f['dataset_manifest'])
    for split,key in (('train','train_feeder_args'),('val','val_feeder_args'),('rr','test_reduced_feeder_args'),('fr','test_full_feeder_args')):
        cfg[key]['dataroot']=data['splits'][split]['dataroot']
    validate_config(cfg,require_bound=True)
    return immutable_json(run_dir(case,root)/'meta/config.resolved.yaml',cfg)

def authorize_train(root,s,config_path):
    verify_registration(root,s);cfg=read_config(config_path);case=validate_config(cfg,require_bound=True)
    state=_state(root,s);ready=read_json(camp(root,s)/'preflight.json');window=local_window(root,s)
    if utc(utcnow())>=window.train_finish_utc:raise ValueError('P40 optimizer window is closed')
    if (case.server!=s or state.get('active_run')!=case.run_id or state.get('status')!='TRAINING'
        or not ready.get('complete') or ready['source_identity']!=source_identity(root)
        or ready['dataset_manifest_sha256']!=object_sha(read_json(cfg['gfp40']['dataset_manifest']))):
        raise ValueError('Current controller/local preflight has not authorized this case')
    block=block_for(case.block_id);receipt=read_json(camp(root,s)/'admissions'/f'{case.block_id}.json')
    admission(window,receipt['at_utc'],block,receipt=receipt,diagnostic_hours=0.)
    for d in case.depends_on:
        if not state['runs'].get(case_for(d).run_id,{}).get('complete'):raise ValueError('Parent not officially complete')
    previous=block.cases[:block.cases.index(case)]
    if any(not state['runs'].get(c.run_id,{}).get('terminal') for c in previous):raise ValueError('Previous arm not terminal')
    return case

def evaluation_debt(root,s):
    total=0.
    for c in cases_for(s):
        wd=run_dir(c,root)
        if read(wd/'official/summary.json').get('complete'):continue
        records=read(wd/'official/raw_grid.json').get('records',[]);seen={r['update'] for r in records}
        n=sum(t not in seen and (wd/f'candidates/{t}/identity.json').is_file() for t in grid_steps(c.updates))
        total+=n*max([r.get('seconds',0)/3600 for r in records]+[.02])*1.15
    return total
def retry_uploads(root,s):
    from gfp40.upload import upload_run
    result={}
    for c in cases_for(s):
        if utc(utcnow())>=local_window(root,s).deadline_utc:break
        wd=run_dir(c,root)
        if not read(wd/'official/summary.json').get('complete') or read(wd/'official/upload_receipt.json').get('readback_verified'):continue
        try:result[c.run_id]=upload_run(c.run_id,root,activated=True)
        except Exception as exc:
            result[c.run_id]=dict(status='UPLOAD_PENDING',reason=f'{type(exc).__name__}: {exc}',at_utc=utcnow())
            atomic_json(wd/'official/upload_status.json',result[c.run_id])
    return result

def _run_case(root,s,state,case,window,upload):
    path=resolve_config(root,case);wd=run_dir(case,root);row=state['runs'].setdefault(case.run_id,{})
    saved=read(wd/'meta/training_status.json')
    if saved.get('training_complete'):
        if saved.get('actual_updates')!=case.updates:raise ValueError('Completed horizon differs')
        row['training_complete']=True
    if not row.get('training_complete'):
        from gfp40.training import recover_exact_endpoint
        state.update(status='TRAINING',active_run=case.run_id);_save(root,s,state)
        if not recover_exact_endpoint(root,wd):
            while True:
                resume=(wd/'last/training_state.pt').is_file()
                args=['train','--config',str(path)]+(['--resume'] if resume else [])
                code,hours=_command(root,s,args,camp(root,s)/'logs'/f'{case.run_id}.train.log',window.deadline_utc.isoformat())
                row['wall_hours']=row.get('wall_hours',0.)+hours
                append_event(camp(root,s)/'attempts.jsonl','TRAIN_EXIT',run_id=case.run_id,code=code,hours=hours)
                if code==74 and row.get('retries',0)<2 and (wd/'last/identity.json').is_file():
                    row['retries']=row.get('retries',0)+1;_save(root,s,state);continue
                break
        else:code=0
        saved=read(wd/'meta/training_status.json')
        row.update(training_complete=saved.get('training_complete',False),actual_updates=saved.get('actual_updates',0))
        if not row['training_complete']:
            failure=saved.get('status','PARTIAL_TIME_LIMIT' if code==75 else 'BLOCKED_INTEGRITY')
            paused=failure in ('PAUSED_CONTROL','PAUSED_SIGNAL','PAUSED_DIVERGENCE','ABORTED_VAL_DIVERGENCE')
            row.update(status=failure,terminal=not paused)
            if paused:state.update(status='PAUSED_SAFE',reason=failure)
            _save(root,s,state);return 'PAUSED_SAFE' if paused else False
    state['status']='POSTRUN';_save(root,s,state)
    code,hours=_command(root,s,['postrun','--config',str(path)],camp(root,s)/'logs'/f'{case.run_id}.postrun.log',window.deadline_utc.isoformat())
    row['wall_hours']=row.get('wall_hours',0.)+hours
    summary=read(wd/'official/summary.json')
    row.update(complete=summary.get('complete',False),terminal=True,
        status='OFFICIAL_EVAL_COMPLETE' if summary.get('complete') else 'EVAL_PENDING')
    if row['complete']:
        state['observations'].append(dict(server=s,updates=case.updates,arm=case.arm,family=case.calibration_family,
            complete=True,wall_hours=row['wall_hours'],includes_eval_io_diagnostics=True))
        if upload:retry_uploads(root,s)
    _save(root,s,state);return True

def ready_case(block,state):
    for c in block.cases:
        row=state['runs'].get(c.run_id,{})
        if row.get('terminal'):continue
        if any(not state['runs'].get(case_for(d).run_id,{}).get('complete') for d in c.depends_on):return None
        return c
    return None

def _done(block,state):return [c.run_id for c in block.cases if state['runs'].get(c.run_id,{}).get('terminal')]
def outstanding(root,s,state,exclude=None):
    return sum(estimate_block(b,state['observations'],_done(b,state)) for b in blocks_for(s)
        if b.block_id!=exclude and state['blocks'].get(b.block_id,{}).get('admitted')
        and not state['blocks'][b.block_id].get('terminal'))
def pending_cache_hours(root,s,state):
    folder=camp(root,s);available=set();ready=set()
    for family in INITIAL_FAMILIES[s]:
        receipt=read(folder/'families'/f'{family}.json')
        if not receipt:continue
        if sha256(receipt['manifest'])!=receipt['manifest_sha256']:raise ValueError('Published family receipt changed')
        manifest=read_json(receipt['manifest'])
        if manifest.get('family')!=family:raise ValueError('Wrong family receipt')
        ready.add(family);available.update(manifest['gammas'])
    needed={c.calibration_family for c in cases_for(s) if c.arm!='NATIVE0'
        and not state['blocks'].get(c.block_id,{}).get('terminal')}
    missing=needed-ready;gammas=set(g for f in missing for g in family_for(f)['gammas'])-available
    # Cold generation bound is never reduced merely because setup ran slowly.
    # .75h per new gamma + .20h per exact family pooling is an engineering
    # reservation floor, not a measured GPU claim; later complete timings raise it.
    observations=state.get('family_observations',[])
    per=max([.75]+[v['wall_hours']/v['new_gamma_count']*1.15 for v in observations if v.get('new_gamma_count',0)>0])
    required=len(gammas)*per+len(missing)*.2
    return required

def block_cache_requirement(root,s,block,state):
    gammas=set()
    for c in block.cases:
        if c.arm=='NATIVE0' or state['runs'].get(c.run_id,{}).get('terminal'):continue
        family=c.calibration_family
        if not read(camp(root,s)/'families'/f'{family}.json'):gammas.update(family_for(family)['gammas'])
    return bool(gammas),len(gammas)

def repay_pending(root,s,state,window,upload):
    for c in cases_for(s):
        row=state['runs'].get(c.run_id,{})
        if row.get('training_complete') and not row.get('complete') and row.get('postrun_retries',0)<2:
            row['postrun_retries']=row.get('postrun_retries',0)+1;_save(root,s,state)
            _run_case(root,s,state,c,window,upload)

def run_replay(root,s,state,window):
    if state.get('replay_complete') or state.get('replay_attempts',0)>=3:return False
    pair={'s3':('A02','A03'),'s4':('B03','B04'),'s5':('C01','C02')}[s]
    if not all(state['runs'].get(case_for(c).run_id,{}).get('complete') for c in pair):return False
    if utc(utcnow())>=window.train_finish_utc:return False
    code,hours=_command(root,s,['replay','--ctrl',str(run_dir(pair[0],root)/'meta/config.resolved.yaml'),
        '--mix',str(run_dir(pair[1],root)/'meta/config.resolved.yaml')],camp(root,s)/'logs/replay.log',window.train_finish_utc.isoformat())
    state['diagnostic_hours']=state.get('diagnostic_hours',0)+hours;state['replay_complete']=code==0
    state['replay_attempted']=True;state['replay_attempts']=state.get('replay_attempts',0)+1
    state['replay_status']='COMPLETE' if code==0 else 'DIAGNOSTICS_PENDING' if state['replay_attempts']<3 else 'DIAGNOSTICS_FAILED'
    _save(root,s,state);return True

def run(root=ROOT,server='s3',bindings=None,upload=True):
    from gfp40.resources import idle_evidence,assess_block
    root=Path(root).resolve();s=server;folder=camp(root,s)
    with locked(folder/'runner.lock'):
        state=_state(root,s);window=local_window(root,s)
        try:
            verify_registration(root,s)
            if utc(utcnow())<window.t0_utc:raise ValueError('Before actual t0')
            while not idle_evidence()['idle']:
                state['status']='WAIT_EXISTING_WORK';_save(root,s,state)
                if utc(utcnow())>=window.admission_cutoff_utc:return 75
                if read(folder/'control.json').get('command')=='STOP_NOW_SAFE':return 75
                time.sleep(15)
            if utc(utcnow())>=window.train_finish_utc:return closeout(root,s,state,window,upload)
            if read(folder/'control.json').get('command') in ('STOP_NOW_SAFE','STOP_AFTER_BLOCK'):return 75
            args=['preflight']+(['--bindings',str(bindings)] if bindings else [])
            code,hours=_command(root,s,args,folder/'logs/preflight.log',window.train_finish_utc.isoformat())
            state['setup_hours']=state.get('setup_hours',0)+hours;_save(root,s,state)
            if code==75:state['status']='PAUSED_PREPARATION';_save(root,s,state);return 75
            if code:raise ValueError('P40 preflight failed; see logs/preflight.log')
            parents=read_json(folder/'parent_bindings.json')
            while utc(utcnow())<window.train_finish_utc:
                command=read(folder/'control.json').get('command')
                if command=='STOP_NOW_SAFE':state['status']='PAUSED_SAFE';_save(root,s,state);return 75
                if not state.get('replay_complete') and state.get('replay_attempts',0)<3:
                    run_replay(root,s,state,window)
                pending=[b for b in blocks_for(s) if not state['blocks'].get(b.block_id,{}).get('terminal')]
                for b in pending:
                    if len(_done(b,state))==len(b.cases):state['blocks'].setdefault(b.block_id,{}).update(terminal=True,status='BLOCK_FINISHED')
                    for c in b.cases:
                        dependencies=[state['runs'].get(case_for(d).run_id,{}) for d in c.depends_on]
                        if any(v.get('terminal') and not v.get('complete') and
                            (not v.get('training_complete') or v.get('postrun_retries',0)>=2) for v in dependencies):
                            state['blocks'].setdefault(b.block_id,{}).update(terminal=True,status='BLOCKED_DEPENDENCY')
                pending=[b for b in pending if not state['blocks'].get(b.block_id,{}).get('terminal')]
                if not pending:break
                started={b.block_id for b in pending if state['blocks'].get(b.block_id,{}).get('admitted')}
                if command=='STOP_AFTER_BLOCK' and not started:state['status']='PAUSED_SAFE';_save(root,s,state);return 75
                cache_hours=pending_cache_hours(root,s,state)
                diag=0. if state.get('replay_complete') else max(0.,DIAGNOSTIC_HOURS-state.get('diagnostic_hours',0))
                available=(window.train_finish_utc-utc(utcnow())).total_seconds()/3600-cache_hours-diag-evaluation_debt(root,s)
                unstarted=[b for b in pending if b.block_id not in started]
                _,dropped=trim_unstarted(s,unstarted,available-outstanding(root,s,state),state['observations'])
                for name in dropped:state['blocks'][name]=dict(terminal=True,status='NOT_ADMITTED_PRIORITY_BUDGET')
                chosen=None
                for b in pending:
                    if b.block_id in dropped:continue
                    if command=='STOP_AFTER_BLOCK' and b.block_id not in started:continue
                    c=ready_case(b,state)
                    if c is None:continue
                    missing=[v.parent_id for v in b.cases if v.parent_id in PARENTS and parents.get(v.parent_id,{}).get('status')!='BOUND']
                    if missing:
                        state['blocks'][b.block_id]=dict(terminal=True,status='BLOCKED_PARENT',parents=missing);continue
                    receipt=read(folder/'admissions'/f'{b.block_id}.json') or None
                    admit=admission(window,utcnow(),b,observations=state['observations'],completed=_done(b,state),receipt=receipt,
                        outstanding_hours=outstanding(root,s,state,b.block_id),debt_hours=evaluation_debt(root,s),cache_hours=cache_hours,diagnostic_hours=diag)
                    append_event(folder/'admit_stop_debt_ledger.jsonl','ADMISSION',evidence=admit)
                    if not admit['allowed']:
                        state['blocks'].setdefault(b.block_id,{}).update(terminal=True,status=admit['reason']);continue
                    family=c.calibration_family
                    need=c.arm!='NATIVE0' and not read(folder/'families'/f'{family}.json')
                    cache_needed,gamma_count=block_cache_requirement(root,s,b,state)
                    capacity=assess_block(root,[v for v in b.cases if v.run_id not in _done(b,state)],needs_mixed=cache_needed,gamma_count=gamma_count)
                    if not capacity['allowed']:state.update(status='WAIT_LOCAL_RESOURCE',resource=capacity);_save(root,s,state);return 75
                    if not receipt:immutable_json(folder/'admissions'/f'{b.block_id}.json',admit)
                    state['blocks'][b.block_id]=dict(admitted=True,terminal=False,status='ADMITTED');_save(root,s,state)
                    if need:
                        known=set(g for f in INITIAL_FAMILIES[s] if read(folder/'families'/f'{f}.json') for g in family_for(f)['gammas'])
                        args=['family','--family',family]+(['--bindings',str(bindings)] if bindings else [])
                        code,hours=_command(root,s,args,folder/'logs'/f'family_{family}.log',window.train_finish_utc.isoformat())
                        state['family_hours']=state.get('family_hours',0)+hours;_save(root,s,state)
                        if code==75:state['status']='PAUSED_CALIBRATION';_save(root,s,state);return 75
                        if code:raise ValueError('Family cache/calibration failed; no subset fallback')
                        state.setdefault('family_observations',[]).append(dict(family=family,wall_hours=hours,
                            new_gamma_count=len(set(family_for(family)['gammas'])-known)))
                        _save(root,s,state)
                    chosen=c;break
                if chosen:
                    if not idle_evidence()['idle']:state['status']='WAIT_LOCAL_RESOURCE';_save(root,s,state);return 75
                    outcome=_run_case(root,s,state,chosen,window,upload)
                    if outcome=='PAUSED_SAFE':return 75
                    if not outcome:state['blocks'][chosen.block_id].update(terminal=True,status='FAILED_OR_PARTIAL')
                    _save(root,s,state);continue
                repay_pending(root,s,state,window,upload)
                if not state.get('replay_complete'):run_replay(root,s,state,window)
                if all(state['blocks'].get(b.block_id,{}).get('terminal') for b in blocks_for(s)):break
                if utc(utcnow())>=window.admission_cutoff_utc and not outstanding(root,s,state):break
                state['status']='WAIT_LOCAL_DEPENDENCY';_save(root,s,state);time.sleep(15)
            if not state.get('replay_complete'):run_replay(root,s,state,window)
            return closeout(root,s,state,window,upload)
        except (ValueError,OSError,RuntimeError) as exc:
            state.update(status='BLOCKED_INTEGRITY',reason=f'{type(exc).__name__}: {exc}');_save(root,s,state)
            append_event(folder/'attempts.jsonl','BLOCKED',reason=state['reason'])
            from gfp40.reporting import rebuild
            rebuild(root,s);return 1

def closeout(root,s,state,window,upload):
    state['status']='CLOSEOUT';_save(root,s,state)
    for c in cases_for(s):
        wd=run_dir(c,root)
        if (wd/'meta/training_status.json').is_file() and not read(wd/'official/summary.json').get('complete'):
            if utc(utcnow())>=window.deadline_utc:break
            code,hours=_command(root,s,['postrun','--config',str(wd/'meta/config.resolved.yaml')],folder_log(root,s,c),window.deadline_utc.isoformat())
            row=state['runs'].setdefault(c.run_id,{});row['wall_hours']=row.get('wall_hours',0)+hours
            if read(wd/'official/summary.json').get('complete'):row.update(complete=True,status='OFFICIAL_EVAL_COMPLETE')
    if upload and utc(utcnow())<window.deadline_utc:retry_uploads(root,s)
    state.update(status='CLOSED',active_run=None);_save(root,s,state)
    from gfp40.reporting import rebuild
    rebuild(root,s);return 0
def folder_log(root,s,c):return camp(root,s)/'logs'/f'{c.run_id}.closeout.log'

def start(root=ROOT,server='s3',foreground=False,bindings=None,upload=True):
    verify_lane(server)
    window=local_window(root,server,initialize=True)
    if utc(utcnow())<window.t0_utc:raise ValueError('Cannot start before local actual t0')
    register_runtime(root,server);folder=camp(root,server)
    if foreground:return dict(exit_code=run(root,server,bindings,upload))
    with locked(folder/'startup.lock'):
        try:
            with locked(folder/'runner.lock'):pass
        except BlockingIOError:return dict(status='ALREADY_RUNNING')
        from gfp40.deployment import process_start
        old=read(folder/'runner.pid.json')
        if old.get('ticks') and process_start(old.get('pid'))==old['ticks']:return dict(status='ALREADY_RUNNING',pid=old['pid'])
        args=[sys.executable,'-u','tools/gfp40_runner.py','run','--server',server]
        if bindings:args+=['--bindings',str(bindings)]
        if not upload:args+=['--no-upload']
        with (folder/'runner.log').open('a') as stream:
            child=subprocess.Popen(args,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        atomic_json(folder/'runner.pid.json',dict(pid=child.pid,ticks=process_start(child.pid)))
    return dict(status='SUBMITTED',pid=child.pid,log=str(folder/'runner.log'))
