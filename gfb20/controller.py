"""Finite whole-block B20 runner; no Teacher training, old queue mutation or kills."""
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import time
from gfb20.common import (ROOT,CAMPAIGN_ID,apply_runtime_policy,atomic_json,camp,immutable_json,
    locked,object_sha,read,read_json,read_config,run_dir,sha256,source_identity,utcnow,append_event)
from gfb20.plan import (REGISTRY,SERVERS,blocks_for,block_for,build_config,case_for,cases_for,
    registry_sha256,validate_config,verify_lane,verify_sources,grid_steps)
from gfb20.policy import CampaignWindow,admission,estimate_block,validate_admission,promotion,block_eligible,utc


def window_path(root): return Path(root)/'work_dir/_gfb20/campaign_window.json'


def write_window(root,t0):
    window=CampaignWindow(t0)
    with locked(window_path(root).with_suffix('.lock')):
        immutable_json(window_path(root),window.to_dict())
    return window.to_dict()


def shared_window(root,supplied=None):
    path=window_path(root)
    if supplied:
        window=CampaignWindow.from_dict(read_json(supplied))
        write_window(root,window.t0_utc)
    if not path.is_file(): raise ValueError('Supply the same actual --t0 or --clock-file to all s3-s5; no implicit per-server clock')
    return CampaignWindow.from_dict(read_json(path))


def build(root=ROOT):
    verify_sources(root)
    folder=Path(root)/'config/gfb20'
    for server in SERVERS:
        immutable_json(folder/f'{server}_queue.json',dict(campaign_id=CAMPAIGN_ID,
            server=server,blocks=[b.to_dict() for b in blocks_for(server)]))
        for case in cases_for(server):
            immutable_json(folder/(case.run_id+'.yaml'),build_config(case))
    immutable_json(folder/'frozen_protocol.json',REGISTRY)
    return dict(configs=36,core=18,teacher_training=0,clock_created=False,training_started=False)


def _state(root,server):
    return read(camp(root,server)/'status.json',dict(campaign_id=CAMPAIGN_ID,server=server,
        status='DEFINED_NOT_LAUNCHED',runs={},blocks={},observations=[]))


def _save(root,server,state):
    state['updated_at_utc']=utcnow();atomic_json(camp(root,server)/'status.json',state)


def register_runtime(root,server):
    verify_lane(server);verify_sources(root);apply_runtime_policy(root)
    value=dict(campaign_id=CAMPAIGN_ID,server=server,source_identity=source_identity(root),
               registry_sha256=registry_sha256(),window=shared_window(root).to_dict())
    immutable_json(camp(root,server)/'runtime_binding.json',value)
    immutable_json(camp(root,server)/'frozen_protocol.json',REGISTRY)
    return value


def verify_registration(root,server):
    apply_runtime_policy(root)
    expected=dict(campaign_id=CAMPAIGN_ID,server=server,source_identity=source_identity(root),
                  registry_sha256=registry_sha256(),window=shared_window(root).to_dict())
    if read_json(camp(root,server)/'runtime_binding.json')!=expected:
        raise ValueError('GFB20 registered source/runtime/shared clock changed')
    return expected


def status(root,server):
    return dict(state=_state(root,server),window=read(window_path(root)),
        preflight=read(camp(root,server)/'preflight.json'),queue=[b.to_dict() for b in blocks_for(server)])


def control(root,server,command):
    if command not in ('STOP_NOW_SAFE','STOP_AFTER_BLOCK','CONTINUE'):raise ValueError('Unknown GFB20 control')
    value=dict(command=command,at_utc=utcnow())
    atomic_json(camp(root,server)/'control.json',value)
    append_event(camp(root,server)/'control_ledger.jsonl','OPERATOR_CONTROL',**value)
    return value


def _command(root,server,args,log,deadline):
    """Only our child is signalled, and never hard-killed while saving fullstate."""
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    remaining=(utc(deadline)-utc(utcnow())).total_seconds()
    if remaining<=0:return 75,0.
    start=time.monotonic()
    with Path(log).open('a') as stream:
        process=subprocess.Popen([sys.executable,'-u','tools/gfb20_runner.py',*args,'--server',server,
            '--deadline',deadline],cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
            stdout=stream,stderr=subprocess.STDOUT)
        # Internal deadlines close HDF5/context managers cooperatively. This
        # grace is cleanup only; it never extends the optimizer's fixed18h.
        try:code=process.wait(timeout=remaining+30)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                raise RuntimeError('Own child is preserving past deadline; no forced kill or subsequent GPU work')
            code=75
    return code,(time.monotonic()-start)/3600.


def evaluation_debt(root,server):
    debt=0.
    for case in cases_for(server):
        wd=run_dir(case,root)
        if read(wd/'official/summary.json').get('complete'):continue
        records=read(wd/'official/raw_grid.json').get('records',[])
        seen={r['update'] for r in records}
        pending=[s for s in grid_steps(case.updates) if s not in seen and (wd/f'candidates/{s}/identity.json').is_file()]
        per=max([r.get('seconds',0.)/3600. for r in records]+[1/50])
        debt+=len(pending)*per*1.15
    return debt


def summaries(root,server):
    return {c.case_id:read(run_dir(c,root)/'official/summary.json') for c in cases_for(server)
            if (run_dir(c,root)/'official/summary.json').is_file()}


def promotion_evidence(root,server):
    """B's gradient-scale evidence is mandatory, not implied by metric success."""
    rows=summaries(root,server)
    result=promotion(server,rows)
    if server!='s4' or not result.get('ready'):return result
    from gfb20.diagnostics import low_active_mass
    evidence={};complete=True
    for identifier in ('B01','B02','B03','B04','B05','B06'):
        case=case_for(identifier);wd=run_dir(case,root);probes=[]
        for step in (0,5000,20000):
            path=wd/'diagnostics'/f'probe_{step}.json';record=read(path)
            checkpoint=wd/('restart_fullstates' if step==0 else 'candidates')/str(step)/'identity.json'
            identity=read(checkpoint)
            valid=bool(record.get('gradient') and identity
                and record.get('local_step')==step
                and record.get('model_state_hash')==identity.get('state_hash')
                and record.get('config_sha256')==rows[identifier].get('config_sha256')
                and record.get('source_identity')==rows[identifier].get('source_identity'))
            complete=complete and valid
            if valid:probes.append(record)
        evidence[identifier]=dict(complete=len(probes)==3,
            probe_sha256=[object_sha(p) for p in probes],active_mass=low_active_mass(probes))
    result['mandatory_gradient_evidence']=evidence
    if not complete:
        result.update(ready=False,fresh_signal=False,fresh_allowed_after_negative=False,
                      status='REQUIRED_GRADIENT_DIAGNOSTICS_PENDING')
    elif all(evidence[c]['active_mass']['status']=='LOW_ACTIVE_MASS' for c in ('B02','B05')):
        e=result['contrasts']['S4_E100']
        positive=e['spatial_signal'] and not e['checkpoint_sensitive']
        result.update(selected_profile='E100',selection_override='LOW_ACTIVE_MASS_BOTH_PARENTS',
            fresh_signal=positive,fresh_allowed_after_negative=not positive,
            status='PROMOTED_SIGNAL' if positive else 'FRESH_PATH_TEST_AFTER_NEGATIVE_FT')
    return result


def repay_pending(root,server,state,window,upload):
    """Bounded evaluation-only retries before making a conditional decision."""
    for case in cases_for(server):
        row=state['runs'].get(case.run_id,{})
        if (not row.get('training_complete') or row.get('complete')
            or row.get('postrun_retries',0)>=2):continue
        if utc(utcnow())>=window.train_finish_utc:return
        row['postrun_retries']=row.get('postrun_retries',0)+1;_save(root,server,state)
        _run_case(root,server,state,case,window,upload)


def resolve_config(root,case):
    folder=camp(root,case.server)
    decision=read(folder/'promotion_decision.json')
    cfg=build_config(case,decision.get('selected_profile') if case.profile=='B_SELECTED' else None)
    field=cfg['gfb20']
    field.update(dataset_manifest=str(folder/'dataset_manifest.json'),window_path=str(window_path(root)),
                 runtime_policy_sha256=sha256(Path(root)/'gfb20/runtime_policy.json'))
    if case.profile=='B_SELECTED':field['profile_decision_sha256']=object_sha(decision)
    if case.parent_id in ('P3NEW1','P3NEW2'):
        from gfb20.assets import bind_fresh_parent
        trunk=case_for('A11' if case.parent_id=='P3NEW1' else 'A14')
        parent=bind_fresh_parent(case.parent_id,case.server,run_dir(trunk,root)/'meta/config.resolved.yaml',root)
        parentpath=immutable_json(folder/'parents'/f'{case.parent_id}.json',parent)
    elif case.is_ft:parentpath=folder/'parents'/f'{case.parent_id}.json'
    else:parentpath=None
    field['parent_manifest']=str(parentpath) if parentpath else None
    refpath=folder/'references'/f'{case.reference_key}.json'
    ref=read_json(refpath)
    field['reference_manifest']=str(refpath)
    native,paths=ref['native_manifest'],ref['resolved_artifacts']
    field.update(teacher_checkpoint=paths['teacher_checkpoint'],teacher_sha256=native['teacher_checkpoint_sha256'],
        tau_R=native['tau_R'],q_ref=native['q_ref'],q_cache=paths['q_cache_path'],q_cache_sha256=native['q_cache_sha256'])
    if field['profile'] in ('PANMIX','NAT_MIXCAL'):
        from gfb20.calibration import load_mixed_reference
        mixpath=folder/'mixed/mixed_reference_manifest.json'
        mixed,_=load_mixed_reference(mixpath,native_reference=dict(native,**paths,native_manifest=native))
        field.update(mixed_calibration_manifest=str(mixpath),
            augmentation_manifest=str(folder/'mixed/augmentation_manifest.json'))
        # The native binding stays intact; the trainer explicitly chooses mixed scales/cache.
    data=read_json(field['dataset_manifest'])
    for split,key in (('train','train_feeder_args'),('val','val_feeder_args'),
                      ('rr','test_reduced_feeder_args'),('fr','test_full_feeder_args')):
        cfg[key]['dataroot']=data['splits'][split]['dataroot']
    validate_config(cfg,require_bound=True)
    path=run_dir(case,root)/'meta/config.resolved.yaml'
    immutable_json(path,cfg)
    return path


def authorize_train(root,server,config_path):
    verify_registration(root,server)
    cfg=read_config(config_path);case=validate_config(cfg,require_bound=True)
    if case.server!=server:raise ValueError('Cross-server case rejected')
    state=_state(root,server)
    if state.get('active_run')!=case.run_id or state.get('status')!='TRAINING':
        raise ValueError('Controller has not authorized this training case')
    ready=read_json(camp(root,server)/'preflight.json')
    if not ready.get('complete') or ready['source_identity']!=source_identity(root):
        raise ValueError('Current local preflight required')
    if ready['dataset_manifest_sha256']!=object_sha(read_json(cfg['gfb20']['dataset_manifest'])):
        raise ValueError('Preflight data changed')
    receipt=read_json(camp(root,server)/'admissions'/f'{case.block_id}.json')
    validate_admission(receipt,block_for(case.block_id),shared_window(root))
    block=block_for(case.block_id)
    for previous in block.cases[:block.cases.index(case)]:
        if not state['runs'].get(previous.run_id,{}).get('terminal'):
            raise ValueError('Previous registered arm has not reached a recorded terminal state')
    return case


def prepare_mixed(root,server,deadline):
    if server!='s5':raise ValueError('Mixed preparation only belongs to s5')
    from gfb20.assets import load_reference
    from gfb20.calibration import prepare_mixed_reference
    from g20.data import build_dataset
    folder=camp(root,server)
    teacher,q,ref=load_reference(folder/'references/R5_100.json','cuda',root,deadline=deadline)
    dataset=build_dataset(read_json(folder/'dataset_manifest.json'),'train',root=root)
    path=prepare_mixed_reference(teacher,dataset,ref,folder/'mixed',device='cuda',
        deadline_utc=deadline,source_identity=source_identity(root))
    return dict(manifest=str(path))


def retry_uploads(root,server):
    from gfb20.upload import upload_run
    results={}
    for case in cases_for(server):
        wd=run_dir(case,root)
        if not read(wd/'official/summary.json').get('complete'):continue
        if read(wd/'official/upload_receipt.json').get('readback_verified'):continue
        try:results[case.run_id]=upload_run(case.run_id,root,activated=True)
        except Exception as exc:
            value=dict(status='UPLOAD_PENDING',reason=f'{type(exc).__name__}: {exc}',at_utc=utcnow())
            atomic_json(wd/'official/upload_status.json',value);results[case.run_id]=value
    return results


def _run_case(root,server,state,case,window,upload):
    cfgpath=resolve_config(root,case);wd=run_dir(case,root)
    row=state['runs'].setdefault(case.run_id,{})
    resume=(wd/'last/training_state.pt').is_file() or (wd/'last').is_dir()
    disk_training=read(wd/'meta/training_status.json')
    # The immutable endpoint is validated by postrun; never rerun optimizer
    # merely because the controller crashed after the trainer published status.
    if disk_training.get('training_complete') is True:
        if disk_training.get('actual_updates')!=case.updates:raise ValueError('Completed horizon differs')
        row['training_complete']=True
    if not row.get('training_complete'):
        state.update(status='TRAINING',active_run=case.run_id);row['status']='TRAINING';_save(root,server,state)
        from gfb20.training import recover_exact_endpoint
        recovered=recover_exact_endpoint(root,wd)
        if not recovered:
            while True:
                args=['train','--config',str(cfgpath)]+(['--resume'] if resume else [])
                code,hours=_command(root,server,args,camp(root,server)/'logs'/f'{case.run_id}.train.log',
                                    window.deadline_utc.isoformat())
                row['wall_hours']=row.get('wall_hours',0.)+hours
                append_event(camp(root,server)/'attempts.jsonl','TRAIN_EXIT',run_id=case.run_id,code=code,hours=hours)
                if code==74 and row.get('retries',0)<2:
                    row['retries']=row.get('retries',0)+1
                    resume=(wd/'last/identity.json').is_file();_save(root,server,state);continue
                break
        else:code=0
        training=read(wd/'meta/training_status.json')
        row.update(training_complete=training.get('training_complete',False),actual_updates=training.get('actual_updates',0))
        if code and not row['training_complete']:
            failure=training.get('status','PARTIAL_TIME_LIMIT' if code==75 else 'FAILED_NUMERICAL')
            paused=failure in ('PAUSED_SIGNAL','PAUSED_CONTROL') and utc(utcnow())<window.train_finish_utc
            row.update(status=failure,terminal=not paused)
            if paused:state['status']='PAUSED_SAFE'
            _save(root,server,state)
            if code==75:return 'PAUSED_SAFE' if paused else False
            return True
    state['status']='POSTRUN';_save(root,server,state)
    code,hours=_command(root,server,['postrun','--config',str(cfgpath)],camp(root,server)/'logs'/f'{case.run_id}.postrun.log',
                        window.deadline_utc.isoformat())
    row['wall_hours']=row.get('wall_hours',0.)+hours
    summary=read(wd/'official/summary.json')
    row.update(complete=summary.get('complete',False),terminal=True,
               status='OFFICIAL_EVAL_COMPLETE' if summary.get('complete') else 'EVAL_PENDING')
    if summary.get('complete'):
        state['observations'].append(dict(server=server,updates=case.updates,input_views=case.input_views,
            wall_hours=row['wall_hours'],includes_eval_io_diagnostics=True,complete=True))
        if upload:retry_uploads(root,server)
    _save(root,server,state)
    return True


def run(root=ROOT,server='s3',bindings=None,upload=True):
    from gfb20.resources import idle_evidence,assess_block
    root=Path(root).resolve();folder=camp(root,server)
    with locked(folder/'runner.lock'):
        state=_state(root,server);window=shared_window(root)
        try:
            verify_registration(root,server)
            if utc(utcnow())<window.t0_utc:
                raise ValueError('Actual preparation cannot begin before the shared t0')
            while not idle_evidence()['idle']:
                state['status']='WAIT_EXISTING_WORK';_save(root,server,state)
                if utc(utcnow())>=window.admission_cutoff_utc:return 75
                if read(folder/'control.json').get('command')=='STOP_NOW_SAFE':return 75
                time.sleep(15)
            if utc(utcnow())>=window.train_finish_utc:return closeout(root,server,state,window,upload)
            args=['preflight']+(['--bindings',str(bindings)] if bindings else [])
            code,hours=_command(root,server,args,folder/'logs/preflight.log',window.train_finish_utc.isoformat())
            state['setup_hours']=state.get('setup_hours',0.)+hours
            if code:raise ValueError('Local B20 preflight failed; see logs/preflight.log')
            parent_bindings=read_json(folder/'parent_bindings.json')
            data=read_json(folder/'dataset_manifest.json')
            immutable_json(folder/'source_and_data_manifest.json',dict(source_identity=source_identity(root),
                dataset_manifest_sha256=object_sha(data),parents=parent_bindings))
            if server=='s5' and not (folder/'mixed/mixed_reference_manifest.json').is_file():
                resource=assess_block(root,block_for('C_PANEL1').cases,needs_mixed=True)
                if not resource['allowed']:raise ValueError('Mixed cache resource admission: '+str(resource['reasons']))
                code,hours=_command(root,server,['mixed'],folder/'logs/mixed.log',window.train_finish_utc.isoformat())
                state['mixed_calibration_hours']=state.get('mixed_calibration_hours',0.)+hours
                _save(root,server,state)
                if code:raise ValueError('Mixed calibration incomplete/failed; no augmented fallback')
            for block in blocks_for(server):
                if state['blocks'].get(block.block_id,{}).get('terminal'):continue
                command=read(folder/'control.json').get('command')
                if command in ('STOP_NOW_SAFE','STOP_AFTER_BLOCK'):
                    state['status']='PAUSED_SAFE';_save(root,server,state);return 75
                if block.stage!='CORE':repay_pending(root,server,state,window,upload)
                decision=promotion_evidence(root,server)
                if decision.get('ready') and not (folder/'promotion_decision.json').is_file():
                    immutable_json(folder/'promotion_decision.json',dict(decision,at_utc=utcnow()))
                if (folder/'promotion_decision.json').is_file():decision=read_json(folder/'promotion_decision.json')
                eligible,reason=block_eligible(block,decision,mixed_calibration_hours=state.get('mixed_calibration_hours',0.))
                if not eligible:
                    state['blocks'][block.block_id]=dict(status=reason,terminal=True);_save(root,server,state);continue
                missing=[c.parent_id for c in block.cases if c.is_ft and c.parent_id not in ('P3NEW1','P3NEW2')
                    and parent_bindings.get(c.parent_id,{}).get('status')!='BOUND']
                if missing:
                    status=('BLOCKED_INTEGRITY' if any(parent_bindings.get(p,{}).get('status')=='BLOCKED_INTEGRITY'
                                                      for p in missing) else 'BLOCKED_MISSING_PARENT')
                    state['blocks'][block.block_id]=dict(status=status,parents=missing,terminal=True)
                    _save(root,server,state);continue
                done=[c.run_id for c in block.cases if state['runs'].get(c.run_id,{}).get('terminal')]
                receipt=read(folder/'admissions'/f'{block.block_id}.json') or None
                admit=admission(window,utcnow(),block,observations=state['observations'],completed=done,
                    receipt=receipt,debt_hours=evaluation_debt(root,server))
                append_event(folder/'admit_stop_debt_ledger.jsonl','ADMISSION',evidence=admit)
                if not admit['allowed']:
                    state['blocks'][block.block_id]=dict(status=admit['reason'],terminal=True);_save(root,server,state);continue
                resource=assess_block(root,[c for c in block.cases if c.run_id not in done])
                if not resource['allowed']:
                    state['blocks'][block.block_id]=dict(status='WAIT_LOCAL_RESOURCE',evidence=resource)
                    _save(root,server,state);return 75
                if receipt is None:immutable_json(folder/'admissions'/f'{block.block_id}.json',admit)
                state['blocks'][block.block_id]=dict(status='ADMITTED',terminal=False);_save(root,server,state)
                for case in block.cases:
                    if state['runs'].get(case.run_id,{}).get('terminal'):continue
                    done=[c.run_id for c in block.cases if state['runs'].get(c.run_id,{}).get('terminal')]
                    suffix=admission(window,utcnow(),block,completed=done,observations=state['observations'],
                        receipt=read_json(folder/'admissions'/f'{block.block_id}.json'),debt_hours=evaluation_debt(root,server))
                    if not suffix['allowed']:break
                    if not idle_evidence()['idle']:
                        state['status']='WAIT_LOCAL_RESOURCE';_save(root,server,state);return 75
                    dependencies=[d for d in str(case.dependencies or '').split(';') if d]
                    if dependencies:repay_pending(root,server,state,window,upload)
                    if any(not state['runs'].get(case_for(d).run_id,{}).get('complete') for d in dependencies):
                        state['runs'][case.run_id]=dict(status='BLOCKED_DEPENDENCY',terminal=True)
                        _save(root,server,state);continue
                    outcome=_run_case(root,server,state,case,window,upload)
                    if outcome=='PAUSED_SAFE':return 75
                    if not outcome:break
                state['blocks'][block.block_id].update(terminal=True,status='BLOCK_FINISHED_OR_PARTIAL')
                _save(root,server,state)
            return closeout(root,server,state,window,upload)
        except (ValueError,OSError,RuntimeError) as exc:
            state.update(status='BLOCKED_INTEGRITY',reason=f'{type(exc).__name__}: {exc}')
            append_event(folder/'attempts.jsonl','BLOCKED',reason=state['reason'])
            _save(root,server,state)
            from gfb20.reporting import rebuild
            rebuild(root,server)
            return 1


def closeout(root,server,state,window,upload):
    state['status']='CLOSEOUT';_save(root,server,state)
    for case in cases_for(server):
        wd=run_dir(case,root)
        if (wd/'meta/training_status.json').is_file() and not read(wd/'official/summary.json').get('complete'):
            if utc(utcnow())>=window.deadline_utc:break
            _command(root,server,['postrun','--config',str(wd/'meta/config.resolved.yaml')],
                camp(root,server)/'logs'/f'{case.run_id}.closeout.log',window.deadline_utc.isoformat())
            if read(wd/'official/summary.json').get('complete'):
                state['runs'].setdefault(case.run_id,{}).update(complete=True,status='OFFICIAL_EVAL_COMPLETE')
    if upload and utc(utcnow())<window.deadline_utc:retry_uploads(root,server)
    state.update(status='CLOSED',active_run=None);_save(root,server,state)
    from gfb20.reporting import rebuild
    rebuild(root,server)
    return 0


def start(root=ROOT,server='s3',foreground=False,bindings=None,upload=True):
    verify_lane(server)
    if utc(utcnow())<shared_window(root).t0_utc:
        raise ValueError('Cannot start before the shared actual t0')
    register_runtime(root,server)
    folder=camp(root,server)
    if foreground:return dict(exit_code=run(root,server,bindings,upload))
    with locked(folder/'startup.lock'):
        try:
            with locked(folder/'runner.lock'):pass
        except BlockingIOError:return dict(status='ALREADY_RUNNING')
        old=read(folder/'runner.pid.json')
        from gfb20.deployment import process_start
        if old.get('ticks') and process_start(old.get('pid'))==old['ticks']:return dict(status='ALREADY_RUNNING',pid=old['pid'])
        args=[sys.executable,'-u','tools/gfb20_runner.py','run','--server',server]
        if bindings:args+=['--bindings',str(bindings)]
        if not upload:args+=['--no-upload']
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'runner.log').open('a') as stream:
            process=subprocess.Popen(args,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                     env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        atomic_json(folder/'runner.pid.json',dict(pid=process.pid,ticks=process_start(process.pid)))
    return dict(status='SUBMITTED',pid=process.pid,log=str(folder/'runner.log'))
