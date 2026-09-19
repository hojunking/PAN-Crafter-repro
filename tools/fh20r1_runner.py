#!/usr/bin/env python
"""FH20R1 finite atomic blocks, minimum qualified 20h (never a deadline)."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fh12_runner import locked,detect_server,gpu_processes
from fh20r1.ledger import CAMPAIGN_ID,camp,read,write,iso,digest,record_interval,ingest_timing,report

HOLD='FH20R1_FUTURE_ADMISSIONS_HOLD_v1'
DONE={'DONE','DONE_REUSED'}
CLOSED={'COMPLETE_20HPLUS','COMPLETE_20HPLUS_WITH_BLOCKED_DONOR','WORK_COMPLETE_UPLOAD_PENDING',
        'CAPACITY_EXHAUSTED_BELOW20','INCOMPLETE_CORE'}


def inventory():
    result=subprocess.run(['ps','-eo','pid=,args='],capture_output=True,text=True,check=True)
    waiting={'mix20h_runner.py','qrecon24_waiter.sh'}
    drain={'fh12_runner.py','_run_cases.sh'}
    workers={'main.py','fh12_train.py','fh12_postrun.py','fh12_calibrate.py','fh12_preflight.py',
             'mix20h_postrun.py','fh20r1_train.py','fh20r1_postrun.py','fh20r1_diagnose.py',
             'fh20r1_preflight.py','fh20r1_reference.py','qrecon24_select.py','eval_fr_paperset.py',
             'noa_eval.py','aligner_scope_audit.py'}
    rows=[]
    for line in result.stdout.splitlines():
        fields=line.split(None,1)
        if len(fields)!=2 or int(fields[0])==os.getpid():continue
        try:args=shlex.split(fields[1])
        except ValueError:continue
        if '-c' in args[:3]:continue
        script=next((Path(s).name for s in args[:3] if Path(s).name in waiting|drain|workers),None)
        if script:rows.append(dict(pid=int(fields[0]),script=script,waiting_controller=script in waiting))
    return rows


def validate_hold(root,server):
    path=Path(root)/'work_dir/_eval_phase/hold.json'; value=read(path)
    if value and (value.get('server')!=server or value.get('protocol_id') not in
                  {HOLD,'FH12_FUTURE_ADMISSIONS_HOLD_v1'}):
        raise ValueError('Unknown/foreign admission hold preserved; cannot safely take over')
    return value


def config_hashes(root,server):
    import yaml
    from fh20r1.plan import cases_for,build_config
    hashes={}
    for case in cases_for(server):
        path=Path(root)/'config'/f'{case.run_id}.yaml'
        if yaml.safe_load(path.read_text())!=build_config(case):raise ValueError(f'Registry/config mismatch: {path}')
        hashes[case.run_id]=digest(path)
    return hashes


def registry_hash(server):
    from fh20r1.plan import registry_sha256
    return registry_sha256()


def prepare(root,server,device='cuda',upload=True,dry_run=False):
    from fh20r1.plan import SOURCE_PLAN,SOURCE_CASES
    root=Path(root); directory=camp(root,server); hashes=config_hashes(root,server)
    old=read(directory/'campaign_budget.json'); held=validate_hold(root,server); processes=inventory()
    if any(p['script']=='_run_cases.sh' for p in processes) and not (root/'work_dir/_qrc24_mix20h/plan_manifest.json').exists():
        raise ValueError('Unknown live generic queue has no verified case-boundary hold; current work untouched')
    identity=dict(campaign_id=CAMPAIGN_ID,server_id=server,registry_sha256=registry_hash(server),
                  config_hashes=hashes,source_plan_sha256=digest(root/SOURCE_PLAN),source_cases_sha256=digest(root/SOURCE_CASES))
    if old and any(old.get(k)!=v for k,v in identity.items()):raise ValueError('Registered campaign identity changed')
    budget=old or dict(identity,started_at_utc=iso(),minimum_effective_hours=20.,hard_deadline=None,
        time_policy='minimum_qualified_interval_union',actual_start_authorized=True,device=device,upload_enabled=upload)
    if dry_run:return dict(status='DRY_RUN',budget_would_be=budget,current_processes=processes,mutates_runtime=False)
    write(directory/'campaign_budget.json',budget)
    if not (directory/'takeover_manifest.json').exists():
        write(directory/'takeover_manifest.json',dict(at_utc=iso(),processes=processes,previous_hold=held,
            policy='preserve_current_training/postrun; only_supersede_future_admissions; no_kill',
            old_fh12_window_sha256=digest(root/'work_dir/_fh12'/server/'window.json')
                 if (root/'work_dir/_fh12'/server/'window.json').exists() else None))
    write(root/'work_dir/_eval_phase/hold.json',dict(protocol_id=HOLD,campaign_id=CAMPAIGN_ID,server=server,
          reason='FH20R1 owns future admissions; old work drains normally',at_utc=iso(),auto_release=False))
    pointer=root/'work_dir/_fh20r1/local_server.txt';tmp=pointer.with_suffix('.tmp');tmp.write_text(server+'\n');os.replace(tmp,pointer)
    return dict(status='REGISTERED',budget=budget)


def load_budget(root,server):
    from fh20r1.plan import SOURCE_PLAN,SOURCE_CASES
    value=read(camp(root,server)/'campaign_budget.json')
    if value.get('campaign_id')!=CAMPAIGN_ID or value.get('server_id')!=server or not value.get('actual_start_authorized'):
        raise ValueError('Use explicit fh20r1_start.sh to authorize new campaign')
    if value.get('hard_deadline') is not None or value.get('minimum_effective_hours')!=20:
        raise ValueError('FH20R1 is minimum effective20h, never a deadline')
    if value['config_hashes']!=config_hashes(root,server) or value['registry_sha256']!=registry_hash(server):
        raise ValueError('Campaign config/registry drift')
    if value['source_plan_sha256']!=digest(Path(root)/SOURCE_PLAN) or value['source_cases_sha256']!=digest(Path(root)/SOURCE_CASES):
        raise ValueError('Campaign source MD/CSV drift')
    return value


def command(root,args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        return subprocess.run([sys.executable]+args,cwd=root,stdout=stream,stderr=subprocess.STDOUT).returncode


def admission_event(root,server,event,**details):
    directory=camp(root,server);directory.mkdir(parents=True,exist_ok=True)
    row=dict(details,campaign_id=CAMPAIGN_ID,server_id=server,event_id=uuid.uuid4().hex,
             at_utc=iso(),event=event)
    with locked(directory/'.events.lock'):
        with (directory/'admission_events.jsonl').open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    return row


def wait_resources(root,server,budget,state,wait=True):
    start=time.time()
    while True:
        held=validate_hold(root,server)
        if held.get('protocol_id')!=HOLD:raise ValueError('FH20R1 admission hold lost')
        busy=[p for p in inventory() if not p['waiting_controller']]
        gpu=gpu_processes() if budget['device']!='cpu' else []
        if not busy and gpu==[]:
            if time.time()>start:
                record_interval(root,server,interval_id='wait:'+uuid.uuid4().hex,kind='waiting',
                    start_utc=start,end_utc=time.time(),evidence='resource inventory; excluded from effective time')
            return True
        state.update(status='DRAINING_EXISTING',active_processes=busy,gpu_processes=gpu)
        write(camp(root,server)/'status.json',state)
        if not wait:return False
        time.sleep(15)


def branch(root,server,condition,reason,evidence=None):
    """s1 immutable; s2 can degrade PRIMARY→declared fallback, never upgrade."""
    path=camp(root,server)/'branch_record.json'; old=read(path)
    if old and 'condition' not in old:old=dict(old,condition=old.get('branch'))
    if old and old['condition']!=condition:
        allowed=server=='s2' and ((old['condition']=='PRIMARY' and condition=='S2_N2PL_REFERENCE_READY') or
            (old['condition'] in {'PRIMARY','S2_N2PL_REFERENCE_READY'} and condition=='S2_DONOR_OR_REFERENCE_UNAVAILABLE'))
        if not allowed:
            raise ValueError('Branch re-selection forbidden')
    value=old if old and old['condition']==condition else dict(condition=condition,reason=reason,
            evidence=evidence,selected_at_utc=iso(),history=old.get('history',[])+([old] if old else []))
    value=dict(value,branch=condition)
    write(path,value);return value


def order_for(block,condition,rows):
    selected=list(block.alternative_orders.get(condition,block.primary_order))
    # F2 baselines have the same ID in both s2 orders and therefore naturally
    # reuse completed results. Completed N2PL members are never relabeled PLH.
    if condition=='S2_DONOR_OR_REFERENCE_UNAVAILABLE':
        from fh20r1.plan import case_for
        completed={case_for(run).student_seed:run for run in block.primary_order
                   if case_for(run).teacher_alias=='N2PL' and rows.get(run,{}).get('status') in DONE}
        selected=[completed.get(case_for(run).student_seed,run)
                  if case_for(run).input_layout=='PLH' else run for run in selected]
    return selected


def official_done(root,run):
    doc=read(Path(root)/'work_dir'/run/'official/postrun_status.json')
    return bool(doc.get('official_complete') and doc.get('actual_updates')==50000)


def should_stop(core_complete,effective_hours,block_running=False):
    return bool(core_complete and effective_hours>=20 and not block_running)


def upload_case(root,case,row,enabled=True):
    """Independent CPU-only delivery; never changes training/ledger identity."""
    wd=Path(root)/'work_dir'/case.run_id
    rc=0
    if enabled:
        rc=command(root,['tools/fh20r1_postrun.py',case.run_id,'--upload','--upload-only'],wd/'fh20r1_upload.log')
    receipt=read(wd/'official/postrun_status.json')
    row['upload_pending']=bool(rc) or not receipt.get('sheet_uploaded',False)
    row['upload_status']='VERIFIED' if not row['upload_pending'] else 'PENDING' if enabled else 'NOT_REQUESTED'
    if rc:row['upload_error']=f'Upload-only subprocess failed with exit_code={rc}; prior receipt is not current verification'
    elif receipt.get('upload_error'):row['upload_error']=receipt['upload_error']
    else:row.pop('upload_error',None)


def readiness(root,server,state,budget,chosen):
    """Record observed gates and the branch-resolved next block before admission."""
    from fh20r1.plan import blocks_for,cases_for,METHOD_REVISION
    directory=camp(root,server); preflight=read(directory/'preflight_report.json')
    bridge_path=directory/'imported_refs'/('F'+server[1:])/'bridge_manifest.json';bridge=read(bridge_path)
    smoke=preflight.get('batch_smoke',{}); parity=bridge.get('parity',{})
    busy=[p for p in inventory() if not p['waiting_controller']]
    gpu=gpu_processes() if budget['device']!='cpu' else []
    checks=dict(actual_start_authorized=budget.get('actual_start_authorized') is True,
        minimum_effective20_no_deadline=budget.get('minimum_effective_hours')==20 and budget.get('hard_deadline') is None,
        origin_forward_step0_parity=parity.get('status')=='PASS',
        preflight_pass=preflight.get('preflight_pass') is True,
        actual_batch48=smoke.get('actual_batch48_smoke_status')=='PASS' and smoke.get('batch_size')==48,
        local_drain_complete=not busy and gpu==[],
        admission_hold_owned=validate_hold(root,server).get('protocol_id')==HOLD,
        source_hashes_match=preflight.get('registry',{}).get('source_plan_sha256')==budget['source_plan_sha256'] and
            preflight.get('registry',{}).get('source_cases_sha256')==budget['source_cases_sha256'])
    if server=='s2' and read(directory/'donor_manifest.json').get('donor_available') and chosen['condition']!='S2_DONOR_OR_REFERENCE_UNAVAILABLE':
        smoke_n2=preflight.get('n2_teacher_smoke',{})
        checks['n2_teacher_batch48']=smoke_n2.get('passed') is True and smoke_n2.get('actual_batch48_smoke_status')=='PASS' and smoke_n2.get('batch_size')==48
    queue=[dict(block_id=b.block_id,tier=b.tier,run_ids=order_for(b,chosen['condition'],state['runs']),
                status=state['blocks'].get(b.block_id,{}).get('status','PENDING')) for b in blocks_for(server)]
    teacher=next((c for c in cases_for(server) if c.role=='T'),None)
    if teacher and chosen['condition']!='S2_DONOR_OR_REFERENCE_UNAVAILABLE':
        queue.insert(0,dict(block_id=teacher.block_id,tier=teacher.tier,run_ids=[teacher.run_id],
            status='DONE' if state['runs'].get(teacher.run_id,{}).get('status') in DONE else 'PENDING'))
    next_block=next((b for b in queue if b['status']!='DONE'),None)
    ready=all(checks.values()) and budget['device']=='cuda'
    result=dict(campaign_id=CAMPAIGN_ID,server_id=server,checked_at_utc=iso(),launch_ready=ready,
        status='READY' if ready else 'CPU_DIAGNOSTIC_ONLY' if budget['device']=='cpu' else 'BLOCKED',
        checks=checks,branch=chosen,device=budget['device'],actual_batch_smoke=smoke,
        origin_forward_step0_parity=parity,origin_teacher_sha256=bridge.get('teacher_checkpoint_sha256'),
        origin_manifest_sha256=bridge.get('origin_manifest_sha256'),
        origin_teacher_run=bridge.get('origin_reference',{}).get('teacher_run_id'),
        execution_release=preflight.get('source_identity',{}).get('git_release'),numeric_method_revision=METHOD_REVISION,
        consumer_source_identity=preflight.get('source_identity'),
        branch_report_sha256=digest(directory/'branch_record.json') if (directory/'branch_record.json').exists() else None,
        queue_sha256=hashlib.sha256(json.dumps(queue,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        yaml_paths=[f'config/{run}.yaml' for run in budget['config_hashes']],
        loss_routing_test_status='PASS' if preflight.get('loss_routing',{}).get('passed') else 'NOT_PASSED',
        counted_categories=['train','eval','calibration','diagnostic'],
        uploader_schema=dict(mapping='header_name',tab_policy='existing_server_gid_only',
            row_identity=['campaign_id','run_id'],reuse_policy='source_link_only_zero_new_time',
            implementation_sha256=digest(Path(root)/'fh20r1/upload.py') if (Path(root)/'fh20r1/upload.py').exists() else None),
        imported_bridge_sha256=digest(bridge_path),preflight_report_sha256=digest(directory/'preflight_report.json'),
        source_plan_sha256=budget['source_plan_sha256'],source_cases_sha256=budget['source_cases_sha256'],
        registry_sha256=budget['registry_sha256'],config_hashes=budget['config_hashes'],
        actual_start_authorized=budget['actual_start_authorized'],minimum_effective_hours=20.,
        hard_deadline=None,effective_hours=report(root,server)['effective_hours'],
        drain_status='COMPLETE' if checks['local_drain_complete'] else 'BUSY',
        current_processes=busy,gpu_processes=gpu,queue=queue,next_atomic_block=next_block,
        next_run_ids=[] if next_block is None else next_block['run_ids'],
        core_run_count=sum(len(b['run_ids']) for b in queue if b['tier']=='CORE'),
        reserve_run_count=sum(len(b['run_ids']) for b in queue if b['tier']=='RESERVE'),
        registered_definition_count=len(cases_for(server)))
    write(directory/'readiness_report.json',result)
    if budget['device']=='cuda' and not ready:raise RuntimeError('FH20R1 readiness gates failed; see readiness_report.json')
    return result


def publish_completion(root,server,state,chosen):
    from fh20r1.plan import blocks_for
    core=all(state['blocks'].get(b.block_id,{}).get('status')=='DONE' for b in blocks_for(server) if b.tier=='CORE')
    timing=report(root,server);pending=[run for run,row in state['runs'].items()
        if row.get('status') in DONE and row.get('upload_pending',True)]
    satisfied=should_stop(core,timing['effective_hours'])
    certified='COMPLETE_20HPLUS_WITH_BLOCKED_DONOR' if server=='s2' and chosen['condition']=='S2_DONOR_OR_REFERENCE_UNAVAILABLE' else 'COMPLETE_20HPLUS'
    final=('WORK_COMPLETE_UPLOAD_PENDING' if pending else certified) if satisfied else 'CAPACITY_EXHAUSTED_BELOW20' if core else 'INCOMPLETE_CORE'
    state['status']=final;write(camp(root,server)/'status.json',state)
    write(camp(root,server)/'completion_report.json',dict(status=final,core_complete=core,
        work_complete=satisfied,all_requested_results_uploaded=not pending,pending_upload_run_ids=pending,
        certifiable_status_when_uploaded=certified if satisfied else None,branch=chosen,**timing,
        runs=state['runs'],blocks=state['blocks']))
    return final


def admit_disk(root,server,state,block_id,run_ids):
    """Reserve a complete atomic block; never delete candidates to make space."""
    directory=camp(root,server);disk=read(directory/'preflight_report.json').get('disk',{})
    estimates=disk.get('per_run_estimate_bytes',{})
    reuse=read(directory/'reuse_manifest.json').get('runs',{})
    pending=[run for run in run_ids if state['runs'].get(run,{}).get('status') not in DONE
             and not (reuse.get(run,{}).get('validated') and reuse.get(run,{}).get('official_complete'))]
    if any(run not in estimates for run in pending):raise ValueError('Missing atomic-block disk estimate; preflight must cover alternatives/reserves')
    needed=sum(estimates[run] for run in pending);safety=disk.get('minimum_safety_bytes',2*1024**3)
    free=shutil.disk_usage(root).free
    result=dict(at_utc=iso(),block_id=block_id,run_ids=run_ids,pending_run_ids=pending,
        free_bytes=free,additional_estimate_bytes=needed,minimum_safety_bytes=safety,
        passed=free>=needed+safety,policy='full pending block upper bound; original artifacts never deleted')
    write(directory/'disk_admission.json',result)
    if not result['passed']:
        state.update(status='DISK_BLOCKED',disk_admission=result);write(directory/'status.json',state)
    admission_event(root,server,'BLOCK_DISK_ADMISSION' if result['passed'] else 'BLOCK_DISK_REJECTED',**result)
    return result['passed']


def run_case(root,server,case,state,budget):
    wd=Path(root)/'work_dir'/case.run_id; row=state['runs'].setdefault(case.run_id,dict(status='PENDING'))
    if row['status'] in DONE:
        admission_event(root,server,'CASE_SKIPPED_DONE',run_id=case.run_id,block_id=case.block_id,status=row['status'])
        return True
    if row['status']=='FAILED':return False
    if case.role=='S' and case.teacher_alias=='N2PL':
        reference=camp(root,server)/'imported_refs/N2PL/bridge_manifest.json'
        if not reference.exists() or read(reference).get('status') in {'UNAVAILABLE','BLOCKED_REFERENCE'}:
            row['status']='REFERENCE_UNAVAILABLE'
            admission_event(root,server,'CASE_REFERENCE_UNAVAILABLE',run_id=case.run_id,block_id=case.block_id)
            return None
        if not read(reference).get('complete'):
            raise ValueError('Present N2PL bridge is invalid; integrity errors must not silently fall back')
    reuse=read(camp(root,server)/'reuse_manifest.json').get('runs',{}).get(case.run_id,{})
    if reuse:
        if not reuse.get('validated') or not reuse.get('official_complete') or not reuse.get('identity_sha256'):
            raise ValueError('Unvalidated semantic reuse is forbidden')
        write(wd/'meta/reuse_reference.json',dict(campaign_id=CAMPAIGN_ID,run_id=case.run_id,**reuse))
        row.update(status='DONE_REUSED',reuse=reuse,credited_new_seconds=0)
        admission_event(root,server,'CASE_SKIPPED_REUSE',run_id=case.run_id,block_id=case.block_id,
                        source_run_id=reuse.get('source_run_id'),identity_sha256=reuse['identity_sha256'],credited_new_seconds=0)
        upload_case(root,case,row,budget.get('upload_enabled',False));return True
    if not official_done(root,case.run_id):
        trained=read(wd/'meta/training_status.json').get('training_complete',False)
        finishing=trained and not read(wd/'official/raw_grid.json').get('complete')
        if not trained or finishing:
            resume=(wd/'last/training_state.pt').exists()
            if (row.get('training_started') or finishing) and not resume:
                row.update(status='FAILED',reason='interrupted_without_exact_fullstate');return False
            row.update(status='TRAINING',training_started=True);write(camp(root,server)/'status.json',state)
            args=['tools/fh20r1_train.py','--config',f'config/{case.run_id}.yaml','--device',budget['device']]
            if resume:args.append('--resume')
            admission_event(root,server,'CASE_RESUME' if resume else 'CASE_START',run_id=case.run_id,
                            block_id=case.block_id,finishing_evaluation_only=finishing)
            rc=command(root,args,wd/'fh20r1_train.log');ingest_timing(root,server,case.run_id)
            admission_event(root,server,'CASE_TRAIN_RETURN',run_id=case.run_id,block_id=case.block_id,exit_code=rc)
            if rc==69 and case.teacher_alias=='N2PL':row['status']='REFERENCE_UNAVAILABLE';return None
            if rc:
                row.update(status='PAUSED' if rc==75 else 'FAILED',exit_code=rc);return False
        before=read(wd/'official/postrun_status.json');start=time.time()
        rc=command(root,['tools/fh20r1_postrun.py',case.run_id,'--device',budget['device']],wd/'fh20r1_postrun.log')
        if rc or not official_done(root,case.run_id):row.update(status='EVALUATION_PENDING',exit_code=rc);return False
        if not before.get('official_complete'):
            record_interval(root,server,interval_id='postrun:'+case.run_id,kind='eval',start_utc=start,end_utc=time.time(),
                            run_id=case.run_id,evidence=str((wd/'official/postrun_status.json').relative_to(root)))
    ingest_timing(root,server,case.run_id)
    row.update(status='DONE',training_complete=True)
    admission_event(root,server,'CASE_COMPLETE',run_id=case.run_id,block_id=case.block_id,
                    actual_updates=50000,official_complete=True)
    upload_case(root,case,row,budget.get('upload_enabled',False))
    return True


def stage(root,server,state,name,args,output,kind=None):
    if state.setdefault('stages',{}).get(name,{}).get('complete'):return read(output)
    was_complete=read(output).get('complete',False)
    start=time.time();rc=command(root,args,camp(root,server)/(name+'.log'));result=read(output)
    if rc or not result.get('complete',result.get('preflight_pass',False)):
        raise RuntimeError(f'{name} failed rc={rc}; see {name}.log')
    if kind and not was_complete and not result.get('reused',False):
        record_interval(root,server,interval_id='stage:'+name,kind=kind,start_utc=start,end_utc=time.time(),
                        evidence=str(Path(output).relative_to(root)))
    state['stages'][name]=dict(complete=True,output=str(output));write(camp(root,server)/'status.json',state)
    return result


def run_campaign(root,server,wait=True):
    from fh20r1.plan import blocks_for,case_for,cases_for
    root=Path(root);directory=camp(root,server);budget=load_budget(root,server)
    with locked(directory/'.runner.lock'):
        state=read(directory/'status.json',dict(status='REGISTERED',runs={},blocks={},stages={}))
        if state.get('status') in CLOSED:return 0
        if not wait_resources(root,server,budget,state,wait):return 3
        stage(root,server,state,'import',['tools/fh20r1_reference.py','import','--server',server,'--device',budget['device']],
              directory/'imported_refs'/('F'+server[1:])/'bridge_manifest.json')
        stage(root,server,state,'preflight',['tools/fh20r1_preflight.py','--server',server,'--device',budget['device']],directory/'preflight_report.json')
        diagnostic=stage(root,server,state,'diagnostic',['tools/fh20r1_diagnose.py','--server',server,'--device',budget['device']],directory/'diagnostics/report.json','diagnostic')
        chosen=read(directory/'branch_record.json')
        if chosen and 'condition' not in chosen:chosen=branch(root,server,chosen['branch'],chosen.get('reason','local diagnostic'))
        if not chosen:
            condition='PRIMARY' if server=='s2' else diagnostic.get('branch','STANDARD')
            if server=='s1' and condition not in {'S1_A_SUPPORT','S1_NO_A_SUPPORT_OR_INCONCLUSIVE'}:
                raise ValueError('s1 diagnostic did not establish an admissible branch')
            chosen=branch(root,server,condition,diagnostic.get('branch_reason','local diagnostic'),str(directory/'diagnostics/report.json'))
        # Once a verified N2 reference exists, disappearance of the original
        # donor must not reselect the experiment branch on a process restart.
        n2=read(directory/'imported_refs/N2PL/bridge_manifest.json')
        if server=='s2' and chosen['condition']=='S2_N2PL_REFERENCE_READY' and n2 and not n2.get('complete'):
            raise ValueError('Previously ready N2PL bridge is invalid; refusing silent fallback')
        if server=='s2' and chosen['condition'] in {'PRIMARY','S2_N2PL_REFERENCE_READY'} and not (
                chosen['condition']=='S2_N2PL_REFERENCE_READY' and n2.get('complete')):
            rc=command(root,['tools/fh20r1_reference.py','donor','--server',server],directory/'donor.log')
            if rc:raise RuntimeError('Donor probe failed without an explicit BLOCKED_DONOR record')
            donor=read(directory/'donor_manifest.json')
            if not donor.get('donor_available'):
                chosen=branch(root,server,'S2_DONOR_OR_REFERENCE_UNAVAILABLE','BLOCKED_DONOR')
            else:
                teacher=next(c for c in cases_for(server) if c.role=='T')
                readiness(root,server,state,budget,chosen)
                if not admit_disk(root,server,state,teacher.block_id,[teacher.run_id]):return 3
                admission_event(root,server,'BLOCK_START',block_id=teacher.block_id,tier=teacher.tier,
                                branch=chosen['condition'],preparation=True)
                teacher_ok=run_case(root,server,teacher,state,budget)
                if teacher_ok is None:
                    chosen=branch(root,server,'S2_DONOR_OR_REFERENCE_UNAVAILABLE','explicit N2 reference unavailable')
                elif not teacher_ok:
                    state['status']='PAUSED' if state['runs'][teacher.run_id]['status']=='PAUSED' else 'BLOCK_FAILED'
                    admission_event(root,server,'BLOCK_INCOMPLETE',block_id=teacher.block_id,status=state['status'])
                    write(directory/'status.json',state);return 2
                else:
                    try:
                        stage(root,server,state,'calibration',['tools/fh20r1_reference.py','calibrate','--server',server,
                              '--teacher-run',teacher.run_id,'--device',budget['device']],directory/'imported_refs/N2PL/bridge_manifest.json','calibration')
                        chosen=branch(root,server,'S2_N2PL_REFERENCE_READY','verified N2 exact50K reference')
                        admission_event(root,server,'BLOCK_COMPLETE',block_id=teacher.block_id,tier=teacher.tier,
                                        run_ids=[teacher.run_id],preparation=True)
                        upload_case(root,teacher,state['runs'][teacher.run_id],budget.get('upload_enabled',False))
                    except RuntimeError as error:
                        # Absent/unavailable calibration can take the declared
                        # fallback, but present corrupted evidence is not absence.
                        failed_bridge=read(directory/'imported_refs/N2PL/bridge_manifest.json')
                        if failed_bridge.get('status') not in {'UNAVAILABLE','BLOCKED_REFERENCE'}:
                            raise
                        chosen=branch(root,server,'S2_DONOR_OR_REFERENCE_UNAVAILABLE',str(error))
        readiness(root,server,state,budget,chosen)
        blocks=blocks_for(server)
        for block in blocks:
            if state['blocks'].get(block.block_id,{}).get('status')=='DONE':continue
            if block.tier=='RESERVE' and should_stop(True,report(root,server)['effective_hours']):
                admission_event(root,server,'RESERVE_REMAINDER_SKIPPED',first_block_id=block.block_id,
                                reason='CORE complete and qualified effective time >=20h');break
            row=state['blocks'].setdefault(block.block_id,{});row.update(status='RUNNING',tier=block.tier)
            admission_event(root,server,'BLOCK_START',block_id=block.block_id,tier=block.tier,branch=chosen['condition'])
            while True:
                order=order_for(block,chosen['condition'],state['runs']);row['run_ids']=order
                if not admit_disk(root,server,state,block.block_id,order):
                    row['status']='WAITING_DISK';write(directory/'status.json',state);return 3
                write(directory/'status.json',state)
                changed=False
                for run in order:
                    if not wait_resources(root,server,budget,state,wait):return 3
                    ok=run_case(root,server,case_for(run),state,budget)
                    write(directory/'status.json',state)
                    if ok is None:
                        chosen=branch(root,server,'S2_DONOR_OR_REFERENCE_UNAVAILABLE','N2PL unavailable before new Student')
                        readiness(root,server,state,budget,chosen);changed=True;break
                    if not ok:
                        state['status']='PAUSED' if state['runs'][run]['status']=='PAUSED' else 'BLOCK_FAILED'
                        row['status']='INCOMPLETE';write(directory/'status.json',state);return 2
                if not changed:break
            row['status']='DONE';write(directory/'status.json',state)
            admission_event(root,server,'BLOCK_COMPLETE',block_id=block.block_id,tier=block.tier,run_ids=row['run_ids'])
        publish_completion(root,server,state,chosen)
    return 0


def launch(root,server):
    directory=camp(root,server)
    try:
        with locked(directory/'.runner.lock'):pass
    except BlockingIOError:return dict(status='ALREADY_RUNNING')
    with (directory/'runner.log').open('a') as stream:
        proc=subprocess.Popen([sys.executable,str(Path(root)/'tools/fh20r1_runner.py'),'run','--server',server],
            cwd=root,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    return dict(status='RUNNER_SUBMITTED',pid=proc.pid)


def retry_upload(root,server):
    from fh20r1.plan import case_for
    directory=camp(root,server)
    with locked(directory/'.runner.lock'):
        state=read(directory/'status.json')
        if not state:raise ValueError('Campaign has no completed runs to upload')
        for run,row in state.get('runs',{}).items():
            if row.get('status') in DONE:upload_case(root,case_for(run),row)
        write(directory/'status.json',state)
        if (directory/'completion_report.json').exists():
            publish_completion(root,server,state,read(directory/'branch_record.json'))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['start','run','status','retry-upload'])
    p.add_argument('--server',choices=['s1','s2','s3','s4','s5'])
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'],help='Actual campaigns require cuda; cpu is dry-run diagnostics only')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--no-upload',action='store_true');a=p.parse_args()
    server=detect_server(ROOT,a.server);directory=camp(ROOT,server)
    if a.action=='status':
        print(json.dumps(dict(budget=read(directory/'campaign_budget.json'),state=read(directory/'status.json'),
             timing=report(ROOT,server) if (directory/'campaign_budget.json').exists() else {}),indent=2));return 0
    if a.action=='start':
        if not a.dry_run and a.device!='cuda':raise ValueError('Actual FH20R1 campaigns require CUDA; use preflight --check-only for CPU diagnostics')
        if a.dry_run:result=prepare(ROOT,server,a.device,not a.no_upload,True)
        else:
            with locked(directory/'.start.lock'):
                result=prepare(ROOT,server,a.device,not a.no_upload);result['launch']=launch(ROOT,server)
        print(json.dumps(result,indent=2));return 0
    if a.action=='retry-upload':
        return retry_upload(ROOT,server)
    if read(directory/'campaign_budget.json').get('device')!='cuda':
        raise ValueError('Actual FH20R1 campaigns require a CUDA-authorized budget')
    return run_campaign(ROOT,server)


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,RuntimeError,BlockingIOError) as error:
        print(f'FH20R1: {error}',file=sys.stderr);sys.exit(2)
