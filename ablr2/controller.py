"""Durable, sensor-local adaptive campaign. No clock or launch at import/build.

Training is admitted one registered run at a time. Entire five-sweep panels are
registered before observing results, and lease renewal never reinitializes them.
"""
from dataclasses import asdict, replace
import datetime as dt
import math
import os
import re
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

import yaml

from ablr2.common import (ROOT, CAMPAIGN_ID, append_event, apply_runtime_policy, atomic_json,
    before_deadline, camp, immutable_json, locked, object_sha, read, read_json, read_config, run_dir,
    runtime_context, RuntimePaused, sha256, source_identity, utcnow)
from ablr2.plan import (SERVERS, LANES, RECIPES, GRAPH, MAIN_CASES, cases_for, case_for,
    full_wave, recheck_cases, screen_cases, build_config, validate_config, verify_sources,
    registry_document, registry_sha256, SensorSpec, verify_lane, validate_case,
    verify_extension_sources,campaign_id,BOOT_SEED_BASES)

POLICY = dict(schema='ABLR2X_SERVICE_POLICY_v1', default_lease_hours=None,
    service_horizon='UNTIL_OPERATOR_STOP',explicit_continuous_authorization=True,
    automatic_lease_renewal=False, max_campaign_cycles=None, max_technical_retries=2,
    runtime_safety_factor=1.15, retain_all_artifacts=True, bootstrap_run_hours={'T':8.,'S':8.},
    bootstrap_calibration_hours=3., bootstrap_estimates='CONSERVATIVE_UNMEASURED_NOT_A_SPEED_CLAIM',
    master_seed=20260921, checkpoint_sensitive_definition='VAL_GAIN_AND_EXACT50K_INVERSE_REGISTERED_GAIN',
    test_scope='TEST_AWARE_DEV', optional_cases_enabled=False)
STOP_COMMANDS = ('STOP_AFTER_RUN','STOP_AFTER_SWEEP','PAUSE_AFTER_BLOCK','STOP_NOW_SAFE','CONTINUE')


def _save(root, server, state):
    state['updated_at_utc'] = utcnow()
    atomic_json(camp(root,server) / 'state.json',state)


def _state(root, server):
    value = read(camp(root,server) / 'state.json')
    if not value: raise ValueError('Run build before opening the lane')
    if (value.get('campaign_id'),value.get('server')) != (campaign_id(server),server):
        raise ValueError('Wrong campaign state')
    return value


def _register_stage(root, server, state, cases, stage_id, kind, **meta):
    """Write the complete immutable task list, then publish its state pointer."""
    cases=tuple(cases)
    if not re.fullmatch('[A-Za-z0-9_]+',stage_id) or not cases or len({c.run_id for c in cases})!=len(cases):
        raise ValueError('Safe stage ID and unique complete task list required')
    for case in cases:
        validate_case(case)
        if case.server_id!=server: raise ValueError('Cannot register another server lane')
        if case.phase!=kind: raise ValueError('Stage cannot mix campaign phases')
    if kind in ('BOOT5','REFRESH5','VERIFY5'):
        sweeps={c.sweep for c in cases}
        if len(cases)!=100 or len(sweeps)!=5: raise ValueError('Full wave requires all100 registered tasks')
        for sweep in sweeps:
            members=[c for c in cases if c.sweep==sweep]
            if {c.case_id for c in members}!=set(MAIN_CASES)|{'TPLUS','TZERO'} or len(members)!=20:
                raise ValueError('Full wave requires both Teachers and all18 components per sweep')
    elif kind=='RECHECK5':
        relation=GRAPH[meta['relation_id']]
        components={relation['parent'],relation['child'],'C07'}
        if len({c.sweep for c in cases})!=5 or any(c.role!='S' for c in cases):
            raise ValueError('Recheck needs five complete Student pairs')
        for sweep in {c.sweep for c in cases}:
            group=[c for c in cases if c.sweep==sweep]
            if len(group)!=len(components) or {c.case_id for c in group}!=components:
                raise ValueError('Recheck parent/child/FULL tasks incomplete')
    elif kind=='FIT_ROUND':
        if {c.sweep for c in cases}!={'P01','P02'}: raise ValueError('Fitting needs both fixedDEVblocks')
    else: raise ValueError('Unknown registered stage kind')
    folder = camp(root,server)
    document = dict(stage_id=stage_id,kind=kind,cases=[asdict(c) for c in cases],**meta)
    immutable_json(folder / 'stages' / f'{stage_id}.json',document)
    registry = read(folder / 'cases.json', {'cases':{}})
    for case in cases:
        old = registry['cases'].get(case.run_id)
        if old is not None and old != asdict(case): raise ValueError('Registered task changed')
        registry['cases'][case.run_id] = asdict(case)
    atomic_json(folder / 'cases.json',registry)
    if not any(s['stage_id'] == stage_id for s in state['stages']):
        state['stages'].append(dict(stage_id=stage_id,kind=kind,run_ids=[c.run_id for c in cases],
                                  complete=False,**meta))
        append_event(folder / 'task_ledger.jsonl','REGISTER_STAGE',**document)
    state['active_stage'] = stage_id
    _save(root,server,state)


def build(root=ROOT, server=None):
    """Design publication only: no runtime lease, data cache, GPU or training."""
    verify_sources(root)
    verify_extension_sources(root)
    servers = SERVERS if server is None else (server,)
    for lane in servers:
        verify_lane(lane)
        folder = camp(root,lane)
        with locked(folder / 'registration.lock'):
            immutable_json(folder / 'design_ablr2x.json',registry_document())
            immutable_json(folder / 'policy_ablr2x.json',POLICY)
            if (folder / 'state.json').is_file():
                # No cursor, old stage list, seed ledger, cost or lease rewrite.
                from ablr2.extension import sync
                state=_state(root,lane)
                _sync_seed_log(folder,read_json(folder/'seed_ledger.json'))
                sync(root,lane,state)
                continue
            seeds=read(folder / 'seed_ledger.json',[])
            jobs=cases_for(lane)
            seed_resolution=None
            if lane=='s3':
                from ablr2.seeds import resolve_gf2_boot
                jobs,seed_resolution=resolve_gf2_boot(root)
            for case in jobs:
                key=f'BOOT5|{lane}|{case.sweep}|{case.role}'
                existing=[r for r in seeds if r['key']==key]
                if len(existing)>1 or (existing and existing[0]['seed']!=case.seed): raise ValueError('BOOT seed ledger changed')
                if not existing:
                    allocation=next((r for r in seed_resolution['mapping'] if r['sweep']==case.sweep and r['role']==case.role),None) if seed_resolution else None
                    collided=bool(allocation and allocation['resolved_seed']!=allocation['authored_seed'])
                    seeds.append(dict(key=key,seed=case.seed,phase='BOOT5',sensor=case.sensor,
                        selection='PERFORMANCE_INDEPENDENT_SHA256_COLLISION_ONLY' if collided else 'AUTHOR_SUPPLIED_CSV',
                        **(dict(boot_seed_resolution_sha256=object_sha(seed_resolution)) if collided else {})))
            atomic_json(folder / 'seed_ledger.json',seeds)
            _sync_seed_log(folder,seeds)
            state = dict(campaign_id=campaign_id(lane),server=lane,sensor=LANES[lane],status='DEFINED_NOT_LAUNCHED',
                incumbent='R00',recipe_revision='r000',cycle=0,stages=[],runs={},observations=[],
                rechecked={},tried_recipes=[],verification_recipes=[],low_information_cycles=0)
            _register_stage(root,lane,state,jobs,'BOOT5','BOOT5',recipe_id='R00',recipe_revision='r000')
    return dict(campaign_ids={s:campaign_id(s) for s in servers},servers=list(servers),training_cases=100*len(servers),
                activated=False,lease_created=False)


def _sync_seed_log(folder,seeds):
    import json
    path=folder / 'seed_ledger.jsonl'
    records=[json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
    if len({r['key'] for r in seeds})!=len(seeds) or len({r['key'] for r in records})!=len(records):
        raise ValueError('Seed ledger contains duplicate allocation keys')
    expected={r['key']:r for r in seeds}
    if any(r['key'] not in expected or r['seed']!=expected[r['key']]['seed'] for r in records):
        raise ValueError('Seed allocation audit log differs from immutable allocations')
    logged={r['key'] for r in records}
    for row in seeds:
        if row['key'] not in logged: append_event(path,'REGISTER_SEED',**row)


def grant_lease(root, server, hours=72., *, now=None):
    """Explicit operator action only; record extensions without resetting spent time."""
    if not math.isfinite(hours) or hours <= 0 or hours > 72:
        raise ValueError('Explicit lease duration must be within (0,72] hours')
    build(root,server)
    folder = camp(root,server)
    current = dt.datetime.fromisoformat(now or utcnow())
    if current.tzinfo is None: raise ValueError('Lease needs timezone')
    with locked(folder / 'lease.lock'):
        previous=read(folder / 'lease.json')
        expires = current + dt.timedelta(hours=hours)
        if previous and expires < dt.datetime.fromisoformat(previous['expires_utc']):
            raise ValueError('Renewal cannot shorten an active lease; use a stop command')
        value=dict(campaign_id=campaign_id(server),server=server,sensor=LANES[server],
            granted_at_utc=current.isoformat(),expires_utc=expires.isoformat(),
            first_granted_at_utc=previous.get('first_granted_at_utc',current.isoformat()),
            sequence=previous.get('sequence',0)+1,operator_action=True,automatic_renewal=False)
        immutable_json(folder / 'leases' / f'{value["sequence"]:06}.json',value)
        atomic_json(folder / 'lease.json',value)
        append_event(folder / 'lease_ledger.jsonl','OPERATOR_LEASE',lease=value)
    return value


def control(root, server, command):
    if command not in STOP_COMMANDS: raise ValueError('Unknown stop command')
    verify_lane(server)
    if (camp(root,server)/'state.json').is_file():_state(root,server)
    value=dict(command=command,at_utc=utcnow(),operator_action=True)
    atomic_json(camp(root,server) / 'control.json',value)
    append_event(camp(root,server) / 'control_ledger.jsonl','OPERATOR_CONTROL',**value)
    return value


def _lease(root,server):
    from ablr2.common import authorization_context
    authorization,_=authorization_context(root,server)
    return authorization


def register_runtime(root,server):
    from ablr2.plan import campaign_id,verify_extension_sources
    apply_runtime_policy(root)
    verify_sources(root)
    verify_extension_sources(root)
    value=dict(campaign_id=campaign_id(server),server=server,source_identity=source_identity(root),
               registry_sha256=registry_sha256(),policy_sha256=object_sha(POLICY))
    immutable_json(camp(root,server) / 'registration_ablr2x.json',value)
    return value


def verify_registration(root,server):
    from ablr2.plan import campaign_id
    apply_runtime_policy(root)
    previous=read_json(camp(root,server) / 'registration_ablr2x.json')
    expected=dict(campaign_id=campaign_id(server),server=server,source_identity=source_identity(root),
                  registry_sha256=registry_sha256(),policy_sha256=object_sha(POLICY))
    if previous != expected: raise ValueError('Registered source/runtime/policy changed')
    return previous


def resolve_config(root,case):
    from ablr2.references import reference_path, validate_reference, teacher_endpoint
    folder=camp(root,case.server_id)
    data_path=folder / 'dataset_manifest.json'
    data=read_json(data_path)
    path=run_dir(case.run_id,root) / 'meta/config.resolved.yaml'
    if path.is_file():
        existing=read_config(path)
        if validate_config(existing,require_bound=True)!=case:
            raise ValueError('Existing config differs from registered case')
        if object_sha(read_json(existing['ablr2']['dataset_manifest']))!=object_sha(data):
            raise ValueError('Existing config uses different native data')
        return path
    cfg=build_config(case)
    field=cfg['ablr2']
    field.update(dataset_manifest=str(data_path),
        sensor_spec=SensorSpec(case.sensor,tuple(data['band_order'])).to_dict(),
        runtime_policy_sha256=sha256(Path(root) / 'ablr2/runtime_policy.json'))
    for split,key in (('train','train_feeder_args'),('val','val_feeder_args'),
                      ('rr','test_reduced_feeder_args'),('fr','test_full_feeder_args')):
        cfg[key]['dataroot']=data['splits'][split]['dataroot']
    if case.requires_teacher:
        if case.requires_calibration:
            manifest,_,_,paths=validate_reference(case.reference_id,case.server_id,root,dataset_manifest=data)
            field.update(reference_manifest=str(reference_path(case.reference_id,case.server_id,root)),
                teacher_checkpoint=str(paths['teacher_checkpoint']),teacher_sha256=manifest['teacher_checkpoint_sha256'],
                tau_R=manifest['tau_R'],q_ref=manifest['q_ref'],q_cache=str(paths['q_cache_path']),
                q_cache_sha256=manifest['q_cache_sha256'],s_bar=manifest['s_bar'],s_bar_sha256=manifest['s_bar_sha256'])
        else:
            teacher,_,_,identity,_,paths=teacher_endpoint(run_dir(case.teacher_run_id,root),root,case.server_id)
            if teacher.reference_id != case.reference_id: raise ValueError('Clone/output-only Teacher mismatch')
            field.update(teacher_checkpoint=str(paths['candidate'] / 'model.safetensors'),teacher_sha256=identity['model_sha256'])
    validate_config(cfg,require_bound=True)
    # JSON is valid YAML; immutable publication avoids partial configuration files.
    immutable_json(path,cfg)
    return path


def authorize_train(root,server,config_path):
    verify_registration(root,server)
    cfg=read_config(config_path)
    case=validate_config(cfg,require_bound=True)
    if case.server_id != server or case_for(case.run_id,root) != case:
        raise ValueError('Trainer must use the exact preregistered local case')
    from ablr2.seeds import validate_boot_admission
    validate_boot_admission(root,case)
    runtime_context(cfg,root)
    state=_state(root,server)
    if state.get('status') != 'TRAINING' or state.get('active_run') != case.run_id:
        raise ValueError('Controller has not admitted this trainer')
    ready=read_json(camp(root,server) / 'preflight_ablr2x.json')
    if (not ready.get('complete') or ready['source_identity'] != source_identity(root)
            or ready['dataset_manifest_sha256'] != object_sha(read_json(cfg['ablr2']['dataset_manifest']))):
        raise ValueError('Missing or changed local P0 proof')
    return case


def _command(root,server,args,log,deadline):
    log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','tools/ablr2_runner.py',*args,'--server',server]
    if deadline:command+=['--deadline',deadline]
    started=time.monotonic()
    with log.open('a') as stream:
        child=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        # Trainer and evaluators save/check their own boundaries. Never kill a writer.
        code=child.wait()
    return code,(time.monotonic()-started)/3600.


def estimate_hours(case,state):
    values=[r['hours'] for r in state['observations'] if r.get('completed') and r['role']==case.role
            and r['sensor']==case.sensor and r['server']==case.server_id and r.get('kind')=='RUN']
    measured=max(values[-3:]) if values else POLICY['bootstrap_run_hours'][case.role]
    if case.role=='T':
        values=[r['hours'] for r in state['observations'] if r.get('completed')
            and r.get('kind')=='CALIBRATION' and r['sensor']==case.sensor and r['server']==case.server_id]
        measured+=max(values[-3:]) if values else POLICY['bootstrap_calibration_hours']
    return measured*POLICY['runtime_safety_factor']


def _record_action(root,server,state,case,action,args):
    from ablr2.migration import execution_root
    execution=execution_root(root,case)
    deadline=_lease(root,server)['expires_utc']
    if execution.resolve()!=Path(root).resolve():
        original_lease=read(camp(root,server)/'lease.json')
        if (original_lease.get('server')!=server or original_lease.get('sensor')!=case.sensor
                or not original_lease.get('expires_utc') or not before_deadline(original_lease['expires_utc'])):
            raise RuntimePaused('WAIT_ORIGINAL_FINITE_LEASE: until-stop approval does not extend original admitted jobs')
        deadline=original_lease['expires_utc']
    folder=camp(root,server)
    row=state['runs'].setdefault(case.run_id,{})
    count=row.get(action+'_attempts',0)+1
    row[action+'_attempts']=count
    _save(root,server,state)
    append_event(folder / 'all_attempts.jsonl','ACTION_STARTED',run_id=case.run_id,action=action,attempt=count,
                 teacher_seed=case.teacher_seed,student_seed=case.student_seed,case=asdict(case),
                 execution_root=str(execution),execution_deadline=deadline)
    code,hours=_command(execution,server,args,folder / 'logs' / f'{case.run_id}.{action}.log',deadline)
    row[action+'_hours']=row.get(action+'_hours',0.)+hours
    append_event(folder / 'all_attempts.jsonl','ACTION_ENDED',run_id=case.run_id,action=action,attempt=count,
                 exit_code=code,hours=hours)
    _save(root,server,state)
    if code in (75,124): raise RuntimePaused('SAFE_PAUSE: '+action)
    if code==74:
        retries=row.get(action+'_technical_retries',0)
        if retries>=POLICY['max_technical_retries']: raise RuntimePaused('TECHNICAL_RETRIES_EXHAUSTED')
        row[action+'_technical_retries']=retries+1
        _save(root,server,state)
        if action=='train':
            args=[a for a in args if a!='--resume']
            wd=run_dir(case.run_id,root)
            if (wd / 'last/training_state.pt').is_file(): args+=['--resume']
            elif (wd / 'meta/training_start_manifest.json').is_file():
                raise ValueError('Transient failure has no full-state recovery; do not restart seed')
        return _record_action(root,server,state,case,action,args)
    if code!=0 and not (code==2 and action=='postrun'):
        raise ValueError(f'{action} failed with {code}; retain this seed and inspect its log')
    return code


def _maybe_reuse(root,server,state,case,cfg):
    """Reuse exact DEV screen jobs with a checked receipt, never as a new seed."""
    from ablr2.common import execution_identity
    from ablr2.analysis import case_report
    if case.phase!='FIT_ROUND' or case.role!='S': return False
    wd=run_dir(case.run_id,root)
    if (wd / 'meta/training_start_manifest.json').exists(): return False
    signature=execution_identity(cfg,root)
    for run,row in state['runs'].items():
        if run==case.run_id or not row.get('complete') or row.get('reused'): continue
        other=case_for(run,root)
        if other.role!='S' or other.seed!=case.seed: continue
        physical=run_dir(run,root)
        other_cfg=read_config(physical / 'meta/config.resolved.yaml')
        other_signature=execution_identity(other_cfg,root)
        if other_signature!=signature: continue
        observation=case_report(other,root=root)
        if observation['source_identity']!=source_identity(root): raise ValueError('Cannot reuse a different execution release')
        receipt=dict(schema='ABLR2_REUSE_v1',logical_case=asdict(case),physical_run_id=run,
            logical_execution_signature=signature,physical_execution_signature=other_signature,
            source_identity=source_identity(root),data_sha256=observation['data_sha256'],
            physical_raw_grid_sha256=sha256(physical / 'official/raw_grid.json'))
        immutable_json(wd / 'meta/reuse.json',receipt)
        state['runs'][case.run_id].update(complete=True,reused=True,physical_run_id=run,status='NUMERICALLY_IDENTICAL_REUSE')
        append_event(camp(root,server) / 'all_attempts.jsonl','REUSE_NOT_NEW_OBSERVATION',
                     run_id=case.run_id,physical_run_id=run,execution_signature=signature)
        _save(root,server,state)
        return True
    return False


def _run_case(root,server,state,case,upload):
    from ablr2.references import reference_path,validate_reference
    folder,wd=camp(root,server),run_dir(case.run_id,root)
    row=state['runs'].setdefault(case.run_id,{})
    if case.case_id=='C17':
        from ablr2.extension import prepare_anchor
        anchor=prepare_anchor(case,root)
        if not state['runs'].get(anchor.run_id,{}).get('complete'):
            raise ValueError('PAIR_REPAIR anchor requires its own admission before C17')
    cfg_path=resolve_config(root,case)
    cfg=read_config(cfg_path)
    if _maybe_reuse(root,server,state,case,cfg): return
    if (wd / 'candidates/50000/identity.json').is_file():
        from ablr2.training import recover_exact_endpoint
        recover_exact_endpoint(root,wd)
    training=read(wd / 'meta/training_status.json')
    if not training.get('training_complete'):
        args=['train','--config',str(cfg_path)]
        if (wd / 'last/training_state.pt').is_file(): args+=['--resume']
        elif (wd / 'meta/training_start_manifest.json').is_file():
            raise ValueError('Interrupted run lacks a complete restart state; refusing seed restart')
        state.update(status='TRAINING',active_run=case.run_id)
        _save(root,server,state)
        _record_action(root,server,state,case,'train',args)
        training=read(wd / 'meta/training_status.json')
    if not training.get('training_complete') or training.get('actual_updates')!=50000:
        raise ValueError('Trainer did not publish a validated exact50K endpoint')
    state['status']='EVALUATING'; _save(root,server,state)
    official=read(wd / 'official/postrun_status.json')
    if not official.get('official_complete'):
        args=['postrun','--run',case.run_id]+(['--upload'] if upload else [])
        _record_action(root,server,state,case,'postrun',args)
    official=read(wd / 'official/postrun_status.json')
    if not official.get('official_complete'): raise ValueError('Official grid incomplete')
    row['upload_pending']=upload and not official.get('sheet_uploaded')
    if case.role=='T':
        state['status']='CALIBRATING'; _save(root,server,state)
        if not reference_path(case.reference_id,server,root).is_file():
            _record_action(root,server,state,case,'calibration',['calibrate','--run',case.run_id])
        validate_reference(case.reference_id,server,root,dataset_manifest=read_json(folder / 'dataset_manifest.json'))
        row['reference_ready']=True
    if not row.get('complete'):
        # Timing includes train/inline validation/official postrun, separately recorded calibration.
        state['observations'].append(dict(kind='RUN',server=server,sensor=case.sensor,role=case.role,
            hours=row.get('train_hours',0.)+row.get('postrun_hours',0.),completed=True))
        if case.role=='T':
            state['observations'].append(dict(kind='CALIBRATION',server=server,sensor=case.sensor,role='T',
                hours=row.get('calibration_hours',0.),completed=True))
    row.update(complete=True,status='COMPLETE',completed_at_utc=utcnow())
    _save(root,server,state)


def _cases(stage,root): return [case_for(run,root) for run in stage['run_ids']]


def _seed_wave(root,server,state,phase,stage_id):
    folder=camp(root,server)
    seeds=read_json(folder / 'seed_ledger.json'); n=len(seeds)
    cases=full_wave(server,phase,state['incumbent'],state['recipe_revision'],stage_id,
                    POLICY['master_seed'],seeds)
    atomic_json(folder / 'seed_ledger.json',seeds)
    _sync_seed_log(folder,seeds)
    _register_stage(root,server,state,cases,stage_id,phase,recipe_id=state['incumbent'],recipe_revision=state['recipe_revision'])


def _teacher_panels(cases):
    return [{c.case_id:c for c in cases if c.role=='T' and c.sweep==sweep}
            for sweep in sorted({c.sweep for c in cases})]


def _schedule_fit(root,server,state,relation):
    from ablr2.policy import choose_recipes
    recipes=choose_recipes(relation,state['tried_recipes'])
    if not recipes:
        state['alert']='PLATEAU_UNRESOLVED'
        return _seed_wave(root,server,state,'REFRESH5',f'REFRESH{state["cycle"]:04}')
    boot=state['stages'][0]
    pool=_teacher_panels(_cases(boot,root))[:2]
    boot_cases=_cases(boot,root)
    blocks=[dict(teacher_seed=p['TPLUS'].seed,
                 student_seed=next(c.seed for c in boot_cases if c.case_id=='C07' and c.sweep==p['TPLUS'].sweep),
                 teachers=p) for p in pool]
    ids=tuple(dict.fromkeys((state['incumbent'],)+recipes))
    revisions={r:'r'+r[1:].zfill(3) for r in ids}
    stage_id=f'FIT{state["cycle"]:04}'
    cases=screen_cases(server,ids,revisions,stage_id,blocks,relation,root=root)
    aliases={}
    for case in cases:
        if case.role!='T': continue
        for run,row in state['runs'].items():
            if not row.get('complete'): continue
            old=case_for(run,root)
            if old.role=='T' and old.seed==case.seed and old.lambda_con==case.lambda_con:
                from ablr2.references import validate_reference
                validate_reference(old.reference_id,server,root,dataset_manifest=read_json(camp(root,server)/'dataset_manifest.json'))
                aliases[case.run_id]=old
                break
    if aliases:
        adjusted=[]
        for case in cases:
            if case.run_id in aliases: continue
            if case.teacher_run_id in aliases:
                owner=aliases[case.teacher_run_id]
                case=replace(case,teacher_run_id=owner.run_id,reference_id=owner.reference_id)
            adjusted.append(case)
        cases=tuple(adjusted)
        immutable_json(camp(root,server) / 'stages' / (stage_id+'.teacher_aliases.json'),
            {logical:asdict(actual) for logical,actual in aliases.items()})
    _register_stage(root,server,state,cases,stage_id,'FIT_ROUND',incumbent=state['incumbent'],
        candidates=list(recipes),relation_id=relation)


def _analyze_and_advance(root,server,state,stage):
    from ablr2.analysis import analyze_wave,case_report
    from ablr2.policy import (choose_relation,classify_relation,recheck_outcome,
                             screen_candidate,choose_recipe)
    from ablr2.extension import stage_cases
    folder=camp(root,server)
    cases=stage_cases(root,server,stage,completed_only=True,state=state) if stage['kind'] in ('BOOT5','REFRESH5','VERIFY5') else _cases(stage,root)
    threshold_path=folder / 'thresholds_v1.json'
    thresholds=read(threshold_path) or None
    if stage['kind'] in ('BOOT5','REFRESH5','VERIFY5'):
        report=analyze_wave(cases,thresholds=thresholds,root=root)
        if thresholds is None:
            thresholds=report['thresholds']; immutable_json(threshold_path,thresholds)
        stage['report_path']=report['report_path']
        if stage['kind']=='VERIFY5':
            state['verification_result']=report['report_path']
            result='VERIFY_SUPPORTED_DEV' if report.get('flow',{}).get('alert')=='FLOW_CANDIDATE_DEV' else 'VERIFY_NEGATIVE'
            immutable_json(folder / 'reports' / stage['stage_id'] / 'verification_decision.json',
                dict(status=result,report_path=report['report_path'],decision_rule='REGISTERED_FLOW_CONDITION',
                     statistical_significance=False,independent_test=False,repeat_same_recipe_forbidden=True))
            stage['verification_status']=result
            restore=state.pop('verify_resume_recipe',None)
            if restore:
                state['incumbent'],state['recipe_revision']=restore['recipe_id'],restore['recipe_revision']
            state['active_stage']=state.pop('resume_stage_after_verify')
            _save(root,server,state)
            return
        previous_relations=state.get('latest_relations')
        previous_full=state.get('latest_full_metrics')
        full=[r['VAL'] for r in report['panelrows'] if r['case_id']=='C07']
        current_full=dict(HQNR=median(r['HQNR'] for r in full),E_val=median(r['E_val'] for r in full)) if full else None
        stable=bool(previous_relations) and all(previous_relations.get(k,{}).get('classification')==v['classification']
                                                for k,v in report['relations'].items())
        improvement=bool(previous_full and current_full) and (current_full['E_val'] < previous_full['E_val']*.997
                        or current_full['HQNR'] > previous_full['HQNR']+thresholds['delta_H'])
        state['low_information_cycles']=(state.get('low_information_cycles',0)+1) if stable and not improvement else 0
        if state['low_information_cycles']>=3: state['alert']='LOW_INFORMATION_GAIN'
        state['latest_full_metrics']=current_full
        state['latest_full_stage']=stage['stage_id']
        # Keep each complete matrix; never overwrite a poor panel with a new one.
        state['latest_relations']=report['relations']
        from ablr2.analysis import cumulative_balanced_reports
        reports=[read_json(s['report_path']) for s in state['stages']
                 if s.get('report_path') and s['kind'] in ('BOOT5','REFRESH5')
                 and s.get('recipe_revision')==stage.get('recipe_revision')]
        # Original17 and extended18 are different balanced matrices. Likewise,
        # old receipts are not relabelled as the new numerical source.
        groups={}
        for item in reports:
            identity=object_sha(dict(sensor=item.get('sensor'),server=item.get('server'),
                recipe=item.get('recipe_id'),source=item.get('source_identity'),data=item.get('data_sha256'),
                components=item.get('component_cases',[f'C{i:02}' for i in range(17)])))
            groups.setdefault(identity,[]).append(item)
        paths=[]
        for identity,group in groups.items():
            cumulative=cumulative_balanced_reports(group)
            path=folder/'reports/cumulative_ablr2x'/f'{state["recipe_revision"]}_{identity}.json'
            atomic_json(path,cumulative);paths.append(str(path))
        atomic_json(folder/'reports'/f'{state["recipe_revision"]}_cumulative_ablr2x_index.json',
            dict(groups=paths,original_cumulative_preserved=True,mixed_coverage_or_source=False))
        state['cycle']+=1
        relation=choose_relation(report['relations'],state['rechecked'].get(state['recipe_revision'],[]))
        if relation is not None:
            seeds=read_json(folder / 'seed_ledger.json'); n=len(seeds)
            stage_id=f'RECHECK{state["cycle"]:04}_{relation}'
            jobs=recheck_cases(server,relation,state['incumbent'],state['recipe_revision'],stage_id,
                _teacher_panels(cases),POLICY['master_seed'],seeds)
            atomic_json(folder / 'seed_ledger.json',seeds)
            _sync_seed_log(folder,seeds)
            return _register_stage(root,server,state,jobs,stage_id,'RECHECK5',relation_id=relation,
                recipe_id=state['incumbent'],recipe_revision=state['recipe_revision'],teacher_pool=stage['stage_id'])
        return _schedule_fit(root,server,state,None)
    if stage['kind']=='RECHECK5':
        relation=stage['relation_id']; graph=GRAPH[relation]
        records={c.run_id:case_report(c,root=root) for c in cases}
        from ablr2.analysis import pair_panels
        paired=pair_panels(cases,relation,root=root)
        report=classify_relation(paired['pairs'],thresholds,paired['exact_pairs'])
        report['outcome']=recheck_outcome(state['latest_relations'][relation],report)
        report.update(relation_id=relation,independent_panel_claim=False,balanced_teacher_pool=stage['teacher_pool'],
                      records=records)
        path=folder / 'reports' / stage['stage_id'] / 'recheck.json'; immutable_json(path,report)
        done=state['rechecked'].setdefault(state['recipe_revision'],[])
        if relation not in done: done.append(relation)
        stage['report_path']=str(path)
        return _schedule_fit(root,server,state,relation)
    if stage['kind']=='FIT_ROUND':
        controls={c.run_id:case_report(c,root=root) for c in cases if c.role=='S'}
        records={(c.recipe_id,c.sweep):controls[c.run_id]['VAL'] for c in cases if c.case_id=='C07'}
        reports=[]
        for recipe in stage['candidates']:
            pairs=[dict(parent=records[(stage['incumbent'],s)],child=records[(recipe,s)]) for s in ('P01','P02')]
            reports.append(screen_candidate(recipe,pairs,thresholds))
        chosen=choose_recipe(stage['incumbent'],reports)
        path=folder / 'reports' / stage['stage_id'] / 'recipe_screen.json'
        immutable_json(path,dict(incumbent=stage['incumbent'],chosen=chosen,candidates=reports,
            promotion='PROVISIONAL_DEV_REQUIRES_COMPLETE_REFRESH5',test_aware=True,
            control_observations=controls,control_gap_used_for_selection=False))
        append_event(folder / 'recipe_ledger.jsonl','FIT_DECISION',stage_id=stage['stage_id'],
                     previous=state['incumbent'],chosen=chosen,report_sha256=sha256(path))
        state['tried_recipes']=list(dict.fromkeys(state['tried_recipes']+stage['candidates']))
        state['incumbent']=chosen;state['recipe_revision']='r'+chosen[1:].zfill(3)
        stage['report_path']=str(path)
        return _seed_wave(root,server,state,'REFRESH5',f'REFRESH{state["cycle"]:04}')
    raise ValueError('Unknown campaign phase')


def request_verify(root,server):
    """Queue one locked VERIFY5 at a block boundary; never replace a failed verify."""
    folder=camp(root,server)
    with locked(folder / 'operator.lock'):
        state=_state(root,server)
        revision=state['recipe_revision']
        if revision in state['verification_recipes']: raise ValueError('VERIFY5 already registered for this exact recipe')
        threshold=read_json(folder / 'thresholds_v1.json')
        request=dict(recipe_id=state['incumbent'],recipe_revision=revision,
            thresholds_sha256=object_sha(threshold),source_identity=source_identity(root),requested_at_utc=utcnow())
        immutable_json(folder / 'verify_requests' / f'{revision}.json',request)
        atomic_json(folder / 'verify_request.json',request)
        return request


def _activate_pending_verify(root,server,state):
    """Resolve an operator request before any newly registered block is admitted."""
    folder=camp(root,server)
    request=read(folder / 'verify_request.json')
    if not request or request['recipe_revision'] in state['verification_recipes']: return False
    stage=next(s for s in state['stages'] if s['stage_id']==state['active_stage'])
    if any(state['runs'].get(run,{}).get('train_attempts') or state['runs'].get(run,{}).get('complete')
           for run in stage['run_ids']): return False  # Finish its preregistered block first.
    if (request['source_identity']!=source_identity(root)
            or request['thresholds_sha256']!=object_sha(read_json(folder / 'thresholds_v1.json'))):
        raise ValueError('Locked VERIFY5 source/threshold identity changed')
    state['verification_recipes'].append(request['recipe_revision'])
    state['resume_stage_after_verify']=state['active_stage']
    incumbent,revision=state['incumbent'],state['recipe_revision']
    state['verify_resume_recipe']=dict(recipe_id=incumbent,recipe_revision=revision)
    state['incumbent'],state['recipe_revision']=request['recipe_id'],request['recipe_revision']
    try:
        _seed_wave(root,server,state,'VERIFY5','VERIFY_'+request['recipe_revision'])
    finally:
        state['incumbent'],state['recipe_revision']=incumbent,revision
    _save(root,server,state)
    return True


def status(root=ROOT,server='s1'):
    folder=camp(root,server)
    state=read(folder / 'state.json',{'status':'NOT_BUILT'})
    debts=read(folder/'extension/debts.json',{'debts':{}})['debts']
    return dict(state=state,lease=read(folder / 'lease.json'),control=read(folder / 'control.json'),
                authorization=read(folder/'authorization.json'),
                handover=read(folder/'migration/waiter_status.json'),
                extension_debts={name:sum(r.get('status')==name for r in debts.values()) for name in
                    ('PENDING','WAIT_C03','READY','REFERENCE_UNAVAILABLE','COMPLETE')},
                training_started=any(r.get('train_attempts',0)>0 for r in state.get('runs',{}).values()))


def retry_uploads(root=ROOT,server='s1'):
    """Retry evaluated rows only; failed networking never restarts training."""
    from ablr2.upload import spool_run,retry_pending
    folder=camp(root,server)
    outcomes={}
    with locked(folder / 'upload_retry.lock'):
        # Do not rewrite controller state from this independent operator command.
        # The official receipts are authoritative and the runner reconciles them.
        state=_state(root,server)
        spooled=0
        for run,row in state['runs'].items():
            if not row.get('complete') or row.get('reused'): continue
            report=read(run_dir(run,root) / 'official/postrun_status.json')
            if not report.get('official_complete') or report.get('sheet_uploaded'): continue
            if (folder/'upload_outbox/pending'/f'{run}.json').exists():continue
            if spooled>=8:break
            spooled+=1
            try:
                spool_run(run,root=root)
            except Exception as exc:
                outcomes[run]=dict(error=f'{type(exc).__name__}: {exc}')
        outcomes.update(retry_pending(root,server,limit=8,activated=True))
    return outcomes


def run(root=ROOT,server='s1',upload=True):
    root=Path(root).resolve();folder=camp(root,server)
    from ablr2.resources import idle_evidence,assess_case,assess_block
    from ablr2.extension import sync,ready_debts,stage_cases
    with locked(folder / 'runner.lock'):
        state=_state(root,server)
        try:
            verify_registration(root,server);_lease(root,server)
            sync(root,server,state)

            def admit(case,previous=None):
                if case.case_id=='C17':
                    from ablr2.extension import prepare_anchor
                    original=case_for(case.run_id.replace('_C17_','_C03_'),root)
                    if not state['runs'].get(original.run_id,{}).get('complete'):admit(original,previous)
                    anchor=prepare_anchor(case,root)
                    if not state['runs'].get(anchor.run_id,{}).get('complete'):admit(anchor,previous)
                lease=_lease(root,server)
                command=read(folder/'control.json').get('command')
                if command=='STOP_NOW_SAFE' or (command=='STOP_AFTER_RUN' and state.get('last_completed_run')):
                    raise RuntimePaused(command)
                if command=='STOP_AFTER_SWEEP' and (previous is None or previous.sweep!=case.sweep):
                    raise RuntimePaused(command)
                expiry=lease.get('expires_utc')
                remaining=(dt.datetime.fromisoformat(expiry)-dt.datetime.now(dt.timezone.utc)).total_seconds()/3600 if expiry else None
                estimate=estimate_hours(case,state)
                if remaining is not None and remaining<estimate:raise RuntimePaused('LEASE_ADMISSION_RESERVE_INSUFFICIENT')
                if not idle_evidence()['idle']:raise RuntimePaused('WAIT_LOCAL_RESOURCE')
                resources=assess_case(root,case)
                atomic_json(folder/'resources'/f'{case.run_id}.json',resources)
                if not resources['allowed']:raise RuntimePaused('WAIT_LOCAL_RESOURCE: '+str(resources['reasons']))
                append_event(folder/'admission_ledger.jsonl','ADMITTED',run_id=case.run_id,
                    lease_sequence=lease['sequence'],remaining_hours=remaining,reserved_hours=estimate,
                    service_mode=lease.get('service_mode','FINITE_LEASE'))
                _run_case(root,server,state,case,upload)
                if case.case_id!='C17':state['extension_debt_streak']=0
                state['last_completed_run']=case.run_id;_save(root,server,state)

            def drain_debts(previous=None):
                allowance=max(0,2-state.get('extension_debt_streak',0))
                if not allowance:return
                ready=ready_debts(root,server,state)
                if read(folder/'control.json').get('command')=='STOP_AFTER_SWEEP':
                    ready=[pair for pair in ready if previous is not None and pair[0].sweep==previous.sweep]
                selected=ready[:allowance]
                jobs=[job for debt,anchor in selected for job in (anchor,debt)
                      if not state['runs'].get(job.run_id,{}).get('complete')]
                if jobs:
                    resources=assess_block(root,jobs)
                    atomic_json(folder/'resources/next_c17_debt_block.json',resources)
                    if not resources['allowed']:raise RuntimePaused('WAIT_LOCAL_RESOURCE: '+str(resources['reasons']))
                for debt,anchor in selected:
                    streak=state.get('extension_debt_streak',0)
                    for job in (anchor,debt):
                        if not state['runs'].get(job.run_id,{}).get('complete'):admit(job,previous)
                    state['extension_debt_streak']=streak+1;_save(root,server,state)

            while True:
                command=read(folder / 'control.json').get('command')
                if command=='STOP_NOW_SAFE': raise RuntimePaused(command)
                lease=_lease(root,server)
                idle=idle_evidence()
                if idle['idle']: break
                state.update(status='WAIT_LOCAL_RESOURCE',resource_evidence=idle);_save(root,server,state)
                time.sleep(15)
            code,hours=_command(root,server,['preflight'],folder / 'logs/preflight.log',lease['expires_utc'])
            if code in (75,124): raise RuntimePaused('PREFLIGHT_LEASE_END')
            if code: raise ValueError('Local preflight failed; inspect logs/preflight.log')
            state['setup_hours']=state.get('setup_hours',0.)+hours
            if upload: retry_uploads(root,server)
            while True:
                lease=_lease(root,server)
                _activate_pending_verify(root,server,state)
                stage=next(s for s in state['stages'] if s['stage_id']==state['active_stage'])
                sync(root,server,state)
                cases=stage_cases(root,server,stage)
                remaining_cases=[c for c in cases if not state['runs'].get(c.run_id,{}).get('complete')]
                if remaining_cases:
                    resources=assess_block(root,remaining_cases)
                    atomic_json(folder/'resources'/(stage['stage_id']+'.block.json'),resources)
                    if not resources['allowed']:raise RuntimePaused('WAIT_LOCAL_RESOURCE: '+str(resources['reasons']))
                for index,case in enumerate(cases):
                    if state['runs'].get(case.run_id,{}).get('complete'): continue
                    previous=cases[index-1] if index else None
                    # At most two old C17 debts at each original-job boundary.
                    # Missing references remain visible debts, never starve core work.
                    drain_debts(previous)
                    if state['runs'].get(case.run_id,{}).get('complete'):continue
                    if case.case_id=='C17' and stage['kind'] in ('BOOT5','REFRESH5','VERIFY5'):
                        # Full-panel C17 is handled only by the bounded debt
                        # dispatcher; unavailable refs/fairness cannot admit a third.
                        continue
                    admit(case,previous)
                drain_debts(cases[-1])
                from ablr2.analysis import refresh_extended_reports
                for old_id,outcome in refresh_extended_reports(root,server,state).items():
                    old_stage=next(s for s in state['stages'] if s['stage_id']==old_id)
                    old_stage['legacy_core17_complete']=outcome.get('legacy_core17_complete',old_stage.get('legacy_core17_complete',old_stage.get('complete',False)))
                    old_stage['extended18_complete']=outcome['extended18_complete']
                    if outcome.get('report_path'):old_stage['extended_report_path']=outcome['report_path']
                if stage['kind'] in ('BOOT5','REFRESH5','VERIFY5'):
                    stage['legacy_core17_complete']=all(state['runs'].get(c.run_id,{}).get('complete') for c in cases if c.case_id!='C17')
                    stage['extended18_complete']=all(state['runs'].get(c.run_id,{}).get('complete') for c in cases)
                stage['complete']=True
                state['status']='ANALYZE';_save(root,server,state)
                if upload: retry_uploads(root,server)
                if read(folder / 'control.json').get('command') in ('PAUSE_AFTER_BLOCK','STOP_AFTER_SWEEP','STOP_AFTER_RUN','STOP_NOW_SAFE'):
                    raise RuntimePaused('BLOCK_BOUNDARY_STOP')
                _analyze_and_advance(root,server,state,stage)
                _activate_pending_verify(root,server,state)
                _save(root,server,state)
                from ablr2.reporting import rebuild
                rebuild(root,server)
        except RuntimePaused as exc:
            state.update(status='PAUSED_SAFE',reason=str(exc));_save(root,server,state)
            from ablr2.reporting import rebuild
            rebuild(root,server)
            return 75
        except (ValueError,RuntimeError,OSError,TimeoutError) as exc:
            state.update(status='BLOCKED_NUMERICS',reason=f'{type(exc).__name__}: {exc}')
            append_event(folder / 'all_attempts.jsonl','BLOCKED',reason=state['reason'])
            _save(root,server,state)
            from ablr2.reporting import rebuild
            rebuild(root,server)
            return 1


def start(root=ROOT,server='s1',foreground=False,upload=True):
    folder=camp(root,server)
    waiting=False
    with locked(folder / 'startup.lock'):
        from ablr2.resources import process_start as _process_start
        from ablr2.migration import assert_ready,start_waiter
        old=read(folder / 'runner.pid.json')
        if old.get('process_start_ticks') and _process_start(old.get('pid'))==old['process_start_ticks']:
            if (folder/'runtime_release.json').exists() and not (folder/'migration/transition.json').exists():
                waiting=True
            else:return dict(status='ALREADY_SUBMITTED',pid=old['pid'])
        if not waiting:
            try:
                with locked(folder / 'runner.lock'): pass
            except BlockingIOError:return dict(status='ALREADY_RUNNING')
            try:
                assert_ready(root,server)
                if server=='s3':
                    from ablr2.handoff import verify_ready
                    verify_ready(root,server)
            except RuntimePaused:waiting=True
        if waiting:
            if not foreground:return start_waiter(root,server,foreground=False,upload=upload)
        else:
            command=read(folder/'control.json').get('command')
            if command in STOP_COMMANDS and command!='CONTINUE':
                return dict(status='OPERATOR_STOPPED',command=command,exit_code=75,automatic_restart=False)
            from ablr2.migration import ensure_source_bridge
            ensure_source_bridge(root,server)
            if read(folder/'control.json').get('command') not in (None,'CONTINUE'):
                return dict(status='OPERATOR_STOPPED',exit_code=75,automatic_restart=False)
            build(root,server);_lease(root,server);register_runtime(root,server)
            if foreground:
                atomic_json(folder/'runner.pid.json',dict(pid=os.getpid(),process_start_ticks=_process_start(os.getpid())))
            else:
                args=[sys.executable,'-u','tools/ablr2_runner.py','run','--server',server]+([] if upload else ['--no-upload'])
                with (folder / 'runner.log').open('a') as stream:
                    process=subprocess.Popen(args,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                             env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
                atomic_json(folder / 'runner.pid.json',dict(pid=process.pid,process_start_ticks=_process_start(process.pid)))
                return dict(status='SUBMITTED',pid=process.pid,log=str(folder / 'runner.log'))
    # Never hold startup.lock throughout a long foreground waiter/controller.
    if waiting:return start_waiter(root,server,foreground=True,upload=upload)
    return dict(exit_code=run(root,server,upload))
