"""Persistent server-local infinite iterator, exact resumption and typed failures."""
from pathlib import Path
import subprocess
import sys
import time

from g23sens.common import (ROOT,camp,run_dir,read,read_json,atomic_json,immutable_json,
    locked,utcnow,object_sha,source_identity,verify_server,RuntimePaused,sha256)
from g23sens.plan import (CAMPAIGN,cycle_cases,next_cursor,build_config,validate_case,verify_sources)

TECHNICAL_RETRIES=2


class ResumeUnusable(RuntimeError):
    pass


def resume_available(work,cfg):
    path=work/'last/identity.json'
    if not path.exists():return False
    import json
    try:identity=read_json(path)
    except json.JSONDecodeError as exc:raise ResumeUnusable('Incomplete last identity') from exc
    expected=dict(config_sha256=object_sha(cfg),source_identity=cfg['g23sens']['source_identity'],
                  bindings_sha256=cfg['g23sens']['binding_sha256'])
    if any(identity.get(k)!=v for k,v in expected.items()):
        raise ValueError('Existing resume belongs to another source/config/binding; no fresh overwrite')
    state=work/'last/training_state.pt'
    if (identity.get('full_state') is not True or not state.is_file()
            or identity.get('training_state_sha256')!=sha256(state)):
        raise ResumeUnusable('Last full-state publication/checksum is incomplete')
    return True


def authorize_train(root,case,config_path):
    """Reject ad-hoc main.py bypasses of preflight, admission and local ownership."""
    validate_case(case);folder=camp(root,case['server'])
    state=read(folder/'state.json');preflight=read(folder/'preflight.json')
    if (state.get('active_run_id')!=case['run_id'] or state.get('status')!='ADMITTED'
            or state.get('active_case_spec_sha256')!=case['case_spec_sha256']
            or preflight.get('status')!='PASSED' or preflight.get('source_identity')!=source_identity(root)):
        raise PermissionError('No matching preflighted local admission')
    attempt=state['runs'][case['run_id']]['attempt']
    if Path(config_path).resolve()!=(run_dir(case,root,attempt)/'config.json').resolve():
        raise PermissionError('Training config escaped the admitted attempt')
    if (folder/'STOP_NOW_SAFE').exists():raise RuntimePaused('Safe pause before training')


def initial_state(server, bindings):
    return dict(campaign_id=CAMPAIGN,server=verify_server(server),cycle=0,position=0,
        active_run_id=None,active_case_spec_sha256=None,attempt=0,
        locked_bindings_sha256=object_sha(bindings),runs={},status='READY')


def validate_state(state, server, bindings):
    if (state['campaign_id']!=CAMPAIGN or state['server']!=server
            or state['locked_bindings_sha256']!=object_sha(bindings)):
        raise ValueError('Persistent campaign/server/bindings changed')
    case=cycle_cases(server,state['cycle'])[state['position']]
    if state.get('active_run_id') not in (None,case['run_id']):
        raise ValueError('Active run differs from persistent cursor')
    if state.get('active_run_id') and state.get('active_case_spec_sha256')!=case['case_spec_sha256']:
        raise ValueError('Active case specification changed')
    return case


def _save(folder,state):
    state['updated_at_utc']=utcnow();atomic_json(folder/'state.json',state)


def phase_worker(phase, config, case, resume=False, root=ROOT):
    command=[sys.executable,str(Path(root)/'tools/g23sens_runner.py'),phase,
             '--server',case['server'],'--config',str(config)]
    if resume:command.append('--resume')
    log=Path(config).parent/(phase+'.log')
    with log.open('a') as stream:
        return subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT).returncode


def _complete(case,root,attempt):
    from g23sens.postrun import verify_summary_for_upload
    return verify_summary_for_upload(case['run_id'],root=root,attempt=attempt)


def _advance(state,case,status):
    state['runs'][case['run_id']]['status']=status
    state.update(next_cursor(case['cycle'],case['position_in_cycle']))
    state.update(active_run_id=None,active_case_spec_sha256=None,attempt=0,status='READY')


def _queue_results(root,server,case,attempt,no_upload):
    from g23sens.upload import queue_run,flush_outbox
    queue_run(root,case['run_id'],attempt=attempt)
    if not no_upload:
        try:flush_outbox(root,server,activated=True)
        except Exception as exc:
            atomic_json(camp(root,server)/'upload_error.json',dict(error=str(exc),at_utc=utcnow()))
            print('Sheet deferred; verified local results retained:',exc,flush=True)


def _failure(work,case,attempt,kind,code):
    training=read(work/'meta/training_status.json')
    atomic_json(work/'meta/attempt_failure.json',dict(kind=kind,status='TECHNICAL_FAILED',
        code=code,reason='See immutable attempt phase log and controller failure ledger',
        attempt=attempt,run_id=case['run_id'],at_utc=utcnow(),final=True,
        actual_updates=training.get('actual_updates')))


def run(root,server,*,activated=False,no_upload=False,worker=None):
    """No finite deadline, performance threshold, other-server dependency or reroll."""
    if not activated:raise PermissionError('Explicit start required')
    root=Path(root);verify_server(server);folder=camp(root,server)
    from g23sens.assets import validate_bindings
    bindings=read_json(folder/'runtime_bindings.json')
    if bindings is None:raise ValueError('Run preflight before admission')
    worker=worker or (lambda phase,config,case,resume=False:phase_worker(phase,config,case,resume,root))
    with locked(folder/'runner.lock'):
        state=read(folder/'state.json',initial_state(server,bindings))
        while True:
            case=validate_state(state,server,bindings)
            if (folder/'STOP_NOW_SAFE').exists() or ((folder/'STOP_AFTER_RUN').exists() and not state['active_run_id']):
                state['status']='PAUSED_BY_OPERATOR';_save(folder,state);return 75
            from g23sens.resources import ensure_space
            ensure_space(folder);verify_sources(root)
            validate_bindings(bindings,root=root,server=server,rehash=True)
            item=state['runs'].setdefault(case['run_id'],dict(attempt=0,status='NEW',failures=[]))
            if item['status'] in ('BASE_DIVERGED','TECHNICAL_FAILED','EVALUATION_FAILED'):
                state['status']='PAUSE_BLOCK';_save(folder,state);return 3
            if item['status']=='DIVERGED':
                _queue_results(root,server,case,item['attempt'],no_upload)
                _advance(state,case,'DIVERGED');_save(folder,state);continue
            attempt=item['attempt'];work=run_dir(case,root,attempt)
            cfg=build_config(case,root,bindings,attempt)
            immutable_json(work/'case.json',case);immutable_json(work/'config.json',cfg)
            state.update(active_run_id=case['run_id'],active_case_spec_sha256=case['case_spec_sha256'],attempt=attempt)
            _save(folder,state)
            # Even a restart after evaluation must recheck bytes before skipping training.
            if (work/'official/COMPLETE.json').exists():
                _complete(case,root,attempt);item['status']='EVALUATED'
            if item['status'] not in ('TRAINED','EVALUATED'):
                from g23sens.handoff import ensure_gpu_idle
                import os
                ensure_gpu_idle(os.environ.get('PANCRAFTER_G23SENS_GPU_UUID'))
                try:resume=resume_available(work,cfg)
                except ResumeUnusable as exc:
                    item['failures'].append(dict(kind='TRAIN_TECHNICAL',attempt=attempt,code=74,reason=str(exc),at_utc=utcnow()))
                    _failure(work,case,attempt,'TRAIN_TECHNICAL',74)
                    _queue_results(root,server,case,attempt,no_upload)
                    if len([f for f in item['failures'] if f['kind']=='TRAIN_TECHNICAL'])>TECHNICAL_RETRIES:
                        item['status']='TECHNICAL_FAILED';state['status']='PAUSE_BLOCK';_save(folder,state);return 74
                    item.update(attempt=attempt+1,status='RETRY_TRAIN');_save(folder,state);continue
                item['status']='TRAINING';state['status']='ADMITTED';_save(folder,state)
                code=worker('train',work/'config.json',case,resume=resume)
                if code==75:
                    state['status']='PAUSED_SAFE';_save(folder,state);return 75
                if code==3:
                    item['failures'].append(dict(kind='NUMERICAL_DIVERGENCE',attempt=attempt,at_utc=utcnow()))
                    if case['case_id']=='BASE':
                        item['status']='BASE_DIVERGED';state['status']='PAUSE_BLOCK';_save(folder,state)
                        _queue_results(root,server,case,attempt,no_upload);return 3
                    item['status']='DIVERGED';_save(folder,state)
                    _queue_results(root,server,case,attempt,no_upload)
                    _advance(state,case,'DIVERGED');_save(folder,state)
                    from g23sens.reporting import build_report
                    build_report(root,server);continue
                if code!=0:
                    item['failures'].append(dict(kind='TRAIN_TECHNICAL',attempt=attempt,code=code,at_utc=utcnow()))
                    if code not in (74,137,-9) or len([f for f in item['failures'] if f['kind']=='TRAIN_TECHNICAL'])>TECHNICAL_RETRIES:
                        item['status']='TECHNICAL_FAILED';state['status']='PAUSE_BLOCK';_save(folder,state)
                        _failure(work,case,attempt,'TRAIN_TECHNICAL',code)
                        _queue_results(root,server,case,attempt,no_upload);return code
                    # A valid full state is retried in place. Never overwrite a partial fresh attempt.
                    if not (work/'last/identity.json').exists():
                        _failure(work,case,attempt,'TRAIN_TECHNICAL',code)
                        _queue_results(root,server,case,attempt,no_upload)
                        item['attempt']+=1
                    item['status']='RETRY_TRAIN';_save(folder,state);continue
                item['status']='TRAINED';_save(folder,state)
            if item['status']!='EVALUATED':
                code=worker('postrun',work/'config.json',case)
                if code==75:
                    state['status']='PAUSED_SAFE';_save(folder,state);return 75
                if code!=0:
                    item['failures'].append(dict(kind='EVALUATION',attempt=attempt,code=code,at_utc=utcnow()))
                    _save(folder,state)
                    if len([f for f in item['failures'] if f['kind']=='EVALUATION'])>TECHNICAL_RETRIES:
                        item['status']='EVALUATION_FAILED';state['status']='PAUSE_BLOCK';_save(folder,state)
                        _failure(work,case,attempt,'EVALUATION',code)
                        _queue_results(root,server,case,attempt,no_upload);return code
                    continue  # TRAINED is deliberately retained; no training repetition.
                _complete(case,root,attempt);item['status']='EVALUATED';_save(folder,state)
            # Outbox errors never trigger another training/evaluation attempt.
            _queue_results(root,server,case,attempt,no_upload)
            _advance(state,case,'COMPLETE');_save(folder,state)
            from g23sens.reporting import build_report
            build_report(root,server)
