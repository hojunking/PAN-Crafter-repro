"""Server-local paired seven-case cycles, without deadlines or remote locks."""
from pathlib import Path
import subprocess
import sys

from maina_hqnr.common import (ROOT, camp, run_dir, read, read_json, read_config,
    atomic_json, immutable_json, locked, utcnow, object_sha, source_identity,
    verify_server, RuntimePaused, sha256)
from maina_hqnr.plan import (CAMPAIGN_ID, cycle_cases, next_cursor, build_config,
                             validate_case)

TECHNICAL_RETRIES = 2


class ResumeUnusable(RuntimeError):
    pass


def resume_available(work, cfg):
    identity_path = work / 'last/identity.json'
    if not identity_path.exists():
        return False
    import json
    try:
        identity = read_json(identity_path)
    except json.JSONDecodeError as error:
        raise ResumeUnusable('Incomplete full-state identity') from error
    field = cfg['maina_hqnr']
    expected = dict(config_sha256=object_sha(cfg), source_identity=field['source_identity'],
                    bindings_sha256=field['binding_sha256'])
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError('Resume source/config/bindings changed; no fresh overwrite')
    state = work / 'last/training_state.pt'
    if (identity.get('full_state') is not True or not state.is_file()
            or sha256(state) != identity.get('training_state_sha256')):
        raise ResumeUnusable('Full-state bytes are incomplete/corrupt')
    return True


def authorize_train(root, case, config_path):
    validate_case(case)
    folder = camp(root, case['server'])
    state, preflight = read(folder / 'state.json'), read(folder / 'preflight.json')
    if (state.get('active_run_id') != case['run_id'] or state.get('status') != 'ADMITTED'
            or state.get('active_case_spec_sha256') != case['case_spec_sha256']
            or preflight.get('status') != 'PASSED'
            or preflight.get('source_identity') != source_identity(root)):
        raise PermissionError('No matching preflighted MAIN-A local admission')
    attempt = state['runs'][case['run_id']]['attempt']
    if Path(config_path).resolve() != (run_dir(case, root, attempt) / 'config.json').resolve():
        raise PermissionError('Config is outside the admitted attempt')
    cfg = read_config(config_path); meta = cfg.get('maina_hqnr', {})
    bindings = read_json(folder / 'runtime_bindings.json')
    from maina_hqnr.common import selected_gpu_uuid
    if (meta.get('case') != case or meta.get('attempt') != attempt
            or meta.get('source_identity') != preflight['source_identity']
            or bindings is None or meta.get('binding_sha256') != object_sha(bindings)
            or meta.get('binding_sha256') != state.get('locked_bindings_sha256')
            or meta.get('binding_sha256') != preflight.get('bindings_sha256')
            or preflight.get('gpu_uuid') != selected_gpu_uuid()):
        raise PermissionError('Admission preflight/config/assets/GPU identity differs')
    if (folder / 'STOP_NOW_SAFE').exists():
        raise RuntimePaused('Safe pause before training')


def initial_state(server, bindings, source=None):
    return dict(campaign_id=CAMPAIGN_ID, server=verify_server(server), cycle=0, position=0,
                active_run_id=None, active_case_spec_sha256=None, attempt=0, runs={}, status='READY',
                locked_bindings_sha256=object_sha(bindings), source_identity=source)


def validate_state(state, server, bindings):
    if (state['campaign_id'] != CAMPAIGN_ID or state['server'] != server
            or state['locked_bindings_sha256'] != object_sha(bindings)):
        raise ValueError('Persistent campaign/server/asset bindings changed')
    case = cycle_cases(server, state['cycle'])[state['position']]
    if state.get('active_run_id') not in (None, case['run_id']):
        raise ValueError('Active run differs from local cursor')
    if state.get('active_run_id') and state.get('active_case_spec_sha256') != case['case_spec_sha256']:
        raise ValueError('Active case specification changed')
    return case


def _save(folder, state):
    state['updated_at_utc'] = utcnow()
    atomic_json(folder / 'state.json', state)


def phase_worker(phase, config, case, resume=False, root=ROOT):
    command = [sys.executable, str(Path(root) / 'tools/maina_hqnr_runner.py'), phase,
               '--server', case['server'], '--config', str(config)]
    if resume:
        command.append('--resume')
    with (Path(config).parent / (phase + '.log')).open('a') as stream:
        return subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT).returncode


def _complete(work, root):
    from maina_hqnr.postrun import verify_summary_for_upload
    return verify_summary_for_upload(work / 'config.json', root=root)


def _queue_results(work, root, server, no_upload):
    from maina_hqnr.upload import enqueue_run, flush_outbox
    enqueue_run(work / 'config.json', root=root)
    if not no_upload:
        try:
            flush_outbox(root, server, activated=True)
        except Exception as error:
            atomic_json(camp(root, server) / 'upload_error.json', dict(error=str(error), at_utc=utcnow()))
            print('Upload deferred; durable local outbox retained:', error, flush=True)


def _advance(state, case, status):
    state['runs'][case['run_id']]['status'] = status
    state.update(next_cursor(case['server'], case['cycle'], case['position_in_cycle']))
    state.update(active_run_id=None, active_case_spec_sha256=None, attempt=0, status='READY')


def _failure(work, case, attempt, phase, code):
    status = read(work / 'meta/training_status.json')
    atomic_json(work / 'meta/attempt_failure.json', dict(status='TECHNICAL_FAILED', phase=phase,
        code=code, attempt=attempt, run_id=case['run_id'], actual_updates=status.get('actual_updates'),
        at_utc=utcnow(), reason='See retained phase log; case and seed are not changed'))


def run(root, server, *, activated=False, no_upload=False, worker=None):
    if not activated:
        raise PermissionError('Explicit MAIN-A start required')
    root = Path(root); verify_server(server); folder = camp(root, server)
    from maina_hqnr.assets import validate_bindings
    from maina_hqnr.resources import ensure_space
    bindings = read_json(folder / 'runtime_bindings.json')
    if bindings is None:
        raise RuntimePaused('PAUSED_ASSET_MISSING: verified F1 bindings absent')
    worker = worker or (lambda phase, config, case, resume=False:
                        phase_worker(phase, config, case, resume, root))
    with locked(folder / 'runner.lock'):
        source = source_identity(root)
        state = read(folder / 'state.json', initial_state(server, bindings, source))
        while True:
            source = source_identity(root)
            if state.get('source_identity') != source:
                state['status'] = 'PAUSED_SOURCE_CHANGED'; _save(folder, state)
                raise RuntimePaused('Source changed: preserve this block; create a new explicit revision, never mix sources')
            case = validate_state(state, server, bindings)
            if ((folder / 'STOP_NOW_SAFE').exists()
                    or ((folder / 'STOP_AFTER_RUN').exists() and not state['active_run_id'])):
                state['status'] = 'PAUSED_BY_OPERATOR'; _save(folder, state); return 75
            ensure_space(folder)
            validate_bindings(bindings, root=root, server=server, rehash=True)
            item = state['runs'].setdefault(case['run_id'], dict(attempt=0, status='NEW', failures=[]))
            if item['status'] in ('BASE_DIVERGED', 'TECHNICAL_FAILED', 'PAUSED_BASE_TECHNICAL', 'PENDING_EVAL_NOT_COMPARABLE'):
                state['status'] = item['status']; _save(folder, state); return 75
            attempt = item['attempt']; work = run_dir(case, root, attempt)
            cfg = build_config(case, root, bindings, attempt)
            immutable_json(work / 'case.json', case); immutable_json(work / 'config.json', cfg)
            state.update(active_run_id=case['run_id'], active_case_spec_sha256=case['case_spec_sha256'], attempt=attempt)
            _save(folder, state)
            if (work / 'official/COMPLETE.json').exists():
                _complete(work, root); item['status'] = 'EVALUATED'
            if item['status'] not in ('TRAINED', 'EVALUATED'):
                from maina_hqnr.handoff import ensure_gpu_idle
                import os
                ensure_gpu_idle(os.environ.get('PANCRAFTER_MAINA_GPU_UUID'))
                try:
                    resume = resume_available(work, cfg)
                    item['status'] = 'TRAINING'; state['status'] = 'ADMITTED'; _save(folder, state)
                    code = worker('train', work / 'config.json', case, resume=resume)
                except ResumeUnusable as error:
                    code = 74
                    item['failures'].append(dict(kind='RESUME_UNUSABLE', reason=str(error), attempt=attempt, at_utc=utcnow()))
                if code == 75:
                    state['status'] = 'PAUSED_SAFE'; _save(folder, state); return 75
                if code == 3:
                    item['failures'].append(dict(kind='NUMERICAL_DIVERGENCE', attempt=attempt, at_utc=utcnow()))
                    item['status'] = 'BASE_DIVERGED' if case['code'] == 'BASE' else 'DIVERGED'
                    _save(folder, state); _queue_results(work, root, server, no_upload)
                    if case['code'] == 'BASE':
                        state['status'] = 'BASE_DIVERGED'; _save(folder, state); return 3
                    _advance(state, case, 'DIVERGED'); _save(folder, state); continue
                if code != 0:
                    item['failures'].append(dict(kind='TRAIN_TECHNICAL', code=code, attempt=attempt, at_utc=utcnow()))
                    _failure(work, case, attempt, 'train', code)
                    _queue_results(work, root, server, no_upload)
                    failures = sum(row['kind'] == 'TRAIN_TECHNICAL' for row in item['failures'])
                    if case['code'] == 'BASE' or code not in (74, 137, -9) or failures > TECHNICAL_RETRIES:
                        item['status'] = 'PAUSED_BASE_TECHNICAL' if case['code'] == 'BASE' else 'TECHNICAL_FAILED'
                        state['status'] = item['status']; _save(folder, state); return code
                    # Every fresh retry is a new immutable attempt of the same seed.
                    item.update(attempt=attempt + 1, status='RETRY_TRAIN'); _save(folder, state); continue
                item['status'] = 'TRAINED'; _save(folder, state)
            if item['status'] != 'EVALUATED':
                code = worker('postrun', work / 'config.json', case)
                if code == 75:
                    state['status'] = 'PAUSED_SAFE'; _save(folder, state); return 75
                if code != 0:
                    item['failures'].append(dict(kind='EVALUATION', code=code, attempt=attempt, at_utc=utcnow()))
                    item['status'] = 'PENDING_EVAL_NOT_COMPARABLE'; state['status'] = item['status']
                    _save(folder, state)
                    _queue_results(work, root, server, no_upload)
                    # Keep trained candidates; explicit retry --phase postrun never retrains.
                    return code
                _complete(work, root); item['status'] = 'EVALUATED'; _save(folder, state)
            _queue_results(work, root, server, no_upload)
            _advance(state, case, 'COMPLETE'); _save(folder, state)
            from maina_hqnr.reporting import build_report
            build_report(root, server)


def retry(root, server, phase):
    """Explicit operator recovery; never reroll a seed or erase an old attempt."""
    folder = camp(root, verify_server(server))
    with locked(folder / 'runner.lock'):
        state = read(folder / 'state.json'); run_id = state.get('active_run_id')
        if not run_id:
            raise ValueError('No failed active case')
        item = state['runs'][run_id]
        if phase == 'postrun':
            if item['status'] != 'PENDING_EVAL_NOT_COMPARABLE':
                raise ValueError('Only pending evaluation can retry postrun')
            item['status'] = 'TRAINED'
        elif phase == 'train':
            if item['status'] not in ('PAUSED_BASE_TECHNICAL', 'TECHNICAL_FAILED'):
                raise ValueError('Only a technical failure may start a fresh retry')
            item.update(attempt=item['attempt'] + 1, status='RETRY_TRAIN')
        else:
            raise ValueError('Unknown retry phase')
        state['status'] = 'READY'; _save(folder, state)
    return state
