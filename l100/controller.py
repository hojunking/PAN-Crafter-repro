"""Finite, additive LOCAL-T controller with one shared clock and local references."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from l100.common import (ROOT, CAMPAIGN_ID, apply_runtime_policy, atomic_json, camp,
    immutable_json, locked, object_sha, read, read_json, sha256, source_identity, utcnow)
from l100.plan import (CORE_CASES, SERVERS, blocks_for, build_config, case_for, cases_for,
    registry_document, registry_sha256, verify_sources, validate_config, grid_steps)
from l100.policy import CampaignWindow, estimate_block_hours, evaluate_admission


class TrainingTimeLimit(RuntimeError):
    """Expected partial endpoint at the fixed optimizer deadline, not corruption."""


class OwnTrainerStillPreserving(RuntimeError):
    """Own child remains alive; no further GPU work is safe in this controller."""


def shared_window(root, supplied=None):
    candidates = ([Path(supplied)] if supplied else []) + [Path(root) / 'work_dir/_l100/campaign_window.json']
    paths = [p for p in candidates if p.is_file()]
    if not paths:
        raise ValueError('Set one shared actual --t0 for all three servers; no per-server automatic clock')
    window = CampaignWindow.from_dict(read_json(paths[0]))
    if any(CampaignWindow.from_dict(read_json(path)) != window for path in paths[1:]):
        raise ValueError('Existing LOCAL-T clock differs; refusing a new 20h window')
    return window


def write_window(root, t0_utc):
    window = CampaignWindow(t0_utc)
    path = Path(root) / 'work_dir/_l100/campaign_window.json'
    # Never reset an earlier L100 clock. Unknown v1 run lineage requires an
    # explicit migration rather than silently reinitializing its Students.
    for old in (Path(root) / 'l100/campaign_window.json', Path(root) / 'work_dir/_l100_v1/campaign_window.json'):
        if old.is_file() and read_json(old).get('t0_utc') != window.to_dict()['t0_utc']:
            raise ValueError('Earlier L100 actual clock exists: ' + str(old))
    with locked(path.parent / 'clock.lock'):
        immutable_json(path, window.to_dict())
    return dict(path=str(path), **window.to_dict(), training_started=False)


def build_artifacts(root=ROOT):
    """Explicit deterministic generation, without clock creation or launch."""
    root = Path(root)
    verify_sources(root)
    target = root / 'config/l100'
    target.mkdir(parents=True, exist_ok=True)
    immutable_json(target / 'L100_DesignRegistry_MD_Derived.json', registry_document())
    for server in SERVERS:
        immutable_json(target / f'{server}_queue.json', dict(campaign_id=CAMPAIGN_ID,
            server_id=server, blocks=[b.to_dict() for b in blocks_for(server)]))
    for case in CORE_CASES:
        path, expected = target / (case.run_id + '.yaml'), build_config(case)
        if path.exists() and yaml.safe_load(path.read_text()) != expected:
            raise ValueError('Generated case changed: ' + case.run_id)
        if not path.exists():
            path.write_text(yaml.safe_dump(expected, sort_keys=False))
    return dict(configs=len(CORE_CASES), deferred=4, activated=False, clock_created=False,
                registry_sha256=registry_sha256())


def _state(root, server):
    return read(camp(root, server) / 'status.json', dict(campaign_id=CAMPAIGN_ID,
        server_id=server, status='DEFINED_NOT_STARTED', runs={}, blocks={}, observations=[]))


def _save(root, server, state):
    state['updated_at_utc'] = utcnow()
    atomic_json(camp(root, server) / 'status.json', state)


def status(root, server):
    state = _state(root, server)
    try:
        window = shared_window(root).to_dict()
    except (ValueError, FileNotFoundError) as error:
        window = dict(status='UNBOUND', reason=str(error))
    return dict(state=state, window=window, queue=[b.to_dict() for b in blocks_for(server)],
                preflight=read(camp(root, server) / 'preflight.json'),
                training_started=any(r.get('training_started') for r in state['runs'].values()))


def verify_registration(root, server):
    apply_runtime_policy(root)
    path = camp(root, server) / 'registration.json'
    registered = read_json(path)
    if (registered.get('campaign_id') != CAMPAIGN_ID or registered.get('server_id') != server
            or registered.get('source_identity') != source_identity(root)
            or registered.get('registry_sha256') != registry_sha256()
            or registered.get('window') != shared_window(root).to_dict()):
        raise ValueError('LOCAL-T registered source/recipe/clock differs; no live mutation')
    return registered


def authorize_train(root, server, config_path):
    verify_registration(root, server)
    cfg = yaml.safe_load(Path(config_path).read_text())
    case = validate_config(cfg, require_bound=True)
    if case.server_id != server:
        raise ValueError('Wrong local server for this case')
    ready = read_json(camp(root, server) / 'preflight.json')
    if not ready.get('complete') or ready.get('source_identity') != source_identity(root):
        raise ValueError('Current local P0 receipt is required')
    data = read_json(cfg['l100']['dataset_manifest'])
    if ready['dataset_manifest_sha256'] != object_sha(data):
        raise ValueError('P0 data binding changed')
    receipt = read_json(camp(root, server) / 'admissions' / (case.block_id + '.json'))
    state = _state(root, server)
    if (state.get('status') != 'RUNNING'
            or state['runs'].get(case.run_id, {}).get('status') != 'TRAINING'
            or state['blocks'].get(case.block_id, {}).get('status') != 'ADMITTED'):
        raise ValueError('Controller has not authorized this active training case')
    block = next(b for b in blocks_for(server) if b.block_id == case.block_id)
    pending = [c.run_id for c in block.cases if not state['runs'].get(c.run_id, {}).get('endpoint_complete')]
    if not pending or pending[0] != case.run_id:
        raise ValueError('Trainer is not the next registered local case')
    for previous in block.cases[:block.cases.index(case)]:
        if previous.role == 'T' and not state['runs'].get(previous.run_id, {}).get('calibrated'):
            raise ValueError('Earlier local Teacher calibration must complete before this case')
    from l100.policy import _validate_admitted
    _validate_admitted(receipt, block, shared_window(root))
    return case


def _resolve_config(root, case):
    from l100.references import reference_path, validate_reference
    folder = camp(root, case.server_id)
    cfg = build_config(case)
    data_path = folder / 'dataset_manifest.json'
    data = read_json(data_path)
    cfg['l100'].update(dataset_manifest=str(data_path), runtime_policy_sha256=sha256(Path(root) / 'l100/runtime_policy.json'))
    for split, key in (('train', 'train_feeder_args'), ('val', 'val_feeder_args'),
                       ('rr', 'test_reduced_feeder_args'), ('fr', 'test_full_feeder_args')):
        cfg[key]['dataroot'] = data['splits'][split]['dataroot']
    if case.role == 'S':
        manifest, _, _, paths = validate_reference(case.reference_id, case.server_id, root, dataset_manifest=data)
        cfg['l100'].update(reference_manifest=str(reference_path(case.reference_id, case.server_id, root)),
            teacher_checkpoint=str(paths['teacher_checkpoint']), teacher_sha256=manifest['teacher_checkpoint_sha256'],
            tau_R=manifest['tau_R'], q_ref=manifest['q_ref'], q_cache=str(paths['q_cache_path']),
            q_cache_sha256=manifest['q_cache_sha256'])
    validate_config(cfg, require_bound=True)
    path = Path(root) / 'work_dir' / case.run_id / 'meta/config.resolved.yaml'
    if path.exists() and yaml.safe_load(path.read_text()) != cfg:
        raise ValueError('Started run bindings differ; refusing hot swap: ' + case.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def _command(root, args, log, deadline):
    """Each child has bounded campaign time; logs survive nonzero exit."""
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    remaining = (dt.datetime.fromisoformat(deadline) - dt.datetime.now(dt.timezone.utc)).total_seconds()
    if remaining <= 0:
        return 124, 0.
    started = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    with Path(log).open('a') as stream:
        process = subprocess.Popen([sys.executable, '-u', 'tools/l100_runner.py', *args],
            cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.terminate()  # Own child only; trainer preserves at update boundary.
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                if args and args[0] == 'train':
                    raise OwnTrainerStillPreserving('Own trainer is still preserving full-state; refusing a forced kill or another case')
                process.kill()
                process.wait()
            code = 124
    return code, (time.monotonic() - started) / 3600.


def evaluation_debt(root, server):
    total = 0.
    for case in cases_for(server):
        wd = Path(root) / 'work_dir' / case.run_id
        if read(wd / 'official/postrun_status.json').get('official_complete'):
            continue
        records = read(wd / 'official/raw_grid.json').get('records', [])
        evaluated = {r['update'] for r in records}
        pending = [s for s in grid_steps(case.updates) if s not in evaluated
                   and (wd / f'candidates/{s}/model.safetensors').is_file()]
        if pending:
            known = [r.get('seconds') for r in records if isinstance(r.get('seconds'), (int, float))
                     and not isinstance(r.get('seconds'), bool) and r['seconds'] >= 0]
            measured = (max(known) * len(pending) / 3600. * 1.15) if known else 0.
            total += max(len(pending) / 50. * 1.15, measured)
    return total


def _observation(case, component, hours, completed=True):
    return dict(server_id=case.server_id, sensor='GF2', role=case.role, updates=case.updates,
        width=case.width, depth=list(case.depth), profile=case.profile, component=component,
        hours=hours, completed=completed, evaluation_valid=completed)


def _event(root, server, case, name, evidence_path, timestamp=None):
    path = camp(root, server) / 'events' / case.run_id / (name + '.json')
    binding = dict(campaign_id=CAMPAIGN_ID, server_id=server, case_id=case.case_id,
        run_id=case.run_id, event=name, horizon_updates=case.updates,
        evidence_path=str(evidence_path), evidence_sha256=sha256(evidence_path))
    previous = read(path)
    if previous:
        if any(previous.get(key) != value for key, value in binding.items()):
            raise ValueError('Immutable lifecycle event evidence changed: ' + name)
        return previous
    return read_json(immutable_json(path, dict(binding, at_utc=timestamp or utcnow(),
        timestamp_source='saved_artifact' if timestamp else 'first_verified_ready')))


def _training_events(root, server, case, row):
    wd = Path(root) / 'work_dir' / case.run_id
    start_path = wd / 'meta/training_start_manifest.json'
    if start_path.is_file():
        start = read_json(start_path)
        if start.get('run_id') != case.run_id or start.get('campaign_id') != CAMPAIGN_ID:
            raise ValueError('Lifecycle training-start evidence belongs to another run')
        started = start.get('started_at_utc')
        if not started:
            raise ValueError('Actual training-start timestamp is missing')
        event = _event(root, server, case, 'teacher_started' if case.role == 'T' else 'student_started', start_path, started)
        row['started_at_utc'] = event['at_utc']
        row['training_started'] = True
    training = read(wd / 'meta/training_status.json')
    identity_path = wd / 'candidates' / str(case.updates) / 'identity.json'
    if training.get('training_complete') and training.get('actual_updates') == case.updates and identity_path.is_file():
        identity = read_json(identity_path)
        ended = identity.get('saved_at_utc') or training.get('training_completed_at_utc')
        event = _event(root, server, case, 'teacher_complete' if case.role == 'T' else 'student_complete', identity_path, ended)
        row['completed_at_utc'] = event['at_utc']


def _ensure_calibration(root, server, case, state, window):
    from l100.references import reference_path, validate_reference
    ref = reference_path(case.reference_id, server, root)
    if not ref.is_file():
        code, hours = _command(root, ['calibrate', '--server', server, '--run', case.run_id],
            camp(root, server) / 'logs' / (case.run_id + '.calibration.log'), window.deadline_utc.isoformat())
        state['observations'].append(_observation(case, 'CALIBRATION', hours, code == 0))
        if code:
            raise RuntimeError(f'Local calibration failed ({code}); see {case.run_id}.calibration.log')
    validate_reference(case.reference_id, server, root,
        dataset_manifest=read_json(camp(root, server) / 'dataset_manifest.json'))
    state['runs'].setdefault(case.run_id, {})['calibrated'] = True
    ready = _event(root, server, case, 'reference_ready', ref)
    state['runs'][case.run_id]['reference_ready_at_utc'] = ready['at_utc']
    _save(root, server, state)


def _run_case(root, server, case, state, window):
    wd = Path(root) / 'work_dir' / case.run_id
    row = state['runs'].setdefault(case.run_id, {})
    config = _resolve_config(root, case)
    _recover_final(root, case)
    training = read(wd / 'meta/training_status.json')
    _training_events(root, server, case, row)
    if row.get('train_attempt_started_utc'):
        elapsed = max(0., (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(row.pop('train_attempt_started_utc'))).total_seconds() / 3600.)
        row['train_subprocess_hours'] = row.get('train_subprocess_hours', 0.) + elapsed
        row['conservative_recovered_wall_hours'] = row.get('conservative_recovered_wall_hours', 0.) + elapsed
    if row.get('endpoint_complete'):
        if not training.get('training_complete') or training.get('actual_updates') != case.updates:
            raise ValueError('Controller endpoint state differs from saved actual training completion')
        if case.role == 'T':
            _ensure_calibration(root, server, case, state, window)
        return
    if not training.get('training_complete'):
        if dt.datetime.now(dt.timezone.utc) >= window.train_finish_utc:
            row.update(status='PARTIAL_TIME_LIMIT')
            _save(root, server, state)
            raise TrainingTimeLimit('Optimizer window closed; saved candidates require evaluation-only closeout')
        args = ['train', '--server', server, '--config', str(config)]
        if (wd / 'last/training_state.pt').is_file():
            args.append('--resume')
        elif (wd / 'meta/training_start_manifest.json').is_file() or row.get('training_started'):
            raise ValueError('Interrupted training lacks a full-state resume; refusing fresh restart')
        row.update(status='TRAINING', training_started=True, train_attempt_started_utc=utcnow())
        row.setdefault('submitted_at_utc', row['train_attempt_started_utc'])
        _save(root, server, state)
        code, hours = _command(root, args, camp(root, server) / 'logs' / (case.run_id + '.train.log'),
                               window.train_finish_utc.isoformat())
        row['train_subprocess_hours'] = row.get('train_subprocess_hours', 0.) + hours
        row.pop('train_attempt_started_utc', None)
        _save(root, server, state)
        if code not in (0, 75, 124):
            raise RuntimeError(f'Trainer failed with exit {code}; saved artifacts retained without fresh restart')
        _recover_final(root, case)
        training = read(wd / 'meta/training_status.json')
        _training_events(root, server, case, row)
    else:
        code = 0
    # Endpoint existence is verified by its evaluator; exit code alone is not completion.
    if (not training.get('training_complete') or training.get('actual_updates') != case.updates
            or not (wd / f'candidates/{case.updates}/model.safetensors').is_file()):
        row.update(status='PARTIAL_TIME_LIMIT' if code in (75, 124) else 'BLOCKED_LOCAL_FAILURE', exit_code=code)
        _save(root, server, state)
        if code in (75, 124):
            raise TrainingTimeLimit(f'{case.run_id}: exact endpoint not complete (exit={code})')
        raise RuntimeError(f'{case.run_id}: exact endpoint not complete (exit={code})')
    code, endpoint_hours = _command(root, ['endpoint', '--server', server, '--run', case.run_id],
        camp(root, server) / 'logs' / (case.run_id + '.endpoint.log'), window.deadline_utc.isoformat())
    endpoint = read(wd / 'official/postrun_status.json')
    if code or endpoint.get('endpoint_complete') is not True or endpoint.get('actual_updates') != case.updates:
        raise RuntimeError('Exact endpoint validation failed: ' + case.run_id)
    row.update(endpoint_complete=True, status='OFFICIAL_EVAL_PENDING')
    row['endpoint_hours'] = row.get('endpoint_hours', 0.) + endpoint_hours
    _save(root, server, state)
    if case.role == 'T':
        _ensure_calibration(root, server, case, state, window)
    else:
        # Students normally already evaluated their candidate grids inline.
        _postrun(root, server, case, state, window)


def _postrun(root, server, case, state, window):
    wd = Path(root) / 'work_dir' / case.run_id
    previous = read(wd / 'official/postrun_status.json')
    receipt = read(wd / 'official/upload_receipt.json')
    row = state['runs'].setdefault(case.run_id, {})
    if previous.get('official_complete') and previous.get('sheet_uploaded') and receipt.get('readback_verified'):
        row.update(status='UPLOAD_VERIFIED', official_complete=True, upload_pending=False)
        _save(root, server, state)
        return
    action = 'upload' if previous.get('official_complete') else 'postrun'
    args = [action, '--server', server, '--run', case.run_id] + (['--upload'] if action == 'postrun' else [])
    code, hours = _command(root, args,
        camp(root, server) / 'logs' / (case.run_id + '.postrun.log'), window.deadline_utc.isoformat())
    row['postrun_hours'] = row.get('postrun_hours', 0.) + hours
    report = read(wd / 'official/postrun_status.json')
    receipt = read(wd / 'official/upload_receipt.json')
    if report.get('official_complete'):
        uploaded = not code and report.get('sheet_uploaded') is True and receipt.get('readback_verified') is True
        row.update(status='OFFICIAL_EVAL_COMPLETE', official_complete=True,
                   upload_pending=not uploaded)
        total = row.get('train_subprocess_hours', 0.) + row.get('endpoint_hours', 0.) + row['postrun_hours']
        if not row.get('timing_observation_recorded'):
            state['observations'].append(_observation(case, 'TRAIN_EVAL', total))
            row['timing_observation_recorded'] = True
    else:
        row.update(status='OFFICIAL_EVAL_PENDING', postrun_exit_code=code)
    _save(root, server, state)


def _recover_final(root, case):
    wd = Path(root) / 'work_dir' / case.run_id
    cfg_path = wd / 'meta/config.resolved.yaml'
    if cfg_path.is_file() and (wd / 'candidates' / str(case.updates)).is_dir():
        from l100.training import recover_final_status
        return recover_final_status(wd, yaml.safe_load(cfg_path.read_text()), root)
    return False


def _closeout(root, server, state, window):
    from l100.resources import idle_evidence
    blocked = state.get('status') == 'BLOCKED_LOCAL_INTEGRITY_OR_RUNTIME'
    # A legacy job or a still-saving own trainer is never overlapped with GPU eval.
    while window.remaining_hours(utcnow()) > 0:
        idle = idle_evidence()
        if idle['idle']:
            break
        state.update(status='WAIT_CLOSEOUT_RESOURCE', idle_evidence=idle)
        _save(root, server, state)
        time.sleep(15)
    for case in cases_for(server):
        if window.remaining_hours(utcnow()) <= 0:
            break
        wd = Path(root) / 'work_dir' / case.run_id
        if not (wd / 'meta/training_start_manifest.json').is_file():
            continue
        _recover_final(root, case)
        training = read(wd / 'meta/training_status.json')
        if training.get('training_complete') and training.get('actual_updates') == case.updates:
            row = state['runs'].get(case.run_id, {})
            if not row.get('endpoint_complete') or (case.role == 'T' and not row.get('calibrated')):
                _run_case(root, server, case, state, window)
            _postrun(root, server, case, state, window)
        else:
            code, hours = _command(root, ['partial', '--server', server, '--run', case.run_id],
                camp(root, server) / 'logs' / (case.run_id + '.partial.log'), window.deadline_utc.isoformat())
            state['runs'].setdefault(case.run_id, {}).update(status='PARTIAL_TIME_LIMIT',
                partial_postrun_exit_code=code, partial_postrun_hours=hours)
    state['evaluation_debt_hours'] = evaluation_debt(root, server)
    state['unstarted_run_ids'] = [c.run_id for c in cases_for(server)
        if not (Path(root) / 'work_dir' / c.run_id / 'meta/training_start_manifest.json').is_file()]
    state['core_endpoint_count'] = sum(bool(state['runs'].get(c.run_id, {}).get('endpoint_complete')) for c in cases_for(server))
    for block in blocks_for(server):
        complete = [bool(state['runs'].get(run, {}).get('endpoint_complete')) for run in block.run_ids]
        entry = state['blocks'].setdefault(block.block_id, dict(run_ids=list(block.run_ids)))
        entry['pair_status'] = 'ENDPOINTS_COMPLETE' if all(complete) else 'PAIR_INCOMPLETE'
    if blocked:
        state['status'] = 'BLOCKED_LOCAL_INTEGRITY_OR_RUNTIME'
    else:
        active = [row for run, row in state['runs'].items() if run not in state['unstarted_run_ids']]
        if any(not row.get('endpoint_complete') for row in active):
            state['status'] = 'PARTIAL_TIME_LIMIT'
        elif state['evaluation_debt_hours'] > 0 or any(not row.get('official_complete') for row in active):
            state['status'] = 'OFFICIAL_EVAL_PENDING'
        elif any(row.get('upload_pending') for row in active):
            state['status'] = 'UPLOAD_PENDING'
        elif state['unstarted_run_ids']:
            state['status'] = 'PARTIAL_QUEUE'
        else:
            state['status'] = 'FINITE_QUEUE_FINISHED'
    _save(root, server, state)
    return 0 if state['status'] == 'FINITE_QUEUE_FINISHED' else 1


def run(root=ROOT, server='s3'):
    root, folder = Path(root).resolve(), camp(root, server)
    with locked(folder / 'runner.lock'):
        verify_registration(root, server)
        window, state = shared_window(root), _state(root, server)
        from l100.resources import idle_evidence, assess_block
        while window.remaining_hours(utcnow()) > 2:
            idle = idle_evidence()
            if idle['idle']:
                break
            state.update(status='WAIT_LOCAL_RESOURCE', idle_evidence=idle)
            _save(root, server, state)
            time.sleep(15)
        if window.remaining_hours(utcnow()) <= 2:
            state['status'] = 'CLOSEOUT'
            return _closeout(root, server, state, window)
        try:
            code, hours = _command(root, ['preflight', '--server', server], folder / 'logs/preflight.log',
                                   window.train_finish_utc.isoformat())
            state['setup_hours'] = state.get('setup_hours', 0.) + hours
            if code:
                raise RuntimeError('Local P0 failed; inspect logs/preflight.log')
            for block in blocks_for(server):
                admission_path = folder / 'admissions' / (block.block_id + '.json')
                admission = read(admission_path) or None
                if state['blocks'].get(block.block_id, {}).get('status', '').startswith('NOT_ADMITTED'):
                    continue
                for case in block.cases:
                    completed = {c.run_id for c in block.cases if state['runs'].get(c.run_id, {}).get('endpoint_complete')}
                    calibrated = {c.reference_id for c in block.cases if c.role == 'T' and state['runs'].get(c.run_id, {}).get('calibrated')}
                    if case.run_id in completed and (case.role == 'S' or case.reference_id in calibrated):
                        continue
                    remaining = estimate_block_hours(block, state['observations'], completed, calibrated)
                    decision = evaluate_admission(window, utcnow(), block, remaining,
                        evaluation_debt(root, server), admitted_receipt=admission)
                    atomic_json(folder / 'admission_checks' / (block.block_id + '.json'), decision)
                    if not decision['allowed']:
                        state['blocks'][block.block_id] = dict(status=decision['reason'],
                            pair_status='PAIR_INCOMPLETE', run_ids=list(block.run_ids))
                        _save(root, server, state)
                        break
                    resources = assess_block(root, [c for c in block.cases if c.run_id not in completed])
                    atomic_json(folder / 'resources' / (block.block_id + '.json'), resources)
                    if not resources['allowed']:
                        raise RuntimeError('Insufficient local resources: ' + str(resources['reasons']))
                    if admission is None:
                        immutable_json(admission_path, decision)
                        admission = decision
                    state['blocks'][block.block_id] = dict(status='ADMITTED', run_ids=list(block.run_ids))
                    state['status'] = 'RUNNING'
                    _save(root, server, state)
                    _run_case(root, server, case, state, window)
                else:
                    state['blocks'][block.block_id] = dict(status='ENDPOINTS_COMPLETE', run_ids=list(block.run_ids))
            state['status'] = 'CLOSEOUT'
        except OwnTrainerStillPreserving as error:
            state.update(status='WAIT_OWN_TRAINER_PRESERVING', error=str(error))
            _save(root, server, state)
            return 1
        except TrainingTimeLimit as error:
            state.update(status='PARTIAL_TIME_LIMIT', time_limit_reason=str(error))
            _save(root, server, state)
        except (ValueError, RuntimeError, OSError, TimeoutError) as error:
            state.update(status='BLOCKED_LOCAL_INTEGRITY_OR_RUNTIME', error=str(error))
            _save(root, server, state)
        # Teacher full-grid debt never delays its first local Student. It remains
        # explicit and is repaid before the 20h deadline; no unfinished upload.
        return _closeout(root, server, state, window)


def _process_start(pid):
    try:
        # Linux start-time ticks disambiguate a recycled PID from our child.
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, ValueError, IndexError, TypeError):
        return None


def _submitted_alive(folder):
    submitted = read(folder / 'runner.pid.json')
    expected = submitted.get('process_start_ticks')
    return bool(expected and _process_start(submitted.get('pid')) == expected)


def start(root, server, *, t0=None, window_path=None, foreground=False, dry_run=False):
    root, folder = Path(root).resolve(), camp(root, server)
    if dry_run:
        return status(root, server)
    verify_sources(root)
    apply_runtime_policy(root)
    if t0:
        write_window(root, t0)
    window = shared_window(root, window_path)
    with locked(root / 'work_dir/_l100/clock.lock'):
        immutable_json(root / 'work_dir/_l100/campaign_window.json', window.to_dict())
    if dt.datetime.now(dt.timezone.utc) < window.t0_utc or window.remaining_hours(utcnow()) <= 0:
        raise ValueError('Actual shared window is not currently open')
    with locked(folder / 'registration.lock'):
        identity = dict(campaign_id=CAMPAIGN_ID, server_id=server,
            registry_sha256=registry_sha256(), source_identity=source_identity(root), window=window.to_dict())
        immutable_json(folder / 'registration.json', identity)
    if foreground:
        return dict(exit_code=run(root, server))
    with locked(folder / 'startup.lock'):
        # Do not hold runner.lock while spawning: the child's nonblocking lock
        # acquisition otherwise races its own parent and exits before starting.
        try:
            with locked(folder / 'runner.lock'):
                pass
        except BlockingIOError:
            return dict(status='ALREADY_RUNNING', window=window.to_dict())
        if _submitted_alive(folder):
            return dict(status='ALREADY_SUBMITTED', window=window.to_dict())
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'runner.log').open('a') as stream:
            process = subprocess.Popen([sys.executable, '-u', 'tools/l100_runner.py', 'run', '--server', server],
                cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        atomic_json(folder / 'runner.pid.json', dict(pid=process.pid, submitted_at_utc=utcnow(),
            process_start_ticks=_process_start(process.pid)))
    return dict(pid=process.pid, status='SUBMITTED', log=str(folder / 'runner.log'), window=window.to_dict())
