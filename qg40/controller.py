"""Finite QG40 scheduler: common wall-clock, local dependencies, atomic blocks."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from qg40.common import (ROOT, CAMPAIGN_ID, append_event, atomic_json, camp,
                        immutable_json, locked, object_sha, read, read_json,
                        sha256, source_identity, utcnow)
from qg40.plan import (CASES, SERVERS, SHEET_TABS, SensorSpec, active_cases,
                      blocks_for, build_config, case_for, cases_for, registry_document,
                      registry_sha256, teacher_for, verify_sources as validate_definition)
from qg40.policy import (CampaignWindow, admission, c3_eligible, choose_student_trial,
                         promote_screen)

HOLD = 'QG40_FUTURE_ADMISSIONS_HOLD_v1'
DONE = {'OFFICIAL_EVAL_COMPLETE', 'UPLOAD_VERIFIED', 'ARCHIVED'}


def shared_window(root, supplied=None):
    canonical = Path(root) / 'work_dir/_qg40/campaign_window.json'
    packaged = Path(root) / 'qg40/campaign_window.json'
    paths = [path for path in (canonical, packaged) if path.is_file()]
    if supplied:
        paths.insert(0, Path(supplied))
    if not paths:
        raise ValueError('Shared campaign window missing; set one common t0, never local-server now')
    window = CampaignWindow.from_dict(read_json(paths[0]))
    if any(CampaignWindow.from_dict(read_json(path)) != window for path in paths[1:]):
        raise ValueError('Shared QG40 window differs from existing runtime/deployment; refusing clock reset')
    return window


def write_window(root, t0_utc):
    window = CampaignWindow(t0_utc)
    path = Path(root) / 'work_dir/_qg40/campaign_window.json'
    if (path.is_file() or (Path(root) / 'qg40/campaign_window.json').is_file()) and shared_window(root) != window:
        raise ValueError('Shared QG40 window already fixed; refusing clock reset')
    immutable_json(path, window.to_dict())
    return dict(path=str(path), **window.to_dict(), training_started=False)


def build_artifacts(root=ROOT):
    """Explicit deterministic generation; never register or activate a campaign."""
    root = Path(root)
    validation = validate_definition()
    target = root / 'config/qg40'
    target.mkdir(parents=True, exist_ok=True)
    immutable_json(target / 'QG40_Registry.json', registry_document())
    for case in CASES:
        path = target / (case.run_id + '.yaml')
        expected = build_config(case)
        if path.exists() and yaml.safe_load(path.read_text()) != expected:
            raise ValueError('Existing generated case differs: ' + case.run_id)
        if not path.exists():
            path.write_text(yaml.safe_dump(expected, sort_keys=False))
    return dict(validation=validation, configs=len(CASES), activated=False)


def branches(root, server):
    folder = camp(root, server)
    return dict(screen=read(folder / 'branch_receipt.json').get('profile', 'BASE'),
                confirm=read(folder / 'screen_promotion.json').get('confirm_profile', 'BASE'),
                enable_c3=read(folder / 'c3_branch_receipt.json').get('admit', False), include_reserve=True)


def _state(root, server):
    return read(camp(root, server) / 'status.json', dict(campaign_id=CAMPAIGN_ID,
                server_id=server, status='DEFINED_NOT_DEPLOYED', runs={}, blocks={}, observations=[]))


def effective_queue(root, server, state=None):
    state = state or _state(root, server)
    if state['status'] in ('WINDOW_CLOSED', 'FINITE_QUEUE_FINISHED'):
        return []
    queue = []
    for block in blocks_for(server, **branches(root, server)):
        saved = state['blocks'].get(block.block_id, {})
        if saved.get('status') in {'DONE', 'NOT_ADMITTED_BUDGET', 'NOT_ADMITTED_CUTOFF'}:
            continue
        order = saved.get('run_ids', list(block.run_ids))
        if saved and order != list(block.run_ids):
            raise ValueError('Admitted QG40 block membership/order changed')
        pending = [run for run in order if state['runs'].get(run, {}).get('status') not in DONE]
        if pending:
            queue.append(dict(block_id=block.block_id, run_ids=order, pending_run_ids=pending,
                              admitted=saved.get('status') == 'ADMITTED', is_c3=block.is_c3))
    return sorted(queue, key=lambda row: not row['admitted'])


def evaluation_debt(root, server, state=None):
    """Outstanding valid candidate evaluations, never hidden by a failed evaluator."""
    state = state or _state(root, server)
    debt = 0.
    for run, row in state['runs'].items():
        if row.get('status') in DONE:
            continue
        wd = Path(root) / 'work_dir' / run
        grid = read(wd / 'official/raw_grid.json')
        evaluated = {r['update'] for r in grid.get('records', [])}
        pending = [p for p in (wd / 'candidates').glob('*') if p.name.isdigit()
                   and int(p.name) not in evaluated and (p / 'model.safetensors').is_file()]
        # Conservative fallback is one hour for a full grid until same-server
        # actual evaluation observations exist; never zero for unknown work.
        samples = [r['hours'] for r in state.get('observations', [])
                   if r.get('component') == 'EVAL' and r.get('completed') and r.get('evaluation_valid')]
        full_grid = max(samples, default=1.) * 1.25
        debt += len(pending) / 50 * full_grid
        if row.get('status') == 'TRAIN50K_COMPLETE':
            debt += .1  # selected-point publication/profiling/readback allowance
    return debt


def status(root, server, *, window_path=None):
    state = _state(root, server)
    queue = effective_queue(root, server, state)
    try:
        window = shared_window(root, window_path)
        remaining, timing = window.remaining_hours(utcnow()), window.to_dict()
    except (ValueError, FileNotFoundError) as error:
        remaining, timing = None, dict(status='UNBOUND', reason=str(error))
    rows = cases_for(server)
    return dict(campaign_id=CAMPAIGN_ID, server_id=server, state=state['status'],
                registry_sha256=registry_sha256(), definitions=len(CASES), local_definitions=len(rows),
                local_baseline_count=sum(c.baseline_id_preserved for c in rows),
                branch=branches(root, server), next_two_run_ids=[r for b in queue for r in b['pending_run_ids']][:2],
                queue=queue, teacher_reference=rows[0].reference_id, sheet_tab=SHEET_TABS[server],
                window=timing, remaining_hours=remaining, evaluation_debt_hours=evaluation_debt(root, server, state),
                preflight=read(camp(root, server) / 'preflight.json'),
                registered=(camp(root, server) / 'registration.json').exists(),
                training_started=any(r.get('training_started') for r in state['runs'].values()))


def start(root, server, *, spec_path=None, window_path=None, dry_run=False, device='cuda', foreground=False):
    root = Path(root).resolve()
    validate_definition()
    if dry_run:
        return status(root, server, window_path=window_path)
    if device != 'cuda':
        raise ValueError('Production QG40 start requires CUDA; CPU is for isolated tests')
    if Path('/.dockerenv').exists() and not foreground:
        raise ValueError('Use a dedicated persistent container with start --foreground, not a retiring WV3 container')
    window = shared_window(root, window_path)
    if dt.datetime.now(dt.timezone.utc) < window.t0_utc:
        raise ValueError('Shared preparation start is in the future')
    if window.remaining_hours(utcnow()) <= 0:
        raise ValueError('Common 40h window already ended; clock cannot be restarted')
    folder = camp(root, server)
    binding = Path(spec_path) if spec_path else root / 'config/qg40' / (cases_for(server)[0].sensor + '_sensor.json')
    if spec_path or binding.is_file():
        spec = SensorSpec.from_dict(read_json(binding))
    else:
        from qg40.bootstrap import default_sensor_spec
        spec = default_sensor_spec(root, cases_for(server)[0].sensor)
    if spec.sensor != cases_for(server)[0].sensor or not spec.is_bound:
        raise ValueError('Actual local sensor paths/band/source must be bound before registration')
    with locked(folder / '.registration.lock'):
        shared = root / 'work_dir/_qg40/campaign_window.json'
        immutable_json(shared, window.to_dict())
        original_hold = read(root / 'work_dir/_eval_phase/hold.json')
        known_holds = {None, 'FH20R1_FUTURE_ADMISSIONS_HOLD_v1', 'FH20R1_R2_BOUNDARY_HOLD_v1', HOLD}
        # Verify current known protocol spelling from unchanged legacy module.
        from tools.fh20r1_runner import HOLD as legacy_hold
        known_holds.add(legacy_hold)
        if ((original_hold and not original_hold.get('protocol_id'))
                or original_hold.get('protocol_id') not in known_holds):
            raise ValueError('Foreign hold owner preserved; explicit coordination needed')
        expected = dict(campaign_id=CAMPAIGN_ID, server_id=server, device=device,
                        window=window.to_dict(), sensor_spec_sha256=object_sha(spec.to_dict()),
                        registry_sha256=registry_sha256(), source_identity=source_identity(root))
        existing = read(folder / 'registration.json')
        if existing and existing.get('identity') != expected:
            raise ValueError('Existing registration differs; exact-resume identity cannot be rewritten')
        if not existing:
            existing = dict(identity=expected, registered_at_utc=utcnow(), original_wv3_hold=original_hold)
            atomic_json(folder / 'registration.json', existing)
            immutable_json(folder / 'sensor_spec.json', spec.to_dict())
        hold = dict(protocol_id=HOLD, campaign_id=CAMPAIGN_ID, server=server, auto_release=False,
                    reason='QG40: finish active WV3 case; do not admit new WV3 cases')
        atomic_json(root / 'work_dir/_eval_phase/hold.json', hold)
        from qg40.deployment import install_recovery, spawn
        try:
            recovery = install_recovery(root, server)
        except Exception as error:
            recovery = dict(status='WARNING', error=str(error), automatic_recovery=False)
        atomic_json(folder / 'recovery.json', recovery)
    if foreground:
        return run(root, server)
    return dict(status='QG40_REQUESTED', recovery=recovery, launch=spawn(root, server), window=window.to_dict())


def verify_registration(root, server):
    registration = read_json(camp(root, server) / 'registration.json')
    identity = registration['identity']
    if (identity['campaign_id'] != CAMPAIGN_ID or identity['server_id'] != server
            or identity['registry_sha256'] != registry_sha256()
            or identity['source_identity'] != source_identity(root)
            or identity['window'] != shared_window(root).to_dict()
            or identity['sensor_spec_sha256'] != object_sha(read_json(camp(root, server) / 'sensor_spec.json'))):
        raise ValueError('QG40 registration/source/data/window changed')
    hold = read(Path(root) / 'work_dir/_eval_phase/hold.json')
    if (hold.get('protocol_id') != HOLD or hold.get('server') != server
            or hold.get('campaign_id') != CAMPAIGN_ID):
        raise ValueError('QG40 admission hold no longer owned; never override another owner')
    return registration


def authorize_train(root, server, config_path):
    """The executable training route cannot bypass registration or atomic admission."""
    verify_registration(root, server)
    cfg = yaml.safe_load(Path(config_path).read_text())
    from qg40.training import validate_config
    case = validate_config(cfg)
    if case.server_id != server:
        raise ValueError('Training case belongs to another server')
    state = _state(root, server)
    if state['status'] == 'BLOCKED_INTEGRITY':
        raise ValueError('Integrity failure needs diagnosis, not automatic training retry')
    block = state['blocks'].get(case.block_id, {})
    selected = next((b for b in blocks_for(server, **branches(root, server)) if b.block_id == case.block_id), None)
    if (not selected or block.get('status') != 'ADMITTED' or block.get('run_ids') != list(selected.run_ids)
            or case.run_id not in selected.run_ids or not block.get('reservation', {}).get('allowed')):
        raise ValueError('Case lacks a current registered atomic block admission')
    if state['runs'].get(case.run_id, {}).get('status') != 'RUNNING':
        raise ValueError('Controller has not authorized this active training case')
    receipt = read(camp(root, server) / 'preflight.json')
    if not receipt.get('complete') or receipt.get('source_identity') != source_identity(root):
        raise ValueError('Current numerical release has no completed local preflight')
    return case


def _command(root, args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        return subprocess.run([sys.executable, str(Path(root) / 'tools/qg40_runner.py')] + args,
                              cwd=root, stdout=stream, stderr=subprocess.STDOUT).returncode


def _write_state(root, server, state):
    atomic_json(camp(root, server) / 'status.json', state)


def _resources_available(root, server, cases, state, block_id):
    from qg40.resources import assess_block
    decision = assess_block(root, cases, observations=state['observations'])
    atomic_json(camp(root, server) / 'resources' / (block_id + '.json'), decision)
    if decision['allowed']:
        state['blocks'].get(block_id, {}).pop('resource_wait', None)
        return True
    block = state['blocks'].setdefault(block_id, dict(status='WAIT_LOCAL_RESOURCE',
        run_ids=[c.run_id for c in cases]))
    block['resource_wait'] = decision
    state.update(status='WAIT_LOCAL_RESOURCE', waiting_block=block_id)
    _write_state(root, server, state)
    return False


def _wait_local_idle(root, server, state, window):
    from tools.fh20r1_runner import inventory, gpu_processes
    while window.remaining_hours(utcnow()) > 0:
        busy = [r for r in inventory() if not r.get('waiting_controller')]
        gpu = gpu_processes()
        if not busy and gpu == []:
            return True
        state.update(status='WAIT_LOCAL_RESOURCE', active_processes=busy, gpu_processes=gpu)
        _write_state(root, server, state)
        time.sleep(15)
    return False


def _resolve_config(root, case):
    from qg40.references import reference_path, validate_reference
    folder, wd = camp(root, case.server_id), Path(root) / 'work_dir' / case.run_id
    cfg = build_config(case)
    data_path = folder / 'dataset_manifest.json'
    data = read_json(data_path)
    cfg['qg40'].update(dataset_manifest=str(data_path), sensor_spec_path=str(folder / 'sensor_spec.json'),
                        sensor_spec=read_json(folder / 'sensor_spec.json'))
    for split, key in (('train', 'train_feeder_args'), ('val', 'val_feeder_args'),
                       ('rr', 'test_reduced_feeder_args'), ('fr', 'test_full_feeder_args')):
        cfg[key]['dataroot'] = data['splits'][split]['dataroot']
    if case.role == 'S':
        manifest, _, _, paths = validate_reference(case.reference_id, case.server_id, root)
        cfg['qg40'].update(reference_manifest=str(reference_path(case.reference_id, case.server_id, root)),
                          teacher_checkpoint=paths['teacher_checkpoint'], teacher_sha256=manifest['teacher_checkpoint_sha256'],
                          tau_R=manifest['tau_R'], q_ref=manifest['q_ref'], q_cache=paths['q_cache_path'],
                          q_cache_sha256=manifest['q_cache_sha256'])
    path = wd / 'meta/config.resolved.yaml'
    if path.exists() and yaml.safe_load(path.read_text()) != cfg:
        raise ValueError('Existing run config differs; no fresh overwrite: ' + case.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def ensure_reference(root, server, reference_id, state, window):
    from qg40.references import reference_path, validate_reference, import_reference
    path = reference_path(reference_id, server, root)
    if path.exists():
        validate_reference(reference_id, server, root)
        return True
    owner = teacher_for(reference_id).server_id
    if owner == server:
        return False
    archive = camp(root, server) / 'incoming' / (reference_id + '.tar.gz')
    if archive.is_file():
        code = _command(root, ['import', '--reference', reference_id, '--archive', str(archive),
                              '--server', server], camp(root, server) / 'reference_import.log')
        if code:
            raise ValueError('Local reference import failed; inspect reference_import.log')
        validate_reference(reference_id, server, root)
        return True
    state.update(status='WAIT_REFERENCE', waiting_reference=reference_id, reference_owner=owner,
                 expected_transfer_path=str(archive), deadline_utc=window.deadline_utc.isoformat())
    _write_state(root, server, state)
    return False


def _branch_decisions(root, server, state, window):
    folder = camp(root, server)
    baseline = [c for c in cases_for(server) if c.baseline_id_preserved]
    if not all(state['runs'].get(c.run_id, {}).get('status') in DONE for c in baseline):
        return
    if server in ('s2', 's5') and not (folder / 'branch_receipt.json').exists():
        _command(root, ['branch-evidence', '--server', server], folder / 'branch_evidence.log')
        evidence = read(folder / 'diagnostics/student_trial_evidence.json')
        evidence.update(p0_passed=True, baseline_complete=True)
        decision = choose_student_trial(server, evidence, utcnow())
        immutable_json(folder / 'branch_receipt.json', decision)
    if server in ('s1', 's3') and not (folder / 'c3_branch_receipt.json').exists():
        _command(root, ['branch-evidence', '--server', server], folder / 'branch_evidence.log')
        evidence = read(folder / 'diagnostics/c3_evidence.json')
        choice = c3_eligible(server, evidence)
        c3 = [c for c in cases_for(server) if c.tier == 'P2_TEACHER_CONDITIONAL']
        reservation = admission(window, utcnow(), c3, p0_ready=True,
            evaluation_debt_hours=evaluation_debt(root, server, state),
            observations=state.get('observations', []), c3_evidence_valid=choice['eligible'])
        immutable_json(folder / 'c3_branch_receipt.json', dict(choice, admit=reservation['allowed'], reservation=reservation))
    if server in ('s2', 's5') and not (folder / 'screen_promotion.json').exists():
        screen = branches(root, server)['screen']
        runs = [c for c in active_cases(server, screen=screen) if c.tier in ('P1_SCREEN_BASE', 'P1_SCREEN_ALT')]
        if all(state['runs'].get(c.run_id, {}).get('status') in DONE for c in runs):
            from qg40.postrun import screen_result
            base, alt = [], []
            for case in runs:
                (base if case.profile == 'BASE' else alt).append(screen_result(case.run_id, root))
            immutable_json(folder / 'screen_promotion.json', promote_screen(server, screen, base, alt))


def run_case(root, server, case, state, window):
    wd = Path(root) / 'work_dir' / case.run_id
    row = state['runs'].setdefault(case.run_id, {})
    cfg_path = _resolve_config(root, case)
    trained = read(wd / 'meta/training_status.json').get('training_complete', False)
    if not trained:
        resume = (wd / 'last/training_state.pt').is_file()
        if (row.get('training_started') or (wd / 'meta/training_start_manifest.json').exists()) and not resume:
            raise ValueError('Interrupted run lacks exact fullstate; fresh restart prohibited')
        args = ['train', '--config', str(cfg_path), '--device', 'cuda'] + (['--resume'] if resume else [])
        row.update(status='RUNNING', training_started=True)
        state['status'] = 'RUNNING'
        _write_state(root, server, state)
        code = _command(root, args, wd / 'qg40_train.log')
        if code:
            row['status'] = 'INCOMPLETE_AT_DEADLINE' if window.remaining_hours(utcnow()) <= 0 else 'PAUSED' if code == 75 else 'BLOCKED_INTEGRITY'
            if row['status'] == 'BLOCKED_INTEGRITY':
                state['status'] = 'BLOCKED_INTEGRITY'
            _write_state(root, server, state)
            return False
    training = read(wd / 'meta/training_status.json')
    if training.get('actual_updates') != 50000 or training.get('training_complete') is not True:
        raise ValueError('Trainer returned without an exact50K completion receipt')
    row['status'] = 'TRAIN50K_COMPLETE'
    _write_state(root, server, state)
    if case.role == 'T':
        from qg40.references import reference_path
        if not reference_path(case.reference_id, server, root).is_file():
            cal_start = time.monotonic()
            code = _command(root, ['calibrate', '--run', case.run_id, '--server', server], wd / 'qg40_calibration.log')
            row['calibration_seconds'] = row.get('calibration_seconds', 0.) + time.monotonic() - cal_start
            if code:
                return False
        # Export immediately after exact50K calibration, before optional report debt.
        if case.reference_id in {'QB_TA', 'GF2_TA'}:
            code = _command(root, ['export', '--reference', case.reference_id, '--server', server], wd / 'qg40_export.log')
            if code:
                return False
    post_start = time.monotonic()
    code = _command(root, ['postrun', '--run', case.run_id], wd / 'qg40_postrun.log')
    row['postrun_seconds'] = row.get('postrun_seconds', 0.) + time.monotonic() - post_start
    if code or not read(wd / 'official/postrun_status.json').get('official_complete'):
        row['status'] = 'EVALUATION_PENDING'
        _write_state(root, server, state)
        return False
    row['status'] = 'OFFICIAL_EVAL_COMPLETE'
    code = _command(root, ['upload', '--run', case.run_id], wd / 'qg40_upload.log')
    uploaded = (not code and read(wd / 'official/upload_receipt.json').get('readback_verified') is True
                and read(wd / 'official/postrun_status.json').get('sheet_uploaded') is True)
    row.update(upload_pending=not uploaded, status='UPLOAD_VERIFIED' if uploaded else 'OFFICIAL_EVAL_COMPLETE')
    if not row.get('timing_observation_recorded'):
        measured = read(wd / 'meta/training_status.json')
        components = dict(TRAIN=measured.get('training_seconds', 0.) + measured.get('io_seconds', 0.)
                               + measured.get('diagnostic_seconds', 0.),
                          EVAL=measured.get('evaluation_seconds', 0.) + row.get('postrun_seconds', 0.),
                          CALIBRATION=row.get('calibration_seconds', 0.))
        for component, seconds in components.items():
            if seconds > 0:
                state['observations'].append(dict(server_id=server, sensor=case.sensor, role=case.role,
                    width=case.width, depth=list(case.depth), profile=case.profile,
                    peak_training_memory_bytes=measured.get('peak_memory_bytes') if component == 'TRAIN' else None,
                    memory_scope=measured.get('peak_memory_scope') if component == 'TRAIN' else None,
                    component=component, hours=seconds/3600,
                    completed=True, evaluation_valid=True))
        row['timing_observation_recorded'] = True
    _write_state(root, server, state)
    return True


def _closeout(root, server, state, window):
    records = {}
    enabled = {c.run_id for c in active_cases(server, **branches(root, server))}
    for case in cases_for(server):
        row = state['runs'].get(case.run_id, {})
        block_reason = state['blocks'].get(case.block_id, {}).get('status')
        reason = block_reason if block_reason in {'NOT_ADMITTED_BUDGET', 'NOT_ADMITTED_CUTOFF'} else 'NOT_ADMITTED_BUDGET'
        if state['blocks'].get(case.block_id, {}).get('resource_wait'):
            reason = 'NOT_ADMITTED_RESOURCE'
        if case.tier == 'P2_TEACHER_CONDITIONAL':
            c3 = read(camp(root, server) / 'c3_branch_receipt.json')
            if c3.get('eligible') and not c3.get('admit'):
                reason = c3.get('reservation', {}).get('reason', reason)
                enabled.add(case.run_id)
        records[case.run_id] = dict(row, status=row.get('status') or
            ('NOT_ADMITTED_CONDITION' if case.run_id not in enabled else reason))
    state['status'] = 'WINDOW_CLOSED' if window.remaining_hours(utcnow()) <= 0 else 'FINITE_QUEUE_FINISHED'
    _write_state(root, server, state)
    atomic_json(camp(root, server) / 'closeout.json', dict(campaign_id=CAMPAIGN_ID, server_id=server,
                at_utc=utcnow(), window=window.to_dict(), runs=records,
                fixed_base_separate_from_tuned=True, teacher_lineages_never_pooled=True,
                debt_hours=evaluation_debt(root, server, state)))


def retry_uploads(root, server, state):
    """Completed result delivery is independent of GPU/clock/learning restart."""
    for run, row in state['runs'].items():
        if row.get('status') in DONE and row.get('upload_pending', True):
            wd = Path(root) / 'work_dir' / run
            code = _command(root, ['upload', '--run', run], wd / 'qg40_upload.log')
            verified = (not code and read(wd / 'official/postrun_status.json').get('sheet_uploaded') is True
                        and read(wd / 'official/upload_receipt.json').get('readback_verified') is True)
            row['upload_pending'] = not verified
            if verified:
                row['status'] = 'UPLOAD_VERIFIED'
    _write_state(root, server, state)


def reconcile(root, server, state):
    """Reuse only locally verified exact results; missing Sheet rows never mean fresh."""
    from qg40.upload import row_values
    for case in cases_for(server):
        wd = Path(root) / 'work_dir' / case.run_id
        row = state['runs'].get(case.run_id, {})
        complete = read(wd / 'official/postrun_status.json').get('official_complete') is True
        if complete or row.get('status') in DONE:
            _resolve_config(root, case)
            row_values(case.run_id, root)
            row = state['runs'].setdefault(case.run_id, {})
            row.update(status='OFFICIAL_EVAL_COMPLETE', training_started=True)
            row['upload_pending'] = read(wd / 'official/upload_receipt.json').get('readback_verified') is not True
        elif (wd / 'meta/training_start_manifest.json').exists() and not (wd / 'last/training_state.pt').is_file():
            raise ValueError('Interrupted existing case lacks exact fullstate: ' + case.run_id)
    for block in blocks_for(server, **branches(root, server)):
        if all(state['runs'].get(run, {}).get('status') in DONE for run in block.run_ids):
            state['blocks'][block.block_id] = dict(status='DONE', run_ids=list(block.run_ids))
    _write_state(root, server, state)


def _run(root=ROOT, server='s1'):
    root = Path(root).resolve()
    folder = camp(root, server)
    with locked(folder / '.runner.lock'):
        verify_registration(root, server)
        window, state = shared_window(root), _state(root, server)
        if state['status'] == 'BLOCKED_INTEGRITY' or any(r.get('status') == 'BLOCKED_INTEGRITY' for r in state['runs'].values()):
            return 2
        retry_uploads(root, server, state)
        if state['status'] in ('WINDOW_CLOSED', 'FINITE_QUEUE_FINISHED'):
            return 0
        reconcile(root, server, state)
        # Hold is already published. Wait for the unchanged WV3 owner naturally;
        # never kill or alter its checkpoints, source, budget, or original ledger.
        old_lock = root / 'work_dir/_fh20r1' / server / '.runner.lock'
        while window.remaining_hours(utcnow()) > 0:
            try:
                with locked(old_lock):
                    break
            except BlockingIOError:
                state['status'] = 'WAIT_WV3_CASE_BOUNDARY'
                _write_state(root, server, state)
                time.sleep(15)
        if not _wait_local_idle(root, server, state, window):
            _closeout(root, server, state, window)
            return 0
        if _command(root, ['preflight', '--server', server, '--spec', str(folder / 'sensor_spec.json')],
                    folder / 'preflight.log') or not read(folder / 'preflight.json').get('complete'):
            raise ValueError('Local preflight failed; original training remains untouched')
        state['status'] = 'LOCAL_READY'
        _write_state(root, server, state)
        while window.remaining_hours(utcnow()) > 0:
            verify_registration(root, server)
            retry_uploads(root, server, state)
            _branch_decisions(root, server, state, window)
            queue = effective_queue(root, server, state)
            if not queue:
                break
            item = queue[0]
            cases = [case_for(run) for run in item['run_ids']]
            first = next(c for c in cases if c.run_id in item['pending_run_ids'])
            if first.role == 'S' and not ensure_reference(root, server, first.reference_id, state, window):
                time.sleep(15)
                continue
            if not item['admitted']:
                decision = admission(window, utcnow(), cases, p0_ready=True,
                    evaluation_debt_hours=evaluation_debt(root, server, state), observations=state['observations'],
                    c3_evidence_valid=branches(root, server)['enable_c3'])
                atomic_json(folder / 'admissions' / (item['block_id'] + '.json'), decision)
                if not decision['allowed']:
                    state['blocks'][item['block_id']] = dict(status=decision['reason'], run_ids=item['run_ids'])
                    _write_state(root, server, state)
                    continue
                if not _resources_available(root, server, cases, state, item['block_id']):
                    return 75  # Local hold only; no train, kill, clock reset or integrity waiver.
                state['blocks'][item['block_id']] = dict(status='ADMITTED', run_ids=item['run_ids'], reservation=decision)
                _write_state(root, server, state)
            for case in cases:
                if state['runs'].get(case.run_id, {}).get('status') in DONE:
                    continue
                if window.remaining_hours(utcnow()) <= 0:
                    break
                started = (Path(root) / 'work_dir' / case.run_id / 'meta/training_start_manifest.json').exists()
                if not started and dt.datetime.now(dt.timezone.utc) >= window.admission_cutoff_utc:
                    state['runs'].setdefault(case.run_id, {})['status'] = 'NOT_ADMITTED_CUTOFF'
                    state['blocks'][item['block_id']]['status'] = 'NOT_ADMITTED_CUTOFF'
                    _write_state(root, server, state)
                    break
                if not _wait_local_idle(root, server, state, window):
                    break
                if not run_case(root, server, case, state, window):
                    if window.remaining_hours(utcnow()) <= 0:
                        break
                    return 2
            else:
                state['blocks'][item['block_id']]['status'] = 'DONE'
                _write_state(root, server, state)
                continue
            break
        _closeout(root, server, state, window)
        return 0


def run(root=ROOT, server='s1'):
    try:
        return _run(root, server)
    except BlockingIOError:
        return 3
    except TimeoutError:
        _closeout(root, server, _state(root, server), shared_window(root))
        return 75
    except Exception as error:
        state = _state(root, server)
        state.update(status='BLOCKED_INTEGRITY', reason=f'{type(error).__name__}: {error}')
        _write_state(root, server, state)
        raise


def ensure(root, server):
    verify_registration(root, server)
    folder = camp(root, server)
    state = _state(root, server)
    if state['status'] == 'BLOCKED_INTEGRITY' or any(r.get('status') == 'BLOCKED_INTEGRITY' for r in state['runs'].values()):
        return dict(status='BLOCKED_INTEGRITY', automatic_training_retry=False)
    if state['status'] in ('WINDOW_CLOSED', 'FINITE_QUEUE_FINISHED'):
        with locked(folder / '.runner.lock'):
            retry_uploads(root, server, state)
        return dict(status='CLOSED', pending_uploads=[r for r, row in state['runs'].items() if row.get('upload_pending')])
    try:
        with locked(folder / '.runner.lock'):
            pass
    except BlockingIOError:
        return dict(status='ALREADY_RUNNING')
    from qg40.deployment import spawn
    return spawn(root, server)
