"""Finite G20 scheduler: common wall-clock, local dependencies, atomic blocks."""
from __future__ import annotations

import datetime as dt
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from g20.common import (ROOT, CAMPAIGN_ID, append_event, atomic_json, camp,
                        immutable_json, locked, object_sha, read, read_json,
                        sha256, source_identity, utcnow)
from g20.plan import (CASES, CORE_CASES, SERVERS, SHEET_TABS, SensorSpec, REFERENCES, AXES,
                      blocks_for, build_config, case_for, cases_for, registry_document,
                      registry_sha256, teacher_for, verify_sources as validate_definition,
                      case_from_config, resolve_confirmation, resolve_transfer)
from g20.policy import (CampaignWindow, admission, paired_screen, choose_screen, confirm_gain,
                        lock_transfer as policy_lock_transfer)

HOLD = 'G20_FUTURE_ADMISSIONS_HOLD_v1'
DONE = {'OFFICIAL_EVAL_COMPLETE', 'UPLOAD_VERIFIED', 'ARCHIVED'}


def shared_window(root, supplied=None):
    canonical = Path(root) / 'work_dir/_g20/campaign_window.json'
    packaged = Path(root) / 'g20/campaign_window.json'
    paths = [path for path in (canonical, packaged) if path.is_file()]
    if supplied:
        paths.insert(0, Path(supplied))
    if not paths:
        raise ValueError('Shared campaign window missing; set one common t0, never local-server now')
    window = CampaignWindow.from_dict(read_json(paths[0]))
    if any(CampaignWindow.from_dict(read_json(path)) != window for path in paths[1:]):
        raise ValueError('Shared G20 window differs from existing runtime/deployment; refusing clock reset')
    return window


def write_window(root, t0_utc):
    window = CampaignWindow(t0_utc)
    path = Path(root) / 'work_dir/_g20/campaign_window.json'
    if (path.is_file() or (Path(root) / 'g20/campaign_window.json').is_file()) and shared_window(root) != window:
        raise ValueError('Shared G20 window already fixed; refusing clock reset')
    immutable_json(path, window.to_dict())
    return dict(path=str(path), **window.to_dict(), training_started=False)


def build_artifacts(root=ROOT):
    """Explicit deterministic generation; never register or activate a campaign."""
    root = Path(root)
    validation = validate_definition()
    target = root / 'config/g20'
    target.mkdir(parents=True, exist_ok=True)
    immutable_json(target / 'G20_Registry.json', registry_document())
    for case in CORE_CASES:
        path = target / (case.run_id + '.yaml')
        expected = build_config(case)
        if path.exists() and yaml.safe_load(path.read_text()) != expected:
            raise ValueError('Existing generated case differs: ' + case.run_id)
        if not path.exists():
            path.write_text(yaml.safe_dump(expected, sort_keys=False))
    return dict(validation=validation, configs=len(CORE_CASES), conditional_slots=24, activated=False)


def branches(root, server):
    folder = camp(root, server)
    decision = read(folder / 'screen_selection.json')
    transfer = read(folder / 'transfer_lock.json')
    def shas(ids):
        from g20.references import reference_path, validate_reference
        values = {}
        for ref in ids:
            manifest, _, _, _ = validate_reference(ref, server, root)
            values[ref] = object_sha(manifest)
        return values
    confirm = ()
    if decision.get('selected_candidate'):
        candidate = decision['selected_candidate']
        refs = ({'R0', candidate} if server == 's1' else {'R2', candidate}
                if server == 's2' else {'R0'})
        actual_refs = shas(refs)
        if decision.get('reference_sha256') != actual_refs:
            raise ValueError('Confirmation reference bundle changed after the screen lock')
        confirm = resolve_confirmation(server, decision, actual_refs)
    if transfer:
        actual_refs = shas({transfer['reference_id']})
        if actual_refs[transfer['reference_id']] != transfer.get('reference_sha256'):
            raise ValueError('Transfer reference bundle changed after its lock')
        extra = resolve_transfer(transfer, actual_refs)
    else:
        extra = ()
    return dict(confirmation_cases=confirm, transfer_cases=extra,
                transfer_first=transfer.get('before_confirmation', False))


def active_cases(root, server):
    return tuple(case for block in blocks_for(server, **branches(root, server)) for case in block.cases)


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
        if saved.get('status') == 'DONE' or saved.get('status', '').startswith('NOT_ADMITTED'):
            continue
        order = saved.get('run_ids', list(block.run_ids))
        if saved and order != list(block.run_ids):
            raise ValueError('Admitted G20 block membership/order changed')
        pending = [run for run in order if state['runs'].get(run, {}).get('status') not in DONE]
        if pending:
            queue.append(dict(block_id=block.block_id, run_ids=order, pending_run_ids=pending,
                              admitted=saved.get('status') == 'ADMITTED'))
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
                local_core_count=len(rows), conditional_slots=24,
                branch=branches(root, server), next_two_run_ids=[r for b in queue for r in b['pending_run_ids']][:2],
                queue=queue, teacher_reference_ids=sorted({c.reference_id for c in rows}), sheet_tab=SHEET_TABS[server],
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
        raise ValueError('Production G20 start requires CUDA; CPU is for isolated tests')
    if Path('/.dockerenv').exists() and not foreground:
        raise ValueError('Use a dedicated persistent container with start --foreground, not a retiring WV3 container')
    window = shared_window(root, window_path)
    if dt.datetime.now(dt.timezone.utc) < window.t0_utc:
        raise ValueError('Shared preparation start is in the future')
    if window.remaining_hours(utcnow()) <= 0:
        raise ValueError('Common 20h window already ended; clock cannot be restarted')
    folder = camp(root, server)
    binding = Path(spec_path) if spec_path else root / 'config/g20' / (cases_for(server)[0].sensor + '_sensor.json')
    if spec_path or binding.is_file():
        spec = SensorSpec.from_dict(read_json(binding))
    else:
        from qg40.bootstrap import default_sensor_spec
        spec = default_sensor_spec(root, cases_for(server)[0].sensor)
    if spec.sensor != cases_for(server)[0].sensor or not spec.is_bound:
        raise ValueError('Actual local sensor paths/band/source must be bound before registration')
    with locked(folder / '.registration.lock'):
        shared = root / 'work_dir/_g20/campaign_window.json'
        immutable_json(shared, window.to_dict())
        original_hold = read(root / 'work_dir/_eval_phase/hold.json')
        known_holds = {None, 'FH20R1_FUTURE_ADMISSIONS_HOLD_v1', 'FH20R1_R2_BOUNDARY_HOLD_v1', 'QG40_FUTURE_ADMISSIONS_HOLD_v1', HOLD}
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
                    reason='G20 transition: finish admitted legacy block; stop future legacy admissions')
        immutable_json(folder / 'requested_hold.json', hold)
        from g20.deployment import install_recovery, spawn
        try:
            recovery = install_recovery(root, server)
        except Exception as error:
            recovery = dict(status='WARNING', error=str(error), automatic_recovery=False)
        atomic_json(folder / 'recovery.json', recovery)
    if foreground:
        return run(root, server)
    return dict(status='G20_REQUESTED', recovery=recovery, launch=spawn(root, server), window=window.to_dict())


def verify_registration(root, server):
    registration = read_json(camp(root, server) / 'registration.json')
    identity = registration['identity']
    if (identity['campaign_id'] != CAMPAIGN_ID or identity['server_id'] != server
            or identity['registry_sha256'] != registry_sha256()
            or identity['source_identity'] != source_identity(root)
            or identity['window'] != shared_window(root).to_dict()
            or identity['sensor_spec_sha256'] != object_sha(read_json(camp(root, server) / 'sensor_spec.json'))):
        raise ValueError('G20 registration/source/data/window changed')
    return registration


def authorize_train(root, server, config_path):
    """The executable training route cannot bypass registration or atomic admission."""
    verify_registration(root, server)
    if read(Path(root) / 'work_dir/_eval_phase/hold.json') != read(camp(root, server) / 'requested_hold.json'):
        raise ValueError('G20 has not completed the safe legacy admission handoff')
    cfg = yaml.safe_load(Path(config_path).read_text())
    from g20.training import validate_config
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
        return subprocess.run([sys.executable, str(Path(root) / 'tools/g20_runner.py')] + args,
                              cwd=root, stdout=stream, stderr=subprocess.STDOUT).returncode


def _write_state(root, server, state):
    atomic_json(camp(root, server) / 'status.json', state)


def _resources_available(root, server, cases, state, block_id):
    from g20.resources import assess_block
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
    from g20.transition import legacy_busy
    while window.remaining_hours(utcnow()) > 0:
        busy = [r for r in inventory() if not r.get('waiting_controller')]
        gpu = gpu_processes()
        legacy = legacy_busy(root, server)
        if not busy and gpu == [] and not legacy:
            return True
        state.update(status='WAIT_LOCAL_RESOURCE', active_processes=busy, gpu_processes=gpu,
                     legacy_full_block_busy=legacy)
        _write_state(root, server, state)
        time.sleep(15)
    return False


def _resolve_config(root, case):
    from g20.references import reference_path, validate_reference
    folder, wd = camp(root, case.server_id), Path(root) / 'work_dir' / case.run_id
    cfg = build_config(case)
    data_path = folder / 'dataset_manifest.json'
    data = read_json(data_path)
    cfg['g20'].update(dataset_manifest=str(data_path), sensor_spec_path=str(folder / 'sensor_spec.json'),
                        sensor_spec=read_json(folder / 'sensor_spec.json'))
    for split, key in (('train', 'train_feeder_args'), ('val', 'val_feeder_args'),
                       ('rr', 'test_reduced_feeder_args'), ('fr', 'test_full_feeder_args')):
        cfg[key]['dataroot'] = data['splits'][split]['dataroot']
    if case.role == 'S':
        manifest, _, _, paths = validate_reference(case.reference_id, case.server_id, root, dataset_manifest=data)
        cfg['g20'].update(reference_manifest=str(reference_path(case.reference_id, case.server_id, root)),
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
    from g20.references import reference_path, validate_reference, import_reference, inventory_references, adopt_reference
    path = reference_path(reference_id, server, root)
    if path.exists():
        validate_reference(reference_id, server, root)
        if reference_id == 'R0':
            from g20.parity import verify_r0_parity
            receipt = verify_r0_parity(root, server, 'cuda', window.deadline_utc.isoformat())
            if receipt.get('status') != 'PASS':
                state.update(status='WAIT_REFERENCE_PARITY', parity=receipt)
                _write_state(root, server, state)
                return False
        return True
    if reference_id == 'R0':
        from g20.plan import R0_PINS
        candidates = [row for row in inventory_references(root) if row.get('teacher_checkpoint_sha256') == R0_PINS['teacher_sha256']]
        if candidates:
            adopt_reference(candidates[0]['path'], 'R0', server, root, 'cuda', deadline=window.deadline_utc.isoformat())
            return ensure_reference(root, server, reference_id, state, window)
    owner = REFERENCES[reference_id].server_id
    archive = camp(root, server) / 'incoming' / (reference_id + '.tar.gz')
    if archive.is_file():
        code = _command(root, ['import', '--reference', reference_id, '--archive', str(archive),
                              '--server', server], camp(root, server) / 'reference_import.log')
        if code:
            raise ValueError('Local reference import failed; inspect reference_import.log')
        return ensure_reference(root, server, reference_id, state, window)
    state.update(status='WAIT_REFERENCE', waiting_reference=reference_id, reference_owner=owner,
                 expected_transfer_path=str(archive), deadline_utc=window.deadline_utc.isoformat())
    _write_state(root, server, state)
    return False


def external_references_ready(root, server, cases, state, window):
    """Check R0 even when the first case is a fresh Teacher, before reservation."""
    produced = {case.reference_id for case in cases if case.role == 'T'}
    external = sorted({case.reference_id for case in cases if case.role == 'S'} - produced)
    for reference_id in external:
        if not ensure_reference(root, server, reference_id, state, window):
            return False
    return True


def _branch_decisions(root, server, state, window):
    from g20.postrun import screen_result
    folder = camp(root, server)
    _transfer_summary(root, server, state)
    if (folder / 'screen_selection.json').exists():
        decision = read_json(folder / 'screen_selection.json')
        winner = decision.get('selected_candidate')
        if winner:
            confirm = branches(root, server)['confirmation_cases']
            if confirm and all(state['runs'].get(c.run_id, {}).get('status') in DONE for c in confirm):
                pairs = _pairs(root, server, winner, active_cases(root, server), (93001, 93002, 93011, 93012))
                immutable_json(folder / 'confirmation_result.json', confirm_gain(server, winner, pairs))
        return
    receipts, skipped = [], {}
    for candidate in AXES[server]:
        pairs = _pairs(root, server, candidate, cases_for(server), (93001, 93002), state=state)
        if pairs is None:
            block = state['blocks'].get('S2_C030_REFERENCE', {})
            if server == 's2' and candidate == 'R3' and block.get('status', '').startswith('NOT_ADMITTED'):
                skipped[candidate] = object_sha(block)
                continue
            return
        measured = paired_screen(server, candidate, pairs)
        immutable_json(folder / 'screens' / (candidate + '.json'), measured)
        receipts.append(measured)
    decision = choose_screen(server, receipts, utcnow(), not_admitted=skipped)
    if decision.get('selected_candidate'):
        from g20.references import validate_reference
        candidate = decision['selected_candidate']
        refs = {'R0', candidate} if server == 's1' else {'R2', candidate} if server == 's2' else {'R0'}
        bindings = {ref: object_sha(validate_reference(ref, server, root)[0]) for ref in refs}
        decision['reference_sha256'] = bindings
    immutable_json(folder / 'screen_selection.json', decision)


def _pairs(root, server, candidate, cases, seeds, state=None):
    from g20.postrun import screen_result
    base_ref = 'R2' if server == 's2' else 'R0'
    alt_ref, alt_profile = (candidate, 'BASE') if server in ('s1', 's2') else ('R0', candidate)
    pairs = []
    for seed in seeds:
        def find(ref, profile):
            return next((c for c in cases if c.role == 'S' and c.seed == seed and
                         c.reference_id == ref and c.profile == profile), None)
        base, alt = find(base_ref, 'BASE'), find(alt_ref, alt_profile)
        if not base or not alt or (state is not None and
                any(state['runs'].get(c.run_id, {}).get('status') not in DONE for c in (base, alt))):
            return None
        pairs.append(dict(seed=seed, base=screen_result(base.run_id, root),
                          candidate=screen_result(alt.run_id, root)))
    return pairs


def _verify_evidence_files(root, evidence):
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError('Transfer requires content-backed evidence files')
    for name, expected in evidence.items():
        path = Path(name)
        path = path if path.is_absolute() else Path(root) / path
        if not path.is_file():
            # Different host prefixes never justify editing a locked receipt.
            # A copied content-addressed evidence file must have identical bytes.
            from g20.plan import valid_sha
            if not valid_sha(expected):
                raise ValueError('Invalid transfer evidence SHA256')
            path = Path(root) / 'work_dir/_g20/incoming/evidence' / expected
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('Missing/changed transfer evidence: ' + str(path))


def publish_transfer_selection(root, reference_screen_path, student_screen_path, reference_id,
                               *, before_confirmation=False):
    """Only s1 issues the one portable global transfer choice; no remote writes.

    Remote evidence must first be copied with its content hashes. The receipt
    is then copied to its owner; owners cannot issue a competing global choice.
    """
    verify_registration(root, 's1')
    window = shared_window(root)
    if dt.datetime.now(dt.timezone.utc) >= window.admission_cutoff_utc:
        raise TimeoutError('No new transfer selection after the 16h cutoff')
    reference_screen, student_screen = read_json(reference_screen_path), read_json(student_screen_path)
    for screen in (reference_screen, student_screen):
        _verify_evidence_files(root, screen.get('evidence_files'))
    reference_shas = {pair['candidate']['reference_sha256'] for pair in reference_screen['pairs']}
    if len(reference_shas) != 1:
        raise ValueError('Reference screen seeds used different reference bundles')
    proof = dict(reference_screen.get('evidence_files', {}), **student_screen.get('evidence_files', {}))
    receipt = policy_lock_transfer(reference_screen, student_screen, utcnow(), reference_id=reference_id,
        reference_sha256=next(iter(reference_shas)), local_controls_verified=True,
        pair_identity_verified=True, evidence_files=proof)
    receipt.update(issuer_server='s1', window=window.to_dict(), before_confirmation=bool(before_confirmation),
                   reference_screen=reference_screen, student_screen=student_screen,
                   coordination='ONE_S1_ISSUED_IMMUTABLE_RECEIPT; copy to original scalar owner')
    path = Path(root) / 'work_dir/_g20/transfer_selection.json'
    with locked(path.with_suffix('.lock')):
        old = read(path)
        if old:
            keys = ('reference_screen_sha256', 'student_screen_sha256', 'reference_id',
                    'server_id', 'profile', 'reference_sha256', 'before_confirmation', 'window')
            if any(old.get(key) != receipt.get(key) for key in keys):
                raise ValueError('One global transfer already selected; substitution forbidden')
            return dict(path=str(path), receipt=old, admitted=False)
        immutable_json(path, receipt)
    return dict(path=str(path), receipt=receipt, admitted=False)


def register_transfer(root, server, global_receipt_path, *, runner_lock_held=False):
    """Owner verifies local controls and reserves all four runs before activation."""
    verify_registration(root, server)
    receipt = read_json(global_receipt_path)
    window = shared_window(root)
    if (receipt.get('issuer_server') != 's1' or receipt.get('server_id') != server or
            receipt.get('window') != window.to_dict()):
        raise ValueError('Use the common s1-issued transfer receipt on its original scalar owner')
    local = read_json(camp(root, server) / 'screen_selection.json')
    if local.get('selected_candidate') != receipt.get('profile') or local.get('selected_status') != 'PROMISING_PAIRED':
        raise ValueError('Transfer differs from the locked local winning scalar')
    state = _state(root, server)
    pairs = _pairs(root, server, receipt['profile'], cases_for(server), (93001, 93002), state=state)
    measured = paired_screen(server, receipt['profile'], pairs or ())
    if measured != receipt.get('student_screen'):
        raise ValueError('Transferred screen differs from actual local R0/BASE/XSTAR controls')
    _verify_evidence_files(root, measured['evidence_files'])
    reference_screen = receipt['reference_screen']
    checked = policy_lock_transfer(reference_screen, measured, receipt['selected_at_utc'],
        reference_id=receipt['reference_id'], reference_sha256=receipt['reference_sha256'],
        local_controls_verified=True, pair_identity_verified=True, evidence_files=receipt['evidence_files'])
    if any(receipt.get(key) != value for key, value in checked.items()):
        raise ValueError('Global transfer receipt proof changed')
    incoming = camp(root, server) / 'incoming/transfer_selection.json'
    if not runner_lock_held:
        # Staging does not admit work or alter an already-running pair. The
        # owner checks the receipt at its next atomic boundary under its lock.
        with locked(camp(root, server) / '.transfer_stage.lock'):
            immutable_json(incoming, receipt)
        try:
            with locked(camp(root, server) / '.runner.lock'):
                return register_transfer(root, server, incoming, runner_lock_held=True)
        except BlockingIOError:
            return dict(status='TRANSFER_STAGED_FOR_BOUNDARY', admitted=False, path=str(incoming))
    with nullcontext():
        state = _state(root, server)
        preflight = read(camp(root, server) / 'preflight.json')
        if not preflight.get('complete') or preflight.get('source_identity') != source_identity(root):
            raise ValueError('Current local P0 required before transfer admission')
        confirmation = branches(root, server)['confirmation_cases']
        confirmation_done = bool(confirmation) and all(state['runs'].get(c.run_id, {}).get('status') in DONE for c in confirmation)
        if not receipt.get('before_confirmation') and not confirmation_done:
            return dict(status='WAIT_CONFIRMATION_PRIORITY', admitted=False)
        if receipt.get('before_confirmation') and any(state['blocks'].get(c.block_id, {}).get('status') == 'ADMITTED' for c in confirmation):
            raise ValueError('Transfer ordering must be locked before confirmation admission')
        if receipt.get('before_confirmation') and any(row.get('training_started') for run, row in state['runs'].items()
                                                     if '_CONFIRM_' in run):
            raise ValueError('Transfer cannot overtake a confirmation pair that already started')
        if not ensure_reference(root, server, receipt['reference_id'], state, window):
            raise ValueError('Transfer reference not locally ready; no fallback')
        from g20.references import validate_reference
        actual_sha = object_sha(validate_reference(receipt['reference_id'], server, root)[0])
        if actual_sha != receipt['reference_sha256']:
            raise ValueError('Prepared transfer reference differs from selected reference SHA')
        cases = resolve_transfer(receipt, {receipt['reference_id']: actual_sha})
        decision = admission(window, utcnow(), cases, p0_ready=True,
            evaluation_debt_hours=evaluation_debt(root, server, state), observations=state['observations'])
        if not decision['allowed']:
            return dict(status=decision['reason'], admitted=False, reservation=decision)
        if not _resources_available(root, server, cases, state, cases[0].block_id):
            return dict(status='WAIT_LOCAL_RESOURCE', admitted=False)
        global_path = Path(root) / 'work_dir/_g20/transfer_selection.json'
        with locked(global_path.with_suffix('.lock')):
            immutable_json(global_path, receipt)
            immutable_json(camp(root, server) / 'transfer_lock.json', receipt)
            state['blocks'][cases[0].block_id] = dict(status='ADMITTED', run_ids=[c.run_id for c in cases], reservation=decision)
            state['status'] = 'LOCAL_READY'
            _write_state(root, server, state)
    return dict(status='TRANSFER_ADMITTED', admitted=True, reservation=decision)


def _activate_staged_transfer(root, server, state):
    """Called only by the owner already holding .runner.lock at a boundary."""
    folder = camp(root, server)
    incoming = folder / 'incoming/transfer_selection.json'
    if not incoming.is_file() or (folder / 'transfer_lock.json').is_file():
        return
    previous = read(folder / 'transfer_activation.json')
    digest = sha256(incoming)
    if previous.get('receipt_file_sha256') == digest and previous.get('status') == 'TRANSFER_REJECTED':
        return
    try:
        outcome = register_transfer(root, server, incoming, runner_lock_held=True)
    except (ValueError, FileNotFoundError) as error:
        # A bad optional transfer receipt must not derail the valid local
        # coefficient screen/confirmation. It is never given fallback values.
        outcome = dict(status='TRANSFER_REJECTED', admitted=False, reason=str(error))
    atomic_json(folder / 'transfer_activation.json', dict(outcome, receipt_file_sha256=digest,
                                                        at_utc=utcnow()))
    if outcome.get('admitted'):
        state.clear()
        state.update(_state(root, server))


def _transfer_summary(root, server, state):
    receipt = read(camp(root, server) / 'transfer_lock.json')
    if not receipt:
        return
    cases = active_cases(root, server)
    transferred = [c for c in cases if c.tier == 'CONDITIONAL_TRANSFER']
    if not transferred or any(state['runs'].get(c.run_id, {}).get('status') not in DONE for c in transferred):
        return
    from g20.postrun import screen_result
    records = {}
    for case in cases:
        if (case.role == 'S' and case.seed in (93001, 93002) and case.profile in ('BASE', receipt['profile'])
                and case.reference_id in ('R0', receipt['reference_id'])):
            records[(case.seed, case.reference_id, case.profile)] = screen_result(case.run_id, root)
    contrasts = []
    for seed in (93001, 93002):
        rows = [records[(seed, ref, profile)] for ref, profile in (
            ('R0', 'BASE'), ('R0', receipt['profile']), (receipt['reference_id'], 'BASE'),
            (receipt['reference_id'], receipt['profile']))]
        if any(not all(row[key] == rows[0][key] for key in ('u_init_sha256', 'data_view_sha256', 'source_sha256'))
               for row in rows):
            raise ValueError('Transfer 2x2 lost common U/data/source identity')
        contrasts.append(dict(seed=seed, difference_in_differences={selection: {
            metric: (rows[3][selection][metric] - rows[2][selection][metric]) -
                    (rows[1][selection][metric] - rows[0][selection][metric])
            for metric in ('HQNR', 'ERGAS', 'D_lambda')} for selection in ('exact50k', 'rr_val_selected')}))
    immutable_json(camp(root, server) / 'transfer_result.json', dict(campaign_id=CAMPAIGN_ID,
        server_id=server, receipt_sha256=object_sha(receipt), contrasts=contrasts,
        exploratory=True, test_aware=True, independent_confirmation=False))


def _recover_exact50k(root, wd):
    path = Path(wd) / 'meta/config.resolved.yaml'
    if path.is_file():
        from g20.training import recover_exact50k_status
        return recover_exact50k_status(wd, yaml.safe_load(path.read_text()), root)
    return False


def run_case(root, server, case, state, window):
    wd = Path(root) / 'work_dir' / case.run_id
    row = state['runs'].setdefault(case.run_id, {})
    if row.get('train_wall_attempt_started_utc'):
        # After a controller crash the precise child-exit time is unknown.
        # Count that entire recorded interval conservatively, not as zero.
        previous_start = dt.datetime.fromisoformat(row.pop('train_wall_attempt_started_utc').replace('Z', '+00:00'))
        recovered = max(0., (dt.datetime.now(dt.timezone.utc) - previous_start).total_seconds())
        row['train_wall_seconds'] = row.get('train_wall_seconds', 0.) + recovered
        row['train_wall_recovered_interval_seconds'] = row.get('train_wall_recovered_interval_seconds', 0.) + recovered
    cfg_path = _resolve_config(root, case)
    _recover_exact50k(root, wd)
    trained = read(wd / 'meta/training_status.json').get('training_complete', False)
    if not trained:
        resume = (wd / 'last/training_state.pt').is_file()
        if (row.get('training_started') or (wd / 'meta/training_start_manifest.json').exists()) and not resume:
            raise ValueError('Interrupted run lacks exact fullstate; fresh restart prohibited')
        args = ['train', '--config', str(cfg_path), '--device', 'cuda', '--server', server] + (['--resume'] if resume else [])
        row.update(status='RUNNING', training_started=True, train_wall_attempt_started_utc=utcnow())
        state['status'] = 'RUNNING'
        _write_state(root, server, state)
        wall_started = time.monotonic()
        try:
            code = _command(root, args, wd / 'g20_train.log')
        finally:
            row['train_wall_seconds'] = row.get('train_wall_seconds', 0.) + time.monotonic() - wall_started
            row.pop('train_wall_attempt_started_utc', None)
            _write_state(root, server, state)
        if code:
            row['status'] = 'PARTIAL_TIME_LIMIT' if dt.datetime.now(dt.timezone.utc) >= window.train_finish_utc else 'PAUSED' if code == 75 else 'BLOCKED_INTEGRITY'
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
        from g20.references import reference_path
        if not reference_path(case.reference_id, server, root).is_file():
            cal_start = time.monotonic()
            code = _command(root, ['calibrate', '--run', case.run_id, '--server', server], wd / 'g20_calibration.log')
            row['calibration_seconds'] = row.get('calibration_seconds', 0.) + time.monotonic() - cal_start
            if code:
                return False
        # Export immediately after exact50K calibration, before optional report debt.
        if case.reference_id in {'R1', 'R2', 'R3', 'R4'}:
            code = _command(root, ['export', '--reference', case.reference_id, '--server', server], wd / 'g20_export.log')
            if code:
                return False
    post_start = time.monotonic()
    code = _command(root, ['postrun', '--run', case.run_id, '--server', server], wd / 'g20_postrun.log')
    row['postrun_seconds'] = row.get('postrun_seconds', 0.) + time.monotonic() - post_start
    if code or not read(wd / 'official/postrun_status.json').get('official_complete'):
        row['status'] = 'EVALUATION_PENDING'
        _write_state(root, server, state)
        return False
    row['status'] = 'OFFICIAL_EVAL_COMPLETE'
    code = _command(root, ['upload', '--run', case.run_id, '--server', server], wd / 'g20_upload.log')
    uploaded = (not code and read(wd / 'official/upload_receipt.json').get('readback_verified') is True
                and read(wd / 'official/postrun_status.json').get('sheet_uploaded') is True)
    row.update(upload_pending=not uploaded, status='UPLOAD_VERIFIED' if uploaded else 'OFFICIAL_EVAL_COMPLETE')
    if not row.get('timing_observation_recorded'):
        measured = read(wd / 'meta/training_status.json')
        components = dict(TRAIN=measured.get('training_seconds', 0.) + measured.get('io_seconds', 0.)
                               + measured.get('diagnostic_seconds', 0.),
                          EVAL=measured.get('evaluation_seconds', 0.) + row.get('postrun_seconds', 0.),
                          CALIBRATION=row.get('calibration_seconds', 0.))
        if row.get('train_wall_seconds', 0.) > 0:
            # Includes init/source/reference validation, DataLoader waits,
            # scheduled evaluations, checkpoint I/O and diagnostics. CAL is
            # reserved separately; do not add it to this combined observation.
            components['TRAIN_EVAL'] = row['train_wall_seconds'] + row.get('postrun_seconds', 0.)
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
    active = {c.case_id: c for c in active_cases(root, server)}
    slots = {c.case_id: c for c in cases_for(server, include_conditional=True)}
    slots.update(active)
    for slot in slots.values():
        case = active.get(slot.case_id, slot)
        row = state['runs'].get(case.run_id, {})
        block = state['blocks'].get(case.block_id, {})
        reason = block.get('status') if block.get('status', '').startswith('NOT_ADMITTED') else (
            'NOT_ADMITTED_CONDITION' if not case.is_resolved else 'NOT_ADMITTED_TIME_LIMIT')
        if row.get('training_started') and row.get('status') not in DONE:
            reason = 'EVALUATION_PENDING' if read(Path(root) / 'work_dir' / case.run_id / 'meta/training_status.json').get('training_complete') else 'PARTIAL_TIME_LIMIT'
        records[case.case_id] = dict(row, run_id=case.run_id, status=row.get('status') if row.get('status') in DONE else reason)
    debt = evaluation_debt(root, server, state)
    state['status'] = ('WINDOW_CLOSED' if window.remaining_hours(utcnow()) <= 0 else
                       'CLOSEOUT_EVALUATION_PENDING' if debt > 0 else 'FINITE_QUEUE_FINISHED')
    _write_state(root, server, state)
    atomic_json(camp(root, server) / 'closeout.json', dict(campaign_id=CAMPAIGN_ID, server_id=server,
        at_utc=utcnow(), window=window.to_dict(), runs=records,
        fixed_base_separate_from_tuned=True, teacher_lineages_never_pooled=True,
        debt_hours=debt))


def repay_evaluation_debt(root, server, state, window):
    """18–20h closeout evaluates saved outputs only; never starts/resumes training."""
    if window.remaining_hours(utcnow()) <= 0:
        return
    by_run = {case.run_id: case for case in active_cases(root, server)}
    for run, row in list(state['runs'].items()):
        if row.get('status') in DONE or not row.get('training_started'):
            continue
        if window.remaining_hours(utcnow()) <= 0:
            break
        wd = Path(root) / 'work_dir' / run
        _recover_exact50k(root, wd)
        training = read(wd / 'meta/training_status.json')
        if training.get('training_complete') is True and training.get('actual_updates') == 50000:
            # run_case takes the already-trained path, including missing CAL.
            run_case(root, server, by_run[run], state, window)
        else:
            from g20.postrun import process_partial
            process_partial(run, device='cuda', deadline=window.deadline_utc.isoformat(), root=root)
            row['status'] = 'PARTIAL_TIME_LIMIT'
            row['partial_evaluation'] = read(wd / 'official/postrun_status.json')
            _write_state(root, server, state)
    retry_uploads(root, server, state)
    _branch_decisions(root, server, state, window)


def retry_uploads(root, server, state):
    """Completed result delivery is independent of GPU/clock/learning restart."""
    for run, row in state['runs'].items():
        if row.get('status') in DONE and row.get('upload_pending', True):
            wd = Path(root) / 'work_dir' / run
            code = _command(root, ['upload', '--run', run, '--server', server], wd / 'g20_upload.log')
            verified = (not code and read(wd / 'official/postrun_status.json').get('sheet_uploaded') is True
                        and read(wd / 'official/upload_receipt.json').get('readback_verified') is True)
            row['upload_pending'] = not verified
            if verified:
                row['status'] = 'UPLOAD_VERIFIED'
    _write_state(root, server, state)


def reconcile(root, server, state):
    """Reuse only locally verified exact results; missing Sheet rows never mean fresh."""
    from g20.upload import row_values
    for case in active_cases(root, server):
        wd = Path(root) / 'work_dir' / case.run_id
        _recover_exact50k(root, wd)
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
        if state['status'] == 'CLOSEOUT_EVALUATION_PENDING' or dt.datetime.now(dt.timezone.utc) >= window.train_finish_utc:
            repay_evaluation_debt(root, server, state, window)
            _closeout(root, server, state, window)
            return 0
        from g20.transition import wait_and_claim
        if not wait_and_claim(root, server, window):
            _closeout(root, server, state, window)
            return 75
        if not _wait_local_idle(root, server, state, window):
            _closeout(root, server, state, window)
            return 0
        if dt.datetime.now(dt.timezone.utc) >= window.train_finish_utc:
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
            _activate_staged_transfer(root, server, state)
            queue = effective_queue(root, server, state)
            if not queue:
                break
            item = queue[0]
            cases = list(next(b.cases for b in blocks_for(server, **branches(root, server)) if b.block_id == item['block_id']))
            first = next(c for c in cases if c.run_id in item['pending_run_ids'])
            if not item['admitted'] and not external_references_ready(root, server, cases, state, window):
                if dt.datetime.now(dt.timezone.utc) >= window.admission_cutoff_utc:
                    state['blocks'][item['block_id']] = dict(status='NOT_ADMITTED_CUTOFF', run_ids=item['run_ids'])
                    _write_state(root, server, state)
                    continue
                time.sleep(15)
                continue
            if not item['admitted']:
                decision = admission(window, utcnow(), cases, p0_ready=True,
                    evaluation_debt_hours=evaluation_debt(root, server, state), observations=state['observations'],
                    completed_run_ids=[r for r, row in state['runs'].items() if row.get('status') in DONE])
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
                if dt.datetime.now(dt.timezone.utc) >= window.train_finish_utc:
                    break
                started = (Path(root) / 'work_dir' / case.run_id / 'meta/training_start_manifest.json').exists()
                if not _wait_local_idle(root, server, state, window):
                    break
                if case.role == 'S' and not ensure_reference(root, server, case.reference_id, state, window):
                    return 75
                if not run_case(root, server, case, state, window):
                    if window.remaining_hours(utcnow()) <= 0 or dt.datetime.now(dt.timezone.utc) >= window.train_finish_utc:
                        break
                    return 2 if state['status'] == 'BLOCKED_INTEGRITY' else 75
            else:
                state['blocks'][item['block_id']]['status'] = 'DONE'
                _write_state(root, server, state)
                continue
            break
        repay_evaluation_debt(root, server, state, window)
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
    from g20.deployment import spawn
    return spawn(root, server)
