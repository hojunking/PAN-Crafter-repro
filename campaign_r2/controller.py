"""Priority-only FH20R1 controller using unchanged local numerical executables.

Explicit activation requests a cooperative *case-boundary* handoff, never a kill.
After acquiring the original runner lock we re-read all state: even a block
admitted by the old controller during the handshake must finish in its old order.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
from pathlib import Path
import time
import uuid

from fh12.common import ROOT, atomic_json, object_sha, read_json, sha256, utcnow
from fh20r1 import ledger
from fh20r1.common import source_identity
from fh20r1.plan import active_cases, case_for
from tools import fh20r1_runner as legacy
from campaign_r2.plan import (CAMPAIGN_ID, PRIORITY_REVISION, SOURCE_PLAN,
                              canonical_overlay, planned_queue, require_branch_record,
                              validate_definition)

TRANSFER_HOLD = 'FH20R1_R2_BOUNDARY_HOLD_v1'
SCHEMA = 'FH20R1_R2_CONTROLLER_v1'


def directory(root, server):
    if server not in {'s1', 's2', 's3', 's4', 's5'}:
        raise ValueError('Expected local server s1..s5')
    return ledger.camp(root, server) / 'priority_r2'


def controller_identity(root):
    root = Path(root)
    paths = sorted((root / 'campaign_r2').glob('*.py'))
    paths += sorted((root / 'tools').glob('r2_*.py'))
    paths += sorted((root / 'tools').glob('r2_*.sh'))
    files = {str(p.relative_to(root)): sha256(p) for p in paths if not p.name.startswith('test_')}
    files[SOURCE_PLAN] = sha256(root / SOURCE_PLAN)
    return dict(files=files, content_sha256=object_sha(files), priority_revision=PRIORITY_REVISION)


def operational_identity(root, server):
    """Re-use each server's original (including s2-specific) source contract."""
    root = Path(root)
    budget = legacy.load_budget(root, server)  # Never prepare() or reset the clock.
    if budget.get('device') != 'cuda':
        raise ValueError('R2 can adjust only an existing CUDA-authorized campaign')
    base = ledger.camp(root, server)
    branch = ledger.read(base / 'branch_record.json')
    require_branch_record(server, branch)
    preflight = ledger.read(base / 'preflight_report.json')
    if not preflight.get('preflight_pass'):
        raise ValueError('Existing method preflight is not complete')
    current = source_identity(root)
    if current != preflight.get('source_identity'):
        raise ValueError('Original execution source/runtime changed; no identity rewriting permitted')
    bridge_path = base / 'imported_refs' / ('F' + server[1:]) / 'bridge_manifest.json'
    bridge = ledger.read(bridge_path)
    if not bridge.get('complete') or bridge.get('consumer_source_identity') != current:
        raise ValueError('Existing reference bridge no longer belongs to this numerical runtime')
    definition = validate_definition(root)
    return dict(server_id=server, campaign_id=CAMPAIGN_ID, priority_revision=PRIORITY_REVISION,
                original_source_identity=current, controller_identity=controller_identity(root),
                campaign_budget_sha256=sha256(base / 'campaign_budget.json'),
                branch_record_sha256=sha256(base / 'branch_record.json'),
                preflight_sha256=sha256(base / 'preflight_report.json'),
                reference_bridge_sha256=sha256(bridge_path),
                overlay_sha256=definition['overlay_sha256'])


def ledger_prefix(root, server):
    path = ledger.camp(root, server) / 'active_intervals.jsonl'
    raw = path.read_bytes() if path.exists() else b''
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def verify_registration(root, server):
    registration = ledger.read(directory(root, server) / 'registration.json')
    if (registration.get('schema') != SCHEMA or not registration.get('explicit_start_authorized')
            or registration.get('identity') != operational_identity(root, server)):
        raise ValueError('Missing/stale R2 registration; retain original artifacts and investigate')
    path = ledger.camp(root, server) / 'active_intervals.jsonl'
    original = registration['original_ledger_prefix']
    raw = path.read_bytes() if path.exists() else b''
    if len(raw) < original['bytes'] or hashlib.sha256(raw[:original['bytes']]).hexdigest() != original['sha256']:
        raise ValueError('Original campaign ledger was shortened or rewritten')
    return registration


def hold_path(root):
    return Path(root) / 'work_dir/_eval_phase/hold.json'


def transfer_hold(registration):
    return dict(protocol_id=TRANSFER_HOLD, campaign_id=CAMPAIGN_ID,
                server=registration['identity']['server_id'], priority_revision=PRIORITY_REVISION,
                owner_nonce=registration['owner_nonce'], auto_release=False,
                reason='Finish current training/postrun/upload; R2 takes next admissions without a kill')


def arm_transfer(root, registration):
    current = ledger.read(hold_path(root))
    wanted = transfer_hold(registration)
    if current == wanted:
        return
    if current != registration['original_hold']:
        raise ValueError('Admission hold belongs to another owner; refusing to replace it')
    atomic_json(hold_path(root), wanted)


def restore_owned_hold(root, registration, *, already_applied=False):
    """Call only while owning the original .runner.lock."""
    current = ledger.read(hold_path(root))
    if current == transfer_hold(registration):
        atomic_json(hold_path(root), registration['original_hold'])
    elif not (already_applied and current == registration['original_hold']):
        raise ValueError('R2 transfer nonce/hold changed; original hold will not be restored')


def collect_audit(root, server, *, write_reports=False):
    from campaign_r2.audit import collect
    return collect(root, server, write_reports=write_reports)


def check(root, server, *, write_reports=False):
    identity = operational_identity(root, server)
    audit = collect_audit(root, server, write_reports=write_reports)
    state = ledger.read(ledger.camp(root, server) / 'status.json')
    branch = ledger.read(ledger.camp(root, server) / 'branch_record.json')
    queue = planned_queue(server, state, branch)
    return dict(schema=SCHEMA, priority_revision=PRIORITY_REVISION, identity=identity,
                audit=audit, queue=queue, next_two_run_ids=[run for row in queue
                      for run in row['pending_run_ids']][:2], applied=False,
                current_campaign_time=ledger.report(root, server), mutates_training=False)


def start(root, server, *, dry_run=False):
    root = Path(root)
    pointer = root / 'work_dir/_fh20r1/r2_local_server.txt'
    if pointer.is_symlink() or (pointer.exists() and not pointer.is_file()):
        raise ValueError('Invalid R2 pointer is preserved; no takeover')
    if pointer.is_file() and pointer.read_text().strip() != server:
        raise ValueError('Another R2 server owns this repository')
    preview = check(root, server, write_reports=False)
    if dry_run:
        return preview
    if not preview['audit'].get('admission_integrity_ok'):
        raise ValueError('R2 admission integrity checks failed; existing training left untouched')
    folder = directory(root, server)
    with legacy.locked(folder / '.activation.lock'):
        registration_path = folder / 'registration.json'
        if registration_path.exists():
            registration = verify_registration(root, server)
        else:
            hold = ledger.read(hold_path(root))
            if (hold.get('protocol_id') != legacy.HOLD or hold.get('server') != server
                    or hold.get('campaign_id') != CAMPAIGN_ID):
                raise ValueError('Existing FH20R1 admission hold required; no foreign takeover')
            registration = dict(schema=SCHEMA, identity=preview['identity'], owner_nonce=uuid.uuid4().hex,
                                explicit_start_authorized=True, registered_at_utc=utcnow(),
                                original_hold=hold, original_ledger_prefix=ledger_prefix(root, server),
                                original_time=ledger.report(root, server), original_state=ledger.read(
                                    ledger.camp(root, server) / 'status.json'),
                                original_readiness=ledger.read(ledger.camp(root, server) / 'readiness_report.json'))
            atomic_json(registration_path, registration)
            atomic_json(folder / 'priority_overlay.json', canonical_overlay(root))
        from campaign_r2.deployment import install_watchdog, install_watchdog_cron, spawn
        watchdog = install_watchdog(root)  # Must precede pointer + restoration of normal hold.
        try:
            recovery = install_watchdog_cron(root)
        except Exception as error:
            # Current handoff is still safe without cron. Do not claim unattended
            # recovery when the host cannot read/install its recovery entry.
            recovery = dict(status='WARNING', recover_on_reboot=False, error=str(error))
        watchdog['cron'] = recovery
        atomic_json(folder / 'watchdog_installation.json', watchdog)
        applied = ledger.read(folder / 'applied_readback.json').get('applied') is True
        if not applied:
            arm_transfer(root, registration)
        if not pointer.exists():
            tmp = pointer.with_suffix('.tmp')
            tmp.write_text(server + '\n')
            os.replace(tmp, pointer)
        result = spawn(root, server)
        return dict(status='R2_REQUESTED', priority_revision=PRIORITY_REVISION,
                    launch=result, recovery=recovery, original_campaign_clock_preserved=True,
                    actual_queue_handoff='after running case finishes; admitted block continues first')


@contextlib.contextmanager
def take_runner_lock(root, server, *, wait=True, sleep=time.sleep):
    """Acquire the *same* lock held by the old process; never signal that process."""
    folder = directory(root, server)
    with (ledger.camp(root, server) / '.runner.lock').open('a') as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                atomic_json(folder / 'controller_status.json', dict(status='WAITING_CASE_BOUNDARY',
                            checked_at_utc=utcnow(), original_training_untouched=True))
                if not wait:
                    raise
                sleep(10)
        yield


def reconcile_completed(root, server, state, branch):
    """Never start fresh from a missing Sheet row or trust DONE without evidence."""
    from fh20r1.upload import row_values
    condition = require_branch_record(server, branch)
    for case in active_cases(server, condition):
        wd = Path(root) / 'work_dir' / case.run_id
        row = state['runs'].get(case.run_id, {})
        reuse = (wd / 'meta/reuse_reference.json').exists()
        complete = legacy.official_done(root, case.run_id)
        if row.get('status') in legacy.DONE or complete or reuse:
            row_values(case.run_id, root)  # Original local strict validator; no Sheet access.
            row = state['runs'].setdefault(case.run_id, {})
            row.update(status='DONE_REUSED' if reuse else 'DONE', training_complete=not reuse)
            row.setdefault('upload_pending', True)
        elif wd.exists() and (row.get('training_started') or
                (wd / 'meta/training_start_manifest.json').exists() or (wd / 'candidates').exists()):
            if not (wd / 'last/training_state.pt').exists():
                raise ValueError(f'Existing interrupted run has no exact fullstate; no fresh overwrite: {case.run_id}')
            if row.get('status') == 'FAILED':
                raise ValueError(f'Failed run requires explicit diagnosis, not automatic fresh restart: {case.run_id}')
            state['runs'].setdefault(case.run_id, {}).setdefault('training_started', True)
    # Preserve original membership/order when local files are ahead of runner status.
    from campaign_r2.plan import priority_blocks
    for block in priority_blocks(server):
        order = list(block.order_for(condition))
        if any(run in state['runs'] for run in order):
            saved = state['blocks'].setdefault(block.block_id, dict(status='INCOMPLETE', tier=block.tier,
                                                                    run_ids=order))
            if all(state['runs'].get(run, {}).get('status') in legacy.DONE for run in order):
                saved.update(status='DONE', tier=block.tier, run_ids=order)
        if state['blocks'].get(block.block_id, {}).get('status') == 'DONE' and any(
                state['runs'].get(run, {}).get('status') not in legacy.DONE for run in order):
            raise ValueError(f'Block marked DONE without every member verified: {block.block_id}')
    return state


def retry_completed_uploads(root, server, state, budget):
    """Delivery-only retry for completed blocks omitted from the pending queue."""
    if not budget.get('upload_enabled', True):
        return
    for run, row in state['runs'].items():
        if row.get('status') in legacy.DONE and row.get('upload_pending', True):
            legacy.upload_case(root, case_for(run), row, True)
    atomic_json(ledger.camp(root, server) / 'status.json', state)


def publish_readback(root, server, state, branch, registration, *, readiness=None):
    queue = planned_queue(server, state, branch)
    remaining = queue
    if state.get('status') in legacy.CLOSED:
        queue = []  # Preserved reserve definitions are not scheduled after completion.
        if readiness is None:
            readiness = ledger.read(ledger.camp(root, server) / 'readiness_report.json') or None
        if readiness is not None:
            readiness = dict(readiness, launch_ready=False, status=state['status'])
    active = [run for run, row in state['runs'].items()
              if row.get('status') in {'TRAINING', 'PAUSED', 'EVALUATION_PENDING'}]
    steps = {run: ledger.read(Path(root) / 'work_dir' / run / 'meta/training_status.json').get('actual_updates')
             for run in active}
    completed = sorted(run for run, row in state['runs'].items() if row.get('status') in legacy.DONE)
    uploaded = []
    for run in completed:
        receipt = ledger.read(Path(root) / 'work_dir' / run / 'official/upload_receipt.json')
        if (receipt.get('readback_verified') and receipt.get('run_id') == run
                and receipt.get('campaign_id') == CAMPAIGN_ID
                and not state['runs'][run].get('upload_pending', True)):
            uploaded.append(run)
    bridge = ledger.read(ledger.camp(root, server) / 'imported_refs' / ('F' + server[1:]) / 'bridge_manifest.json')
    verification = ledger.read(directory(root, server) / 'verification_status.json')
    value = dict(schema=SCHEMA, server=server, campaign_id=CAMPAIGN_ID,
                 priority_revision=PRIORITY_REVISION, applied=True, checked_at_utc=utcnow(),
                 method_numeric_changed=False, case_numeric_changed=False,
                 old_campaign_ledger_preserved=True, global_performance_lock=False,
                 branch_from_actual_record=require_branch_record(server, branch),
                 current_run_ids=active, current_admitted_blocks=[r['block_id'] for r in queue if r['admitted']],
                 current_update_by_run=steps,
                 current_run_id=active[0] if len(active) == 1 else None,
                 current_update=steps.get(active[0]) if len(active) == 1 else None,
                 current_admitted_block=next((r['block_id'] for r in queue if r['admitted']), None),
                 teacher_reference_sha256=bridge.get('teacher_checkpoint_sha256'),
                 local_completed_runs=completed, sheet_uploaded_runs=uploaded,
                 sheet_upload_verification='existing local readback receipts; live Sheet not re-read',
                 completed_but_upload_pending=[run for run in completed if run not in uploaded],
                 verification_receipts=dict(
                     path=str((directory(root, server) / 'verification_status.json').relative_to(root)),
                     created_at_utc=verification.get('created_at_utc'),
                     statuses={key: item.get('status') for key, item in verification.get('checks', {}).items()}),
                 next_two_run_ids=[run for row in queue for run in row['pending_run_ids']][:2], queue=queue,
                 preserved_unscheduled_blocks=remaining if state.get('status') in legacy.CLOSED else [],
                 source_runtime_for_active_run_preserved=True,
                 original_source_identity=registration['identity']['original_source_identity'],
                 controller_identity=registration['identity']['controller_identity'],
                 effective_hours_from_campaign_ledger=ledger.report(root, server)['effective_hours'])
    if readiness is not None:
        # Do not leave the legacy path advertising a stale original-order queue.
        readiness = dict(readiness, priority_revision=PRIORITY_REVISION, queue=queue,
                         next_atomic_block=queue[0] if queue else None,
                         next_run_ids=queue[0]['pending_run_ids'] if queue else [],
                         queue_sha256=object_sha(queue), r2_controller_identity=value['controller_identity'])
        atomic_json(ledger.camp(root, server) / 'readiness_report.json', readiness)
        atomic_json(directory(root, server) / 'readiness_report.json', readiness)
    atomic_json(directory(root, server) / 'applied_readback.json', value)
    return value


def execute_queue(root, server, state, budget, branch, registration):
    """No branch selection or Teacher preparation: retain the recorded F1–F5 branch."""
    from campaign_r2.plan import priority_blocks
    branch = dict(branch, condition=require_branch_record(server, branch))
    lookup = {b.block_id: b for b in priority_blocks(server)}
    while True:
        verify_registration(root, server)
        retry_completed_uploads(root, server, state, budget)
        queue = planned_queue(server, state, branch)
        if not queue:
            break
        item = queue[0]
        block = lookup[item['block_id']]
        core_complete = all(state['blocks'].get(b.block_id, {}).get('status') == 'DONE'
                            for b in lookup.values() if b.tier == 'CORE')
        # Grandfather admitted reserves even if their first member crossed 20h.
        if (block.tier == 'RESERVE' and not item['admitted'] and core_complete
                and legacy.should_stop(True, ledger.report(root, server)['effective_hours'])):
            legacy.admission_event(root, server, 'R2_RESERVE_REMAINDER_SKIPPED',
                                   first_block_id=block.block_id, priority_revision=PRIORITY_REVISION)
            break
        if not legacy.wait_resources(root, server, budget, state, True):
            return 3
        readiness = legacy.readiness(root, server, state, budget, branch)
        publish_readback(root, server, state, branch, registration, readiness=readiness)
        row = state['blocks'].setdefault(block.block_id, {})
        if not legacy.admit_disk(root, server, state, block.block_id, item['run_ids']):
            return 3
        row.update(status='RUNNING', tier=block.tier, run_ids=item['run_ids'])
        state['priority_revision'] = PRIORITY_REVISION
        atomic_json(ledger.camp(root, server) / 'status.json', state)
        legacy.admission_event(root, server, 'R2_BLOCK_CONTINUE' if item['admitted'] else 'BLOCK_START',
                               block_id=block.block_id, tier=block.tier, branch=branch.get('condition', branch.get('branch')),
                               run_ids=item['run_ids'], priority_revision=PRIORITY_REVISION)
        publish_readback(root, server, state, branch, registration)
        for run in item['run_ids']:
            verify_registration(root, server)
            if state['runs'].get(run, {}).get('status') in legacy.DONE:
                if state['runs'][run].get('upload_pending'):
                    legacy.upload_case(root, case_for(run), state['runs'][run], budget.get('upload_enabled', True))
                continue
            if not legacy.wait_resources(root, server, budget, state, True):
                return 3
            ok = legacy.run_case(root, server, case_for(run), state, budget)
            atomic_json(ledger.camp(root, server) / 'status.json', state)
            if ok is not True:
                row['status'] = 'INCOMPLETE'
                state['status'] = 'PAUSED' if state['runs'][run].get('status') == 'PAUSED' else 'BLOCK_FAILED'
                atomic_json(ledger.camp(root, server) / 'status.json', state)
                return 2  # Never auto-select N2/A10 or replace a failed reference.
        row['status'] = 'DONE'
        atomic_json(ledger.camp(root, server) / 'status.json', state)
        legacy.admission_event(root, server, 'BLOCK_COMPLETE', block_id=block.block_id,
                               run_ids=item['run_ids'], priority_revision=PRIORITY_REVISION)
        audit = collect_audit(root, server, write_reports=True)
        if not audit.get('admission_integrity_ok'):
            raise ValueError('Post-block integrity audit failed; new admissions blocked')
        publish_readback(root, server, state, branch, registration)
    legacy.publish_completion(root, server, state, branch)
    collect_audit(root, server, write_reports=True)
    publish_readback(root, server, state, branch, registration)
    return 0


def _run(root, server, *, wait=True):
    root = Path(root)
    folder = directory(root, server)
    with legacy.locked(folder / '.controller.lock'):
        registration = verify_registration(root, server)
        applied = ledger.read(folder / 'applied_readback.json').get('applied') is True
        if not applied:
            arm_transfer(root, registration)  # Recover interruption between registration and hold.
        with take_runner_lock(root, server, wait=wait):
            registration = verify_registration(root, server)
            audit = collect_audit(root, server, write_reports=True)
            if not audit.get('admission_integrity_ok'):
                raise ValueError('Final locked reconciliation failed; no R2 case will start')
            base = ledger.camp(root, server)
            state = ledger.read(base / 'status.json')
            branch = ledger.read(base / 'branch_record.json')
            state = reconcile_completed(root, server, state, branch)
            planned_queue(server, state, branch)  # Includes any late old-controller admission.
            restore_owned_hold(root, registration, already_applied=applied)
            atomic_json(base / 'status.json', state)
            publish_readback(root, server, state, branch, registration)
            if state.get('status') in legacy.CLOSED:
                retry_completed_uploads(root, server, state, legacy.load_budget(root, server))
                legacy.publish_completion(root, server, state, branch)
                publish_readback(root, server, state, branch, registration)
                return 0  # Never reopen completed campaigns or reset their clock.
            return execute_queue(root, server, state, legacy.load_budget(root, server), branch, registration)


def run(root, server, *, wait=True):
    try:
        result = _run(root, server, wait=wait)
    except BlockingIOError:
        return 3  # Another controller/runner retains ownership; no status overwrite.
    except Exception as exc:
        atomic_json(directory(root, server) / 'controller_status.json', dict(
            status='BLOCKED', reason=str(exc), at_utc=utcnow(),
            training_killed=False, original_artifacts_rewritten=False))
        raise
    atomic_json(directory(root, server) / 'controller_status.json', dict(
        status='FINISHED' if result == 0 else 'RETRY_OR_REVIEW_REQUIRED',
        exit_code=result, at_utc=utcnow(), original_campaign_clock_preserved=True))
    return result


def ensure(root, server):
    """Watchdog path; a missing/bad registration never falls back to legacy."""
    pointer = Path(root) / 'work_dir/_fh20r1/r2_local_server.txt'
    if not pointer.is_file() or pointer.read_text().strip() != server:
        raise ValueError('R2 pointer/server mismatch; no legacy fallback')
    verify_registration(root, server)
    path = directory(root, server) / '.controller.lock'
    try:
        with legacy.locked(path):
            pass
    except BlockingIOError:
        return dict(status='ALREADY_RUNNING', server=server)
    from campaign_r2.deployment import spawn
    return spawn(root, server)
