"""R2 handoff tests with temporary runtimes and mocked original executables.

Nothing in this suite modifies live holds, starts training, sends signals, changes
HEAD, accesses Google Sheets, or acquires a lock outside a temporary directory.
"""
import contextlib
import copy
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from campaign_r2 import controller as C
from campaign_r2 import deployment as D
from campaign_r2 import plan as P
from fh20r1 import ledger


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='r2-controller-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Every controller test is offline, including newly added explicit-start
        # paths.  Individual tests may replace this mock with an injected error.
        cron_patch = patch.object(D, 'install_watchdog_cron', return_value={
            'status': 'ALREADY_INSTALLED', 'changed': False,
            'recover_on_reboot': True, 'readback_verified': True,
        })
        self.cron = cron_patch.start()
        self.addCleanup(cron_patch.stop)
        self.server = 's3'
        self.base = ledger.camp(self.root, self.server)
        self.folder = C.directory(self.root, self.server)
        self.folder.mkdir(parents=True)
        self.branch = dict(condition='STANDARD', branch='STANDARD')
        self.budget = dict(campaign_id=P.CAMPAIGN_ID, server_id=self.server,
                          actual_start_authorized=True, device='cuda', upload_enabled=True,
                          started_at_utc='2026-01-01T00:00:00+00:00',
                          minimum_effective_hours=20., hard_deadline=None)
        self.state = dict(status='REGISTERED', runs={}, blocks={}, stages={})
        ledger.write(self.base / 'branch_record.json', self.branch)
        ledger.write(self.base / 'campaign_budget.json', self.budget)
        ledger.write(self.base / 'status.json', self.state)
        self.original_hold = dict(protocol_id=C.legacy.HOLD, server=self.server,
                                  campaign_id=P.CAMPAIGN_ID, auto_release=False)
        ledger.write(C.hold_path(self.root), self.original_hold)
        self.identity = dict(server_id=self.server, campaign_id=P.CAMPAIGN_ID,
                             priority_revision=P.PRIORITY_REVISION,
                             original_source_identity=dict(git_release='pinned-original-head',
                                                           files={'model.py': 'pinned-bytes'}),
                             controller_identity=dict(content_sha256='fixture-controller'))
        self.registration = dict(schema=C.SCHEMA, explicit_start_authorized=True,
                                 owner_nonce='fixture-nonce', identity=self.identity,
                                 original_hold=self.original_hold,
                                 original_ledger_prefix=C.ledger_prefix(self.root, self.server))
        self.blocks = list(P.priority_blocks(self.server))

    def write_registration(self):
        ledger.write(self.folder / 'registration.json', self.registration)

    def block(self, number):
        return next(b for b in self.blocks if b.block_id.endswith(f'_B{number:02d}'))

    def mark_block(self, number, status='DONE'):
        block = self.block(number)
        order = list(block.order_for('STANDARD'))
        self.state['blocks'][block.block_id] = dict(status=status, tier=block.tier, run_ids=order)
        for run in order:
            self.state['runs'][run] = dict(status='DONE' if status == 'DONE' else 'PENDING',
                                            upload_pending=False)
        return block, order

    def complete_all(self):
        for block in self.blocks:
            self.mark_block(int(block.block_id[-2:]))

    def persist_state(self):
        ledger.write(self.base / 'status.json', self.state)

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes()
                for p in self.root.rglob('*') if p.is_file()}

    @contextlib.contextmanager
    def execution_mocks(self, *, hours=20.1, run_case=None, audit_ok=True):
        calls = []

        def succeed(root, server, case, state, budget):
            calls.append(case.run_id)
            state['runs'][case.run_id] = dict(status='DONE', upload_pending=False)
            return True

        with contextlib.ExitStack() as stack:
            mocks = {}
            for name, options in {
                'verify_registration': {'return_value': self.registration},
                'publish_readback': {'return_value': {}},
                'collect_audit': {'return_value': {'admission_integrity_ok': audit_ok}},
            }.items():
                mocks[name] = stack.enter_context(patch.object(C, name, **options))
            for name, options in {
                'wait_resources': {'return_value': True},
                'readiness': {'return_value': {}},
                'admit_disk': {'return_value': True},
                'admission_event': {},
                'run_case': {'side_effect': run_case or succeed},
                'upload_case': {},
                'publish_completion': {},
            }.items():
                mocks[name] = stack.enter_context(patch.object(C.legacy, name, **options))
            stack.enter_context(patch.object(C.ledger, 'report', return_value={'effective_hours': hours}))
            yield calls, mocks


class HoldAndLockTests(Fixture):
    def test_arm_and_restore_only_own_nonce(self):
        C.arm_transfer(self.root, self.registration)
        self.assertEqual(ledger.read(C.hold_path(self.root)), C.transfer_hold(self.registration))
        before = C.hold_path(self.root).read_bytes()
        C.arm_transfer(self.root, self.registration)
        self.assertEqual(C.hold_path(self.root).read_bytes(), before)
        C.restore_owned_hold(self.root, self.registration)
        self.assertEqual(ledger.read(C.hold_path(self.root)), self.original_hold)
        with self.assertRaisesRegex(ValueError, 'nonce/hold'):
            C.restore_owned_hold(self.root, self.registration)
        C.restore_owned_hold(self.root, self.registration, already_applied=True)

    def test_foreign_transfer_nonce_or_hold_is_never_overwritten(self):
        for hold in (dict(C.transfer_hold(self.registration), owner_nonce='someone-else'),
                     dict(self.original_hold, server='s4'), {}):
            with self.subTest(hold=hold):
                ledger.write(C.hold_path(self.root), hold)
                before = C.hold_path(self.root).read_bytes()
                with self.assertRaises(ValueError):
                    C.arm_transfer(self.root, self.registration)
                with self.assertRaises(ValueError):
                    C.restore_owned_hold(self.root, self.registration, already_applied=True)
                self.assertEqual(C.hold_path(self.root).read_bytes(), before)

    def test_old_runner_lock_is_not_stolen_or_signaled(self):
        original_status = (self.base / 'status.json').read_bytes()
        with (self.base / '.runner.lock').open('a') as old:
            fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(C.os, 'kill') as kill:
                with self.assertRaises(BlockingIOError):
                    with C.take_runner_lock(self.root, self.server, wait=False):
                        self.fail('Acquired live old runner lock')
                kill.assert_not_called()
            self.assertEqual(ledger.read(self.folder / 'controller_status.json')['status'],
                             'WAITING_CASE_BOUNDARY')
            self.assertEqual((self.base / 'status.json').read_bytes(), original_status)
        with C.take_runner_lock(self.root, self.server, wait=False):
            pass

    def test_waiting_controller_acquires_only_after_old_lock_release(self):
        with (self.base / '.runner.lock').open('a') as old:
            fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
            calls = []

            def finish_old_child(seconds):
                calls.append(seconds)
                fcntl.flock(old, fcntl.LOCK_UN)

            with patch.object(C.os, 'kill') as kill:
                with C.take_runner_lock(self.root, self.server, sleep=finish_old_child):
                    self.assertEqual(calls, [10])
                kill.assert_not_called()


class StartAndIdentityTests(Fixture):
    def test_dry_run_has_no_files_hold_launch_or_watchdog_effects(self):
        before = self.snapshot()
        with patch.object(C, 'operational_identity', return_value=self.identity), \
                patch.object(C, 'collect_audit', return_value={'admission_integrity_ok': True}), \
                patch.object(D, 'install_watchdog') as install, \
                patch.object(D, 'spawn') as spawn:
            result = C.start(self.root, self.server, dry_run=True)
        self.assertFalse(result['applied'])
        self.assertFalse(result['mutates_training'])
        self.assertEqual(len(result['next_two_run_ids']), 2)
        self.assertEqual(self.snapshot(), before)
        install.assert_not_called()
        spawn.assert_not_called()
        self.cron.assert_not_called()

    def test_failed_initial_audit_never_registers_or_changes_hold(self):
        before = self.snapshot()
        with patch.object(C, 'operational_identity', return_value=self.identity), \
                patch.object(C, 'collect_audit', return_value={'admission_integrity_ok': False}), \
                patch.object(D, 'install_watchdog') as install, \
                patch.object(D, 'spawn') as spawn:
            with self.assertRaisesRegex(ValueError, 'integrity'):
                C.start(self.root, self.server)
        self.assertEqual(self.snapshot(), before)
        install.assert_not_called()
        spawn.assert_not_called()
        self.cron.assert_not_called()

    def test_hold_is_armed_before_watchdog_pointer_becomes_visible(self):
        self.write_registration()
        pointer = self.root / 'work_dir/_fh20r1/r2_local_server.txt'
        real_arm = C.arm_transfer
        observations = []

        def arm(root, registration):
            observations.append(pointer.exists())
            return real_arm(root, registration)

        with patch.object(C, 'operational_identity', return_value=self.identity), \
                patch.object(C, 'collect_audit', return_value={'admission_integrity_ok': True}), \
                patch.object(D, 'install_watchdog', return_value={'changed': True}), \
                patch.object(D, 'spawn', return_value={'status': 'MOCKED'}), \
                patch.object(C, 'arm_transfer', side_effect=arm):
            result = C.start(self.root, self.server)
        self.assertEqual(observations, [False])
        self.assertEqual(pointer.read_text(), self.server + '\n')
        self.assertEqual(result['status'], 'R2_REQUESTED')

    def test_cron_failure_is_a_warning_and_does_not_block_background_launch(self):
        self.write_registration()
        self.cron.side_effect = RuntimeError('fixture crontab permission denied')
        original_training_status = (self.base / 'status.json').read_bytes()
        with patch.object(C, 'operational_identity', return_value=self.identity), \
                patch.object(C, 'collect_audit', return_value={'admission_integrity_ok': True}), \
                patch.object(D, 'install_watchdog', return_value={'changed': True}), \
                patch.object(D, 'spawn', return_value={'status': 'MOCKED', 'pid': 123}) as spawn, \
                patch.object(C.os, 'kill') as kill:
            result = C.start(self.root, self.server)
        self.cron.assert_called_once_with(self.root)
        spawn.assert_called_once_with(self.root, self.server)
        kill.assert_not_called()
        self.assertEqual(result['status'], 'R2_REQUESTED')
        self.assertEqual(result['recovery']['status'], 'WARNING')
        self.assertFalse(result['recovery']['recover_on_reboot'])
        self.assertIn('permission denied', result['recovery']['error'])
        receipt = ledger.read(self.folder / 'watchdog_installation.json')
        self.assertEqual(receipt['cron'], result['recovery'])
        self.assertEqual((self.base / 'status.json').read_bytes(), original_training_status)

    def test_cron_success_is_reported_and_recorded_without_real_cron_access(self):
        self.write_registration()
        with patch.object(C, 'operational_identity', return_value=self.identity), \
                patch.object(C, 'collect_audit', return_value={'admission_integrity_ok': True}), \
                patch.object(D, 'install_watchdog', return_value={'changed': False}), \
                patch.object(D, 'spawn', return_value={'status': 'MOCKED'}) as spawn:
            result = C.start(self.root, self.server)
        self.cron.assert_called_once_with(self.root)
        spawn.assert_called_once()
        self.assertEqual(result['recovery'], self.cron.return_value)
        self.assertTrue(result['recovery']['recover_on_reboot'])
        self.assertEqual(ledger.read(self.folder / 'watchdog_installation.json')['cron'],
                         self.cron.return_value)

    def test_start_refuses_symlink_or_nonfile_pointer_without_changing_hold(self):
        self.write_registration()
        pointer = self.root / 'work_dir/_fh20r1/r2_local_server.txt'
        target = self.root / 'pointer-target.txt'
        target.write_text(self.server + '\n')
        for kind in ('dangling-symlink', 'valid-symlink', 'directory'):
            with self.subTest(kind=kind):
                # Keep failure reports independent if a broken implementation
                # incorrectly replaced the previous fixture pointer.
                if pointer.is_symlink() or pointer.is_file():
                    pointer.unlink()
                ledger.write(C.hold_path(self.root), self.original_hold)
                if kind == 'directory':
                    pointer.mkdir()
                else:
                    pointer.symlink_to(target if kind == 'valid-symlink' else self.root / 'absent')
                before = C.hold_path(self.root).read_bytes()
                with patch.object(C, 'operational_identity', return_value=self.identity), \
                        patch.object(C, 'collect_audit', return_value={'admission_integrity_ok': True}), \
                        patch.object(D, 'install_watchdog', return_value={'changed': False}), \
                        patch.object(D, 'spawn') as spawn:
                    with self.assertRaises(ValueError):
                        C.start(self.root, self.server)
                    spawn.assert_not_called()
                    self.cron.assert_not_called()
                self.assertEqual(C.hold_path(self.root).read_bytes(), before)
                if kind == 'directory':
                    self.assertTrue(pointer.is_dir())
                    pointer.rmdir()
                else:
                    self.assertTrue(pointer.is_symlink())
                    pointer.unlink()
        self.assertEqual(target.read_text(), self.server + '\n')

    def test_registration_accepts_only_append_only_ledger(self):
        path = self.base / 'active_intervals.jsonl'
        path.write_bytes(b'original evidence\n')
        self.registration['original_ledger_prefix'] = C.ledger_prefix(self.root, self.server)
        self.write_registration()
        with patch.object(C, 'operational_identity', return_value=self.identity):
            C.verify_registration(self.root, self.server)
            path.write_bytes(b'original evidence\nnew evidence\n')
            C.verify_registration(self.root, self.server)
            for invalid in (b'', b'altered! evidence\nnew evidence\n'):
                path.write_bytes(invalid)
                with self.assertRaisesRegex(ValueError, 'shortened or rewritten'):
                    C.verify_registration(self.root, self.server)

    def test_changed_original_head_source_or_budget_identity_rejects(self):
        self.write_registration()
        for key, value in (
            ('original_source_identity', {'git_release': 'new-head'}),
            ('campaign_budget_sha256', 'changed-budget'),
            ('branch_record_sha256', 'changed-branch'),
            ('reference_bridge_sha256', 'changed-reference'),
        ):
            identity = dict(self.identity, **{key: value})
            with self.subTest(key=key), patch.object(C, 'operational_identity', return_value=identity):
                with self.assertRaisesRegex(ValueError, 'stale R2 registration'):
                    C.verify_registration(self.root, self.server)


class QueueExecutionTests(Fixture):
    def test_branch_only_record_is_normalized_for_legacy_without_mutation(self):
        self.complete_all()
        self.mark_block(6, status='PENDING')
        branch = {'branch': 'STANDARD'}
        ledger.write(self.base / 'branch_record.json', branch)
        original_bytes = (self.base / 'branch_record.json').read_bytes()
        with self.execution_mocks() as (calls, mocks):
            result = C.execute_queue(self.root, self.server, self.state, self.budget,
                                     branch, self.registration)
        self.assertEqual(result, 0)
        self.assertEqual(branch, {'branch': 'STANDARD'})
        self.assertEqual((self.base / 'branch_record.json').read_bytes(), original_bytes)
        self.assertEqual(mocks['readiness'].call_count, 1)
        passed_branch = mocks['readiness'].call_args.args[4]
        self.assertEqual(passed_branch, {'branch': 'STANDARD', 'condition': 'STANDARD'})
        self.assertIsNot(passed_branch, branch)
        self.assertEqual(calls, list(self.block(6).order_for('STANDARD')))

    def test_completed_block_pending_upload_is_retried_without_training(self):
        self.complete_all()
        run = self.block(1).order_for('STANDARD')[0]
        self.state['runs'][run]['upload_pending'] = True

        def delivered(root, case, row, enabled):
            self.assertEqual(case.run_id, run)
            self.assertTrue(enabled)
            row['upload_pending'] = False

        with self.execution_mocks() as (calls, mocks), \
                patch.object(C, 'retry_completed_uploads', wraps=C.retry_completed_uploads) as retry:
            mocks['upload_case'].side_effect = delivered
            result = C.execute_queue(self.root, self.server, self.state, self.budget,
                                     self.branch, self.registration)
        self.assertEqual(result, 0)
        retry.assert_called_once()
        mocks['upload_case'].assert_called_once()
        mocks['run_case'].assert_not_called()
        mocks['admit_disk'].assert_not_called()
        self.assertEqual(calls, [])
        self.assertFalse(self.state['runs'][run]['upload_pending'])

    def test_closed_readback_schedules_nothing_but_preserves_reserve_definitions(self):
        for block in self.blocks:
            self.mark_block(int(block.block_id[-2:]), status='DONE' if block.tier == 'CORE' else 'PENDING')
        self.state['status'] = 'COMPLETE_20HPLUS'
        original_state = copy.deepcopy(self.state)
        readback = C.publish_readback(self.root, self.server, self.state, self.branch,
                                     self.registration, readiness={'launch_ready': True, 'status': 'READY'})
        self.assertEqual(readback['next_two_run_ids'], [])
        self.assertEqual(readback['queue'], [])
        self.assertEqual(readback['current_admitted_blocks'], [])
        self.assertEqual([row['block_id'] for row in readback['preserved_unscheduled_blocks']],
                         [b.block_id for b in self.blocks if b.tier == 'RESERVE'])
        readiness = ledger.read(self.base / 'readiness_report.json')
        self.assertFalse(readiness['launch_ready'])
        self.assertEqual(readiness['next_run_ids'], [])
        self.assertIsNone(readiness['next_atomic_block'])
        self.assertEqual(readiness['status'], 'COMPLETE_20HPLUS')
        self.assertEqual(self.state, original_state)

    def test_admitted_reserve_finishes_after20h_then_untouched_reserve_skips(self):
        self.complete_all()
        reserve, order = self.mark_block(11, status='RUNNING')
        self.state['runs'][order[0]]['status'] = 'DONE'
        self.mark_block(12, status='PENDING')
        self.mark_block(13, status='PENDING')
        self.mark_block(14, status='PENDING')
        self.persist_state()
        original = (self.base / 'campaign_budget.json').read_bytes()
        with self.execution_mocks(hours=20.1) as (calls, mocks):
            result = C.execute_queue(self.root, self.server, self.state, self.budget,
                                     self.branch, self.registration)
        self.assertEqual(result, 0)
        self.assertEqual(calls, order[1:])
        self.assertEqual(self.state['blocks'][reserve.block_id]['status'], 'DONE')
        self.assertEqual(mocks['admit_disk'].call_count, 1)
        self.assertEqual((self.base / 'campaign_budget.json').read_bytes(), original)
        event_names = [call.args[2] for call in mocks['admission_event'].call_args_list]
        self.assertIn('R2_BLOCK_CONTINUE', event_names)
        self.assertIn('R2_RESERVE_REMAINDER_SKIPPED', event_names)

    def test_failed_run_stops_without_admitting_another_block(self):
        block, order = self.mark_block(1, status='RUNNING')

        def fail(root, server, case, state, budget):
            state['runs'][case.run_id] = dict(status='FAILED')
            return False

        with self.execution_mocks(run_case=fail) as (calls, mocks):
            result = C.execute_queue(self.root, self.server, self.state, self.budget,
                                     self.branch, self.registration)
        self.assertEqual(result, 2)
        self.assertEqual(mocks['run_case'].call_count, 1)
        self.assertEqual(mocks['admit_disk'].call_count, 1)
        self.assertEqual(self.state['blocks'][block.block_id]['status'], 'INCOMPLETE')
        mocks['publish_completion'].assert_not_called()

    def test_failed_postblock_integrity_audit_blocks_next_admission(self):
        self.mark_block(1, status='RUNNING')
        with self.execution_mocks(audit_ok=False) as (calls, mocks):
            with self.assertRaisesRegex(ValueError, 'integrity audit failed'):
                C.execute_queue(self.root, self.server, self.state, self.budget,
                                self.branch, self.registration)
        self.assertEqual(mocks['admit_disk'].call_count, 1)
        self.assertEqual(len(calls), len(self.block(1).order_for('STANDARD')))
        mocks['publish_completion'].assert_not_called()

    def test_disallowed_reference_fallback_stops_unchanged_branch(self):
        self.mark_block(1, status='RUNNING')
        original_branch = copy.deepcopy(self.branch)

        def unavailable(root, server, case, state, budget):
            state['runs'][case.run_id] = dict(status='REFERENCE_UNAVAILABLE')
            return None

        with self.execution_mocks(run_case=unavailable) as (calls, mocks):
            result = C.execute_queue(self.root, self.server, self.state, self.budget,
                                     self.branch, self.registration)
        self.assertEqual(result, 2)
        self.assertEqual(self.branch, original_branch)
        self.assertEqual(mocks['admit_disk'].call_count, 1)


class LockedReconciliationTests(Fixture):
    def test_closed_campaign_retries_completed_uploads_but_never_training(self):
        self.complete_all()
        self.state['status'] = 'WORK_COMPLETE_UPLOAD_PENDING'
        run = self.block(1).order_for('STANDARD')[0]
        self.state['runs'][run]['upload_pending'] = True
        self.persist_state()
        self.write_registration()

        def delivered(root, case, row, enabled):
            row['upload_pending'] = False

        with patch.object(C, 'operational_identity', return_value=self.identity), \
                patch.object(C, 'collect_audit', return_value={'admission_integrity_ok': True}), \
                patch.object(C, 'reconcile_completed', side_effect=lambda root, server, state, branch: state), \
                patch.object(C, 'retry_completed_uploads', wraps=C.retry_completed_uploads) as retry, \
                patch.object(C.legacy, 'load_budget', return_value=self.budget), \
                patch.object(C.legacy, 'upload_case', side_effect=delivered) as upload, \
                patch.object(C.legacy, 'publish_completion') as publish, \
                patch.object(C, 'execute_queue') as execute:
            result = C.run(self.root, self.server, wait=False)
        self.assertEqual(result, 0)
        retry.assert_called_once()
        upload.assert_called_once()
        publish.assert_called_once()
        execute.assert_not_called()
        state = ledger.read(self.base / 'status.json')
        self.assertFalse(state['runs'][run]['upload_pending'])

    def test_final_audit_failure_leaves_transfer_hold_and_never_executes(self):
        self.write_registration()
        with patch.object(C, 'operational_identity', return_value=self.identity), \
                patch.object(C, 'collect_audit', return_value={'admission_integrity_ok': False}), \
                patch.object(C, 'execute_queue') as execute:
            with self.assertRaisesRegex(ValueError, 'Final locked reconciliation'):
                C.run(self.root, self.server, wait=False)
        execute.assert_not_called()
        self.assertEqual(ledger.read(C.hold_path(self.root)), C.transfer_hold(self.registration))

    def test_late_admitted_block_is_reread_after_old_runner_lock(self):
        self.mark_block(1)
        self.mark_block(2)
        self.persist_state()
        self.write_registration()
        observed = []

        @contextlib.contextmanager
        def old_finishes_with_late_admission(root, server, wait=True):
            self.mark_block(3, status='RUNNING')
            self.persist_state()
            yield

        def execute(root, server, state, budget, branch, registration):
            observed.extend(P.planned_queue(server, state, branch))
            return 0

        with patch.object(C, 'operational_identity', return_value=self.identity), \
                patch.object(C, 'collect_audit', return_value={'admission_integrity_ok': True}), \
                patch.object(C, 'take_runner_lock', side_effect=old_finishes_with_late_admission), \
                patch.object(C, 'reconcile_completed', side_effect=lambda root, server, state, branch: state), \
                patch.object(C.legacy, 'load_budget', return_value=self.budget), \
                patch.object(C, 'execute_queue', side_effect=execute):
            result = C.run(self.root, self.server, wait=False)
        self.assertEqual(result, 0)
        self.assertEqual([row['block_id'] for row in observed[:2]],
                         [self.block(3).block_id, self.block(6).block_id])
        readback = ledger.read(self.folder / 'applied_readback.json')
        self.assertEqual(readback['current_admitted_blocks'], [self.block(3).block_id])
        self.assertEqual(readback['next_two_run_ids'], list(self.block(3).order_for('STANDARD')))
        self.assertEqual(ledger.read(C.hold_path(self.root)), self.original_hold)

    def test_done_evidence_is_validated_before_reuse(self):
        block, order = self.mark_block(1)
        with patch('fh20r1.upload.row_values', side_effect=ValueError('checkpoint hash mismatch')) as validate:
            with self.assertRaisesRegex(ValueError, 'checkpoint hash mismatch'):
                C.reconcile_completed(self.root, self.server, self.state, self.branch)
        validate.assert_called_once_with(order[0], self.root)

    def test_interrupted_run_missing_fullstate_never_becomes_fresh(self):
        block, order = self.mark_block(1, status='RUNNING')
        self.state['runs'][order[0]].update(status='PAUSED', training_started=True)
        (self.root / 'work_dir' / order[0]).mkdir()
        with self.assertRaisesRegex(ValueError, 'no exact fullstate'):
            C.reconcile_completed(self.root, self.server, self.state, self.branch)

    def test_ensure_foreign_pointer_never_spawns(self):
        pointer = self.root / 'work_dir/_fh20r1/r2_local_server.txt'
        for content in ('', 's4\n', 's3\ns4\n'):
            pointer.write_text(content)
            with patch.object(D, 'spawn') as spawn:
                with self.assertRaisesRegex(ValueError, 'pointer/server mismatch'):
                    C.ensure(self.root, self.server)
                spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
