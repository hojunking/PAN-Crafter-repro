from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g20 import controller as c
from g20.common import atomic_json, read_json
from g20.plan import blocks_for, build_config, case_for
from g20.policy import CampaignWindow


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.window = CampaignWindow('2050-01-01T00:00:00Z')

    def test_build_generates_only32_resolved_configs(self):
        result = c.build_artifacts(self.root)
        self.assertEqual(result['configs'], 32)
        self.assertEqual(len(list((self.root / 'config/g20').glob('*.yaml'))), 32)
        self.assertFalse((self.root / 'work_dir').exists())
        c.build_artifacts(self.root)

    def test_clock_cannot_restart(self):
        c.write_window(self.root, '2050-01-01T00:00:00Z')
        with self.assertRaises(ValueError):
            c.write_window(self.root, '2050-01-01T01:00:00Z')
        self.assertEqual(c.shared_window(self.root), self.window)

    def test_external_r0_checked_before_fresh_s1_teacher(self):
        cases = blocks_for('s1')[0].cases
        with patch.object(c, 'ensure_reference', return_value=False) as ensure:
            self.assertFalse(c.external_references_ready(self.root, 's1', cases, {}, self.window))
            ensure.assert_called_once_with(self.root, 's1', 'R0', {}, self.window)
        with patch.object(c, 'ensure_reference') as ensure:
            self.assertTrue(c.external_references_ready(self.root, 's2', blocks_for('s2')[0].cases, {}, self.window))
            ensure.assert_not_called()

    def test_all_terminal_notadmitted_reasons_leave_queue(self):
        state = c._state(self.root, 's2')
        for b in blocks_for('s2'):
            state['blocks'][b.block_id] = dict(status='NOT_ADMITTED_PRIORITY_COMPARISON', run_ids=list(b.run_ids))
        self.assertEqual(c.effective_queue(self.root, 's2', state), [])

    def test_legacy_postrun_lock_not_just_gpu_controls_idle(self):
        state = c._state(self.root, 's3')
        with patch('tools.fh20r1_runner.inventory', return_value=[]), patch('tools.fh20r1_runner.gpu_processes', return_value=[]), \
             patch('g20.transition.legacy_busy', side_effect=[True, False]), patch.object(c.time, 'sleep') as sleep:
            self.assertTrue(c._wait_local_idle(self.root, 's3', state, self.window))
            sleep.assert_called_once_with(15)

    def test_reference_lock_cannot_silently_follow_new_bundle(self):
        folder = c.camp(self.root, 's3')
        atomic_json(folder / 'screen_selection.json', dict(selected_candidate='A1', reference_sha256={'R0': 'a' * 64}))
        with patch('g20.references.validate_reference', return_value=({'changed': True}, None, None, None)):
            with self.assertRaisesRegex(ValueError, 'bundle changed'):
                c.branches(self.root, 's3')

    def test_closeout_unstarted_admitted_is_not_reported_running(self):
        state = c._state(self.root, 's3')
        block = blocks_for('s3')[0]
        state['blocks'][block.block_id] = dict(status='ADMITTED', run_ids=list(block.run_ids))
        with patch.object(c, 'evaluation_debt', return_value=0):
            c._closeout(self.root, 's3', state, self.window)
        result = read_json(c.camp(self.root, 's3') / 'closeout.json')
        self.assertEqual(result['runs']['G20-S11']['status'], 'NOT_ADMITTED_TIME_LIMIT')

    def test_remaining_eval_debt_keeps_recovery_enabled(self):
        state = c._state(self.root, 's3')
        with patch.object(c, 'evaluation_debt', return_value=.1):
            c._closeout(self.root, 's3', state, self.window)
        self.assertEqual(state['status'], 'CLOSEOUT_EVALUATION_PENDING')

    def test_repay_debt_never_calls_train_for_partial(self):
        case = case_for('G20-S11')
        state = c._state(self.root, 's3')
        state['runs'][case.run_id] = dict(training_started=True, status='PARTIAL_TIME_LIMIT')
        atomic_json(self.root / 'work_dir' / case.run_id / 'meta/training_status.json',
                    dict(training_complete=False, actual_updates=24240))
        with patch('g20.postrun.process_partial', create=True) as partial, patch.object(c, 'run_case') as train, \
             patch.object(c, 'retry_uploads'), patch.object(c, '_branch_decisions'):
            c.repay_evaluation_debt(self.root, 's3', state, self.window)
            partial.assert_called_once()
            train.assert_not_called()

    def test_repay_complete_run_uses_existing_trained_path(self):
        case = case_for('G20-S11')
        state = c._state(self.root, 's3')
        state['runs'][case.run_id] = dict(training_started=True, status='EVALUATION_PENDING')
        atomic_json(self.root / 'work_dir' / case.run_id / 'meta/training_status.json',
                    dict(training_complete=True, actual_updates=50000))
        with patch('g20.postrun.process_partial', create=True) as partial, patch.object(c, 'run_case') as finish, \
             patch.object(c, 'retry_uploads'), patch.object(c, '_branch_decisions'):
            c.repay_evaluation_debt(self.root, 's3', state, self.window)
            finish.assert_called_once()
            partial.assert_not_called()

    def test_transfer_owner_cannot_issue_coordinator_receipt(self):
        with patch.object(c, 'verify_registration', side_effect=ValueError('No s1 registration')):
            with self.assertRaises(ValueError):
                c.publish_transfer_selection(self.root, 'missing', 'missing', 'R1')

    def test_remote_evidence_can_relocate_by_actual_content_hash(self):
        import hashlib
        content = b'original measured remote artifact'
        digest = hashlib.sha256(content).hexdigest()
        path = self.root / 'work_dir/_g20/incoming/evidence' / digest
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        c._verify_evidence_files(self.root, {'/different-server/receipt.json': digest})
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            c._verify_evidence_files(self.root, {'/different-server/receipt.json': digest})

    def test_staged_transfer_activated_under_existing_runner_lock(self):
        incoming = c.camp(self.root, 's3') / 'incoming/transfer_selection.json'
        atomic_json(incoming, {'placeholder': 'measured receipt checked by register_transfer'})
        state = c._state(self.root, 's3')
        with patch.object(c, 'register_transfer', return_value=dict(status='WAIT_CONFIRMATION_PRIORITY', admitted=False)) as register:
            c._activate_staged_transfer(self.root, 's3', state)
            register.assert_called_once_with(self.root, 's3', incoming, runner_lock_held=True)
        self.assertEqual(read_json(c.camp(self.root, 's3') / 'transfer_activation.json')['status'], 'WAIT_CONFIRMATION_PRIORITY')

    def test_invalid_optional_transfer_does_not_block_valid_base(self):
        incoming = c.camp(self.root, 's3') / 'incoming/transfer_selection.json'
        atomic_json(incoming, {'invalid': True})
        state = c._state(self.root, 's3')
        with patch.object(c, 'register_transfer', side_effect=ValueError('Bad proof')) as register:
            c._activate_staged_transfer(self.root, 's3', state)
            c._activate_staged_transfer(self.root, 's3', state)
            register.assert_called_once()
        self.assertNotEqual(state['status'], 'BLOCKED_INTEGRITY')
        self.assertEqual(read_json(c.camp(self.root, 's3') / 'transfer_activation.json')['status'], 'TRANSFER_REJECTED')

    def test_training_wall_includes_loader_init_and_resumed_attempts(self):
        case = case_for('G20-S11')
        state = c._state(self.root, 's3')
        state['runs'][case.run_id] = dict(train_wall_seconds=5.)
        wd = self.root / 'work_dir' / case.run_id
        def command(root, args, log):
            if args[0] == 'train':
                atomic_json(wd / 'meta/training_status.json', dict(training_complete=True,
                    actual_updates=50000, training_seconds=1., evaluation_seconds=1., io_seconds=1.))
            elif args[0] == 'postrun':
                atomic_json(wd / 'official/postrun_status.json', dict(official_complete=True))
            return 0
        with patch.object(c, '_resolve_config', return_value=wd / 'config.yaml'), \
             patch.object(c, '_command', side_effect=command), \
             patch.object(c.time, 'monotonic', side_effect=[100., 120., 200., 204.]):
            self.assertTrue(c.run_case(self.root, 's3', case, state, self.window))
        row = state['runs'][case.run_id]
        self.assertEqual(row['train_wall_seconds'], 25.)
        self.assertNotIn('train_wall_attempt_started_utc', row)
        combined = next(o for o in state['observations'] if o['component'] == 'TRAIN_EVAL')
        self.assertAlmostEqual(combined['hours'], 29. / 3600.)

    def test_failed_train_attempt_retains_wall_time(self):
        case = case_for('G20-S11')
        state = c._state(self.root, 's3')
        with patch.object(c, '_resolve_config', return_value=self.root / 'config.yaml'), \
             patch.object(c, '_command', return_value=75), \
             patch.object(c.time, 'monotonic', side_effect=[100., 113.]):
            self.assertFalse(c.run_case(self.root, 's3', case, state, self.window))
        self.assertEqual(state['runs'][case.run_id]['train_wall_seconds'], 13.)
        self.assertEqual(state['runs'][case.run_id]['status'], 'PAUSED')


if __name__ == '__main__':
    unittest.main()
