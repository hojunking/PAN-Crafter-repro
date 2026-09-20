import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from g20 import transition as t
from g20.common import atomic_json, camp, read_json, locked
from g20.policy import CampaignWindow


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original = dict(protocol_id='QG40_FUTURE_ADMISSIONS_HOLD_v1')
        self.requested = dict(protocol_id='G20_FUTURE_ADMISSIONS_HOLD_v1')
        self.window = CampaignWindow('2050-01-01T00:00:00Z')
        atomic_json(camp(self.root, 's3') / 'registration.json', dict(original_wv3_hold=self.original))
        atomic_json(camp(self.root, 's3') / 'requested_hold.json', self.requested)
        atomic_json(self.root / 'work_dir/_eval_phase/hold.json', self.original)

    def snapshot(self):
        return dict(blocks={'pair': dict(status='ADMITTED', run_ids=['FIRST', 'FINAL'])},
                    runs={'FIRST': dict(status='OFFICIAL_EVAL_COMPLETE'), 'FINAL': dict(status='RUNNING')})

    def child(self):
        return dict(pid=101, args=['python', 'tools/qg40_runner.py', 'train', '--config', '/work/FINAL.yaml'],
                    started_epoch=time.time() - 100)

    def status(self, update=10000):
        atomic_json(self.root / 'work_dir/FINAL/meta/training_status.json',
                    dict(actual_updates=update, training_complete=False))

    def test_terminal_only_complete_prior_cases_and_current_training(self):
        self.status()
        self.assertIsNotNone(t.terminal_training(self.root, 's3', self.snapshot(), [self.child()]))
        snapshot = self.snapshot()
        snapshot['runs']['FIRST']['status'] = 'RUNNING'
        self.assertIsNone(t.terminal_training(self.root, 's3', snapshot, [self.child()]))

    def test_stale_status_cannot_authorize_new_child_before_hold_check(self):
        self.status()
        child = self.child() | dict(started_epoch=time.time() + 1)
        self.assertIsNone(t.terminal_training(self.root, 's3', self.snapshot(), [child]))
        self.assertIsNone(t.terminal_training(self.root, 's3', self.snapshot(), [self.child() | dict(started_epoch=None)]))

    def test_near_complete_or_absent_child_waits_boundary(self):
        self.status(49000)
        self.assertIsNone(t.terminal_training(self.root, 's3', self.snapshot(), [self.child()]))
        self.status()
        self.assertIsNone(t.terminal_training(self.root, 's3', self.snapshot(), []))

    def test_idle_no_pending_block_can_claim_without_kill(self):
        with patch.object(t, 'processes', return_value=[]):
            self.assertTrue(t.wait_and_claim(self.root, 's3', self.window))
        self.assertEqual(read_json(self.root / 'work_dir/_eval_phase/hold.json'), self.requested)
        self.assertEqual(read_json(camp(self.root, 's3') / 'transition.json')['boundary'], 'old_runner_idle')

    def test_idle_incomplete_admitted_block_is_not_abandoned(self):
        atomic_json(self.root / 'work_dir/_qg40/s3/status.json', self.snapshot())
        with patch.object(t, 'processes', return_value=[]), patch.object(t.time, 'sleep', side_effect=RuntimeError('one poll')):
            with self.assertRaisesRegex(RuntimeError, 'one poll'):
                t.wait_and_claim(self.root, 's3', self.window)
        self.assertEqual(read_json(self.root / 'work_dir/_eval_phase/hold.json'), self.original)

    def test_full_legacy_runner_lock_remains_busy_without_gpu(self):
        with locked(self.root / 'work_dir/_qg40/s3/.runner.lock'):
            with patch.object(t, 'processes', return_value=[]):
                self.assertTrue(t.legacy_busy(self.root, 's3'))
        with locked(self.root / 'work_dir/_fh20r1/s3/.runner.lock'):
            with patch.object(t, 'processes', return_value=[]):
                self.assertTrue(t.legacy_busy(self.root, 's3'))

    def test_foreign_hold_never_overwritten(self):
        foreign = dict(protocol_id='DIFFERENT_USER_HOLD')
        atomic_json(self.root / 'work_dir/_eval_phase/hold.json', foreign)
        with self.assertRaises(ValueError):
            t.wait_and_claim(self.root, 's3', self.window)
        self.assertEqual(read_json(self.root / 'work_dir/_eval_phase/hold.json'), foreign)

    def test_last_child_crash_is_not_successful_drain(self):
        atomic_json(camp(self.root, 's3') / 'transition.json',
                    dict(status='DRAINING_FINAL_LEGACY_CASE', evidence={'last_admitted_run': 'FINAL'}))
        self.status(24000)
        with patch.object(t, 'processes', return_value=[]):
            self.assertTrue(t.legacy_busy(self.root, 's3'))
            atomic_json(self.root / 'work_dir/FINAL/meta/training_status.json',
                        dict(actual_updates=50000, training_complete=True))
            self.assertTrue(t.legacy_busy(self.root, 's3'))
            atomic_json(self.root / 'work_dir/FINAL/official/postrun_status.json', dict(official_complete=True))
            self.assertFalse(t.legacy_busy(self.root, 's3'))


if __name__ == '__main__':
    unittest.main()
