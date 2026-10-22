"""CPU-only deadline, whole-group admission and original-donor reuse tests."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from directshift48.reporting import atomic_json
from panda_scale30 import plan
from panda_scale30 import runtime as R
from panda_scale30 import donor as D


class FakeClock:
    def __init__(self):
        self.wall, self.monotonic = 200000., 50000.

    def __call__(self):
        return self.wall

    def advance(self, seconds):
        self.wall += seconds
        self.monotonic += seconds


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = FakeClock()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.real_code_pin = R.code_pin
        self.stack.enter_context(patch.object(R, 'code_pin', return_value={'synthetic.py': 'd' * 64}))
        self.stack.enter_context(patch.object(R, 'evaluator_identity', return_value={'protocol': 'SYNTHETIC_ONLY'}))
        self.stack.enter_context(patch.object(R.time, 'monotonic', side_effect=lambda: self.clock.monotonic))
        self.stack.enter_context(patch.object(R.shutil, 'disk_usage', return_value=SimpleNamespace(free=10 * 1024**3)))
        self.stack.enter_context(patch.object(R.torch.cuda, 'synchronize', side_effect=AssertionError('No GPU work')))
        self.stack.enter_context(patch.object(R.torch.cuda, 'empty_cache', side_effect=AssertionError('No GPU work')))
        self.campaign = R.Campaign('s1', output=self.temp.name, device='cpu', clock=self.clock)

    def group(self, name):
        return next(g for g in self.campaign.config['groups'] if g['name'] == name)

    def measured(self, case, update_p90=.1):
        cfg = plan.resolve_case('s1', case)
        self.campaign.state['timing'][self.campaign.timing_key(cfg)] = {
            'updates_observed': 1000, 'update_p90': update_p90,
            'eval_seconds': 2., 'save_seconds': 1.,
        }

    def test_budget_persists_across_restart_and_clock_rollback(self):
        c = self.campaign
        started, deadline = c.state['started_unix'], c.deadline_unix
        self.clock.advance(7200.)
        c.persist()
        restarted = R.Campaign('s1', output=self.temp.name, device='cpu', clock=self.clock)
        self.assertEqual(restarted.state['started_unix'], started)
        self.assertEqual(restarted.deadline_unix, deadline)
        self.assertEqual(restarted.elapsed_seconds(), 7200.)
        self.clock.wall -= 600.
        self.clock.monotonic += 100.
        self.assertEqual(restarted.elapsed_seconds(), 7300.)
        restarted.persist()
        again = R.Campaign('s1', output=self.temp.name, device='cpu', clock=self.clock)
        self.assertGreaterEqual(again.elapsed_seconds(), 7300.)
        self.assertEqual(again.deadline_unix, deadline)

    def test_deadline_reserves_closeout_and_stop_disk_fail_closed(self):
        c = self.campaign
        limit = c.config['budget']['total_seconds'] - c.config['budget']['closeout_seconds']
        self.clock.advance(limit - 1.)
        c.check()
        self.clock.advance(1.)
        with self.assertRaisesRegex(R.BudgetExpired, 'DURABLE'):
            c.check()
        c.closeout = True
        c.check()
        self.clock.advance(c.config['budget']['closeout_seconds'])
        with self.assertRaisesRegex(R.BudgetExpired, 'DURABLE'):
            c.check()
        c.stop_requested = True
        with self.assertRaisesRegex(R.BudgetExpired, 'STOP_REQUESTED'):
            c.check()
        c.stop_requested = False
        with patch.object(R.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(R.BudgetExpired, 'DISK_CLOSEOUT'):
                c.check()

    def optional_group(self):
        return next(group for group in self.campaign.config['groups'] if group['optional'])

    def test_optional_group_requires_local_measured_timing_and_complete_budget(self):
        c, group = self.campaign, self.optional_group()
        self.assertFalse(c.admit_group(group))
        self.assertFalse(c.state['groups'][-1]['measured_operator_timing'])
        self.measured('S020BASE')
        self.assertTrue(c.admit_group(group))
        required = c.state['groups'][-1]['required_seconds']
        self.assertGreater(required, 0.)
        limit = c.config['budget']['total_seconds'] - c.config['budget']['closeout_seconds']
        self.clock.advance(limit - required + 1.)
        self.assertFalse(c.admit_group(group))
        self.assertEqual(c.state['runs'], [])

    def test_mandatory_groups_allow_explicit_unmeasured_prior(self):
        for group in self.campaign.config['groups']:
            if group['optional']:
                continue
            self.assertTrue(self.campaign.admit_group(group))
            receipt = self.campaign.state['groups'][-1]
            self.assertFalse(receipt['optional'])
            self.assertFalse(receipt['measured_operator_timing'])
            self.assertEqual(len(receipt['pending']), len(group['runs']))

    def test_all_positive_scales_and_tuning_share_one_single_view_operator(self):
        c = self.campaign
        keys = set()
        for server in ('s1', 's3'):
            for group in plan.groups(server):
                for case, replica in group['runs']:
                    cfg = plan.resolve_case(server, case, replica)
                    keys.add(c.timing_key(cfg))
                    self.assertEqual(cfg['student_views'], 1)
                    self.assertEqual(cfg['rotation_consistency_weight'], 0.)
        self.assertEqual(len(keys), 1)
        self.assertNotIn(c.timing_key(plan.resolve_case('s1', plan.DONORS[0])), keys)
        self.measured('S020BASE', .4)
        cfg = plan.resolve_case('s1', 'S010BASE')
        expected = (50000 * .4 + len(cfg['evaluation_steps']) * 3.
                    + c.config['budget']['calibration_reserve_seconds']) * c.config['budget']['safety_factor']
        self.assertAlmostEqual(c.estimate_seconds(cfg), expected)
        for changes in ({'student_views':2}, {'rotation_consistency_weight':.05}):
            with self.assertRaisesRegex(ValueError, 'single-view'):
                c.timing_key({**cfg, **changes})

    def test_pending_skip_refuses_any_started_member(self):
        c, group = self.campaign, self.optional_group()
        request = {'request_id': 'explicit-change', 'reason': 'Skip optional repeat',
                   'skip_groups': [group['name']]}
        atomic_json(c.root / 'pending_control.json', request)
        self.assertTrue(c.pending_skip(group))
        other = next(g for g in self.campaign.config['groups'] if g['name'] != group['name'])
        self.assertFalse(c.pending_skip(other))
        cfg = plan.resolve_case('s1', *group['runs'][0])
        for status in ('RUNNING', 'STOPPED', 'FAILED', 'COMPLETED'):
            c.state['runs'] = [{'run_id': cfg['run_id'], 'status': status}]
            self.assertTrue(c.group_started(group))
            with self.assertRaisesRegex(ValueError, 'Cannot skip started group'):
                c.pending_skip(group)

    def test_code_changes_block_continuation_without_mutating_state(self):
        c = self.campaign
        before = deepcopy(c.state)
        c.verify_code()
        with patch.object(R, 'code_pin', return_value={'synthetic.py': 'e' * 64}):
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                c.verify_code()
        self.assertEqual(c.state, before)

    def test_code_pin_contains_new_and_historical_shared_training_sources(self):
        pins = self.real_code_pin()
        for name in ('panda_scale30/spec.py', 'panda_scale30/method.py',
                     'panda_scale30/calibration.py', 'panda_scale30/engine.py',
                     'panda_scale30/donor.py', 'panda_scale30/plan.py',
                     'panda_mature30/donor.py', 'panda_mature30/selection.py',
                     'panda_band30/reconstruction.py', 'pa/warp.py',
                     'panda_midpoint/resampling.py', 'ta2/registration.py'):
            self.assertIn(name, pins)
            self.assertRegex(pins[name], r'^[0-9a-f]{64}$')

    def test_improvement_keeps_hqnr_primary_even_if_ergas_target_is_missed(self):
        best = {'HQNR': .959, 'ERGAS': 2.04, 'completed_update': 25000}
        evaluation = {'fr': {'metrics': {'HQNR': .960}},
                      'rr': {'metrics': {'ERGAS': 2.3, 'SCC': .98}},
                      'state_before': {'completed_update': 26000}}
        row, reasons = R.improvement(evaluation, {'best_selected': best, 'best_hqnr': best})
        self.assertEqual(row['ERGAS'], 2.3)
        self.assertIn('BEST_HQNR_THEN_ERGAS', reasons)


class DonorWrapperTests(unittest.TestCase):
    def test_cross_server_receipt_rejected_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / plan.donor_receipt_path('s1')
            path.parent.mkdir(parents=True)
            atomic_json(path, {'server': 's3'})
            before = path.read_bytes()
            with patch.object(D, 'validate_receipt', side_effect=lambda value: value):
                with self.assertRaisesRegex(ValueError, 'same-server'):
                    D.source_receipt('s1', name)
            self.assertEqual(path.read_bytes(), before)

    def test_receipt_is_mandatory_and_validation_checked_before_original_loader(self):
        registered = SimpleNamespace(sources=object())
        with self.assertRaisesRegex(ValueError, 'mandatory'):
            D.load_donor('unused.pt', 'a' * 64, registered)
        receipt = {'validation_inventory': {'id': 'synthetic'}}
        with patch.object(D, 'validate_inventory', side_effect=ValueError('wrong inventory')):
            with patch.object(D, 'original_load_donor') as loader:
                with self.assertRaisesRegex(ValueError, 'wrong inventory'):
                    D.load_donor('unused.pt', 'a' * 64, registered, selector_receipt=receipt)
                loader.assert_not_called()

    def test_exact_original_loader_receives_immutable_receipt_and_sha(self):
        registered = SimpleNamespace(sources=object())
        receipt = {'validation_inventory': {'id': 'synthetic'}}
        original = deepcopy(receipt)
        donor = SimpleNamespace(binding={'source': 'original'}, checkpoint_metadata={'source': 'original'})
        with patch.object(D, 'validate_inventory') as validate:
            with patch.object(D, 'original_load_donor', return_value=donor) as loader:
                actual = D.load_donor('/original.pt', 'b' * 64, registered,
                    server='s3', device='cpu', selector_receipt=receipt)
        validate.assert_called_once_with(receipt['validation_inventory'], registered.sources)
        loader.assert_called_once_with('/original.pt', 'b' * 64, registered, server='s3',
            device='cpu', include_teacher=True, selector_receipt=receipt)
        self.assertIs(actual, donor)
        self.assertEqual(receipt, original)
        self.assertEqual(actual.binding, {'source': 'original'})
        self.assertEqual(actual.checkpoint_metadata['consumer_campaign_id'], plan.CAMPAIGN_ID)


if __name__ == '__main__':
    unittest.main()
