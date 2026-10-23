"""Unlimited runtime, explicit queue amendments, and scoped transition guards."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from directshift48.reporting import atomic_json
from panda_schedule30 import plan, runtime as R, transition as T


class Clock:
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
        self.clock = Clock()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(R, 'code_pin', return_value={'synthetic.py':'d'*64}))
        self.stack.enter_context(patch.object(R, 'evaluator_identity', return_value={'protocol':'SYNTHETIC'}))
        self.stack.enter_context(patch.object(R.time, 'monotonic', side_effect=lambda:self.clock.monotonic))
        self.stack.enter_context(patch.object(R.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)))
        self.campaign = R.Campaign('s1', output=self.temp.name, device='cpu', clock=self.clock)

    def test_no_wall_deadline_even_after_restart_and_long_pause(self):
        c = self.campaign
        started = c.state['started_unix']
        self.clock.advance(365*24*3600)
        c.check()
        c.persist()
        restarted = R.Campaign('s1', output=self.temp.name, device='cpu', clock=self.clock)
        restarted.check()
        self.assertEqual(restarted.state['started_unix'], started)
        self.assertIsNone(restarted.deadline_unix)
        self.assertGreaterEqual(restarted.elapsed_seconds(), 365*24*3600)
        self.clock.wall -= 600
        self.clock.monotonic += 100
        self.assertGreaterEqual(restarted.elapsed_seconds(), 365*24*3600+100)

    def test_stop_and_disk_guards_remain(self):
        c = self.campaign
        c.stop_requested = True
        with self.assertRaisesRegex(R.BudgetExpired, 'STOP_REQUESTED'):
            c.check()
        c.stop_requested = False
        with patch.object(R.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(R.BudgetExpired, 'DISK_CLOSEOUT'):
                c.check()

    def test_all_declared_groups_admitted_without_measured_timing(self):
        c = self.campaign
        self.clock.advance(1e9)
        for group in c.config['groups']:
            self.assertTrue(c.admit_group(group))
            self.assertIsNone(c.state['groups'][-1]['remaining_seconds'])
            self.assertFalse(c.state['groups'][-1]['measured_operator_timing'])
        self.assertEqual(c.state['runs'], [])

    def test_fork_estimate_charges_only_additional_updates(self):
        c = self.campaign
        cfg = plan.resolve_case('s1', 'S020F25C35')
        c.state['timing'][c.timing_key(cfg)] = {
            'updates_observed':1000, 'update_p90':.2, 'eval_seconds':2., 'save_seconds':1.}
        expected = (10000*.2 + len(cfg['evaluation_steps'])*3.
                    + c.config['budget']['calibration_reserve_seconds'])*c.config['budget']['safety_factor']
        self.assertAlmostEqual(c.estimate_seconds(cfg), expected)

    def test_skip_refuses_started_group(self):
        c = self.campaign
        group = c.config['groups'][0]
        atomic_json(c.root/'pending_control.json', {'request_id':'user-amendment',
                    'reason':'explicit skip', 'skip_groups':[group['name']]})
        self.assertTrue(c.pending_skip(group))
        cfg = plan.resolve_case('s1', *group['runs'][0])
        c.state['runs'].append({'run_id':cfg['run_id'], 'status':'STOPPED'})
        with self.assertRaisesRegex(ValueError, 'Cannot skip started'):
            c.pending_skip(group)

    def test_source_changes_fail_closed(self):
        original = deepcopy(self.campaign.state)
        with patch.object(R, 'code_pin', return_value={'synthetic.py':'e'*64}):
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                self.campaign.verify_code()
        self.assertEqual(original, self.campaign.state)

    def test_fork_trace_requires_only_local_suffix_and_preserves_discarded_tail(self):
        directory = Path(self.temp.name)/'fork_trace'
        directory.mkdir()
        R.recover_trace(directory, 25000, 25000)
        rows = [{'step':step,'local_update':step-25000,'seconds_including_IO':.1}
                for step in range(25001, 25004)]
        path = directory/'train.jsonl'
        original = ''.join(json.dumps(row)+'\n' for row in rows)
        path.write_text(original)
        R.recover_trace(directory, 25002, 25000)
        self.assertEqual([json.loads(line)['step'] for line in path.read_text().splitlines()], [25001,25002])
        backup = next(directory.glob('train_before_resume_*.jsonl'))
        self.assertEqual(backup.read_text(), original)
        with self.assertRaisesRegex(ValueError, 'incomplete/noncontiguous'):
            R.recover_trace(directory, 25003, 25000)

    def test_fork_trace_rejects_copied_source_prefix(self):
        directory = Path(self.temp.name)/'wrong_trace'
        directory.mkdir()
        (directory/'train.jsonl').write_text(json.dumps({'step':1,'local_update':1,'seconds_including_IO':.1})+'\n')
        with self.assertRaisesRegex(ValueError, 'incomplete/noncontiguous'):
            R.recover_trace(directory, 25001, 25000)


class TransitionTests(unittest.TestCase):
    def run_row(self, case, replica='R1', status='RUNNING', **extra):
        return {'case':case, 'replica':replica, 'status':status, **extra}

    def test_s1_moves_pending_r2_controls_without_relabeling_started_runs(self):
        campaign = {'runs':[self.run_row('S040BASE')]}
        result = T.pending_amendment('s1', campaign)
        self.assertEqual(result['skip_groups'], list(T.SKIP_PENDING['s1']))
        self.assertEqual(campaign['runs'][0]['status'], 'RUNNING')
        self.assertEqual(T.pending_amendment('s1', campaign, result), result)

    def test_s3_skips_only_pending_combination_repeat(self):
        for case in ('S020BASE', 'S020H050', 'S020LR150', 'S020LR150H050'):
            result = T.pending_amendment('s3', {'runs':[self.run_row(case)]})
            self.assertEqual(result['skip_groups'], ['08_s020lr150h050_r2'])
        result = T.pending_amendment('s3', {'runs':[self.run_row('S020LR150H050', 'R2')]})
        self.assertEqual(result['skip_groups'], [])

    def test_s3_does_not_stop_unrelated_active_run(self):
        container = {'Running':True}
        for row in (self.run_row('S020BASE', 'R2'), self.run_row('S020LR150H050', 'R1')):
            self.assertEqual(T.transition_decision('s3', container, {'runs':[row]}), 'WAITING_PREVIOUS')
        self.assertEqual(T.transition_decision('s3', container,
            {'runs':[self.run_row('S020LR150H050', 'R2')]}), 'STOP_SELECTED_RUNNING')

    def test_only_explicit_clean_stop_can_transfer(self):
        container = {'Running':False, 'Status':'exited', 'ExitCode':0}
        campaign = {'status':'STOPPED_SAFE', 'reason':'STOP_REQUESTED',
                    'runs':[self.run_row('S040BASE', status='STOPPED', poisoned=False)]}
        self.assertEqual(T.transition_decision('s1', container, campaign), 'VERIFY_SELECTED_STOP_CHECKPOINT')
        for changes in ({'case':'S020BASE'}, {'poisoned':True}, {'status':'FAILED'},
                        {'stop_checkpoint_error':'disk failure'}):
            bad = deepcopy(campaign)
            bad['runs'][0].update(changes)
            with self.assertRaises(RuntimeError):
                T.transition_decision('s1', container, bad)

    def test_abnormal_exit_and_incomplete_history_block_launch(self):
        campaign = {'status':'FINISHED_REGISTERED_QUEUE', 'runs':[]}
        with self.assertRaises(RuntimeError):
            T.transition_decision('s1', {'Running':False, 'Status':'exited', 'ExitCode':1}, campaign)
        campaign['runs'] = [self.run_row('S020BASE', status='STOPPED')]
        with self.assertRaises(RuntimeError):
            T.transition_decision('s1', {'Running':False, 'Status':'exited', 'ExitCode':0}, campaign)

    def test_s3_completed_predecessor_with_completed_excluded_repeat_is_accepted(self):
        campaign = {'status':'FINISHED_REGISTERED_QUEUE',
                    'runs':[self.run_row('S020LR150H050','R2',status='COMPLETED')]}
        self.assertEqual(T.pending_amendment('s3', campaign)['skip_groups'], [])
        self.assertEqual(T.transition_decision('s3',
                         {'Running':False,'Status':'exited','ExitCode':0},campaign), 'READY_TO_CHECK_GPU')

    def test_s3_budget_stop_accepted_only_for_exact_excluded_repeat(self):
        container = {'Running':False,'Status':'exited','ExitCode':0}
        campaign = {'status':'STOPPED_SAFE','reason':'DURABLE_30H_DEADLINE',
                    'runs':[self.run_row('S020LR150H050','R2',status='STOPPED',poisoned=False)]}
        self.assertEqual(T.transition_decision('s3',container,campaign), 'VERIFY_SELECTED_STOP_CHECKPOINT')
        campaign['runs'][0]['case'] = 'S020BASE'
        with self.assertRaises(RuntimeError):
            T.transition_decision('s3',container,campaign)

    def test_source_path_translation_preserves_workspace_scope(self):
        self.assertEqual(T.local_path(Path('/local/repo'), '/workspace/work_dir/source.pt'),
                         Path('/local/repo/work_dir/source.pt'))


if __name__ == '__main__':
    unittest.main()
