#!/usr/bin/env python
"""CPU/temp-only scheduling regressions; never launch trainers or external APIs."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import mix20h_runner as R
from kdv.mix20h_plan import cases_for


class RunnerTests(unittest.TestCase):
    def setUp(self):
        check = patch.object(R, 'verify_assets', return_value={'fixture': True})
        check.start(); self.addCleanup(check.stop)
        workers = patch.object(R, 'active_gpu_workers', return_value=[])
        workers.start(); self.addCleanup(workers.stop)
    def test_pair_reservation_not_all_candidate_runs(self):
        for server, run_h in [('s1', 2.795), ('s2', 2.96), ('s3', 1.75), ('s4', 1.805), ('s5', 3.84)]:
            self.assertAlmostEqual(R.estimate_hours(server, []), run_h)
            self.assertTrue(R.admission(19, run_h)['admitted'])
            self.assertFalse(R.admission(run_h + .5, run_h)['admitted'])
            self.assertTrue(R.admission(2 * run_h + .5, run_h)['admitted'])

    def test_wallclock_includes_backlog_and_no_duplicate_postrun_charge(self):
        self.assertFalse(R.admission(6.0, 2.795, backlog_h=.1)['admitted'])
        self.assertAlmostEqual(R.estimate_hours('s1', [2., 3.]), 3.3)
        self.assertAlmostEqual(R.estimate_hours('s5', [1.8]), 3.84)

    def test_preparation_reserve_releases_elapsed_time_without_double_charge(self):
        start = 1000.
        self.assertEqual(R.prep_remaining_hours(start, start), .5)
        self.assertEqual(R.prep_remaining_hours(start - 60, start), .5)
        self.assertEqual(R.prep_remaining_hours(start + 900, start), .25)
        self.assertEqual(R.prep_remaining_hours(start + 1800, start), 0.)
        self.assertEqual(R.prep_remaining_hours(start + 7200, start), 0.)
        initial = R.admission(6.3, 2.795, prep_h=R.prep_remaining_hours(start, start))
        self.assertFalse(initial['admitted'])
        self.assertAlmostEqual(initial['required_hours'], 6.59)
        later = R.admission(6.3, 2.795, prep_h=R.prep_remaining_hours(start + 1800, start))
        self.assertTrue(later['admitted'])
        self.assertAlmostEqual(later['required_hours'], 6.09)

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError): R.timestamp('2026-09-18T10:00:00')
        self.assertEqual(R.timestamp('2026-09-18T10:00:00+09:00'), R.timestamp('2026-09-18T01:00:00Z'))

    def test_report_keeps_noeligible_pair_and_legacy_max_separate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); pair = cases_for('s1')[:2]
            state = {'runs': {}}
            for index, c in enumerate(pair):
                state['runs'][c.run_id] = dict(training_complete=True, status='upload_pending',
                    postrun=dict(official_target_complete=True, exact50k_complete=True))
                wd = root/'work_dir'/c.run_id
                R.write_json(wd/'meta/mix20h_run_manifest.json', {'pair_status': 'pair_verified'})
                R.write_json(wd/'best_hqnr_meta.json', dict(step=2020, hqnr=.9589 if index == 0 else .958))
                R.write_text(wd/'checkpoint_metrics.csv', 'step,raw_original.hqnr\n1010,.959\n2020,.9589\n')
                target = dict(target_status='no_eligible', target=None, joint_pass=False) if index else dict(
                    target_status='official', target=dict(ergas=2.05, hqnr=.9586), joint_pass=False)
                R.write_json(wd/'results'/f'qrecon24_target_selection_{R.SELECTOR_ID}.json', target)
                R.write_json(wd/'results/exact50k_official_AON.json', dict(official_complete=True, ergas=2.1+index*.01))
            R.write_json(root/R.CAMP/'status.json', state)
            with patch.object(R, 'load_plan', return_value=dict(server_id='s1', deadline_at_utc=R.iso())):
                result = R.report(root)
            self.assertEqual(result['completion_summary']['base']['training_complete'], 2)
            self.assertEqual(result['completion_summary']['base']['completed_verified_pairs'], 1)
            self.assertEqual(result['profile_summary']['G23']['h_pass_runs'], 1)
            self.assertEqual(result['profile_summary']['B20A03']['h_pass_runs'], 0)
            self.assertEqual(result['runs'][0]['raw_max']['hqnr'], .959)
            self.assertEqual(result['runs'][0]['legacy_best']['hqnr'], .9589)
            self.assertEqual(result['runs'][0]['target']['target']['hqnr'], .9586)
            self.assertIsNone(result['paired_comparisons'][0]['target_E_B_minus_G'])
            self.assertAlmostEqual(result['paired_comparisons'][0]['exact50k_E_B_minus_G'], .01)

    def _fixture(self, root):
        (root / 'work_dir').mkdir()
        R.write_text(root / 'work_dir/cases_queue.txt', 'OLD_PENDING\n')
        ids = [x.run_id for x in cases_for('s1')]
        hashes = dict.fromkeys(ids, 'sha')
        inv = dict(processes=[dict(run_id='OLD_ACTIVE', pid=123)],
                   files={'work_dir/cases_queue.txt': {'content': 'OLD_PENDING\n', 'sha256': 'old'}},
                   new_runs={r: {'status': 'unstarted'} for r in ids})
        return hashes, inv

    def test_prepare_dry_run_has_no_mutations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes, inv = self._fixture(root)
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'inventory', return_value=inv):
                result = R.prepare(root, 's1', None, dry_run=True)
            self.assertEqual(result['status'], 'NEEDS_SHARED_START_AT')
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_migration_preserves_history_and_deadline_survives_restart(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes, inv = self._fixture(root)
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'inventory', return_value=inv), patch.object(R.time, 'time', return_value=1000):
                first = R.prepare(root, 's1', R.iso(1000), dry_run=False)
                p = R.load_plan(root)
                self.assertEqual(R.timestamp(p['deadline_at_utc']), 73000)
                self.assertEqual((root / R.CAMP / 'before/work_dir/cases_queue.txt').read_text(), 'OLD_PENDING\n')
                self.assertLess(R.timestamp((root / 'work_dir/cases_deadline.txt').read_text().strip()), 1000)
                again = R.prepare(root, 's1', None, dry_run=False)
                self.assertEqual(again['status'], 'ALREADY_PREPARED')
                with self.assertRaises(ValueError):
                    R.prepare(root, 's1', R.iso(1010), dry_run=False)
                self.assertEqual(R.load_plan(root)['deadline_at_utc'], p['deadline_at_utc'])

    def test_collision_and_eval_hold_block_apply(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes, inv = self._fixture(root)
            inv['new_runs'][next(iter(hashes))] = {'status': 'completed'}
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'inventory', return_value=inv):
                with self.assertRaisesRegex(ValueError, 'COLLISION'):
                    R.prepare(root, 's1', R.iso(), False)
            self.assertFalse((root / R.CAMP).exists())

    def test_ended_campaign_never_restarts_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes, inv = self._fixture(root)
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'inventory', return_value=inv):
                R.prepare(root, 's1', R.iso(), False)
            R.write_json(root / R.CAMP / 'status.json', {'status': 'STOPPED_BUDGET'})
            with patch.object(R, 'run_command', side_effect=AssertionError('must not launch')):
                self.assertEqual(R.run_campaign(root), 0)

    def test_shell_syntax_and_old_queue_guards(self):
        for name in ['_run_cases.sh', '_watchdog.sh', 'qrecon24_switch.sh', 'qrecon24_waiter.sh', 'campaign_start.sh', '_upload.sh', 'mix20h_switch.sh', 'mix20h_start.sh']:
            path = ROOT / 'tools' / name
            subprocess.run(['bash', '-n', str(path)], check=True)
            if name != '_upload.sh':
                self.assertIn('mix20h', path.read_text())

    def test_postrun_failure_retries_without_retraining(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes, inv = self._fixture(root)
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'inventory', return_value=inv):
                R.prepare(root, 's1', R.iso(), False)
            calls = []; eval_attempts = {}
            def command(repo, cmd, log, env=None):
                calls.append(cmd)
                if 'tools/run.sh' in cmd:
                    return 0
                run = cmd[2]
                eval_attempts[run] = eval_attempts.get(run, 0) + 1
                rc = 2 if run == next(iter(hashes)) and eval_attempts[run] == 1 else 0
                R.write_json(root / 'work_dir' / run / 'results/mix20h_postrun_status.json',
                             dict(sheet_uploaded=rc == 0, official_target_complete=rc == 0, exact50k_complete=rc == 0))
                return rc
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'gpu_processes', return_value=[]), patch.object(R, 'run_command', side_effect=command):
                self.assertEqual(R.run_campaign(root, upload=True), 0)
                R.run_campaign(root, upload=True)
            self.assertEqual(sum('tools/run.sh' in c for c in calls), 8)
            self.assertEqual(eval_attempts[next(iter(hashes))], 2)
            self.assertEqual(R.read_json(root / R.CAMP / 'status.json')['status'], 'DONE')

    def test_budget_stop_does_not_start_pairmate_or_resume_automatically(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes, inv = self._fixture(root)
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'inventory', return_value=inv):
                R.prepare(root, 's1', R.iso(), False)
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'gpu_processes', return_value=[]), patch.object(R, 'run_command', return_value=5) as command:
                R.run_campaign(root); R.run_campaign(root)
            self.assertEqual(command.call_count, 1)
            self.assertEqual(R.read_json(root / R.CAMP / 'status.json')['status'], 'STOPPED_BUDGET')

    def test_interrupted_50k_export_reenters_main_only_with_exact_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes, inv = self._fixture(root)
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'inventory', return_value=inv):
                R.prepare(root, 's1', R.iso(), False)
            run = next(iter(hashes)); wd = root/'work_dir'/run
            R.write_json(wd/'meta/mix20h_run_manifest.json', dict(training_complete=True, actual_updates=50000))
            states = {name: {'status': 'finished'} for name in hashes if name != run}
            R.write_json(root/R.CAMP/'status.json', dict(status='RUNNING', runs=states))
            commands = []
            def command(repo, cmd, log, env=None):
                commands.append(cmd)
                if 'tools/run.sh' not in cmd:
                    R.write_json(wd/'results/mix20h_postrun_status.json', {'sheet_uploaded': True})
                return 0
            resume = wd/'checkpoint-50000'
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'gpu_processes', return_value=[]), patch.object(R, 'checkpoint_for_resume', return_value=resume), patch.object(R, 'run_command', side_effect=command):
                R.run_campaign(root)
            self.assertEqual(commands[0], ['bash', 'tools/run.sh', run, '--resume', str(resume)])
            self.assertEqual(len(commands), 2)
            self.assertEqual(R.read_json(root/R.CAMP/'budget.json')['end_to_end_hours'], [])

    def test_new_hold_at_case_boundary_does_not_start_mate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes, inv = self._fixture(root)
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'inventory', return_value=inv):
                R.prepare(root, 's1', R.iso(), False)
            commands = []
            def command(repo, cmd, log, env=None):
                commands.append(cmd)
                if 'tools/run.sh' not in cmd:
                    R.write_json(root/'work_dir'/cmd[2]/'results/mix20h_postrun_status.json', {'sheet_uploaded': True})
                    R.write_json(root/'work_dir/_eval_phase/hold.json', {'reason': 'fixture diagnosis'})
                return 0
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'gpu_processes', return_value=[]), patch.object(R, 'run_command', side_effect=command):
                self.assertEqual(R.run_campaign(root), 3)
            self.assertEqual(sum('tools/run.sh' in cmd for cmd in commands), 1)
            self.assertEqual(R.read_json(root/R.CAMP/'status.json')['status'], 'WAIT_RESOURCE')
            self.assertFalse((root/'work_dir'/list(hashes)[1]).exists())

    def test_partial_migration_recovers_without_resetting_clock(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes, inv = self._fixture(root); original = R.write_json
            def crash(path, data):
                if Path(path).name == 'budget.json':
                    raise OSError('simulated power failure')
                original(path, data)
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'inventory', return_value=inv):
                t0 = R.iso()
                with patch.object(R, 'write_json', side_effect=crash), self.assertRaises(OSError):
                    R.prepare(root, 's1', t0, False)
                with self.assertRaisesRegex(ValueError, 'incomplete migration'):
                    R.load_plan(root)
                R.prepare(root, 's1', None, False)
                self.assertEqual(R.load_plan(root)['campaign_started_at_utc'], t0)
                for name in ('budget.json', 'status.json', 'pair_reservations.json'):
                    self.assertTrue((root / R.CAMP / name).exists())
                self.assertEqual((root / R.CAMP / 'before/work_dir/cases_queue.txt').read_text(), 'OLD_PENDING\n')

    def test_partial_checkpoint_not_selected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); run = cases_for('s1')[0].run_id
            names = ('model.safetensors', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl',
                     'custom_checkpoint_0.pkl', 'custom_checkpoint_1.pkl', 'custom_checkpoint_2.pkl')
            older = root / 'work_dir' / run / 'checkpoint-10000'
            newer = root / 'work_dir' / run / 'checkpoint-20000'
            for n in names: R.write_text(older / n, 'fixture')
            for n in names[:2]: R.write_text(newer / n, 'fixture')
            self.assertEqual(R.checkpoint_for_resume(root, run), older)

    def test_committed_candidate_prevents_rolling_back_recorded_grid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); run = cases_for('s1')[0].run_id
            names = ('model.safetensors', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl',
                     'custom_checkpoint_0.pkl', 'custom_checkpoint_1.pkl', 'custom_checkpoint_2.pkl')
            old = root/'work_dir'/run/'checkpoint-10000'
            candidate = root/'work_dir'/run/'candidates/step-12120'
            for p in (old, candidate):
                for name in names: R.write_text(p/name, 'fixture')
            self.assertEqual(R.checkpoint_for_resume(root, run), old)
            R.write_json(candidate/'mix20h_evaluation_complete.json', dict(
                run_id=run, step=12120, checkpoint_sha256=R.digest(candidate/'model.safetensors')))
            self.assertEqual(R.checkpoint_for_resume(root, run), candidate)
            R.write_text(candidate/'model.safetensors', 'partial different checkpoint')
            self.assertEqual(R.checkpoint_for_resume(root, run), old)

    def test_checkpoint_selection_uses_completed_updates_not_touch_time(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); run = cases_for('s1')[0].run_id
            R.write_text(root / 'config' / (run + '.yaml'),
                         'eval_epoch: 5\nkdv:\n  candidate_grid_id: GRID1010_50K_v1\n')
            required = ('model.safetensors', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl',
                        'custom_checkpoint_0.pkl', 'custom_checkpoint_1.pkl', 'custom_checkpoint_2.pkl')
            def checkpoint(name):
                p = root / 'work_dir' / run / name
                for f in required:
                    R.write_text(p / f, 'fixture')
                if name.startswith('checkpoint-budget-'):
                    R.write_json(p / 'mix20h_resume.json', {'step': int(name.split('-')[-1])})
                return p
            older = checkpoint('checkpoint-10000')
            newer = checkpoint('checkpoint-20000')
            os.utime(older, (99999999, 99999999))
            os.utime(newer, (100, 100))
            self.assertEqual(R.checkpoint_for_resume(root, run), newer)
            epoch = checkpoint('epoch-125')  # 125 * (1010 / 5) = 25250
            os.utime(epoch, (50, 50))
            self.assertEqual(R.checkpoint_for_resume(root, run), epoch)
            budget = checkpoint('checkpoint-budget-25251')
            os.utime(budget, (25, 25))
            for ambiguous in ('checkpoint-copy-49999', 'epoch-999999', 'checkpoint-50001'):
                checkpoint(ambiguous)
            self.assertEqual(R.checkpoint_for_resume(root, run), budget)

    def test_epoch_without_config_grid_identity_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); run = cases_for('s1')[0].run_id
            required = ('model.safetensors', 'optimizer.bin', 'scheduler.bin', 'random_states_0.pkl',
                        'custom_checkpoint_0.pkl', 'custom_checkpoint_1.pkl', 'custom_checkpoint_2.pkl')
            for f in required:
                R.write_text(root / 'work_dir' / run / 'epoch-125' / f, 'fixture')
            self.assertIsNone(R.checkpoint_for_resume(root, run))

    def test_deadline_between_pairmates_leaves_mate_unstarted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes, inv = self._fixture(root); clock = [1000.]
            with patch.object(R, 'validate_configs', return_value=hashes), \
                 patch.object(R, 'inventory', return_value=inv), patch.object(R.time, 'time', return_value=clock[0]):
                R.prepare(root, 's1', R.iso(clock[0]), False)
            deadline = R.timestamp(R.load_plan(root)['deadline_at_utc'])
            first, mate = list(hashes)[:2]; calls = []
            def command(repo, cmd, log, env=None):
                calls.append(cmd)
                if 'tools/run.sh' in cmd:
                    return 0
                clock[0] = deadline + 1  # first run's official evaluation overran
                R.write_json(root / 'work_dir' / cmd[2] / 'results/mix20h_postrun_status.json',
                             dict(sheet_uploaded=True, official_target_complete=True, exact50k_complete=True))
                return 0
            with patch.object(R, 'validate_configs', return_value=hashes), \
                 patch.object(R, 'gpu_processes', return_value=[]), \
                 patch.object(R, 'run_command', side_effect=command), \
                 patch.object(R.time, 'time', side_effect=lambda: clock[0]):
                R.run_campaign(root, upload=True)
                R.run_campaign(root, upload=True)  # cannot restart or reset the deadline
            state = R.read_json(root / R.CAMP / 'status.json')
            self.assertEqual(state['status'], 'BUDGET_CLOSED')
            self.assertEqual(state['budget_not_started_run_id'], mate)
            self.assertEqual(set(state['runs']), {first})
            self.assertFalse((root / 'work_dir' / mate).exists())
            self.assertEqual(len(calls), 2)  # first train + first postrun only
            self.assertEqual(R.load_plan(root)['deadline_at_utc'], R.iso(deadline))

    def test_resumed_tails_do_not_shrink_full_run_reservation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes, inv = self._fixture(root)
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'inventory', return_value=inv):
                R.prepare(root, 's1', R.iso(), False)
            clock = [R.time.time()]
            def command(repo, cmd, log, env=None):
                clock[0] += 180
                if 'tools/run.sh' not in cmd:
                    R.write_json(root/'work_dir'/cmd[2]/'results/mix20h_postrun_status.json', {'sheet_uploaded': True})
                return 0
            with patch.object(R, 'validate_configs', return_value=hashes), patch.object(R, 'gpu_processes', return_value=[]), patch.object(R, 'checkpoint_for_resume', return_value=Path('/fixture/checkpoint-49000')), patch.object(R, 'run_command', side_effect=command), patch.object(R.time, 'time', side_effect=lambda: clock[0]):
                R.run_campaign(root)
            self.assertEqual(R.read_json(root/R.CAMP/'budget.json')['end_to_end_hours'], [])


if __name__ == '__main__':
    unittest.main()
