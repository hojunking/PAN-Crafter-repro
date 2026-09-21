import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import yaml

from l100 import controller as c
from l100.common import atomic_json, camp, locked, object_sha
from l100.plan import blocks_for, build_config, case_for
from l100.policy import CampaignWindow, evaluate_admission, estimate_block_hours


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.server = 's4'; self.case = case_for('L100I1-T03')
        self.folder = camp(self.root, self.server)
        self.window = CampaignWindow(dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1))
        atomic_json(self.root / 'work_dir/_l100/campaign_window.json', self.window.to_dict())

    def state(self):
        return c._state(self.root, self.server)

    def bound_config(self, case=None):
        case = case or self.case
        cfg = build_config(case)
        data = self.folder / 'dataset_manifest.json'; atomic_json(data, {'example': True})
        cfg['l100'].update(dataset_manifest=str(data), runtime_policy_sha256='a' * 64)
        path = self.root / 'work_dir' / case.run_id / 'meta/config.resolved.yaml'
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(yaml.safe_dump(cfg))
        return path

    def completed_artifacts(self, case=None):
        case = case or self.case; path = self.bound_config(case); wd = path.parent.parent
        atomic_json(wd / 'meta/training_start_manifest.json', dict(campaign_id=c.CAMPAIGN_ID,
            run_id=case.run_id, started_at_utc=self.window.t0_utc.isoformat()))
        atomic_json(wd / 'meta/training_status.json', dict(training_complete=True, actual_updates=case.updates))
        final = wd / 'candidates' / str(case.updates); final.mkdir(parents=True)
        (final / 'model.safetensors').write_bytes(b'test-only-placeholder')
        atomic_json(final / 'identity.json', dict(saved_at_utc=(self.window.t0_utc + dt.timedelta(minutes=30)).isoformat()))
        return path, wd

    def test_shared_clock_cannot_reset(self):
        with self.assertRaises(ValueError):
            c.write_window(self.root, dt.datetime.now(dt.timezone.utc).isoformat())
        self.assertEqual(c.shared_window(self.root), self.window)

    def test_start_does_not_hold_child_runner_lock(self):
        def spawn(*args, **kwargs):
            with locked(self.folder / 'runner.lock'):
                pass
            return Mock(pid=os.getpid())
        with patch.object(c, 'verify_sources'), patch.object(c, 'apply_runtime_policy'), \
                patch.object(c, 'source_identity', return_value={'test': True}), \
                patch.object(c.subprocess, 'Popen', side_effect=spawn) as popen:
            first = c.start(self.root, self.server)
            second = c.start(self.root, self.server)
        self.assertEqual(first['status'], 'SUBMITTED')
        self.assertEqual(second['status'], 'ALREADY_SUBMITTED')
        popen.assert_called_once()

    def test_start_existing_runner_returns_without_spawn(self):
        with patch.object(c, 'verify_sources'), patch.object(c, 'apply_runtime_policy'), \
                patch.object(c, 'source_identity', return_value={'test': True}), \
                patch.object(c.subprocess, 'Popen') as popen, locked(self.folder / 'runner.lock'):
            result = c.start(self.root, self.server)
        self.assertEqual(result['status'], 'ALREADY_RUNNING'); popen.assert_not_called()

    def test_dry_start_does_not_register_or_launch(self):
        before = sorted(str(p) for p in self.root.rglob('*'))
        with patch.object(c, 'apply_runtime_policy') as runtime, patch.object(c.subprocess, 'Popen') as popen:
            result = c.start(self.root, self.server, dry_run=True)
        self.assertFalse(result['training_started']); runtime.assert_not_called(); popen.assert_not_called()
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob('*')))

    def test_at_18h_runs_evaluation_closeout_not_preflight(self):
        late = CampaignWindow(dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=18.5))
        atomic_json(self.root / 'work_dir/_l100/campaign_window.json', late.to_dict())
        with patch.object(c, 'verify_registration'), patch.object(c, '_command') as command, \
                patch.object(c, '_closeout', return_value=7) as closeout:
            self.assertEqual(c.run(self.root, self.server), 7)
        command.assert_not_called(); closeout.assert_called_once()

    def test_saved_complete_endpoint_never_reenters_optimizer(self):
        cfg_path, wd = self.completed_artifacts()
        actions = []
        def command(root, args, log, deadline):
            actions.append(args[0])
            self.assertNotEqual(args[0], 'train')
            atomic_json(wd / 'official/postrun_status.json', dict(endpoint_complete=True, actual_updates=self.case.updates))
            return 0, .01
        state = self.state()
        with patch.object(c, '_resolve_config', return_value=cfg_path), patch.object(c, '_recover_final'), \
                patch.object(c, '_command', side_effect=command), patch.object(c, '_ensure_calibration'):
            c._run_case(self.root, self.server, self.case, state, self.window)
        self.assertEqual(actions, ['endpoint'])
        self.assertTrue(state['runs'][self.case.run_id]['endpoint_complete'])
        self.assertTrue((self.folder / 'events' / self.case.run_id / 'teacher_started.json').is_file())
        self.assertTrue((self.folder / 'events' / self.case.run_id / 'teacher_complete.json').is_file())

    def test_closeout_restarts_missing_local_calibration_after_endpoint(self):
        self.completed_artifacts()
        state = self.state()
        state['runs'][self.case.run_id] = dict(endpoint_complete=True, calibrated=False)
        with patch('l100.resources.idle_evidence', return_value={'idle': True}), \
                patch.object(c, '_recover_final'), patch.object(c, '_run_case') as case_runner, \
                patch.object(c, '_postrun'):
            c._closeout(self.root, self.server, state, self.window)
        case_runner.assert_called_once_with(self.root, self.server, self.case, state, self.window)

    def test_weights_alone_do_not_authorize_endpoint(self):
        cfg_path, wd = self.completed_artifacts()
        atomic_json(wd / 'meta/training_status.json', dict(training_complete=False, actual_updates=99900))
        with patch.object(c, '_resolve_config', return_value=cfg_path), patch.object(c, '_recover_final'), \
                patch.object(c, '_command') as command:
            with self.assertRaisesRegex(ValueError, 'full-state resume'):
                c._run_case(self.root, self.server, self.case, self.state(), self.window)
        command.assert_not_called()

    def test_zero_exit_without_endpoint_proof_is_not_completion(self):
        cfg_path, _wd = self.completed_artifacts()
        with patch.object(c, '_resolve_config', return_value=cfg_path), patch.object(c, '_recover_final'), \
                patch.object(c, '_command', return_value=(0, .01)), patch.object(c, '_ensure_calibration'):
            with self.assertRaisesRegex(RuntimeError, 'endpoint validation'):
                c._run_case(self.root, self.server, self.case, self.state(), self.window)

    def test_partial_time_limit_is_not_an_integrity_failure(self):
        cfg_path = self.bound_config()
        with patch.object(c, '_resolve_config', return_value=cfg_path), patch.object(c, '_recover_final'), \
                patch.object(c, '_command', return_value=(75, .01)):
            state = self.state()
            with self.assertRaises(c.TrainingTimeLimit):
                c._run_case(self.root, self.server, self.case, state, self.window)
        self.assertEqual(state['runs'][self.case.run_id]['status'], 'PARTIAL_TIME_LIMIT')

    def test_expected_pause_reaches_partial_closeout(self):
        with patch.object(c, 'verify_registration'), \
                patch('l100.resources.idle_evidence', return_value={'idle': True}), \
                patch('l100.resources.assess_block', return_value={'allowed': True}), \
                patch.object(c, '_command', return_value=(0, .01)), \
                patch.object(c, '_run_case', side_effect=c.TrainingTimeLimit('saved pause')), \
                patch.object(c, '_closeout', return_value=1) as closeout:
            self.assertEqual(c.run(self.root, self.server), 1)
        self.assertEqual(closeout.call_args.args[2]['status'], 'PARTIAL_TIME_LIMIT')

    def test_still_preserving_trainer_never_starts_gpu_closeout(self):
        with patch.object(c, 'verify_registration'), \
                patch('l100.resources.idle_evidence', return_value={'idle': True}), \
                patch('l100.resources.assess_block', return_value={'allowed': True}), \
                patch.object(c, '_command', return_value=(0, .01)), \
                patch.object(c, '_run_case', side_effect=c.OwnTrainerStillPreserving('saving')), \
                patch.object(c, '_closeout') as closeout:
            self.assertEqual(c.run(self.root, self.server), 1)
        closeout.assert_not_called()
        self.assertEqual(self.state()['status'], 'WAIT_OWN_TRAINER_PRESERVING')

    def test_timeout_requests_checkpoint_without_forced_trainer_kill(self):
        process = Mock()
        process.wait.side_effect = c.subprocess.TimeoutExpired('train', 1)
        with patch.object(c.subprocess, 'Popen', return_value=process):
            with self.assertRaises(c.OwnTrainerStillPreserving):
                c._command(self.root, ['train'], self.folder / 'test.log',
                           self.window.train_finish_utc.isoformat())
        process.terminate.assert_called_once(); process.kill.assert_not_called()

    def test_actual_start_event_is_not_reset_on_resume(self):
        _cfg, wd = self.completed_artifacts(); row = {}
        c._training_events(self.root, self.server, self.case, row)
        saved = (self.folder / 'events' / self.case.run_id / 'teacher_started.json').read_bytes()
        c._training_events(self.root, self.server, self.case, row)
        self.assertEqual(saved, (self.folder / 'events' / self.case.run_id / 'teacher_started.json').read_bytes())
        self.assertEqual(row['started_at_utc'], self.window.t0_utc.isoformat())

    def test_evaluation_debt_uses_correct_postrun_status_path(self):
        wd = self.root / 'work_dir' / self.case.run_id
        for step in (2020, 4040):
            path = wd / 'candidates' / str(step); path.mkdir(parents=True)
            (path / 'model.safetensors').write_bytes(b'x')
        self.assertAlmostEqual(c.evaluation_debt(self.root, self.server), 2 / 50 * 1.15)
        atomic_json(wd / 'official/postrun_status.json', dict(official_complete=True))
        self.assertEqual(c.evaluation_debt(self.root, self.server), 0.)

    def test_evaluation_debt_uses_slower_measured_eval(self):
        wd = self.root / 'work_dir' / self.case.run_id
        path = wd / 'candidates/4040'; path.mkdir(parents=True)
        (path / 'model.safetensors').write_bytes(b'x')
        atomic_json(wd / 'official/raw_grid.json', dict(records=[dict(update=2020, seconds=3600)]))
        self.assertAlmostEqual(c.evaluation_debt(self.root, self.server), 1.15)

    def test_verified_upload_does_not_rerun_evaluation(self):
        wd = self.root / 'work_dir' / self.case.run_id
        atomic_json(wd / 'official/postrun_status.json', dict(official_complete=True, sheet_uploaded=True))
        atomic_json(wd / 'official/upload_receipt.json', dict(readback_verified=True))
        with patch.object(c, '_command') as command:
            c._postrun(self.root, self.server, self.case, self.state(), self.window)
        command.assert_not_called()

    def test_partial_closeout_calls_no_training(self):
        cfg_path = self.bound_config(); wd = cfg_path.parent.parent
        atomic_json(wd / 'meta/training_start_manifest.json', dict(run_id=self.case.run_id))
        atomic_json(wd / 'meta/training_status.json', dict(training_complete=False, actual_updates=1000))
        state = self.state(); state['runs'][self.case.run_id] = dict(training_started=True)
        with patch('l100.resources.idle_evidence', return_value={'idle': True}), \
                patch.object(c, '_recover_final'), patch.object(c, '_command', return_value=(0, .01)) as command:
            c._closeout(self.root, self.server, state, self.window)
        self.assertEqual(command.call_args.args[1][0], 'partial')
        self.assertEqual(state['status'], 'PARTIAL_TIME_LIMIT')

    def test_unstarted_core_cases_never_claim_queue_finished(self):
        state = self.state()
        with patch('l100.resources.idle_evidence', return_value={'idle': True}):
            self.assertEqual(c._closeout(self.root, self.server, state, self.window), 1)
        self.assertEqual(state['status'], 'PARTIAL_QUEUE')
        self.assertTrue(state['unstarted_run_ids'])

    def test_upload_pending_never_claims_queue_finished(self):
        _cfg, wd = self.completed_artifacts()
        atomic_json(wd / 'official/postrun_status.json', dict(official_complete=True))
        state = self.state()
        state['runs'][self.case.run_id] = dict(training_started=True, endpoint_complete=True,
            calibrated=True, official_complete=True, upload_pending=True)
        with patch('l100.resources.idle_evidence', return_value={'idle': True}), \
                patch.object(c, 'cases_for', return_value=(self.case,)), \
                patch.object(c, '_recover_final'), patch.object(c, '_postrun'):
            self.assertEqual(c._closeout(self.root, self.server, state, self.window), 1)
        self.assertEqual(state['status'], 'UPLOAD_PENDING')

    def test_authorize_requires_active_controller_state(self):
        path = self.bound_config()
        release = {'source': 'test'}
        atomic_json(self.folder / 'preflight.json', dict(complete=True, source_identity=release,
            dataset_manifest_sha256=object_sha({'example': True})))
        block = blocks_for(self.server)[0]
        receipt = evaluate_admission(self.window, self.window.t0_utc, block, estimate_block_hours(block))
        atomic_json(self.folder / 'admissions' / (block.block_id + '.json'), receipt)
        with patch.object(c, 'verify_registration'), patch.object(c, 'source_identity', return_value=release):
            with self.assertRaisesRegex(ValueError, 'not authorized'):
                c.authorize_train(self.root, self.server, path)
            state = self.state(); state['status'] = 'RUNNING'
            state['runs'][self.case.run_id] = dict(status='TRAINING')
            state['blocks'][block.block_id] = dict(status='ADMITTED')
            c._save(self.root, self.server, state)
            self.assertEqual(c.authorize_train(self.root, self.server, path), self.case)


if __name__ == '__main__':
    unittest.main()
