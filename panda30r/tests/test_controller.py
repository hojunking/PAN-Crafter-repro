"""Bounded controller contracts using mocks: no datasets, GPU, Sheets or training."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from panda30r import plan
from panda30r.runtime import Campaign, BudgetExpired, improvement, recover_trace
from ta2.evaluation import file_sha


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch('panda30r.runtime.code_pin', return_value={'test': 'a' * 64}), \
                patch('panda30r.runtime.evaluator_identity', return_value={'test': 'b' * 64}):
            self.campaign = Campaign('s3', output=self.temp.name, device='cpu')

    def roomy_disk(self):
        return patch('panda30r.runtime.shutil.disk_usage', return_value=SimpleNamespace(free=10**12))

    def test_durable_elapsed_includes_downtime_and_never_rolls_back(self):
        c = self.campaign
        c.state['started_unix'] = 100
        c.initial_elapsed, c.tick = 30, 500
        c.state['elapsed_seconds'] = 40
        with patch('panda30r.runtime.time.time', return_value=180), \
                patch('panda30r.runtime.time.monotonic', return_value=510):
            self.assertEqual(c.elapsed(), 80)
        # Wall clock rollback cannot extend the remaining budget.
        with patch('panda30r.runtime.time.time', return_value=120), \
                patch('panda30r.runtime.time.monotonic', return_value=520):
            self.assertEqual(c.elapsed(), 80)
        with patch('panda30r.runtime.time.time', return_value=130), \
                patch('panda30r.runtime.time.monotonic', return_value=560):
            self.assertEqual(c.elapsed(), 90)

    def test_reconstruction_of_campaign_keeps_original_clock(self):
        c = self.campaign
        c.state['started_unix'] = 100
        c.state['elapsed_seconds'] = 250
        c.state_path.write_text(json.dumps(c.state))
        with patch('panda30r.runtime.code_pin', return_value={'test': 'a' * 64}), \
                patch('panda30r.runtime.evaluator_identity', return_value={'test': 'b' * 64}), \
                patch('panda30r.runtime.time.time', return_value=500):
            again = Campaign('s3', output=self.temp.name, device='cpu')
        self.assertEqual(again.state['started_unix'], 100)
        self.assertEqual(again.initial_elapsed, 400)

    def test_closeout_and_stop_request_abort_without_mutations(self):
        c = self.campaign
        with patch.object(c, 'elapsed', return_value=30*3600-1800):
            with self.assertRaises(BudgetExpired):
                c.check()
            c.check(reserve=False)
        c.stop_requested = True
        with patch.object(c, 'elapsed', return_value=0):
            with self.assertRaisesRegex(BudgetExpired, 'STOP_REQUESTED'):
                c.check(reserve=False)

    def test_admission_never_shrinks_a_seed_pair_to_fit(self):
        c = self.campaign
        group = next(g for g in plan.groups('s3') if g['name'] == 'M_G')
        with patch.object(c, 'estimate', return_value=100), \
                patch.object(c, 'remaining_for_work', return_value=230), self.roomy_disk():
            result = c.admit([group], name='PAIR')
        self.assertEqual(result['pending_runs'], 2)
        self.assertEqual(result['estimated_seconds'], 240)
        self.assertEqual(result['status'], 'SKIPPED_BUDGET_OR_DISK')
        self.assertFalse(c.state['runs'])

    def test_completed_seed_excluded_but_remaining_seed_still_admitted(self):
        c = self.campaign
        group = next(g for g in plan.groups('s3') if g['name'] == 'M_G')
        cfg = plan.resolve_case('s3', 'M_G', 'R1')
        c.state['runs'].append({'run_id': cfg['run_id'], 'status': 'COMPLETED'})
        with patch.object(c, 'estimate', return_value=100), \
                patch.object(c, 'remaining_for_work', return_value=130), self.roomy_disk():
            result = c.admit([group], name='PAIR_REMAINDER')
        self.assertEqual(result['pending_runs'], 1)
        self.assertEqual(result['estimated_seconds'], 120)
        self.assertEqual(result['status'], 'ADMITTED')

    def test_calibration_reserved_once_per_donor_not_once_per_seed(self):
        c = self.campaign
        group = next(g for g in plan.groups('s3') if g['name'] == 'M_Q')
        with patch.object(c, 'estimate', return_value=100), \
                patch.object(c, 'calibration_overhead', return_value=600) as overhead, \
                patch.object(c, 'remaining_for_work', return_value=10000), self.roomy_disk():
            result = c.admit([group], name='Q_PAIR')
        overhead.assert_called_once_with('M')
        self.assertEqual(result['calibration_seconds'], 600)
        self.assertEqual(result['estimated_seconds'], 960)

    def test_disk_reserve_rejects_pair_even_with_time_available(self):
        c = self.campaign
        group = next(g for g in plan.groups('s3') if g['name'] == 'M_G')
        with patch.object(c, 'estimate', return_value=1), \
                patch.object(c, 'remaining_for_work', return_value=10000), \
                patch('panda30r.runtime.shutil.disk_usage', return_value=SimpleNamespace(free=1024)):
            result = c.admit([group], name='DISK')
        self.assertEqual(result['status'], 'SKIPPED_BUDGET_OR_DISK')

    def test_already_timed_pilot_does_not_build_another_trainer(self):
        c = self.campaign
        cfg = plan.resolve_case('s3', 'M_Q')
        c.state['timing']['stage2_q'] = {'update_p90': .1}
        with patch.object(c, 'build', side_effect=AssertionError('pilot replayed')):
            c.preflight_cases([cfg])

    def test_same_seed_retained_parent_required_and_sha_checked(self):
        c = self.campaign
        cfg = plan.resolve_case('s3', 'T40_EPS', 'R2')
        parent = plan.resolve_case('s3', 'M_Q', 'R2')
        path = c.root / 'retained.pt'
        path.write_bytes(b'SYNTHETIC-NOT-A-MODEL')
        candidate = {'checkpoint_path': str(path), 'file_sha256': file_sha(path)}
        c.state['runs'].append({'run_id': parent['run_id'], 'status': 'COMPLETED', 'directory': str(c.root)})
        with patch.object(c, 'candidate', return_value=candidate) as lookup:
            c.parent(cfg)
        lookup.assert_called_once_with(c.root, 40000)
        self.assertEqual(cfg['parent_binding']['seed'], plan.SEEDS['R2'])
        self.assertEqual(cfg['parent_binding']['step'], 40000)
        path.write_bytes(b'CHANGED')
        with patch.object(c, 'candidate', return_value=candidate):
            with self.assertRaisesRegex(ValueError, 'fork checkpoint changed'):
                c.parent(cfg)

    def test_every_official_step_saved_before_publication_and_latest_updated(self):
        c = self.campaign
        cfg = plan.resolve_case('s3', 'M_G')
        # Synthetic miniature controller exercise only; no scientific trainer built.
        cfg.update(planned_horizon=2, evaluation_steps=[1, 2])
        trainer = SimpleNamespace(updates=0, poisoned=False, config=deepcopy(cfg), stage=object())
        stream = object()
        events = []
        def trace(*args):
            trainer.updates += 1
            return {'loss': 1.}, .001
        def evaluate(t):
            events.append(('evaluate', t.updates))
            return {'state_before': {'completed_update': t.updates}, 'fr': {'metrics': {'HQNR': .95}},
                    'rr': {'metrics': {'ERGAS': 2.1, 'SCC': .98}}, 'elapsed_seconds': .1}
        def save(t, st, path, config, evaluation=None):
            events.append(('save', t.updates))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'SYNTHETIC-NOT-A-MODEL')
            artifact = {'checkpoint_path': str(path), 'file_sha256': file_sha(path),
                        'model_state_sha256': 'a' * 64, 'completed_update': t.updates}
            path.with_suffix('.json').write_text(json.dumps({'artifact': artifact, 'config': config,
                'code_pin': c.code, 'evaluator_pin': c.evaluator_pin, 'evaluation': evaluation}))
            return artifact
        publication_timestamps = []
        def emit(root, config, evaluation, artifact=None, timestamp=None):
            step = evaluation['state_before']['completed_update']
            events.append(('publish', step))
            publication_timestamps.append(timestamp)
            self.assertTrue(Path(artifact['checkpoint_path']).exists())
            # The persisted resume pointer already refers to the same state.
            pointer = Path(artifact['checkpoint_path']).parent / 'latest_resume.json'
            self.assertEqual(json.loads(pointer.read_text())['completed_update'], step)
        c.state['timing']['stage2_gt'] = {'update_p90': .1, 'eval_seconds': .1}
        with patch.object(c, 'build', return_value=(trainer, stream)), \
                patch.object(c, 'trace', side_effect=trace), patch.object(c, 'evaluate', side_effect=evaluate), \
                patch.object(c, '_save', side_effect=save), patch.object(c, 'release'), \
                patch.object(c, 'verify_code'), patch('panda30r.reporting.emit_status'), \
                patch('panda30r.reporting.emit_evaluation', side_effect=emit), self.roomy_disk():
            c.run_case(cfg)
        self.assertEqual(events, [('evaluate',1),('save',1),('publish',1),
                                  ('evaluate',2),('save',2),('publish',2)])
        status = c.completed(cfg)
        self.assertEqual(status['step'], 2)
        selected = json.loads((Path(status['directory']) / 'models/selection.json').read_text())
        self.assertEqual([row['completed_update'] for row in selected['saved']], [1, 2])
        # Equal metrics keep the first selector, but the later checkpoint remains.
        self.assertEqual(selected['best_selected']['completed_update'], 1)
        # Mimic a process dying after publishing the final checkpoint, before
        # recording COMPLETED. Resume must not repeat training or change row ID/date.
        status['status'] = 'RUNNING'
        status['step'] = 1
        with patch.object(c, 'build', return_value=(trainer, stream)), \
                patch.object(c, '_restore', return_value=(trainer, stream)), \
                patch.object(c, 'trace', side_effect=AssertionError('repeated optimizer update')), \
                patch.object(c, 'evaluate', side_effect=AssertionError('repeated evaluation')), \
                patch.object(c, 'release'), patch('panda30r.reporting.emit_status'), \
                patch('panda30r.reporting.emit_evaluation', side_effect=emit), \
                patch('panda30r.evaluation.model_state_hash', return_value='a'*64):
            c.run_case(cfg)
        self.assertEqual(publication_timestamps[-1], publication_timestamps[-2])
        self.assertIsNotNone(publication_timestamps[-1])
        self.assertEqual(c.completed(cfg)['step'], 2)

    def test_orphan_atomic_checkpoint_recovers_evaluation_sidecar(self):
        from panda30r.tests.test_reporting import evaluation
        c = self.campaign
        cfg = plan.resolve_case('s3', 'M_Q')
        directory = c.root / 'runs' / cfg['run_id'] / 'attempt_01'
        models = directory / 'models'
        models.mkdir(parents=True)
        ev = evaluation(cfg, step=1000)
        ev['recorded_at'] = '2026-10-18T01:00:00+00:00'
        for split in ('rr', 'fr'):
            for row in ev[split]['per_scene']:
                row['raw_prediction_sha256'] = 'd' * 64
        state = ev['state_before']['model_state_sha256']
        payload = {'config': cfg, 'pins': {'code': c.code, 'evaluator': c.evaluator_pin},
                   'trainer': {'updates': 1000}, 'model_state_sha256': state, 'evaluation': ev}
        path = models / 'step_0001000.pt'
        torch.save(payload, path)
        artifact = c.resume_artifact(directory, cfg)
        self.assertTrue(artifact['recovered_before_sidecar'])
        self.assertEqual(artifact['completed_update'], 1000)
        trainer = SimpleNamespace(updates=1000, stage=object())
        with patch('panda30r.evaluation.model_state_hash', return_value=state):
            recovered, evidence = c.recover_evaluated_checkpoint(path, trainer, cfg)
        self.assertEqual(evidence['recorded_at'], ev['recorded_at'])
        self.assertEqual(recovered['file_sha256'], file_sha(path))
        self.assertTrue(path.with_suffix('.json').is_file())
        self.assertEqual(c.resume_artifact(directory, cfg)['file_sha256'], artifact['file_sha256'])

    def test_corrupt_latest_orphan_never_falls_back_to_earlier_checkpoint(self):
        c = self.campaign
        cfg = plan.resolve_case('s3', 'M_Q')
        directory = c.root / 'runs' / cfg['run_id'] / 'attempt_01'
        models = directory / 'models'
        models.mkdir(parents=True)
        for step in (1000, 2000):
            payload = {'config': cfg, 'pins': {'code': c.code, 'evaluator': c.evaluator_pin},
                       'trainer': {'updates': step}}
            if step == 2000:
                payload['pins']['evaluator'] = {'foreign': 'c' * 64}
            torch.save(payload, models / f'step_{step:07d}.pt')
        with self.assertRaisesRegex(ValueError, 'Orphan complete checkpoint provenance'):
            c.resume_artifact(directory, cfg)

    def test_crashed_training_trace_prefix_preserves_discarded_work(self):
        directory = self.campaign.root / 'trace-test'
        directory.mkdir()
        path = directory / 'train.jsonl'
        lines = [json.dumps({'step': step, 'seconds_including_IO': .1}) + '\n' for step in (1, 2, 3)]
        original = ''.join(lines) + '{partial'
        path.write_text(original)
        recover_trace(directory, 2)
        self.assertEqual(path.read_text(), ''.join(lines[:2]))
        backups = list(directory.glob('train_before_resume_*.jsonl'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), original)

    def test_remaining_estimate_uses_durable_saved_progress_not_live_counter(self):
        c = self.campaign
        cfg = plan.resolve_case('s3', 'M_Q')
        directory = c.root / 'partial'
        (directory / 'models').mkdir(parents=True)
        (directory / 'models/latest_resume.json').write_text(json.dumps({'completed_update':40000}))
        c.state['runs'].append({'run_id':cfg['run_id'],'status':'RUNNING','step':45000,
                               'directory':str(directory)})
        c.state['timing']['stage2_q'] = {'update_p90': .1, 'eval_seconds': 10., 'checkpoint_seconds': 5.}
        remaining_evaluations = sum(step >= 40000 for step in cfg['evaluation_steps'])
        self.assertEqual(c.estimate(cfg), 10000*.1 + remaining_evaluations*15)

    def test_scientific_gate_uses_ergas_not_scc(self):
        def ev(h,e,scc,step):
            return {'state_before':{'completed_update':step},'fr':{'metrics':{'HQNR':h}},
                    'rr':{'metrics':{'ERGAS':e,'SCC':scc}}}
        before = {'HQNR':.961,'ERGAS':2.2,'SCC':1.,'completed_update':1000}
        row, reasons = improvement(ev(.959,2.1,.5,2000), {'best_selected':before,'best_hqnr':before})
        self.assertIn('HQNR_GATE_ERGAS_SELECTOR', reasons)
        self.assertNotIn('BEST_HQNR_BACKUP', reasons)
        self.assertEqual(row['ERGAS'],2.1)


if __name__ == '__main__':
    unittest.main()
