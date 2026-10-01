"""Mock-only campaign integration checks: no GPU, real metrics or Sheet calls."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pan_shared.common import canonical_sha, file_sha, read_json
from pan_shared.evaluate import PROTOCOL
from pan_shared.registry import build_registry
from pan_shared.runtime import CampaignRuntime, owned, freeze
from pan_shared.safety import SafetyStop


class RuntimeTests(unittest.TestCase):
    def make_runtime(self, root):
        instance = CampaignRuntime.__new__(CampaignRuntime)
        instance.root, instance.device, instance.credentials = root, 'cpu', None
        instance.manifest = dict(source_sha256='a' * 64, environment_sha256='b' * 64,
                                 dataset_sha256='c' * 64, subset_sha256='d' * 64)
        instance.dataset_manifest = {'sensors': {sensor: {'splits': {'val': {'count': count}}}
            for sensor, count in [('WV3', 1080), ('GF2', 2201), ('QB', 1905)]}}
        instance.subsets = {'subset_sha256': 'd' * 64, 'sensors': {sensor: {
            'train_probe_ids': list(range(128)), 'val_probe_ids': list(range(128))}
            for sensor in ('WV3', 'GF2', 'QB')}}
        instance.dataset = Mock(side_effect=lambda sensor, split: list(range(2201)))
        instance.try_sync = Mock(return_value={'status': 'OUTBOX_ONLY'})
        return instance

    def make_trainer_checkpoint_reports(self, root, shared=False):
        run = next(row for row in build_registry()['runs'] if row['repeat'] == 2 and row['case_id'] == ('C00' if shared else 'S01'))
        weights = root / 'fixture_weights.bin'
        weights.write_bytes(b'not-a-real-model; benchmark invocation is mocked')
        checkpoint = {'model_path': str(weights), 'checkpoint_sha256': file_sha(weights),
                      'completed_step': 50000, 'exposures': {'WV3': {'samples': 2400000}}}
        identity = dict(run, architecture_sha256='e' * 64, source_sha256='a' * 64,
                        data_sha256='c' * 64, subset_sha256='d' * 64)
        trainer = SimpleNamespace(config=run, identity=identity, sensors=tuple(run['sensors']),
                                  step=50000, microbatch=48, model=object(), sampler=Mock())
        trainer.sampler.exposures.return_value = checkpoint['exposures']
        trainer.state = {'times': dict(training=2.0, validation=1.0, benchmark=0.0, data_wait=.5, checkpoint=.2),
                         'candidate_count': 6}
        reports = {sensor: {'identity': {'checkpoint_sha256': checkpoint['checkpoint_sha256'], 'evaluator_sha256': 'f' * 64},
            'protocol': copy.deepcopy(PROTOCOL),
            'rr': {'n_scenes': 20, 'ergas': 2.0, 'scc': .98, 'sam': 2.1, 'psnr': 35., 'ssim': .99,
                   'q8' if sensor == 'WV3' else 'q4': .95, 'rmse': 4.0, 'cc': .99},
            'fr': {'n_scenes': 20, 'hqnr': .95, 'd_s': .02, 'd_lambda': .03, 'jqm': .9}}
            for sensor in trainer.sensors}
        return trainer, checkpoint, reports

    def benchmark(self, runtime, trainer, checkpoint, reports):
        with patch('pan_shared.evaluate.benchmark_checkpoint', return_value=reports), \
             patch('pan_shared.evaluate.evaluator_sha', return_value='f' * 64), \
             patch('pan_shared.evaluate.profile_model', return_value={'scope': 'mock-only', 'params_m': 2.1}), \
             patch('pan_shared.sheets.SheetsUploader.enqueue') as enqueue:
            result = runtime.benchmark(trainer, checkpoint, 'EXACT')
            return result, enqueue

    def test_full_validation_rejects_partial_sensor_count(self):
        with tempfile.TemporaryDirectory() as name:
            runtime = self.make_runtime(Path(name))
            trainer = SimpleNamespace(sensors=('WV3', 'GF2', 'QB'), model=object(), microbatch=48)
            report = {'complete': True, 'sensors': {
                'WV3': {'n_samples': 1080, 'mean_l1': .1}, 'GF2': {'n_samples': 2200, 'mean_l1': .1},
                'QB': {'n_samples': 1905, 'mean_l1': .1}}}
            with patch('pan_shared.evaluate.validation', return_value=report):
                with self.assertRaisesRegex(ValueError, 'Partial'):
                    runtime.validate(trainer, 'validation')
            report['sensors']['GF2']['n_samples'] = 2201
            with patch('pan_shared.evaluate.validation', return_value=report):
                self.assertIs(runtime.validate(trainer, 'validation'), report)

    def test_probes_are_fixed_unaugmented_nonselector(self):
        with tempfile.TemporaryDirectory() as name:
            runtime = self.make_runtime(Path(name))
            trainer = SimpleNamespace(sensors=('WV3',), model=object(), microbatch=48)
            with patch('pan_shared.evaluate.validation', return_value={'complete': True}) as evaluate:
                result = runtime.validate(trainer, 'probe')
            self.assertFalse(result['selector_candidate'])
            self.assertEqual(evaluate.call_count, 2)
            for call in evaluate.call_args_list:
                self.assertEqual(len(call.args[1]['WV3']), 128)

    def test_immutable_observation_retry_replays_timestamp_and_cost(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            runtime = self.make_runtime(root)
            trainer, checkpoint, reports = self.make_trainer_checkpoint_reports(root)
            result, _ = self.benchmark(runtime, trainer, checkpoint, reports)
            path = Path(result['WV3']['observation'])
            original_bytes = path.read_bytes()
            trainer.state['times']['benchmark'] = 800
            second, enqueue = self.benchmark(runtime, trainer, checkpoint, reports)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(result, second)
            self.assertEqual(enqueue.call_args.args[0], read_json(path))

    def test_benchmark_rejects_partial_sensor_reports_before_writes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            runtime = self.make_runtime(root)
            trainer, checkpoint, reports = self.make_trainer_checkpoint_reports(root, shared=True)
            reports.pop('GF2')
            with self.assertRaises((ValueError, SafetyStop)):
                self.benchmark(runtime, trainer, checkpoint, reports)
            self.assertFalse((root / 'observations').exists())

    def test_benchmark_rejects_wrong_report_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            runtime = self.make_runtime(root)
            trainer, checkpoint, reports = self.make_trainer_checkpoint_reports(root)
            reports['WV3']['identity']['checkpoint_sha256'] = '0' * 64
            with self.assertRaises((ValueError, SafetyStop)):
                self.benchmark(runtime, trainer, checkpoint, reports)
            self.assertFalse((root / 'observations').exists())

    def test_corrupted_existing_metrics_do_not_silently_replay(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            runtime = self.make_runtime(root)
            trainer, checkpoint, reports = self.make_trainer_checkpoint_reports(root)
            result, _ = self.benchmark(runtime, trainer, checkpoint, reports)
            path = Path(result['WV3']['observation'])
            damaged = read_json(path)
            damaged['rr']['ergas'] += 1
            path.write_text(json.dumps(damaged))
            with self.assertRaises((ValueError, SafetyStop)):
                self.benchmark(runtime, trainer, checkpoint, reports)

    def test_sync_successful_list_receipts_are_logged(self):
        with tempfile.TemporaryDirectory() as name:
            runtime = self.make_runtime(Path(name))
            del runtime.try_sync
            runtime.credentials = 'not-read-mocked-credentials'
            expected = [{'status': 'VERIFIED', 'result_id': 'a' * 64}]
            with patch('pan_shared.runtime.sync_sheet', return_value=expected):
                result = runtime.try_sync()
            self.assertIsInstance(result, dict)
            self.assertTrue((runtime.root / 'sheet_sync.jsonl').exists())

    def test_owned_symlink_escape_and_immutable_conflict(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'work').mkdir(); (root / 'outside').mkdir()
            (root / 'work' / 'escape').symlink_to(root / 'outside', target_is_directory=True)
            with self.assertRaises(SafetyStop):
                owned(root / 'work', 'escape', 'report.json')
            path = root / 'work' / 'asset.json'
            freeze(path, {'x': 1})
            with self.assertRaises(SafetyStop):
                freeze(path, {'x': 2})
            self.assertEqual(read_json(path), {'x': 1})
