import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from panda_rb.common import ROOT, atomic_json, read
from panda_rb.plan import PLAN_DIR, run_dir, training_runs
from panda_rb.controller import run
from panda_rb.evaluation import seal
from panda_rb.reporting import stats, summarize, package


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root / PLAN_DIR).parent.mkdir(parents=True)
        (self.root / PLAN_DIR).symlink_to(ROOT / PLAN_DIR, target_is_directory=True)
        self.calls = []

    def tearDown(self): self.temp.cleanup()

    def _run(self, train_override=None):
        def train(run_id, *args, **kwargs):
            self.calls.append(('train', run_id))
            if train_override: return train_override(run_id)
            atomic_json(run_dir(run_id, self.root) / 'checkpoints/selection_manifest.json', {'test': True})
        def native(run_id, *args, **kwargs):
            self.calls.append(('native', run_id)); return seal(dict(complete=True, file_hashes={}))
        def stress(run_id, binding, mode, *args, **kwargs):
            self.calls.append((mode, run_id)); return seal(dict(complete=True, file_hashes={}))
        with contextlib.ExitStack() as stack:
            for target, value in [('panda_rb.common.docker_required', lambda: None),
                                  ('panda_rb.preflight.prepare', lambda *a, **k: {'passed': True}),
                                  ('panda_rb.training.train_run', train),
                                  ('panda_rb.evaluation.evaluate_native', native),
                                  ('panda_rb.evaluation.evaluate_stress', stress),
                                  ('panda_rb.reporting.summarize', lambda **k: dict(completed_students=8, completed_curves=16, failed_students=0))]:
                stack.enter_context(patch(target, value))
            return run('s1', root=self.root)

    def test_finite_order_and_stop(self):
        self._run()
        expected = []
        for row in training_runs('s1'):
            expected.extend((stage, row['run_id']) for stage in ('train', 'native', 'A_ON', 'A_ZERO_INFERENCE_ONLY'))
        self.assertEqual(self.calls, expected)
        self.assertEqual(read(self.root / 'work_dir/_panda_rb/20260928/B01/control/s1/status.json')['state'], 'STOP_FOR_REVIEW')
        self.assertFalse((self.root / 'work_dir/_panda_rb/local_owner.json').exists())

    def test_numerical_failure_never_replaces_seed(self):
        first = training_runs('s1')[0]['run_id']
        def numerical(run_id):
            if run_id == first: raise FloatingPointError('synthetic NaN')
        self._run(numerical)
        self.assertEqual([stage for stage, ident in self.calls if ident == first], ['train'])
        self.assertEqual(len([1 for stage, _ in self.calls if stage == 'train']), 8)
        self.assertEqual(read(run_dir(first, self.root) / 'meta/controller_failure.json')['kind'], 'NUMERICAL_FAILURE')

    def test_technical_failure_stops_without_replacement(self):
        def broken(_): raise OSError('synthetic disk error')
        with self.assertRaises(OSError): self._run(broken)
        self.assertEqual(len(self.calls), 1)
        with self.assertRaisesRegex(RuntimeError, 'retry-technical'): self._run()

    def test_signal_pause_never_evaluates_partial_checkpoint(self):
        result = self._run(lambda _: 75)
        self.assertEqual(result['state'], 'PAUSED_SIGNAL')
        self.assertEqual(len(self.calls), 1)

    def test_no_s2_admission(self):
        with self.assertRaises(ValueError): run('s2', root=self.root)


class ReportingTests(unittest.TestCase):
    def test_sample_std_and_no_nan_drop(self):
        self.assertEqual(stats([1, 3]), dict(n=2, mean=2, sample_std=2**.5))
        self.assertEqual(stats([]), dict(n=0, mean=None, sample_std=None))
        self.assertIsNone(stats([1])['sample_std'])
        with self.assertRaises(ValueError): stats([1, float('nan')])

    def test_empty_report_is_not_complete_or_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / PLAN_DIR).parent.mkdir(parents=True)
            (root / PLAN_DIR).symlink_to(ROOT / PLAN_DIR, target_is_directory=True)
            report = summarize(root, ['s1'])
            self.assertEqual(report['completed_students'], 0)
            self.assertEqual(report['expected_students'], 8)
            self.assertFalse(report['complete'])
            with self.assertRaisesRegex(ValueError, 'No measured'): package(root, ['s1'], root / 'fake.zip')
            self.assertFalse((root / 'fake.zip').exists())


if __name__ == '__main__': unittest.main()
