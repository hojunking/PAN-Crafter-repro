from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from qg40.common import read_json
from qg40.exposure import record_batch, validate_counts, exposure_report
from qg40.p1 import bounded_deadline, record_training_diagnostics


class P1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = torch.nn.Linear(1, 1)
        for p in self.model.parameters():
            p.grad = torch.ones_like(p)
        self.args = dict(update=10000, device='cpu', profile='BASE', reference={},
                        q_cache=None, consistency_weight=1e-4, identity={'source_identity': {'test': True}})

    def invoke(self, capture, sizes):
        with patch('qg40.diagnostics.capture_diagnostics', capture), \
                patch.dict('sys.modules', {'qg40.alignment_diagnostics': SimpleNamespace(capture_alignment_sizes=sizes)}):
            return record_training_diagnostics(self.root, self.model, None, {'train': object()}, **self.args)

    def test_analysis_error_is_incomplete_not_training_gate_and_is_not_replayed(self):
        before = torch.get_rng_state().clone()
        def fail(*args, **kwargs):
            torch.rand(4)
            self.model.eval()
            for p in self.model.parameters():
                p.grad = None
                p.requires_grad_(False)
            raise TimeoutError('bounded analysis expired')
        capture = Mock(side_effect=fail)
        sizes = Mock(side_effect=ValueError('missing P1 scene'))
        self.invoke(capture, sizes)
        doc = read_json(self.root / 'diagnostics/required_10000.json')
        self.assertEqual(doc['status'], 'INCOMPLETE')
        self.assertFalse(doc['branch_admissible'])
        self.assertFalse(doc['base_training_blocked'])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(self.model.training)
        self.assertTrue(all(p.requires_grad and torch.all(p.grad == 1) for p in self.model.parameters()))
        self.invoke(capture, sizes)
        self.assertEqual(capture.call_count, 1)
        self.assertEqual(sizes.call_count, 1)

    def test_measured_arrays_are_checked_on_resume_not_silently_replaced(self):
        capture = Mock(return_value=({'status': 'MEASURED'}, {'x': np.ones(2)}))
        sizes = Mock(return_value={'status': 'MEASURED'})
        self.invoke(capture, sizes)
        self.invoke(capture, sizes)
        self.assertEqual(capture.call_count, 1)
        (self.root / 'diagnostics/update_10000.npz').chmod(0o644)
        (self.root / 'diagnostics/update_10000.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.invoke(capture, sizes)

    def test_mutated_weights_remain_a_fatal_integrity_error(self):
        def mutate(*args, **kwargs):
            with torch.no_grad():
                self.model.weight.add_(1)
            raise ValueError('analysis failure')
        with self.assertRaisesRegex(ValueError, 'mutated model'):
            self.invoke(Mock(side_effect=mutate), Mock(return_value={'status': 'MEASURED'}))

    def test_receipt_io_error_does_not_stop_base_or_publish_evidence(self):
        capture = Mock(return_value=({'status': 'MEASURED'}, {'x': np.ones(2)}))
        sizes = Mock(return_value={'status': 'MEASURED'})
        before = torch.get_rng_state().clone()
        with patch('qg40.p1.atomic_json', side_effect=PermissionError('diagnostics-only read-only')), \
                self.assertLogs('qg40.p1', level='WARNING') as logs:
            elapsed = self.invoke(capture, sizes)
        self.assertGreaterEqual(elapsed, 0)
        self.assertTrue(any('receipt unavailable' in line for line in logs.output))
        self.assertFalse((self.root / 'diagnostics/update_10000.json').exists())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(p.requires_grad and torch.all(p.grad == 1) for p in self.model.parameters()))

    def test_receipt_io_failure_cannot_hide_weight_mutation(self):
        def mutate(*args, **kwargs):
            with torch.no_grad():
                self.model.weight.add_(1)
            raise ValueError('analysis failure')
        with patch('qg40.p1.atomic_json', side_effect=PermissionError('read-only')), \
                self.assertLogs('qg40.p1', level='WARNING'), \
                self.assertRaisesRegex(ValueError, 'mutated model'):
            self.invoke(Mock(), Mock(side_effect=mutate))

    def test_analysis_deadline_never_extends_shared_window(self):
        self.assertEqual(bounded_deadline('2000-01-01T00:00:00Z'), '2000-01-01T00:00:00+00:00')
        with self.assertRaises(ValueError):
            bounded_deadline('2000-01-01T00:00:00')

    def test_exposure_counts_exact_repeats_views_and_resume(self):
        counts = torch.zeros((5, 4), dtype=torch.int64)
        metadata = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
        record_batch(counts, metadata)
        restored = validate_counts(counts, 5, 1, 4)
        record_batch(restored, metadata)
        self.assertEqual(restored.sum().item(), 8)
        self.assertEqual(counts.sum().item(), 4)
        report = exposure_report(restored, 4, 1, 1)
        self.assertEqual(report['dropped_per_epoch'], 1)
        self.assertEqual(report['unseen_base'], 1)
        self.assertEqual(report['per_view_presentations'], [2, 2, 2, 2])
        with self.assertRaises(ValueError):
            validate_counts(restored, 5, 1, 4)


if __name__ == '__main__':
    unittest.main()
