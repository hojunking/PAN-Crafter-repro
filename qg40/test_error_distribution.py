import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from qg40.error_distribution import measure_error_distribution, write_error_distribution
from qg40.test_reference_parity import Dataset, Teacher


class ErrorDistributionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model, self.train, self.val = Teacher(), Dataset(), Dataset()
        self.ids = np.asarray([1, 3, 0, 2])

    def measure(self, **kwargs):
        return measure_error_distribution(self.model, self.train, self.val, self.ids, .2,
            raw_calibration_median=.2, device='cpu', batch_size=2, synthetic_test=True, **kwargs)

    def test_full_pixel_quantiles_per_band_bias_texture_and_tau_unchanged(self):
        report = self.measure()
        self.assertEqual(report['status'], 'MEASURED')
        self.assertEqual(report['splits']['train_calibration']['actual_samples'], 4)
        self.assertEqual(report['splits']['validation']['actual_samples'], 20)
        expected = []
        for i in self.ids:
            gt, _, ms, lp, pan, _ = self.train.base(i)
            y = self.model(pan[None], ms[None], lp[None])['y'][0].detach()
            expected.append((y-gt).abs().mean(0).numpy())
        values = report['splits']['train_calibration']['all']['error_quantiles']
        np.testing.assert_allclose(list(values.values()), np.percentile(np.asarray(expected), [1, 5, 25, 50, 75, 95, 99]))
        for split in report['splits'].values():
            self.assertEqual(split['low_texture']['samples'] + split['high_texture']['samples'], split['actual_samples'])
        self.assertEqual(report['tau_R'], .2)
        self.assertFalse(report['tau_refitted'])
        self.assertFalse(report['tau_floor_used'])
        self.assertTrue(self.model.training)
        self.assertTrue(self.model.scale.requires_grad)

    def test_bounded_budget_labels_partial_counts_in_both_domains(self):
        with patch('qg40.error_distribution.check_deadline', side_effect=[None, None, TimeoutError('budget')]):
            report = self.measure()
        self.assertEqual(report['status'], 'PARTIAL_BUDGET')
        self.assertEqual(report['splits']['train_calibration']['actual_samples'], 2)
        self.assertEqual(report['splits']['validation']['actual_samples'], 2)

    def test_p1_model_failure_writes_nonblocking_receipt(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(self.model, 'forward', side_effect=ValueError('diagnostic fault')):
            path = Path(temp) / 'report.json'
            report = write_error_distribution(path, self.model, self.train, self.val, self.ids, .2,
                raw_calibration_median=.2, device='cpu', synthetic_test=True)
            self.assertEqual(report['status'], 'DIAGNOSTIC_FAILED')
            self.assertTrue(path.is_file())
            self.assertFalse(report['p0_gate'])
            self.assertEqual(report['tau_R'], .2)

    def test_no_time_left_records_zero_samples_not_fake_pass(self):
        report = self.measure(deadline='2000-01-01T00:00:00Z')
        self.assertEqual(report['status'], 'PARTIAL_BUDGET')
        self.assertTrue(all(r['actual_samples'] == 0 for r in report['splits'].values()))


if __name__ == '__main__':
    unittest.main()
