"""Calibration regression against the existing pooled median/AXIS16 math."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from l100.calibration import compute_calibration, calibrate
from qg40.calibration import compute_calibration as existing, select_calibration_indices


class ConstantTeacher(nn.Module):
    bands = 4
    def predict_delta(self, pan, base):
        return torch.zeros(len(pan), 2, device=pan.device)
    def forward(self, pan, ms, lp):
        return {'y': torch.zeros(len(pan), 4, *pan.shape[-2:], device=pan.device)}


class TinyDataset:
    has_gt, base_count = True, 3
    def __len__(self):
        return self.base_count
    def base(self, index):
        gt = torch.full((4, 8, 8), float(index + 1)); gt[:, :7] = index / 10
        return gt, gt, torch.zeros(4, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, 8, 8), torch.tensor([index, 0])
    def get_view(self, index, rot, augment):
        return self.base(index)


class CalibrationTests(unittest.TestCase):
    def test_math_matches_unchanged_calibration(self):
        model, data, indices = ConstantTeacher(), TinyDataset(), np.arange(3)
        a, aa = existing(model, data, indices, device='cpu', batch_size=16, synthetic_test=True)
        b, bb = compute_calibration(model, data, indices, device='cpu', synthetic_test=True)
        self.assertEqual(b['schema'], 'L100_CALIBRATION_v2')
        self.assertEqual(b['batch_size'], 16)
        for key, value in a.items():
            if key != 'schema':
                self.assertEqual(b[key], value, key)
        for key in aa:
            np.testing.assert_array_equal(aa[key], bb[key])
        self.assertEqual(b['q_ref'], .46875)  # Constant/high valid q is not a quality rejection.

    def test_production_batch64_rejected_before_loading_assets(self):
        with patch('l100.calibration.teacher_endpoint') as endpoint:
            with self.assertRaisesRegex(ValueError, 'batch16'):
                calibrate('anything', root='/not-used', server='s4', device='cpu', batch_size=64)
            endpoint.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'batch16'):
            compute_calibration(ConstantTeacher(), TinyDataset(), np.arange(3), device='cpu', batch_size=64)

    def test_shared3072_ids_are_exact_original_selection(self):
        np.testing.assert_array_equal(select_calibration_indices(19809),
            np.random.default_rng(1234).choice(19809, 3072, replace=False))


if __name__ == '__main__':
    unittest.main()
