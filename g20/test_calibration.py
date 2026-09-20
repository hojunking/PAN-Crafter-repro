import unittest
from unittest.mock import patch
import numpy as np
import torch
from qg40.test_references import ProbeDataset as Dataset, ProbeModel as Constant
from g20.calibration import compute_calibration, select_calibration_indices


class CalibrationTests(unittest.TestCase):
    def test_selection_is_shared3072_seed1234(self):
        ids = select_calibration_indices(4000)
        np.testing.assert_array_equal(ids, np.random.default_rng(1234).choice(4000, 3072, replace=False))
        with self.assertRaises(ValueError):
            select_calibration_indices(3071)

    def test_c8_is_rejected_even_in_synthetic_mode(self):
        model = Constant(); model.bands = 8
        with self.assertRaisesRegex(ValueError, 'C4'):
            compute_calibration(model, Dataset(), np.arange(16), device='cpu', synthetic_test=True)

    def test_wrapper_preserves_numerical_core_and_only_changes_schema(self):
        model = Constant()
        cal = dict(schema='QG40_CALIBRATION_v1', tau_R=.025, q_ref=.46875)
        arrays = {'q': np.full((16, 4), .46875)}
        with patch('g20.calibration._compute', return_value=(cal, arrays)) as core:
            result, q = compute_calibration(model, Dataset(), np.arange(16), device='cpu', synthetic_test=True)
        self.assertEqual(result['schema'], 'G20_CALIBRATION_v1')
        self.assertEqual(result['tau_R'], .025)
        self.assertIs(q, arrays)
        self.assertTrue(core.call_args.kwargs['synthetic_test'])


if __name__ == '__main__':
    unittest.main()
