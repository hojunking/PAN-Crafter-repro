import unittest
from unittest.mock import patch

import torch

from panda_method.calibration import state_dict_sha256
from s2dshift48.aligner import S2DGlobalPANAligner
from s2dshift48.diagnostics import epsilon_response


class ResponseTests(unittest.TestCase):
    def test_zero_head_is_not_falsely_reported_as_equivariant(self):
        aligner = S2DGlobalPANAligner(8).train()
        pan, lms = torch.rand(2, 1, 64, 64), torch.rand(2, 8, 64, 64)
        epsilon = torch.tensor([[0., 0.], [.25, -.5], [1., 0.], [0., 3.]])
        initial = state_dict_sha256(aligner)
        rng = torch.random.get_rng_state().clone()
        report = epsilon_response(aligner, pan, lms, epsilon)
        self.assertEqual(report['rows'][0]['l1_axis_mean_px'], 0.)
        self.assertEqual(report['rows'][1]['l1_axis_mean_px'], .375)
        self.assertEqual(report['rows'][3]['l1_axis_mean_px'], 1.5)
        self.assertEqual(initial, state_dict_sha256(aligner))
        self.assertTrue(aligner.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in aligner.parameters()))

    def test_known_response_sign_and_no_lms_warp(self):
        aligner = S2DGlobalPANAligner(4)
        pan, lms = torch.rand(1, 1, 64, 64), torch.rand(1, 4, 64, 64)
        epsilon = torch.tensor([[.5, -.25]])
        def output(p, m):
            self.assertIs(m, lms)
            self.assertFalse(torch.is_grad_enabled())
            return torch.tensor([[1., 2.]]) if p is pan else torch.tensor([[.5, 2.25]])
        with patch.object(aligner, 'forward', side_effect=output):
            report = epsilon_response(aligner, pan, lms, epsilon)
        self.assertEqual(report['rows'][0]['residual_dy_dx'], [[0., 0.]])

    def test_invalid_epsilons_fail(self):
        aligner = S2DGlobalPANAligner(4)
        pan, lms = torch.rand(1, 1, 64, 64), torch.rand(1, 4, 64, 64)
        for eps in (torch.zeros(0, 2), torch.tensor([[3., 3.]]),
                    torch.tensor([[float('nan'), 0.]]), torch.zeros(1, 2).double()):
            with self.assertRaises(ValueError):
                epsilon_response(aligner, pan, lms, eps)


if __name__ == '__main__':
    unittest.main()
