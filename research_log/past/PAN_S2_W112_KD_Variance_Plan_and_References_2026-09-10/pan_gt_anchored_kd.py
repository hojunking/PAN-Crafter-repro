"""GT-anchored, error-adaptive reconstruction KD for pansharpening.

This is a reference loss implementation, not a full trainer or an alignment model.
Inputs are co-registered NCHW tensors in the same radiometric units.
Ground truth supervision is ALWAYS active. There is no binary target switch.

Modes correspond to the experiment plan:
  gt            : H                         (N0)
  fixed_kd      : H + lambda * K             (R0)
  hard_only     : (1 + alpha*d) * H          (R1)
  teacher_error : (1 + alpha*d)*H + lambda*(1-d)*K        (R2)
  adaptive      : (1 + alpha*d)*H + lambda*(1-d)*a*K      (R3)

Teacher outputs and all routing maps are detached. Student reconstruction errors
remain differentiable in the actual hard and soft losses. Mixed-precision inputs
are promoted to float32 for loss calculation. Float64 inputs retain float64.

Run `python pan_gt_anchored_kd.py` to execute CPU-only mathematical/unit checks.
No satellite training or GPU-memory validation is performed by these tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import unittest
from typing import Dict, Literal, Optional

import torch
from torch import Tensor, nn

Mode = Literal['gt', 'fixed_kd', 'hard_only', 'teacher_error', 'adaptive']
_MODES = {'gt', 'fixed_kd', 'hard_only', 'teacher_error', 'adaptive'}


@dataclass
class ReconstructionKDResult:
    loss: Tensor
    hard: Tensor
    soft: Tensor  # Already includes kd_weight.
    stats: Dict[str, Tensor]
    maps: Optional[Dict[str, Tensor]] = None


class GTAnchoredReconstructionKD(nn.Module):
    """Pixel-wise, shared-across-bands error-adaptive reconstruction loss.

    tau is a positive training-only calibration statistic, fixed for a frozen
    teacher and sensor. Do not recompute it from each batch or a test dataset.
    Inputs must already share the fixed MS/GT frame. Do not warp the prediction
    to the GT before calling this loss.
    """

    def __init__(
        self,
        tau: float,
        *,
        alpha: float = 1.0,
        kd_weight: float = 0.1,
        eps: float = 1e-6,
        mode: Mode = 'adaptive',
        require_gt_dominance: bool = True,
    ) -> None:
        super().__init__()
        for name, value in [('tau', tau), ('alpha', alpha),
                            ('kd_weight', kd_weight), ('eps', eps)]:
            if not math.isfinite(float(value)):
                raise ValueError(f'{name} must be finite.')
        if tau <= 0 or eps <= 0:
            raise ValueError('tau and eps must be positive.')
        if alpha < 0 or kd_weight < 0:
            raise ValueError('alpha and kd_weight must be non-negative.')
        if require_gt_dominance and kd_weight >= 1:
            raise ValueError('Use 0 <= kd_weight < 1 for the GT-dominance bound.')
        if mode not in _MODES:
            raise ValueError(f'Unsupported mode: {mode}. Choose from {_MODES}.')
        self.alpha = float(alpha)
        self.kd_weight = float(kd_weight)
        self.eps = float(eps)
        self.mode = mode
        # Store the calibration constant in state_dict for reproducibility.
        self.register_buffer('tau', torch.tensor(float(tau), dtype=torch.float64))

    def forward(
        self,
        student: Tensor,
        teacher: Tensor,
        target: Tensor,
        *,
        return_maps: bool = False,
    ) -> ReconstructionKDResult:
        if student.ndim != 4 or teacher.shape != student.shape or target.shape != student.shape:
            raise ValueError('student, teacher and target must have identical NCHW shapes.')
        if any(d == 0 for d in student.shape):
            raise ValueError('Empty dimensions are not supported.')
        if not (student.device == teacher.device == target.device):
            raise ValueError('All image tensors must be on the same device.')
        if not all(x.is_floating_point() for x in (student, teacher, target)):
            raise TypeError('All image tensors must be floating point.')

        dtype = torch.float64 if any(x.dtype == torch.float64 for x in (student, teacher, target)) else torch.float32
        s = student.to(dtype=dtype)
        t = teacher.detach().to(dtype=dtype)
        y = target.detach().to(dtype=dtype)
        tau = self.tau.detach().to(device=s.device, dtype=dtype)

        # These are live losses. Do NOT detach them.
        err_gt = (s - y).abs().mean(dim=1, keepdim=True)
        err_teacher = (s - t).abs().mean(dim=1, keepdim=True)

        # Routing signals have no gradient.
        with torch.no_grad():
            e_t = (t - y).abs().mean(dim=1, keepdim=True)
            e_s = err_gt.detach()
            difficulty = e_t / (e_t + tau)
            advantage = (e_s - e_t).clamp_min(0) / (e_s + self.eps)
            advantage = advantage.clamp(max=1.0)
            one = torch.ones_like(difficulty)
            zero = torch.zeros_like(difficulty)

            if self.mode == 'gt':
                w_h, w_k = one, zero
            elif self.mode == 'fixed_kd':
                w_h, w_k = one, self.kd_weight * one
            else:
                w_h = one + self.alpha * difficulty
                if self.mode == 'hard_only':
                    w_k = zero
                elif self.mode == 'teacher_error':
                    w_k = self.kd_weight * (one - difficulty)
                else:
                    w_k = self.kd_weight * (one - difficulty) * advantage

        # Average over ALL pixels, not over only active soft pixels or sum(w_k).
        hard = (w_h * err_gt).mean()
        soft = (w_k * err_teacher).mean()
        loss = hard + soft
        with torch.no_grad():
            stats = {
                'plain_gt_l1': err_gt.detach().mean(),
                'plain_teacher_l1': err_teacher.detach().mean(),
                'teacher_gt_l1': e_t.mean(),
                'difficulty_mean': difficulty.mean(),
                'advantage_mean': advantage.mean(),
                'hard_weight_mean': w_h.mean(),
                'soft_weight_mean': w_k.mean(),
                'soft_positive_fraction': (w_k > 0).to(dtype).mean(),
                'student_better_fraction': (e_s < e_t).to(dtype).mean(),
                'coefficient_ratio_max': (w_k / w_h).amax(),
                'soft_to_hard_loss': soft.detach() / (hard.detach() + self.eps),
            }
            maps = None
            if return_maps:
                maps = {
                    'teacher_error': e_t,
                    'student_error': e_s,
                    'difficulty': difficulty,
                    'advantage': advantage,
                    'hard_weight': w_h,
                    'soft_weight': w_k,
                }
        return ReconstructionKDResult(loss, hard, soft, stats, maps)


class _Tests(unittest.TestCase):
    @staticmethod
    def _tensor(value: float, *, grad: bool = False) -> Tensor:
        return torch.tensor([[[[value]]]], dtype=torch.float64, requires_grad=grad)

    def test_main_formula_matches_hand_calculation(self):
        s, t, y = self._tensor(0.1, grad=True), self._tensor(0.08), self._tensor(0.0)
        result = GTAnchoredReconstructionKD(.01)(s, t, y)
        d = .08 / (.08 + .01)
        a = (.10 - .08) / (.10 + 1e-6)
        expected = (1 + d) * .10 + .1 * (1-d) * a * .02
        self.assertAlmostEqual(result.loss.item(), expected, places=12)

    def test_both_wrong_still_receive_gt_gradient(self):
        s = self._tensor(.08, grad=True)
        result = GTAnchoredReconstructionKD(.01)(s, self._tensor(.08), self._tensor(0))
        result.loss.backward()
        self.assertEqual(result.soft.item(), 0)
        self.assertGreater(s.grad.item(), 1)

    def test_student_better_disables_soft(self):
        result = GTAnchoredReconstructionKD(.01)(self._tensor(.04), self._tensor(.08), self._tensor(0), return_maps=True)
        self.assertEqual(result.maps['soft_weight'].item(), 0)
        self.assertGreater(result.hard.item(), .04)

    def test_exact_gt_has_zero_loss_and_zero_gradient(self):
        s = self._tensor(0, grad=True)
        result = GTAnchoredReconstructionKD(.01)(s, self._tensor(.08), self._tensor(0))
        result.loss.backward()
        self.assertEqual(result.loss.item(), 0)
        self.assertEqual(s.grad.item(), 0)

    def test_teacher_and_routing_are_detached(self):
        s, t, y = self._tensor(.1, grad=True), self._tensor(.02, grad=True), self._tensor(0, grad=True)
        result = GTAnchoredReconstructionKD(.01)(s, t, y, return_maps=True)
        result.loss.backward()
        self.assertIsNone(t.grad)
        self.assertIsNone(y.grad)
        self.assertIsNotNone(s.grad)
        self.assertTrue(all(not m.requires_grad for m in result.maps.values()))

    def test_elementwise_gt_direction_with_conflicting_bands(self):
        # Teacher is better in band-average L1 but worse in channel 1.
        s = torch.tensor([[[[.1]], [[.9]]]], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[[[.2]], [[.1]]]], dtype=torch.float64)
        y = torch.zeros_like(t)
        result = GTAnchoredReconstructionKD(.01, kd_weight=.5)(s, t, y)
        result.loss.backward()
        self.assertTrue(bool((s.grad > 0).all()))

    def test_coefficient_bounds_random_inputs(self):
        g = torch.Generator().manual_seed(2025)
        s = torch.randn(2, 8, 7, 7, generator=g, requires_grad=True)
        t, y = torch.randn(2, 8, 7, 7, generator=g), torch.randn(2, 8, 7, 7, generator=g)
        result = GTAnchoredReconstructionKD(.05)(s, t, y, return_maps=True)
        self.assertTrue(bool((result.maps['hard_weight'] >= 1).all()))
        self.assertTrue(bool((result.maps['hard_weight'] < 2).all()))
        self.assertTrue(bool((result.maps['soft_weight'] >= 0).all()))
        self.assertLessEqual(result.maps['soft_weight'].max().item(), .100001)
        result.loss.backward()
        self.assertTrue(bool((s.grad * (s.detach() - y) > 0).all()))

    def test_ablation_modes(self):
        s, t, y = self._tensor(.1), self._tensor(.08), self._tensor(0)
        for mode in _MODES:
            result = GTAnchoredReconstructionKD(.01, mode=mode)(s, t, y)
            self.assertTrue(bool(torch.isfinite(result.loss)))
        gt = GTAnchoredReconstructionKD(.01, mode='gt')(s, t, y)
        fixed = GTAnchoredReconstructionKD(.01, mode='fixed_kd')(s, t, y)
        self.assertAlmostEqual(gt.loss.item(), .1, places=12)
        self.assertAlmostEqual(fixed.loss.item(), .102, places=12)

    def test_input_and_config_validation(self):
        with self.assertRaises(ValueError):
            GTAnchoredReconstructionKD(0)
        with self.assertRaises(ValueError):
            GTAnchoredReconstructionKD(.01, kd_weight=1)
        with self.assertRaises(ValueError):
            GTAnchoredReconstructionKD(.01)(torch.zeros(1, 8, 4, 4), torch.zeros(1, 1, 4, 4), torch.zeros(1, 8, 4, 4))

    def test_fp16_inputs_compute_loss_in_float32(self):
        s = torch.ones(1, 8, 2, 2, dtype=torch.float16, requires_grad=True)
        t, y = torch.zeros_like(s), torch.zeros_like(s)
        result = GTAnchoredReconstructionKD(.01)(s, t, y)
        self.assertEqual(result.loss.dtype, torch.float32)
        result.loss.backward()
        self.assertTrue(bool(torch.isfinite(s.grad).all()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
