"""GT-anchored, error-adaptive reconstruction KD (plan §6; reference pan_gt_anchored_kd.py, sha256 36a69ad5…).

  gt            : H                                  (REC-N0)
  fixed_kd      : H + beta·K                         (REC-R0)
  hard_only     : (1 + alpha·d)·H                    (REC-R1)
  teacher_error : (1 + alpha·d)·H + beta·(1−d)·K     (REC-R2)
  adaptive      : (1 + alpha·d)·H + beta·(1−d)·a·K   (REC-R3)
d = e_T/(e_T+tau), a = [e_S−e_T]_+/(e_S+eps); Teacher·GT·gate 는 detach, e_S·e_ST 는 live. 모든 픽셀 평균(재정규화 없음).
참조 파일에서 unittest 만 tools/kdv_unit_tests.py 로 옮기고 수식·검증은 그대로다."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

MODES = ('gt', 'fixed_kd', 'hard_only', 'teacher_error', 'adaptive')


@dataclass
class ReconstructionKDResult:
    loss: Tensor
    hard: Tensor
    soft: Tensor                     # kd_weight 포함
    stats: Dict[str, Tensor]
    maps: Optional[Dict[str, Tensor]] = None


class GTAnchoredReconstructionKD(nn.Module):
    """픽셀별(밴드 공유) error-adaptive reconstruction loss. tau 는 frozen Teacher·센서별 train-only calibration 상수."""

    def __init__(self, tau: float, *, alpha: float = 1.0, kd_weight: float = 0.1, eps: float = 1e-6, mode: str = 'adaptive',
                 require_gt_dominance: bool = True) -> None:
        super().__init__()
        for name, value in [('tau', tau), ('alpha', alpha), ('kd_weight', kd_weight), ('eps', eps)]:
            if not math.isfinite(float(value)):
                raise ValueError(f'{name} must be finite.')
        if tau <= 0 or eps <= 0:
            raise ValueError('tau and eps must be positive.')
        if alpha < 0 or kd_weight < 0:
            raise ValueError('alpha and kd_weight must be non-negative.')
        if require_gt_dominance and kd_weight >= 1:
            raise ValueError('Use 0 <= kd_weight < 1 for the GT-dominance bound.')
        if mode not in MODES:
            raise ValueError(f'Unsupported mode: {mode}. Choose from {MODES}.')
        self.alpha = float(alpha); self.kd_weight = float(kd_weight); self.eps = float(eps); self.mode = mode
        self.register_buffer('tau', torch.tensor(float(tau), dtype=torch.float64))     # state_dict 에 calibration 상수 보존

    def forward(self, student: Tensor, teacher: Tensor, target: Tensor, *, return_maps: bool = False) -> ReconstructionKDResult:
        if student.ndim != 4 or teacher.shape != student.shape or target.shape != student.shape:
            raise ValueError('student, teacher and target must have identical NCHW shapes.')
        if any(d == 0 for d in student.shape):
            raise ValueError('Empty dimensions are not supported.')
        if not (student.device == teacher.device == target.device):
            raise ValueError('All image tensors must be on the same device.')
        if not all(x.is_floating_point() for x in (student, teacher, target)):
            raise TypeError('All image tensors must be floating point.')
        dtype = torch.float64 if any(x.dtype == torch.float64 for x in (student, teacher, target)) else torch.float32
        s = student.to(dtype=dtype); t = teacher.detach().to(dtype=dtype); y = target.detach().to(dtype=dtype)
        tau = self.tau.detach().to(device=s.device, dtype=dtype)
        err_gt = (s - y).abs().mean(dim=1, keepdim=True)            # live
        err_teacher = (s - t).abs().mean(dim=1, keepdim=True)       # live
        with torch.no_grad():                                        # routing 은 gradient 없음
            e_t = (t - y).abs().mean(dim=1, keepdim=True)
            e_s = err_gt.detach()
            difficulty = e_t / (e_t + tau)
            advantage = ((e_s - e_t).clamp_min(0) / (e_s + self.eps)).clamp(max=1.0)
            one = torch.ones_like(difficulty); zero = torch.zeros_like(difficulty)
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
        hard = (w_h * err_gt).mean()                                 # 모든 픽셀 평균 (soft active 수로 재정규화하지 않는다)
        soft = (w_k * err_teacher).mean()
        loss = hard + soft
        with torch.no_grad():
            stats = {
                'plain_gt_l1': err_gt.detach().mean(), 'plain_teacher_l1': err_teacher.detach().mean(), 'teacher_gt_l1': e_t.mean(),
                'difficulty_mean': difficulty.mean(), 'advantage_mean': advantage.mean(), 'hard_weight_mean': w_h.mean(),
                'soft_weight_mean': w_k.mean(), 'soft_positive_fraction': (w_k > 0).to(dtype).mean(),
                'student_better_fraction': (e_s < e_t).to(dtype).mean(), 'coefficient_ratio_max': (w_k / w_h).amax(),
                'soft_to_hard_loss': soft.detach() / (hard.detach() + self.eps),
            }
            maps = None
            if return_maps:
                maps = {'teacher_error': e_t, 'student_error': e_s, 'difficulty': difficulty, 'advantage': advantage, 'hard_weight': w_h, 'soft_weight': w_k}
        return ReconstructionKDResult(loss, hard, soft, stats, maps)
