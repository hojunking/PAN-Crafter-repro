"""M12's preregistered routing and mass-matched gate interventions only.

The signed output edge kernel and U/A gradient assignment remain the frozen
FH12 functions. No Teacher/GT/gate is differentiable through this adapter.
"""
from __future__ import annotations

import math
import torch
from pa.losses import output_edge_loss_per_sample

CASES = ('QFULL', 'QMEAN', 'QSHUF', 'QESUR', 'QEDGE', 'QALIGN',
         'H0', 'HSPMEAN', 'NOADV', 'ADVMEAN')


def case_weights(arrays, case_id):
    """Return whole-population (edge, aligner) arrays without renormalization."""
    if case_id not in CASES:
        raise ValueError('Unregistered M12 intervention: ' + str(case_id))
    if case_id in ('QFULL', 'QMEAN', 'QSHUF', 'QESUR'):
        return arrays[case_id], arrays[case_id]
    if case_id == 'QEDGE':
        return arrays['QFULL'], arrays['QMEAN']
    if case_id == 'QALIGN':
        return arrays['QMEAN'], arrays['QFULL']
    return arrays['QFULL'], arrays['QFULL']


def trust_weighted_mean(trust, advantage):
    """Detached per-sample mean preserving sum(t*a); exact-zero safe.

    A positive denominator is never epsilon-clamped: that would change the
    preregistered total soft supervision. Zero trust yields an exact zero.
    """
    t, a = trust.detach(), advantage.detach()
    if t.shape != a.shape or t.ndim != 4 or t.shape[1] != 1:
        raise ValueError('Trust/advantage must match [N,1,H,W]')
    if not bool(torch.isfinite(t).all() & torch.isfinite(a).all()) or bool((t < 0).any()):
        raise FloatingPointError('Invalid trust or advantage')
    denominator = t.flatten(1).sum(1)
    numerator = (t * a).flatten(1).sum(1)
    # where alone still evaluates 0/0 and poisons autograd in other contexts.
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    result = torch.where(denominator > 0, numerator / safe, torch.zeros_like(numerator))
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Nonfinite trust-weighted advantage mean')
    return result[:, None, None, None].detach()


def student_losses(out, teacher_out, gt, tau_rec, w_edge, w_align=None, *, case_id='QFULL'):
    if case_id not in CASES:
        raise ValueError('Unregistered M12 intervention: ' + str(case_id))
    tau = float(tau_rec)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('M12 requires the positive bound F1 tau_rec')
    y, teacher, target = out['y'].float(), teacher_out['y'].float().detach(), gt.float().detach()
    if y.ndim != 4 or y.shape[1] != 8 or y.shape != teacher.shape or y.shape != target.shape:
        raise ValueError('M12 Student/Teacher/GT must match WV3 [B,8,H,W]')
    def detached_weight(value):
        value = torch.as_tensor(value, dtype=torch.float32, device=y.device).detach()
        if value.shape != (len(y),) or not bool(torch.isfinite(value).all()) or not bool(((value > 0) & (value <= 1)).all()):
            raise ValueError('M12 geometry weights must be finite [B] in (0,1]')
        return value
    we = detached_weight(w_edge)
    wa = detached_weight(w_edge if w_align is None else w_align)
    with torch.autocast(device_type=y.device.type, enabled=False):
        es = (y - target).abs().mean(1, keepdim=True)
        et = (teacher - target).abs().mean(1, keepdim=True)
        difficulty = (et / (et + tau)).detach()
        trust = (1. - difficulty).detach()
        advantage = ((es - et).clamp_min(0) / (es + 1e-6)).detach()
        hard_base = (1. + difficulty).detach()
        hard_weight = hard_base
        if case_id == 'H0':
            hard_weight = torch.ones_like(hard_base)
        elif case_id == 'HSPMEAN':
            hard_weight = hard_base.flatten(1).mean(1)[:, None, None, None].detach().expand_as(hard_base)
        applied_advantage = advantage
        if case_id == 'NOADV':
            applied_advantage = torch.ones_like(advantage)
        elif case_id == 'ADVMEAN':
            applied_advantage = trust_weighted_mean(trust, advantage).expand_as(advantage)
        soft_error = (y - teacher).abs().mean(1, keepdim=True)
        hard_i = (hard_weight * es).flatten(1).mean(1)
        # Match the original left-associative FP32 multiplication exactly.
        soft_i = (.1 * trust * applied_advantage * soft_error).flatten(1).mean(1)
        edge_i = output_edge_loss_per_sample(y, target)
        loss_u = (hard_i + soft_i + .002 * we * edge_i).mean()
        loss_a = (wa * hard_i).mean()
    return dict(L_U=loss_u, L_A=loss_a, hard=hard_i.mean(), soft=soft_i.mean(),
                edge=edge_i.mean(), edge_weighted=(.002 * we * edge_i).mean(),
                hard_i=hard_i, soft_i=soft_i, edge_i=edge_i, difficulty=difficulty,
                advantage=advantage, applied_advantage=applied_advantage, trust=trust,
                hard_weight=hard_weight, original_hard_weight=hard_base,
                soft_weight=.1 * trust * applied_advantage,
                q_weights=we, w_edge=we, w_align=wa, case_id=case_id,
                hard_mass_original=hard_base.flatten(1).sum(1),
                hard_mass_applied=hard_weight.flatten(1).sum(1),
                soft_mass_original=(trust * advantage).flatten(1).sum(1),
                soft_mass_applied=(trust * applied_advantage).flatten(1).sum(1))
