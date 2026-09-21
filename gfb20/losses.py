"""B20's registered coefficients; the existing fitting equation is unchanged."""
from __future__ import annotations

import math
import torch

from fh12.losses import routed_student_backward
from pa.losses import output_edge_loss_per_sample

TARGETS = {'BASE': (1., .1, .002), 'H010': (.1, .1, .002),
           'K1': (1., 1., .002), 'E100': (1., .1, .1),
           'NATIVE0': (1., .1, .002), 'NAT_MIXCAL': (1., .1, .002),
           'PANMIX': (1., .1, .002)}


def coefficients(profile, completed_updates):
    if profile not in TARGETS or int(completed_updates) != completed_updates or completed_updates < 0:
        raise ValueError('Bound B20 profile and nonnegative completed-update index required')
    alpha, beta, edge = TARGETS[profile]
    ramp = min(float(completed_updates) / 1000., 1.) if profile in ('K1', 'E100') else 1.
    return dict(alpha=alpha, beta=.1 + ramp * (beta - .1),
                lambda_edge=.002 + ramp * (edge - .002), ramp_fraction=ramp)


def student_losses(out, teacher_out, gt, tau_R, q_weights, profile='BASE',
                   completed_updates=0, *, bands=4):
    coeff = coefficients(profile, completed_updates)
    tau = float(tau_R)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('tau_R must be finite and positive')
    y, teacher, target = out['y'].float(), teacher_out['y'].float().detach(), gt.float().detach()
    if bands != 4 or y.ndim != 4 or y.shape[1] != 4 or y.shape != teacher.shape or y.shape != target.shape:
        raise ValueError('B20 requires matching C4 Student/Teacher/GT')
    weights = torch.as_tensor(q_weights, dtype=torch.float32, device=y.device).detach()
    if weights.shape != (len(y),) or not bool(torch.isfinite(weights).all()) or not bool(((weights > 0) & (weights <= 1)).all()):
        raise ValueError('Detached q weights must be finite [B] in (0,1]')
    with torch.autocast(device_type=y.device.type, enabled=False):
        es = (y - target).abs().mean(dim=1, keepdim=True)
        et = (teacher - target).abs().mean(dim=1, keepdim=True)
        difficulty = (et / (et + tau)).detach()
        advantage = ((es - et).clamp_min(0) / (es + 1e-6)).detach()
        gate = ((1. - difficulty) * advantage).detach()
        discrepancy = (y - teacher).abs().mean(dim=1, keepdim=True)
        hard_map = (1. + coeff['alpha'] * difficulty) * es
        hard_i = hard_map.flatten(1).mean(1)
        # Keep G20's multiplication order for the BASE bitwise regression.
        soft_i = (coeff['beta'] * (1. - difficulty) * advantage * discrepancy).flatten(1).mean(1)
        raw_soft_i = (gate * discrepancy).flatten(1).mean(1)
        edge_i = output_edge_loss_per_sample(y, target)
        weighted_edge_i = coeff['lambda_edge'] * weights * edge_i
        lu = (hard_i + soft_i + weighted_edge_i).mean()
        la = (weights * hard_i).mean()
    return dict(L_U=lu, L_A=la, hard=hard_i.mean(), soft=soft_i.mean(),
                raw_soft=raw_soft_i.mean(), edge=edge_i.mean(),
                edge_weighted=weighted_edge_i.mean(), hard_i=hard_i, soft_i=soft_i,
                edge_i=edge_i, difficulty=difficulty, advantage=advantage,
                soft_gate=gate, q_weights=weights, e_student=es, e_teacher=et,
                hard_map=hard_map, coefficients=coeff)
