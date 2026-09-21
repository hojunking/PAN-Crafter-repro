"""LOCAL-T coefficients on the unchanged C4 objective and gradient routing."""
import math

import torch

from fh12.losses import routed_student_backward
from g20.losses import teacher_loss, student_losses as _existing_student_losses
from l100.plan import PROFILES
from pa.losses import output_edge_loss_per_sample


def student_losses(out, teacher_out, gt, tau_R, q_weights, profile='BASE', *, bands=4):
    # Preserve existing BASE/E1 operation order, not merely algebraic equality.
    if profile in ('BASE', 'E1'):
        return _existing_student_losses(out, teacher_out, gt, tau_R, q_weights, profile, bands=bands)
    if profile not in PROFILES:
        raise ValueError('Unregistered LOCAL-T Student loss profile')
    recipe = PROFILES[profile]
    tau = float(tau_R)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('tau_R must be finite and positive')
    y, teacher, target = out['y'].float(), teacher_out['y'].float().detach(), gt.float().detach()
    if bands != 4 or y.ndim != 4 or y.shape[1] != 4 or y.shape != teacher.shape or y.shape != target.shape:
        raise ValueError('LOCAL-T requires matching C4 Student/Teacher/GT')
    weights = torch.as_tensor(q_weights, dtype=torch.float32, device=y.device).detach()
    if weights.shape != (len(y),) or not bool(torch.isfinite(weights).all()) or not bool(((weights > 0) & (weights <= 1)).all()):
        raise ValueError('q weights must be finite [B] values in (0,1]')
    with torch.autocast(device_type=y.device.type, enabled=False):
        e_student = (y - target).abs().mean(dim=1, keepdim=True)
        e_teacher = (teacher - target).abs().mean(dim=1, keepdim=True)
        difficulty = (e_teacher / (e_teacher + tau)).detach()
        advantage = ((e_student - e_teacher).clamp_min(0) / (e_student + 1e-6)).detach()
        soft_error = (y - teacher).abs().mean(dim=1, keepdim=True)
        hard_i = ((1. + recipe.alpha * difficulty) * e_student).flatten(1).mean(1)
        soft_i = (recipe.beta * (1. - difficulty) * advantage * soft_error).flatten(1).mean(1)
        edge_i = output_edge_loss_per_sample(y, target)
        loss_u = (hard_i + soft_i + recipe.lambda_edge * weights * edge_i).mean()
        loss_a = (weights * hard_i).mean()
    return dict(L_U=loss_u, L_A=loss_a, hard=hard_i.mean(), soft=soft_i.mean(),
                edge=edge_i.mean(), edge_weighted=(recipe.lambda_edge * weights * edge_i).mean(),
                hard_i=hard_i, soft_i=soft_i, edge_i=edge_i, difficulty=difficulty,
                advantage=advantage, q_weights=weights)
