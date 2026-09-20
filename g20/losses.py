"""C4/C8 losses with unchanged band means and independent U/A gradients."""
from __future__ import annotations

import math
import torch
from fh12.losses import routed_student_backward
from pa.losses import output_edge_loss_per_sample
from pa.offset import sample_offsets
from pa.warp import warp_pan

from g20.plan import PROFILES
STUDENT_PROFILES = {key: (p.alpha, p.beta, p.lambda_edge) for key, p in PROFILES.items()}


def teacher_loss(model, out, gt, pan, ms, update_index, generator,
                 consistency_weight=1e-4, *, bands=4):
    if int(update_index) != update_index or update_index < 0:
        raise ValueError("update_index must be a nonnegative 0-based integer")
    if consistency_weight not in (3e-5, 1e-4, 3e-4):
        raise ValueError("Only C030, C100 or C300 Teacher consistency is registered")
    if bands not in (4, 8) or gt.ndim != 4 or gt.shape[1] != bands or out["y"].shape != gt.shape:
        raise ValueError("Teacher/GT must match the explicit band contract")
    active = bool(update_index % 2)
    with torch.autocast(device_type=pan.device.type, enabled=False):
        rec = (out["y"].float() - gt.float().detach()).abs().mean()
        epsilon = torch.zeros(pan.shape[0], 2, device=pan.device)
        off = rec.new_zeros(())
        if active:
            if generator is None or generator.device.type != "cpu":
                raise ValueError("Synthetic corruption needs its independent CPU generator")
            epsilon = sample_offsets(pan.shape[0], 2., generator).to(pan.device)
            with torch.no_grad():
                corrupted = warp_pan(pan.float(), epsilon)
            c_eps = model.predict_delta(corrupted, out["ms_base"].detach())
            off = (c_eps + epsilon - out["delta"].detach()).abs().mean()
        total = rec + (consistency_weight * off if active else 0.)
    return dict(total=total, rec=rec, off=off, offset_active=active,
                offset_weight=consistency_weight if active else 0., epsilon=epsilon)


def student_losses(out, teacher_out, gt, tau_R, q_weights, profile="BASE", *, bands=4):
    if profile not in STUDENT_PROFILES:
        raise ValueError("Unregistered Student loss profile")
    alpha, beta, edge_weight = STUDENT_PROFILES[profile]
    tau = float(tau_R)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau_R must be finite and positive")
    y, teacher, target = out["y"].float(), teacher_out["y"].float().detach(), gt.float().detach()
    if bands not in (4, 8) or y.shape != teacher.shape or y.shape != target.shape or y.ndim != 4 or y.shape[1] != bands:
        raise ValueError("Student/Teacher/GT must match the explicit band contract")
    weights = torch.as_tensor(q_weights, dtype=torch.float32, device=y.device).detach()
    if weights.shape != (len(y),) or not bool(torch.isfinite(weights).all()) or not bool(((weights > 0) & (weights <= 1)).all()):
        raise ValueError("q weights must be finite [B] values in (0,1]")
    with torch.autocast(device_type=y.device.type, enabled=False):
        e_student = (y - target).abs().mean(dim=1, keepdim=True)
        e_teacher = (teacher - target).abs().mean(dim=1, keepdim=True)
        difficulty = (e_teacher / (e_teacher + tau)).detach()
        advantage = ((e_student - e_teacher).clamp_min(0) / (e_student + 1e-6)).detach()
        soft_error = (y - teacher).abs().mean(dim=1, keepdim=True)
        hard_i = ((1. + alpha * difficulty) * e_student).flatten(1).mean(1)
        soft_i = (beta * (1. - difficulty) * advantage * soft_error).flatten(1).mean(1)
        edge_i = output_edge_loss_per_sample(y, target)
        loss_u = (hard_i + soft_i + edge_weight * weights * edge_i).mean()
        loss_a = (weights * hard_i).mean()
    return dict(L_U=loss_u, L_A=loss_a, hard=hard_i.mean(), soft=soft_i.mean(),
                edge=edge_i.mean(), edge_weighted=(edge_weight * weights * edge_i).mean(),
                hard_i=hard_i, soft_i=soft_i, edge_i=edge_i, difficulty=difficulty,
                advantage=advantage, q_weights=weights)
