"""FH12 Teacher offset-only consistency and separately routed G23 objectives."""
from __future__ import annotations

import math

import torch

from pa.losses import output_edge_loss_per_sample
from pa.offset import sample_offsets
from pa.warp import warp_pan


def teacher_loss(model, out, gt, pan, ms, update_index, generator):
    """Native reconstruction every update; A-only offset on odd 0-based updates.

    No synthetic input ever enters U. The only offset target detach is c0;
    native out['delta'] remains live in the reconstruction graph.
    """
    if int(update_index) != update_index or update_index < 0:
        raise ValueError("update_index must be a nonnegative 0-based integer")
    active = bool(update_index % 2 == 1)
    with torch.autocast(device_type=pan.device.type, enabled=False):
        rec = (out["y"].float() - gt.float().detach()).abs().mean()
        epsilon = torch.zeros(pan.shape[0], 2, device=pan.device)
        off = rec.new_zeros(())
        if active:
            if generator is None or generator.device.type != "cpu":
                raise ValueError("FH12 offset corruption requires its separate CPU generator")
            epsilon = sample_offsets(pan.shape[0], 2., generator).to(pan.device)
            with torch.no_grad():
                corrupted = warp_pan(pan.float(), epsilon)
            c_eps = model.predict_delta(corrupted, out["ms_base"].detach())
            off = (c_eps + epsilon - out["delta"].detach()).abs().mean()
        total = rec + (1e-4 * off if active else 0.)
    return dict(total=total, rec=rec, off=off, offset_active=active,
                offset_weight=(1e-4 if active else 0.), epsilon=epsilon)


def student_losses(out, teacher_out, gt, tau_R, q_weights):
    """Return L_U and L_A; calling backward(L_U) is NOT valid FH12 routing.

    Weights/difficulty/advantage/Teacher/GT are detached. Live eS and soft
    residuals retain their gradient. beta=.1 is included exactly once.
    """
    tau = float(tau_R)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("FH12 tau_R must be finite and positive")
    y = out["y"].float()
    teacher = teacher_out["y"].float().detach()
    target = gt.float().detach()
    if y.shape != teacher.shape or y.shape != target.shape or y.ndim != 4 or y.shape[1] != 8:
        raise ValueError("FH12 Student/Teacher/GT must match WV3 [B,8,H,W]")
    weights = torch.as_tensor(q_weights, dtype=torch.float32, device=y.device).detach()
    if weights.shape != (y.shape[0],) or not bool(torch.isfinite(weights).all()) or not bool(((weights > 0) & (weights <= 1)).all()):
        raise ValueError("FH12 q weights must be finite [B] values in (0,1]")
    with torch.autocast(device_type=y.device.type, enabled=False):
        e_student = (y - target).abs().mean(dim=1, keepdim=True)
        e_teacher = (teacher - target).abs().mean(dim=1, keepdim=True)
        difficulty = (e_teacher / (e_teacher + tau)).detach()
        advantage = ((e_student - e_teacher).clamp_min(0) / (e_student + 1e-6)).detach()
        soft_error = (y - teacher).abs().mean(dim=1, keepdim=True)
        hard_i = ((1. + difficulty) * e_student).flatten(1).mean(1)
        soft_i = (.1 * (1. - difficulty) * advantage * soft_error).flatten(1).mean(1)
        edge_i = output_edge_loss_per_sample(y, target)
        loss_u = (hard_i + soft_i + .002 * weights * edge_i).mean()
        loss_a = (weights * hard_i).mean()
    return dict(L_U=loss_u, L_A=loss_a, hard=hard_i.mean(), soft=soft_i.mean(),
                edge=edge_i.mean(), edge_weighted=(.002 * weights * edge_i).mean(),
                hard_i=hard_i, soft_i=soft_i, edge_i=edge_i, difficulty=difficulty,
                advantage=advantage, q_weights=weights)


def routed_student_backward(model, losses, scale=1.0):
    """Assign grad(U)=dL_U/dU and grad(A)=dL_A/dA from one forward graph.

    With fp16 the caller initializes GradScaler by calling scaler.scale(L_U),
    passes scaler.get_scale(), then calls scaler.unscale_/step normally. Do not
    step either optimizer before this function completes both differentiations.
    """
    u = [p for p in model.backbone.parameters() if p.requires_grad]
    a = [p for p in model.aligner.parameters() if p.requires_grad]
    if not u or not a or set(map(id, u)) & set(map(id, a)):
        raise ValueError("FH12 requires distinct nonempty trainable U/A parameter sets")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("gradient scale must be finite and positive")
    grad_u = torch.autograd.grad(losses["L_U"] * scale, u, retain_graph=True, allow_unused=False)
    grad_a = torch.autograd.grad(losses["L_A"] * scale, a, allow_unused=False)
    for param, grad in zip(u + a, grad_u + grad_a):
        param.grad = grad.detach()
    return dict(u_parameters=len(u), a_parameters=len(a), scale=float(scale))
