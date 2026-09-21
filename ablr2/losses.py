"""Catalog-resolved losses and strict independent U/A gradient routing."""
from __future__ import annotations
import math
import torch
from pa.losses import output_edge_loss_per_sample
from pa.offset import sample_offsets
from pa.warp import warp_pan


def teacher_loss(model, out, gt, pan, ms, update_index, generator,
                 consistency_weight=1e-4, *, bands):
    if update_index < 0 or int(update_index) != update_index:
        raise ValueError('Completed updates must be a nonnegative integer')
    if consistency_weight not in (0., 1e-4, 3e-4):
        raise ValueError('Only TZERO, BASE and registered C3 consistency are allowed')
    if bands not in (4, 8) or gt.shape != out['y'].shape or gt.shape[1] != bands:
        raise ValueError('Teacher/GT band shape mismatch')
    active = bool(update_index % 2) and consistency_weight != 0
    rec = (out['y'].float()-gt.float().detach()).abs().mean()
    epsilon, off = pan.new_zeros((len(pan), 2)), rec.new_zeros(())
    if active:
        if generator is None or generator.device.type != 'cpu':
            raise ValueError('Independent CPU corruption generator required')
        epsilon = sample_offsets(len(pan), 2., generator).to(pan.device)
        with torch.no_grad():
            corrupted = warp_pan(pan.float(), epsilon)
        c_eps = model.predict_delta(corrupted, out['ms_base'].detach())
        off = (c_eps + epsilon - out['delta'].detach()).abs().mean()
    return dict(total=rec+consistency_weight*off, rec=rec, off=off,
                offset_active=active, epsilon=epsilon)


def _reliability(mode, y, q_weights, s_bar):
    if mode == 'CONST_HALF':
        return y.new_full((len(y),), .5)
    if mode == 'TRAIN_MEAN':
        if s_bar is None or not math.isfinite(float(s_bar)) or not 0 < float(s_bar) <= 1:
            raise ValueError('Measured full-train-view s_bar required')
        return y.new_full((len(y),), float(s_bar))
    if mode != 'ADAPTIVE' or q_weights is None:
        raise ValueError('Registered reliability mode/cache required')
    weights = torch.as_tensor(q_weights, device=y.device, dtype=torch.float32).detach()
    if weights.shape != (len(y),) or not bool(torch.isfinite(weights).all()) or not bool(((weights > 0)&(weights <= 1)).all()):
        raise ValueError('Reliability must be finite [B] in (0,1]')
    return weights


def student_losses(out, teacher_out, gt, component, tau_R=None, q_weights=None,
                   s_bar=None, *, bands):
    y, target = out['y'].float(), gt.float().detach()
    if bands not in (4, 8) or y.shape != target.shape or y.shape[1] != bands:
        raise ValueError('Student/GT band shape mismatch')
    needs_prediction = bool(component['teacher_predictions_used'])
    if needs_prediction != (teacher_out is not None):
        raise ValueError('Teacher prediction I/O does not match registered component')
    if not needs_prediction and (tau_R is not None or q_weights is not None or s_bar is not None):
        raise ValueError('Teacher-free/clone-only loss cannot consume calibration')
    alpha, beta, edge_weight = [float(component[key]) for key in ('alpha', 'beta', 'lambda_edge')]
    e_s = (y-target).abs().mean(dim=1, keepdim=True)
    difficulty, advantage = torch.zeros_like(e_s), torch.zeros_like(e_s)
    soft_i = y.new_zeros((len(y),))
    if needs_prediction:
        teacher = teacher_out['y'].float().detach()
        if teacher.shape != y.shape:
            raise ValueError('Teacher prediction band shape mismatch')
        e_t = (teacher-target).abs().mean(dim=1, keepdim=True)
        need_d = component['hard_mode'] == 'ADAPTIVE' or component['soft_mode'] == 'SELECTIVE'
        if need_d:
            if tau_R is None or not math.isfinite(float(tau_R)) or float(tau_R) <= 0:
                raise ValueError('Adaptive fitting requires measured positive tau_R')
            difficulty = (e_t/(e_t+float(tau_R))).detach()
        advantage = ((e_s-e_t).clamp_min(0)/(e_s+1e-6)).detach()
        error = (y-teacher).abs().mean(dim=1, keepdim=True)
        if component['soft_mode'] == 'UNIFORM':
            soft_i = (beta*error).flatten(1).mean(1)
        elif component['soft_mode'] == 'SELECTIVE':
            trust = 1.-difficulty if component['soft_trust'] else 1.
            gate = advantage if component['soft_advantage'] else 1.
            soft_i = (beta*trust*gate*error).flatten(1).mean(1)
        elif component['soft_mode'] != 'OFF':
            raise ValueError('Unsupported soft ablation')
    if component['hard_mode'] not in ('PLAIN', 'ADAPTIVE'):
        raise ValueError('Optional train-mean hard study requires a separate release')
    hard_i = ((1.+alpha*difficulty)*e_s).flatten(1).mean(1)
    edge_i = output_edge_loss_per_sample(y, target) if edge_weight else y.new_zeros((len(y),))
    s_e = _reliability(component['q_edge'], y, q_weights, s_bar)
    s_a = _reliability(component['q_aligner'], y, q_weights, s_bar)
    return dict(L_U=(hard_i+soft_i+edge_weight*s_e*edge_i).mean(), L_A=(s_a*hard_i).mean(),
        hard=hard_i.mean(), soft=soft_i.mean(), edge=edge_i.mean(),
        edge_weighted=(edge_weight*s_e*edge_i).mean(), hard_i=hard_i, soft_i=soft_i,
        edge_i=edge_i, difficulty=difficulty, advantage=advantage, q_weights=s_e, q_aligner=s_a)


def routed_student_backward(model, losses):
    u = [p for p in model.backbone.parameters() if p.requires_grad]
    a = [p for p in model.aligner.parameters() if p.requires_grad]
    if not u or set(map(id,u)) & set(map(id,a)):
        raise ValueError('Independent U/A parameter sets required')
    grad_u = torch.autograd.grad(losses['L_U'], u, retain_graph=bool(a), allow_unused=False)
    grad_a = torch.autograd.grad(losses['L_A'], a, allow_unused=False) if a else ()
    for parameter, gradient in zip(u+a, tuple(grad_u)+tuple(grad_a)):
        parameter.grad = gradient.detach()
    return dict(u_parameters=len(u), a_parameters=len(a))
