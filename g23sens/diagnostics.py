"""Fixed-probe diagnostics: no optimizer, .grad, RNG or stream mutation."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import random

import numpy as np
import torch

from fh12.model import state_hash


DIAGNOSTIC_STEPS = (0, 1000, 10000, 25000, 50000)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('A CUDA RNG checkpoint cannot resume on CPU')
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def isolated_probe(*models):
    saved = rng_state()
    modes = [(module, module.training) for model in models for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for model in models for buffer in model.buffers()]
    grads = [(p, None if p.grad is None else p.grad.detach().clone())
             for model in models for p in model.parameters()]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        for p, value in grads:
            p.grad = value
        for module, mode in modes:
            module.training = mode
        restore_rng(saved)


def distribution(value):
    x = torch.as_tensor(value).detach().double().flatten().cpu()
    if x.numel() == 0 or not bool(torch.isfinite(x).all()):
        raise FloatingPointError('Diagnostic distribution is empty or nonfinite')
    return dict(mean=float(x.mean()), min=float(x.min()), max=float(x.max()),
                **{'p%02d' % round(q*100): float(torch.quantile(x, q))
                   for q in (.01, .1, .25, .5, .75, .9, .99)})


def gradient_norm(loss, parameters):
    parameters = [p for p in parameters if p.requires_grad]
    if not parameters or not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(g.detach().double().square().sum())
                         for g in gradients if g is not None))


def fixed_probe(model, teacher, criterion, qweight, batch, optimizer, *, step,
                lambda_edge, initial_aligner_state):
    from g23sens.training import forward_objectives, routed_gradients
    with isolated_probe(model, teacher):
        model.eval()
        teacher.eval()
        device = next(model.parameters()).device
        gt, ms, lp, pan, meta = [batch[k].to(device) for k in ('gt','ms','lpan','pan','meta')]
        out, terms = forward_objectives(model, teacher, criterion, qweight,
                                       gt, ms, lp, pan, meta, lambda_edge)
        initial_aligner=copy.deepcopy(model.aligner)
        initial_aligner.load_state_dict(initial_aligner_state,strict=True)
        initial_aligner.eval().requires_grad_(False)
        with torch.no_grad():
            initial_delta=initial_aligner(model._view(pan.float()),model._view(out['ms_base'].float()))
            correction_change=(out['delta'].detach()-initial_delta).norm(dim=1)
        up = list(model.backbone.parameters())
        ap = list(model.aligner.parameters())
        # Raw computational graphs of K/E may depend on A. They are forbidden
        # contributions only in the optimizer routing, not disconnected graphs.
        gu, ga = routed_gradients(model, terms, retain_graph=True)
        expected = torch.autograd.grad(terms['L_A'], ap, retain_graph=True, allow_unused=True)
        route_match = all((g is None and h is None) or
                          (g is not None and h is not None and torch.equal(g, h))
                          for g, h in zip(ga, expected))
        if not route_match:
            raise AssertionError('A received a gradient other than weighted hard')
        # Explicitly changing the forbidden scalar paths cannot change A's route.
        alternate = dict(terms, L_U=terms['H'].mean()+37*terms['K'].mean()+19*terms['weighted_E'])
        _, alternate_a = routed_gradients(model, alternate, retain_graph=True)
        forbidden_equal = all((g is None and h is None) or
                              (g is not None and h is not None and torch.equal(g, h))
                              for g, h in zip(ga, alternate_a))
        if not forbidden_equal:
            raise AssertionError('Soft/edge entered the applied A gradient')
        shift = []
        for name, tensor in model.aligner.state_dict().items():
            shift.append((tensor.detach().cpu().double()-initial_aligner_state[name].double()).flatten())
        maps = terms['maps']
        return dict(step=int(step), diagnostic_type='FIXED_TRAIN_PROBE_NOT_EVALUATION',
            distribution={name: distribution(value) for name, value in
                dict(d_T=maps['difficulty'], hard_weight=maps['hard_weight'],
                     soft_weight=maps['soft_weight'], q_weight=terms['w'],
                     correction_norm=out['delta'].norm(dim=1),
                     correction_change_from_initial_norm=correction_change).items()},
            soft_positive_fraction=float((maps['soft_weight'] > 0).float().mean()),
            losses=dict(raw_H=float(terms['plain_H'].mean()), weighted_H=float(terms['H'].mean()),
                        raw_K=float(terms['plain_K'].mean()), weighted_K=float(terms['K'].mean()),
                        raw_E=float(terms['E'].mean()), q_weighted_E=float(terms['weighted_E']),
                        lambda_weighted_E=float(lambda_edge*terms['weighted_E']),
                        L_U=float(terms['L_U']), L_A=float(terms['L_A'])),
            gradient_norms=dict(U_hard=gradient_norm(terms['H'].mean(), up),
                U_KD=gradient_norm(terms['K'].mean(), up),
                U_edge_raw=gradient_norm(terms['E'].mean(), up),
                U_edge_applied=gradient_norm(lambda_edge*terms['weighted_E'], up),
                A_applied_hard=gradient_norm(terms['L_A'], ap)),
            lambda_E_actual=float(lambda_edge), q_ref=float(qweight.qref),
            tau_R_actual=float(criterion.tau),
            optimizer_lr={g['name']: float(g['lr']) for g in optimizer.param_groups},
            aligner_change_l2=float(torch.cat(shift).norm()),
            aligner_state_sha256=state_hash(model.aligner.state_dict()),
            gradient_routing=dict(A_hard_only=route_match,
                A_forbidden_soft_edge_offset_invariance=forbidden_equal,
                offset_objective_present=False,
                note='Direct optimizer contributions; raw soft/edge A Jacobians need not be zero'),
            q_asset_stats=qweight.stats)
