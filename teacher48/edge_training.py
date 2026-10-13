"""Explicit Edge-Guided Stage1 update engine, not a campaign/Stage2 runner."""
from __future__ import annotations

import copy
import math

import torch

from panda_method.edge_data import EdgeDataSources
from panda_method.edge_guided import EdgeGuidedStage1, ResidualPANAligner
from panda_method.edge_match import EdgeMatchSpec, EdgeMatcher
from panda_method.reconstruction import build_reconstructor


def build_edge_stage1(sources, *, teacher_spec, match_spec, aligner_view_margin,
                      roi_margin, teacher_seed, aligner_seed, epsilon_seed,
                      boundary_policy, lambda_abs=1., lambda_eps=.001):
    """Bind stored-LMS sources to an explicit P0/bicubic Teacher and matcher."""
    if not isinstance(sources, EdgeDataSources) or not isinstance(match_spec, EdgeMatchSpec):
        raise TypeError('EdgeDataSources and complete EdgeMatchSpec required')
    if teacher_spec.bands != sources.bands or match_spec.sensor != sources.sensor:
        raise ValueError('Sensor/band identities disagree')
    if teacher_spec.input_layout != 'P0' or teacher_spec.base_reference != 'BICUBIC_MS':
        raise ValueError('This Stage1 specification preserves P0/BICUBIC_MS reconstruction')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(aligner_seed)
        aligner = ResidualPANAligner(sources.bands, view_margin=aligner_view_margin)
    teacher = build_reconstructor(teacher_spec, seed=teacher_seed, role='T')
    return EdgeGuidedStage1(aligner, teacher, EdgeMatcher(match_spec),
        sources=sources.training_sources(), roi_margin=roi_margin,
        boundary_policy=boundary_policy, epsilon_seed=epsilon_seed,
        lambda_abs=lambda_abs, lambda_eps=lambda_eps)


def _norm(tensors):
    # Detached FP64 reductions keep the logging path out of the objective.
    value = sum(float(t.detach().double().square().sum()) for t in tensors if t is not None)
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite gradient norm')
    return math.sqrt(value)


def _alignment_log(branch):
    d = branch['match_diagnostics']
    # Preserve per-input vectors/failures, but never dump full edge-score maps.
    return {**{key: branch[key].detach().cpu().tolist() for key in ('b', 'delta', 'u')},
            **{key: d[key].detach().cpu().tolist() for key in
               ('tie_count', 'low_texture', 'boundary_hit', 'flat_score', 'support_count',
                'texture_pan', 'texture_lms')},
            **{key: (None if d[key] is None else d[key].detach().cpu().tolist()) for key in
               ('best_score', 'zero_score', 'second_best_gap', 'score_range')},
            'resolution': d['resolution'], 'roi_margin': d['roi_margin']}


class EdgeStage1Trainer:
    """One RR-only optimizer update, with loss/gradient diagnostics and restart.

    Exact per-term gradients are sampled at an explicitly supplied cadence;
    combined gradients are logged every update. This costs extra backwards on
    diagnostic steps and is not hidden in a future campaign's timing budget.
    Caller owns shuffling/augmentation RNG, scheduler, evaluation and HQNR best.
    """
    def __init__(self, stage, optimizer, *, gradient_log_interval,
                 effective_batch_size=None, microbatch_size=None):
        if not isinstance(stage, EdgeGuidedStage1):
            raise TypeError('New Edge-Guided Stage1 required; no legacy objective')
        if type(gradient_log_interval) is not int or gradient_log_interval < 1:
            raise ValueError('Explicit positive per-term gradient logging cadence required')
        self.stage, self.optimizer = stage, optimizer
        self.gradient_log_interval = gradient_log_interval
        for name, value in (('effective_batch_size', effective_batch_size), ('microbatch_size', microbatch_size)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(name + ' must be a positive integer or explicit unconstrained None')
        if ((effective_batch_size is None) != (microbatch_size is None)
                or effective_batch_size is not None and effective_batch_size % microbatch_size):
            raise ValueError('Pin both divisible effective/micro batch sizes or leave both unconstrained')
        self.effective_batch_size, self.microbatch_size = effective_batch_size, microbatch_size
        self._optimizer_names = self._binding()
        self.updates, self.poisoned = 0, False

    def _binding(self):
        expected = {id(p) for m in (self.stage.aligner, self.stage.teacher) for p in m.parameters()}
        actual = [id(p) for group in self.optimizer.param_groups for p in group['params']]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError('Optimizer must contain all and only this Aligner and Teacher')
        names = {id(p): name for name, p in self.stage.named_parameters()}
        return [[names[id(p)] for p in group['params']] for group in self.optimizer.param_groups]

    def _check_binding(self):
        if self._binding() != self._optimizer_names:
            raise ValueError('Optimizer parameter name/group order changed')

    def step(self, rr):
        return self.step_accumulated([rr])

    def _scope_indices(self):
        names = [name for name, _ in self.stage.aligner.named_parameters()]
        return {'whole': list(range(len(names))),
                'fc2': [i for i, n in enumerate(names) if n.startswith('fc2.')],
                'fc1': [i for i, n in enumerate(names) if n.startswith('fc1.')],
                'body': [i for i, n in enumerate(names) if n.startswith(('joint1.', 'joint2.', 'res.'))],
                'stem': [i for i, n in enumerate(names) if n.startswith(('pan_stem.', 'ms_stem.'))]}

    def _gradient_summary(self, grads, n_a):
        return {'aligner': _norm(grads[:n_a]), 'teacher': _norm(grads[n_a:]),
                'aligner_scopes': {scope: _norm([grads[i] for i in indices])
                                   for scope, indices in self._scope_indices().items()}}

    def step_accumulated(self, microbatches, *, expected_effective_batch=None):
        """Sample-weighted mean across microbatches, then exactly one step.

        Norm/GN/LN operations are sample-local and dropout is zero. The CPU
        epsilon stream advances once per sample independent of partitioning.
        Before optimizer mutation, an aborted effective batch restores epsilon
        state/counters and clears every partial gradient. Optimizer failures
        remain poisoned rather than pretending a partially applied step rolled
        back. No checkpoint or prediction is written by this engine.
        """
        if self.poisoned:
            raise RuntimeError('Failed optimizer update: restore a known complete checkpoint')
        self._check_binding()
        batches = list(microbatches)
        if not batches or any(len(batch.pan) < 1 for batch in batches):
            raise ValueError('Nonempty RR microbatches required')
        total = sum(len(batch.pan) for batch in batches)
        expected = self.effective_batch_size if expected_effective_batch is None else expected_effective_batch
        if expected is not None and (type(expected) is not int or expected <= 0 or total != expected):
            raise ValueError('Effective batch size differs from the pinned contract')
        if self.effective_batch_size is not None and total != self.effective_batch_size:
            raise ValueError('Per-call override cannot change the pinned effective batch')
        if self.microbatch_size is not None and any(len(batch.pan) != self.microbatch_size for batch in batches):
            raise ValueError('Microbatch size differs from the pinned contract')
        if any(batch.pan.shape[1:] != batches[0].pan.shape[1:] for batch in batches):
            raise ValueError('One fixed RR observation shape per effective batch required')
        rng = self.stage._epsilon_rng.get_state().clone()
        calls = self.stage.forward_calls
        self.stage.train()
        self.optimizer.zero_grad(set_to_none=True)
        a = list(self.stage.aligner.parameters())
        t = list(self.stage.teacher.parameters())
        term_grads, term_cosines = None, None
        diagnose = self.updates % self.gradient_log_interval == 0 or self.updates + 1 in (1, 10, 100)
        accumulated = {key: [None] * (len(a)+len(t)) for key in ('rec', 'abs_rr', 'eps_rr')} if diagnose else None
        metrics, support, alignment = {}, {}, {}
        scalar_keys = ('loss', 'rec', 'abs_rr', 'weighted_abs', 'eps_rr', 'weighted_eps_rr')
        support_keys = ('warp_support_fraction', 'native_prewarp_view_support_fraction',
                        'epsilon_prewarp_view_support_fraction', 'epsilon_rr_view_support_fraction')
        try:
            for rr in batches:
                weight = len(rr.pan) / total
                out = self.stage(rr)
                if diagnose:
                    for key in accumulated:
                        grads = torch.autograd.grad(out[key], a+t, retain_graph=True, allow_unused=True)
                        for i, g in enumerate(grads):
                            if g is not None:
                                value = g.detach() * weight
                                if accumulated[key][i] is None:
                                    accumulated[key][i] = value.clone()
                                else:
                                    accumulated[key][i].add_(value)
                (out['loss'] * weight).backward()
                for key in scalar_keys:
                    metrics[key] = metrics.get(key, 0.) + weight * float(out[key].detach())
                for key in support_keys:
                    support[key] = support.get(key, 0.) + weight * float(out[key].detach())
                support.setdefault('spatial_support_ok', []).extend(out['spatial_support_ok'].detach().cpu().tolist())
                for branch in ('native', 'shifted'):
                    logged = _alignment_log(out[branch])
                    merged = alignment.setdefault(branch, {})
                    for key, value in logged.items():
                        if isinstance(value, list):
                            merged.setdefault(key, []).extend(value)
                        elif key in merged and merged[key] != value:
                            raise ValueError('Mixed alignment diagnostic protocols in an effective batch')
                        else:
                            merged[key] = value
            if diagnose:
                accumulated['weighted_abs'] = [None if g is None else g * self.stage.lambda_abs
                                               for g in accumulated['abs_rr']]
                accumulated['weighted_eps_rr'] = [None if g is None else g * self.stage.lambda_eps
                                                  for g in accumulated['eps_rr']]
                term_grads = {key: self._gradient_summary(grads, len(a)) for key, grads in accumulated.items()}
                term_cosines = {}
                for scope, indices in self._scope_indices().items():
                    values = {}
                    for left, right in (('rec', 'weighted_abs'), ('rec', 'weighted_eps_rr'),
                                        ('weighted_abs', 'weighted_eps_rr')):
                        gl, gr = accumulated[left], accumulated[right]
                        nl, nr = _norm([gl[i] for i in indices]), _norm([gr[i] for i in indices])
                        dot = sum(float((gl[i].double()*gr[i].double()).sum()) for i in indices
                                  if gl[i] is not None and gr[i] is not None)
                        values[left+'__'+right] = (None if nl == 0 or nr == 0 else
                                                  max(-1., min(1., dot/(nl*nr))))
                    term_cosines[scope] = values
            norms = {'aligner': _norm([p.grad for p in a]), 'teacher': _norm([p.grad for p in t])}
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in a+t):
                raise FloatingPointError('Nonfinite gradient: no optimizer update')
            # Measured parameter displacement, not gradient-norm times LR.
            before_parameters = [p.detach().clone() for p in a+t]
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            self.stage._epsilon_rng.set_state(rng)
            self.stage.forward_calls = calls
            raise
        try:
            self.optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in a+t):
                raise FloatingPointError('Nonfinite parameter after update')
            update_norms = self._gradient_summary([p.detach()-old for p, old in zip(a+t, before_parameters)], len(a))
        except Exception:
            self.poisoned = True
            raise
        self.updates += 1
        return {'update': self.updates, 'effective_batch_size': total,
                'microbatch_sizes': [len(b.pan) for b in batches], **metrics,
                'gradient_parameter_state_update': self.updates - 1,
                'gradient_norm': norms, 'term_gradient_norm': term_grads,
                'term_gradient_cosine': term_cosines, 'optimizer_update_norm': update_norms,
                'learning_rates': [float(g['lr']) for g in self.optimizer.param_groups],
                'support': support, 'alignment': alignment}

    def state_dict(self):
        if self.poisoned:
            raise RuntimeError('Cannot save a partially failed optimizer update')
        self._check_binding()
        return copy.deepcopy({'schema': 'PANDA_EDGE_STAGE1_TRAINER_V2',
            'stage': self.stage.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'optimizer_parameter_names': self._optimizer_names,
            'optimizer_class': type(self.optimizer).__module__+'.'+type(self.optimizer).__qualname__,
            'gradient_log_interval': self.gradient_log_interval, 'updates': self.updates,
            'effective_batch_size': self.effective_batch_size, 'microbatch_size': self.microbatch_size})

    def load_state_dict(self, state):
        self._check_binding()
        kind = type(self.optimizer).__module__+'.'+type(self.optimizer).__qualname__
        if (state.get('schema') != 'PANDA_EDGE_STAGE1_TRAINER_V2'
                or state.get('optimizer_class') != kind
                or state.get('gradient_log_interval') != self.gradient_log_interval
                or state.get('effective_batch_size') != self.effective_batch_size
                or state.get('microbatch_size') != self.microbatch_size
                or type(state.get('updates')) is not int or state['updates'] < 0):
            raise ValueError('Edge Stage1 trainer checkpoint identity mismatch')
        if state.get('optimizer_parameter_names') != self._optimizer_names:
            raise ValueError('Optimizer checkpoint parameter name/group order mismatch')
        self.poisoned = True
        self.stage.load_state_dict(state['stage'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates, self.poisoned = state['updates'], False
