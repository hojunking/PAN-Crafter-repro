"""RR-only method integration, deliberately not a migrated Teacher48 campaign.

No experiment case, learning rate, capacity, ROI, batch schedule or FR policy is
invented here. Callers supply the existing backbone/spatial definitions and an
optimizer. This module never evaluates, uploads, saves or starts a run by itself.
"""
from __future__ import annotations

import copy

import torch

from panda_method.global_data import GlobalRRDataSources
from panda_method.global_rr import GlobalPANAligner, GlobalRRStage1
from panda_method.losses import AbsoluteGradientLoss
from panda_method.reconstruction import build_reconstructor


def build_global_rr_stage1(sources, *, teacher_spec, spatial_spec, aligner_view_margin,
                          teacher_seed, aligner_seed, epsilon_seed, boundary_policy,
                          calibration_protocol):
    """Join actual source/asset identity to the new model without fake scene IDs."""
    if not isinstance(sources, GlobalRRDataSources):
        raise TypeError('Verified GlobalRRDataSources required')
    if teacher_spec.bands != sources.bands or spatial_spec.sensor != sources.sensor:
        raise ValueError('Backbone/spatial specification and data sensor disagree')
    if teacher_spec.uses_lpan:
        raise ValueError('This stored-LMS adapter supplies P0 only; PL/PH/PLH requires '
                         'a separately pinned LPAN data adapter, not an invented cache')
    asset = sources.asset
    with torch.random.fork_rng(devices=[]):
        # CPU construction must not reseed an already active CUDA campaign.
        torch.random.default_generator.manual_seed(aligner_seed)
        aligner = GlobalPANAligner(sources.bands, view_margin=aligner_view_margin,
            sensor=sources.sensor, train_source_sha256=asset.train_source_sha256,
            prior_asset_sha256=asset.fingerprint(), canonical_prior_dy_dx=asset.vector,
            min_prior_magnitude=asset.min_magnitude, max_prior_magnitude=asset.max_magnitude)
    teacher = build_reconstructor(teacher_spec, seed=teacher_seed, role='T')
    return GlobalRRStage1(aligner, teacher, AbsoluteGradientLoss(spatial_spec),
        sources=sources.training_sources(), epsilon_seed=epsilon_seed,
        boundary_policy=boundary_policy, calibration_protocol=calibration_protocol)


class GlobalRRTrainer:
    """One explicitly supplied RR batch, one optimizer update; no FR stream.

    The caller owns LR scheduling, batching, model selection and official
    evaluation. This small engine fixes the method coefficients to 1/1/.001.
    It does not reuse the old RR+FR BranchObjective or its campaign recipes.
    """
    def __init__(self, stage, optimizer):
        if not isinstance(stage, GlobalRRStage1):
            raise TypeError('Explicit RR-only bGlobal Stage1 required')
        self.stage, self.optimizer = stage, optimizer
        self._optimizer_names = self._parameter_binding()
        self.updates, self.poisoned = 0, False

    def _parameter_binding(self):
        expected = {id(p) for m in (self.stage.aligner, self.stage.teacher) for p in m.parameters()}
        actual = [id(p) for group in self.optimizer.param_groups for p in group['params']]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError('Optimizer must contain all and only this Aligner and Teacher')
        names = {id(p): name for name, p in self.stage.named_parameters()}
        return [[names[id(p)] for p in group['params']] for group in self.optimizer.param_groups]

    def _check_binding(self):
        if self._parameter_binding() != self._optimizer_names:
            raise ValueError('Optimizer parameter-group/name order changed')

    def step(self, rr):
        if self.poisoned:
            raise RuntimeError('Failed optimizer update requires restoring a known checkpoint')
        self._check_binding()
        before_rng = self.stage._epsilon_rng.get_state().clone()
        before_calls = self.stage.forward_calls
        self.stage.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            out = self.stage(rr)
            out['loss'].backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                   for p in self.stage.parameters()):
                raise FloatingPointError('Nonfinite RR method gradients; no optimizer update')
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            self.stage._epsilon_rng.set_state(before_rng)
            self.stage.forward_calls = before_calls
            raise
        try:
            self.optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in self.stage.parameters()):
                raise FloatingPointError('Nonfinite parameter after optimizer update')
        except Exception:
            self.poisoned = True
            raise
        self.updates += 1
        return {'update': self.updates,
                **{key: float(out[key].detach()) for key in
                   ('loss', 'rec', 'abs_rr', 'eps_rr', 'weighted_eps_rr',
                    'warp_support_fraction', 'epsilon_rr_view_support_fraction')}}

    def state_dict(self):
        if self.poisoned:
            raise RuntimeError('Cannot save a partially failed optimizer update')
        self._check_binding()
        return copy.deepcopy({'schema': 'PANDA_BGLOBAL_RR_TRAINER_V1',
            'stage': self.stage.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'optimizer_parameter_names': self._optimizer_names,
            'optimizer_class': type(self.optimizer).__module__ + '.' + type(self.optimizer).__qualname__,
            'updates': self.updates})

    def load_state_dict(self, state):
        self._check_binding()
        kind = type(self.optimizer).__module__ + '.' + type(self.optimizer).__qualname__
        if (state.get('schema') != 'PANDA_BGLOBAL_RR_TRAINER_V1'
                or state.get('optimizer_class') != kind
                or type(state.get('updates')) is not int or state['updates'] < 0):
            raise ValueError('RR-only trainer/checkpoint identity mismatch')
        if state.get('optimizer_parameter_names') != self._optimizer_names:
            raise ValueError('Optimizer checkpoint parameter-group/name order mismatch')
        # Stage validates its method, data, prior and RNG identities before load.
        self.poisoned = True
        self.stage.load_state_dict(state['stage'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates, self.poisoned = state['updates'], False
