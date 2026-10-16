"""Student-only FP32 training, finite-gradient guards and strict new checkpoints."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random

import numpy as np
import torch

from .selection import SELECTION_RULE, resolve_policy

TRAINER_SCHEMA = 'PANDA_S2_CONDQ_STUDENT_TRAINER_V1'


def configure_precision():
    """Explicit per-process scientific policy; importing this file has no effects."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class TrainingSpec:
    lr: float = 1e-4
    weight_decay: float = .01
    betas: tuple = (.9, .999)
    adam_eps: float = 1e-8
    warmup_updates: int = 100
    total_updates: int = 50000
    min_lr: float = 0.
    effective_batch_size: int = 48
    # Actual donor24 baseline: 12 samples x 4 accumulation = 48. The method
    # document's 4 x 12 example is also valid but must be an explicit config.
    microbatch_size: int = 12
    gradient_log_interval: int = 100
    # Full frozen-weight SHA/Adam scans remain mandatory at checkpoints/eval.
    # Production can amortize these CPU/GPU copies; tensor version/frozen/finite
    # gradient guards still run every update. Default preserves strict diagnostics.
    full_audit_interval: int = 1

    def __post_init__(self):
        for name in ('lr', 'weight_decay', 'adam_eps', 'min_lr'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative optimizer parameter required: ' + name)
        if self.lr <= 0 or self.adam_eps <= 0 or self.min_lr > self.lr:
            raise ValueError('Positive LR/epsilon and min_lr <= peak LR required')
        if (len(self.betas) != 2 or any(type(v) not in (int, float) or not math.isfinite(v)
                                      or not 0 <= v < 1 for v in self.betas)):
            raise ValueError('Two finite Adam betas in [0,1) required')
        object.__setattr__(self, 'betas', tuple(self.betas))
        for name in ('total_updates', 'effective_batch_size', 'microbatch_size', 'gradient_log_interval', 'full_audit_interval'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('Positive integer training setting required: ' + name)
        if (type(self.warmup_updates) is not int or not 0 <= self.warmup_updates < self.total_updates
                or self.effective_batch_size % self.microbatch_size):
            raise ValueError('Valid warmup and divisible effective/microbatch sizes required')

    def rate(self, update):
        if type(update) is not int or not 1 <= update <= self.total_updates:
            raise ValueError('Update outside the pinned scheduler horizon')
        if self.warmup_updates and update <= self.warmup_updates:
            return self.lr * update / self.warmup_updates
        phase = (update - self.warmup_updates) / (self.total_updates - self.warmup_updates)
        return self.min_lr + .5 * (self.lr - self.min_lr) * (1 + math.cos(math.pi * phase))


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.random.get_rng_state().clone(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device inventory differs from resume checkpoint')
        torch.cuda.set_rng_state_all(state['cuda'])


def _norm(values):
    squared = sum(float(v.detach().double().square().sum()) for v in values if v is not None)
    if not math.isfinite(squared):
        raise FloatingPointError('Nonfinite Student gradient')
    return math.sqrt(squared)


def _finite_tree(value):
    if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite checkpoint tensor')
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_tree(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError('Nonfinite checkpoint scalar')


class StudentTrainer:
    def __init__(self, stage, training=None, *, evaluation_pin=None, hqnr_target=None):
        from .method import ConditionalQStage2
        if not isinstance(stage, ConditionalQStage2):
            raise TypeError('New ConditionalQStage2 required; legacy E/EQ/MASK2 trainer is incompatible')
        configure_precision()
        self.stage, self.training = stage, training or TrainingSpec()
        if not isinstance(self.training, TrainingSpec):
            raise TypeError('Pinned TrainingSpec required')
        self.optimizer = torch.optim.AdamW(stage.student.parameters(), lr=self.training.lr,
            betas=self.training.betas, eps=self.training.adam_eps, weight_decay=self.training.weight_decay)
        self.updates, self.poisoned = 0, False
        self.evaluator_pin = deepcopy(evaluation_pin)
        self.selection_policy = resolve_policy(stage.spec.sensor, hqnr_target)
        self._initial_selection = self.selection_policy.manifest()
        self._optimizer_names = self._binding()
        self.stage.assert_frozen_binding()
        self._initial_pipeline = deepcopy(stage.pipeline_manifest())
        self._initial_stage_contract = deepcopy(stage.get_extra_state())

    def _binding(self):
        self.stage.assert_optimizer(self.optimizer)
        names = {id(p): name for name, p in self.stage.student.named_parameters()}
        return [[names[id(p)] for p in g['params']] for g in self.optimizer.param_groups]

    def contract(self):
        return {'schema': TRAINER_SCHEMA, 'training': asdict(self.training),
                'pipeline': deepcopy(self._initial_pipeline), 'eval_protocol': deepcopy(self.evaluator_pin),
                'stage_contract': deepcopy(self._initial_stage_contract),
                'optimizer_class': 'torch.optim.adamw.AdamW', 'optimizer_parameters': self._optimizer_names,
                'checkpoint_selection': deepcopy(self._initial_selection), 'precision': 'FP32_no_AMP_no_TF32',
                'loss_reduction': ('all_bands_mean_then_full_BHW_mean'
                    if self.stage.spec.mask_policy == 'full_image_no_warp_valid_mask'
                    else self.stage.spec.hard_soft_reduction),
                'mask_policy': self.stage.spec.mask_policy}

    def assert_contract(self, *, full_audit=True):
        if self.poisoned:
            raise RuntimeError('Trainer poisoned: rebuild and restore a complete checkpoint')
        if self.selection_policy.manifest() != self._initial_selection:
            raise ValueError('Checkpoint selection target/rule changed during training')
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark:
            raise ValueError('Precision policy changed: TF32/benchmark must remain disabled')
        if (self._binding() != self._optimizer_names or self.stage.pipeline_manifest() != self._initial_pipeline
                or self.stage.get_extra_state() != self._initial_stage_contract):
            raise ValueError('Training/model/source/calibration contract changed')
        self.stage._assert_contract()
        self._check_optimizer_state(self.optimizer.state_dict(), self.updates, full_tensor_check=full_audit)
        for module in (self.stage.aligner, self.stage.teacher):
            if module is not None and (module.training or any(p.grad is not None for p in module.parameters())):
                raise ValueError('Frozen donor must stay eval and have no gradients')
        if full_audit:
            self.stage.assert_frozen_binding()

    def _check_optimizer_state(self, state, update, *, full_tensor_check=True):
        if type(self.optimizer) is not torch.optim.AdamW or len(state['param_groups']) != 1:
            raise ValueError('Pinned Student-only AdamW required')
        static = {'betas': self.training.betas, 'eps': self.training.adam_eps,
                  'weight_decay': self.training.weight_decay, 'amsgrad': False, 'maximize': False,
                  'capturable': False, 'differentiable': False, 'foreach': None, 'fused': None}
        group = state['param_groups'][0]
        if any(group.get(k) != v for k, v in static.items()):
            raise ValueError('Optimizer hyperparameters differ from TrainingSpec')
        expected_lr = self.training.rate(update) if update else self.training.lr
        if type(group.get('lr')) not in (float, int) or group['lr'] != expected_lr:
            raise ValueError('Optimizer LR differs from the pinned scheduler state')
        if group.get('params') != self.optimizer.state_dict()['param_groups'][0]['params']:
            raise ValueError('Optimizer checkpoint parameter ordering/count mismatch')
        params = list(self.stage.student.parameters())
        allowed_ids = set(group['params'])
        if set(state['state']) != (allowed_ids if update else set()):
            raise ValueError('Incomplete or unexpected optimizer moment state')
        for pid, item in state['state'].items():
            if set(item) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Unexpected AdamW state schema')
            parameter = params[group['params'].index(pid)]
            for key in ('exp_avg', 'exp_avg_sq'):
                if (not isinstance(item[key], torch.Tensor) or item[key].shape != parameter.shape
                        or item[key].dtype != parameter.dtype
                        or (full_tensor_check and not bool(torch.isfinite(item[key]).all()))):
                    raise ValueError('AdamW moment shape/dtype mismatch')
            if full_tensor_check and not bool((item['exp_avg_sq'] >= 0).all()):
                raise ValueError('AdamW second moments must be nonnegative')
            if not isinstance(item['step'], torch.Tensor) or item['step'].numel() != 1 or float(item['step']) != update:
                raise ValueError('AdamW step counter differs from completed updates')

    def step(self, rr):
        return self.step_accumulated([rr])

    def step_accumulated(self, batches):
        audit = (self.updates == 0 or (self.updates + 1) % self.training.full_audit_interval == 0)
        self.assert_contract(full_audit=audit)
        batches = list(batches)
        total = sum(len(b.pan) for b in batches)
        if (not batches or total != self.training.effective_batch_size or
                any(len(b.pan) != self.training.microbatch_size for b in batches)):
            raise ValueError('Microbatch/effective batch sizes differ from pinned contract')
        if any(b.pan.shape[1:] != batches[0].pan.shape[1:] for b in batches):
            raise ValueError('Same spatial shape required for sample-weighted accumulation')
        rate = self.training.rate(self.updates + 1)
        self.stage.train()
        self.optimizer.zero_grad(set_to_none=True)
        params = list(self.stage.student.parameters())
        diagnostic = self.updates == 0 or (self.updates + 1) % self.training.gradient_log_interval == 0
        components = {key: [None] * len(params) for key in ('hard', 'soft')} if diagnostic else None
        metrics, corrections, bin_counts, ranks, weights = {}, [], None, [], []
        scalar_keys = ('loss', 'hard_weighted', 'soft_unscaled', 'soft_weighted', 'gt_raw',
                       'e_extra_raw', 'q_extra_raw', 'd_e_mean', 'a_T_mean', 'soft_active_fraction',
                       'w_h_mean', 'q_roi_mean', 'g_roi_mean', 'r_q_roi_mean', 'warp_support_fraction')
        try:
            for rr in batches:
                weight = len(rr.pan) / total
                out = self.stage(rr, return_maps=True)
                if not bool(torch.isfinite(out['loss'])):
                    raise FloatingPointError('Nonfinite loss; optimizer not called')
                if diagnostic:
                    for key, field in (('hard', 'hard_weighted'), ('soft', 'soft_weighted')):
                        gradients = (torch.autograd.grad(out[field] * weight, params, retain_graph=True,
                                     allow_unused=True) if out[field].requires_grad else [None] * len(params))
                        for i, grad in enumerate(gradients):
                            if grad is not None:
                                if components[key][i] is None:
                                    components[key][i] = grad.detach().clone()
                                else:
                                    components[key][i].add_(grad.detach())
                (out['loss'] * weight).backward()
                for key in scalar_keys:
                    if key in out and out[key] is not None:
                        x = out[key]
                        number = float(x.detach()) if isinstance(x, torch.Tensor) else float(x)
                        if not math.isfinite(number):
                            raise FloatingPointError('Nonfinite diagnostic: ' + key)
                        metrics[key] = metrics.get(key, 0.) + weight * number
                corrections.extend(out['correction'].detach().cpu().tolist())
                if not bool(out['spatial_support_ok'].all()):
                    raise ValueError('Invalid Scharr support; optimizer not called')
                # Exact effective-batch quantiles, not means of microbatch p95s.
                if 'w_h' in out:
                    values = out['w_h'].detach()
                    if out.get('loss_mask') is not None:
                        values = values[out['loss_mask'].bool()]
                    weights.append(values.flatten().cpu())
                if 'r_q_bar' in out:
                    ranks.append(out['r_q_bar'][..., 8:-8, 8:-8].detach().flatten().cpu())
                if out.get('bin_counts') is not None:
                    counts = torch.as_tensor(out['bin_counts']).cpu()
                    bin_counts = counts.clone() if bin_counts is None else bin_counts + counts
            self.stage._assert_contract()
            if audit:
                self.stage.assert_frozen_binding()
            for module in (self.stage.aligner, self.stage.teacher):
                if module is not None and any(p.grad is not None for p in module.parameters()):
                    raise RuntimeError('Frozen donor received a gradient')
            gradient_norm = _norm(p.grad for p in params)
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        try:
            for group in self.optimizer.param_groups:
                group['lr'] = rate
            self.optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in params):
                raise FloatingPointError('Nonfinite Student weights after update')
            self.stage._assert_contract()
            if audit:
                self.stage.assert_frozen_binding()
            self._check_optimizer_state(self.optimizer.state_dict(), self.updates + 1, full_tensor_check=audit)
        except Exception:
            self.poisoned = True
            raise
        self.updates += 1
        gradient = {'student': gradient_norm, 'aligner': 0., 'teacher': 0., 'diagnostic_computed': diagnostic}
        if diagnostic:
            hard, soft = components['hard'], components['soft']
            hn, sn = _norm(hard), _norm(soft)
            dot = sum(float(h.double().mul(s.double()).sum()) for h, s in zip(hard, soft)
                      if h is not None and s is not None)
            gradient.update(hard_norm=hn, beta_soft_norm=sn,
                            hard_soft_cosine=max(-1., min(1., dot/(hn*sn))) if hn and sn else None)
        if weights:
            values = torch.cat(weights)
            metrics.update(w_h_max=float(values.max()), w_h_p95=float(torch.quantile(values, .95)))
        if ranks:
            metrics['r_q_roi_quantiles'] = torch.quantile(torch.cat(ranks), torch.tensor([0., .25, .5, .75, 1.])).tolist()
        return {'update': self.updates, 'effective_batch_size': total,
                'microbatch_sizes': [len(b.pan) for b in batches], 'learning_rate': rate,
                **metrics, 'gradient': gradient, 'correction_dy_dx': corrections,
                'bin_counts': None if bin_counts is None else bin_counts.tolist(),
                'spatial_support_ok': True, 'mask_policy': self.stage.spec.mask_policy}

    def state_dict(self, *, stream_state=None):
        self.assert_contract()
        return deepcopy({'schema': TRAINER_SCHEMA, 'contract': self.contract(),
            'stage': self.stage.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'scheduler': {'completed_updates': self.updates, 'spec': asdict(self.training)},
            'updates': self.updates, 'rng': rng_state(), 'stream_state': stream_state})

    def load_state_dict(self, state):
        self.assert_contract()
        if (state.get('schema') != TRAINER_SCHEMA or state.get('contract') != self.contract()
                or type(state.get('updates')) is not int
                or not 0 <= state['updates'] <= self.training.total_updates
                or state.get('scheduler') != {'completed_updates': state['updates'], 'spec': asdict(self.training)}):
            raise ValueError('Stage 2 checkpoint contract/schema/schedule mismatch')
        _finite_tree(state['stage'])
        _finite_tree(state['optimizer'])
        self._check_optimizer_state(state['optimizer'], state['updates'])
        current = self.stage.state_dict()
        if set(current) != set(state['stage']):
            raise ValueError('Model checkpoint key mismatch')
        for key, tensor in current.items():
            other = state['stage'][key]
            if isinstance(tensor, torch.Tensor) and (not isinstance(other, torch.Tensor)
                    or tensor.shape != other.shape or tensor.dtype != other.dtype):
                raise ValueError('Model checkpoint shape/dtype mismatch: ' + key)
        try:
            self.stage.load_state_dict(state['stage'], strict=True)
            self.optimizer.load_state_dict(state['optimizer'])
            self.updates = state['updates']
            self.assert_contract()
            restore_rng(state['rng'])
        except Exception:
            self.poisoned = True
            raise
        return deepcopy(state.get('stream_state'))
