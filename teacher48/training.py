"""Disk-free Stage-1 training engine for the Teacher48H scalar campaign.

The controller owns admission, evaluation, durable numeric logs and strict
target-only checkpoint retention. Nothing in this module writes model state.
RR and FR have independent means, samplers and epsilon random streams.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import copy
import math
import random
import time

import numpy as np
import torch

from pa.offset import sample_offsets
from pa.warp import warp_support_mask
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_method.data import ObservationBatch
from panda_method.stages import Stage1, _support
from .plan import (stream_seeds, SEED_DERIVATION_PROTOCOL, STREAM_SEED_NAMESPACE,
                   METHOD_CONTRACT, HEAD_CONDITIONING)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    horizon: int
    lambda_abs: float
    lambda_eps: float
    optimizer: str
    optimizer_betas: tuple
    optimizer_eps: float
    lr_U: float
    lr_A: float
    weight_decay: float
    warmup_updates: int
    min_lr_U: float
    min_lr_A: float
    effective_batch_RR: int
    effective_batch_FR: int
    microbatch_RR: int
    microbatch_FR: int
    precision: str
    scheduler: str

    def __post_init__(self):
        stream_seeds(self.seed)
        for name in ("horizon", "effective_batch_RR", "effective_batch_FR",
                     "microbatch_RR", "microbatch_FR"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.horizon < 2:
            raise ValueError("Cosine horizon must include at least two updates")
        if (type(self.warmup_updates) is not int
                or not 0 <= self.warmup_updates < self.horizon):
            raise ValueError("Require 0 <= warmup_updates < horizon")
        for name in ("lambda_abs", "lambda_eps", "optimizer_eps", "lr_U", "lr_A"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        for name in ("weight_decay", "min_lr_U", "min_lr_A"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.min_lr_U > self.lr_U or self.min_lr_A > self.lr_A:
            raise ValueError("Minimum LR cannot exceed the case peak LR")
        if (len(self.optimizer_betas) != 2
                or any(not math.isfinite(v) or not 0 <= v < 1 for v in self.optimizer_betas)):
            raise ValueError("Two explicit Adam betas in [0,1) required")
        object.__setattr__(self, "optimizer_betas", tuple(self.optimizer_betas))
        if self.optimizer not in ("Adam", "AdamW"):
            raise ValueError("Only explicitly approved Adam/AdamW supported")
        if self.precision != "fp32":
            raise ValueError("Method implementation is FP32; do not silently change precision")
        if self.scheduler != "warmup_cosine":
            raise ValueError("Explicit warmup_cosine baseline required")
        for branch in ("RR", "FR"):
            if getattr(self, f"microbatch_{branch}") > getattr(self, f"effective_batch_{branch}"):
                raise ValueError("Microbatch cannot exceed the effective branch batch")

    @classmethod
    def from_mapping(cls, mapping):
        required = {f.name for f in fields(cls)}
        missing = required - mapping.keys()
        if missing:
            raise ValueError(f"Unpinned training settings: {sorted(missing)}")
        if any(mapping.get(k) not in (None, False, 0)
               for k in ("gradient_clip", "grad_clip", "clip_grad_norm")):
            raise ValueError("No new/adaptive gradient clipping in this campaign")
        return cls(**{name: mapping[name] for name in required})


def learning_rates(config: TrainingConfig, update: int):
    """LR used by a 1-indexed update, including exact endpoint minimum.

    Warmup uses peak*u/warmup. With no warmup the first update is at peak;
    with warmup the warmup endpoint is at peak. The horizon update is at min.
    This indexing convention is recorded in every engine manifest.
    """
    if type(update) is not int or not 1 <= update <= config.horizon:
        raise ValueError("Update outside the pinned scheduler horizon")
    if config.warmup_updates and update <= config.warmup_updates:
        factor = update / config.warmup_updates
        return {"U": config.lr_U * factor, "A": config.lr_A * factor}
    start = max(1, config.warmup_updates)
    phase = (update - start) / (config.horizon - start)
    factor = (1. + math.cos(math.pi * phase)) / 2.
    return {"U": config.min_lr_U + (config.lr_U - config.min_lr_U) * factor,
            "A": config.min_lr_A + (config.lr_A - config.min_lr_A) * factor}


def model_state_hash(stage):
    """A/T tensor state only; RNG, Python metadata and optimizer are separate."""
    return canonical_sha256({"A": state_dict_sha256(stage.aligner),
                             "T": state_dict_sha256(stage.teacher)})


class BranchObjective:
    """Use the vetted method paths with campaign coefficients and split RNG."""

    def __init__(self, stage: Stage1, config: TrainingConfig):
        if not isinstance(stage, Stage1):
            raise TypeError("A verified panda_method.Stage1 is required")
        if stage.aligner.HEAD_CONDITIONING != HEAD_CONDITIONING:
            raise ValueError('Teacher48 requires the v1.3 raw signed current-grid prior head')
        self.stage, self.config = stage, config
        seeds = stream_seeds(config.seed)
        self.generators = {key: torch.Generator(device="cpu").manual_seed(seeds[f"{key}_epsilon"])
                           for key in ("rr", "fr")}
        self.calls = {"rr": 0, "fr": 0}

    def rng_state(self):
        return {"generators": {k: v.get_state().clone() for k, v in self.generators.items()},
                "calls": dict(self.calls)}

    def restore_rng(self, state):
        for key in self.generators:
            self.generators[key].set_state(state["generators"][key].cpu())
        self.calls = dict(state["calls"])

    def _validate(self, batch, branch, training):
        if not all(p.requires_grad for module in (self.stage.aligner, self.stage.teacher)
                   for p in module.parameters()):
            raise ValueError("Full Stage1 requires all shared A and T parameters trainable")
        batch.validate(self.stage.teacher.spec.bands, training=training, require_gt=branch == "rr")
        if batch.lineage.resolution != branch.upper():
            raise ValueError("Observation resolution differs from the objective branch")
        if training:
            self.stage.sources.check(batch, branch.upper())

    def evaluate(self, batch: ObservationBatch, branch: str, *, epsilon=None, training=True):
        if branch not in self.generators:
            raise ValueError("Branch must be rr or fr")
        self._validate(batch, branch, training)
        before = self.rng_state()
        try:
            with torch.autocast(device_type=batch.pan.device.type, enabled=False):
                if branch == "rr":
                    native = self.stage.native_prediction(batch, return_aligner_details=True)
                    correction = native["correction"]
                    head_details = native['aligner_details']
                    rec = (native["y"] - batch.gt.detach().float()).abs().mean()
                    absolute = self.stage.spatial.stage1(native["pan_aligned"], batch.gt)
                    support = _support(batch.pan, correction, self.stage.spatial,
                                       self.stage.boundary_policy)
                else:
                    with torch.no_grad():
                        correction, head_details = self.stage.aligner(batch.pan.float(), batch.native_lms.float(),
                                                        batch.prior_dy_dx, return_details=True)
                    native = None
                    # FR has no absolute/rec term. Its native support is numeric only.
                    support = _support(batch.pan, correction, self.stage.spatial, "report")
                if epsilon is None:
                    epsilon = sample_offsets(len(batch.pan), 2., self.generators[branch]).to(batch.pan.device)
                else:
                    epsilon = epsilon.detach().to(device=batch.pan.device, dtype=torch.float32)
                    if (epsilon.shape != (len(batch.pan), 2)
                            or not bool(torch.isfinite(epsilon).all())
                            or bool((epsilon.norm(dim=-1) > 2. + 1e-6).any())):
                        raise ValueError("Diagnostic epsilon must be finite [B,2] inside radius two")
                eps_loss, c_eps = self.stage._epsilon(batch, correction, epsilon)
                raw = {f"eps_{branch}": eps_loss}
                if branch == "rr":
                    raw.update(rec=rec, abs_rr=absolute)
                weighted = {name: term * (self.config.lambda_abs if name == "abs_rr"
                            else 1. if name == "rec" else self.config.lambda_eps)
                            for name, term in raw.items()}
                loss = sum(weighted.values())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite full-method branch loss")
                mask = warp_support_mask(*batch.pan.shape[-2:], epsilon)
                margin = self.stage.aligner.view_margin
                h, w = batch.pan.shape[-2:]
                view_support = mask[..., margin:h-margin, margin:w-margin].float().mean()
                prior = batch.prior_dy_dx.detach().float()
                residual = correction.detach() - prior
                scaled = prior / prior.abs().amax(dim=-1, keepdim=True)
                direction = scaled / scaled.norm(dim=-1, keepdim=True)
                normal = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
                numeric = {"prior_norm": prior.norm(dim=-1).mean(),
                           "residual_norm": residual.norm(dim=-1).mean(),
                           "correction_norm": correction.detach().norm(dim=-1).mean(),
                           "alpha_mean": (residual * direction).sum(-1).mean(),
                           "beta_mean": (residual * normal).sum(-1).mean(),
                           "warp_support_fraction": support["warp_support_fraction"],
                           "spatial_support_fraction": support["spatial_support_ok"].float().mean(),
                           "epsilon_view_support_fraction": view_support}
                # No clipping/renormalization: log actual current-grid ranges
                # from the same native forward, without another A invocation.
                for axis, index in (('dy', 0), ('dx', 1)):
                    values = prior[:, index]
                    numeric.update({f'prior_{axis}_min': values.min(), f'prior_{axis}_max': values.max(),
                                    f'prior_{axis}_mean': values.mean()})
                for name in ('fc1_preactivation', 'head_hidden'):
                    values = head_details[name].detach()
                    numeric.update({name+'_min': values.min(), name+'_max': values.max(),
                                    name+'_rms': values.square().mean().sqrt(),
                                    name+'_finite_fraction': torch.isfinite(values).float().mean()})
        except Exception:
            self.restore_rng(before)
            raise
        self.calls[branch] += 1
        return {"loss": loss, "raw": raw, "weighted": weighted, "numeric": numeric,
                "epsilon": epsilon, "correction": correction, "c_eps": c_eps, "native": native}


class Trainer:
    """One optimizer boundary per complete independent RR + FR effective batch."""

    def __init__(self, stage: Stage1, config: TrainingConfig):
        self.stage, self.config = stage, config
        self.objective = BranchObjective(stage, config)
        a_parameters, t_parameters = list(stage.aligner.parameters()), list(stage.teacher.parameters())
        if set(map(id, a_parameters)) & set(map(id, t_parameters)):
            raise ValueError("Shared A and T must not alias parameters")
        if any(p.dtype != torch.float32 for p in a_parameters + t_parameters):
            raise ValueError("Pinned FP32 method requires FP32 parameters")
        optimizer_type = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[config.optimizer]
        self.optimizer = optimizer_type(
            [{"params": t_parameters, "lr": config.lr_U, "role": "U"},
             {"params": a_parameters, "lr": config.lr_A, "role": "A"}],
            betas=config.optimizer_betas, eps=config.optimizer_eps, weight_decay=config.weight_decay)
        self.updates = 0
        self.poisoned = False
        self.initial_hashes = {"A": state_dict_sha256(stage.aligner),
                               "T": state_dict_sha256(stage.teacher)}

    def manifest(self):
        return {**METHOD_CONTRACT, "config": asdict(self.config), "streams": stream_seeds(self.config.seed),
                "stream_namespace": STREAM_SEED_NAMESPACE,
                "seed_derivation_protocol": SEED_DERIVATION_PROTOCOL, "probe_seed": 261013900,
                "epsilon_radius": 2., "epsilon_distribution": "UNIFORM_AREA_DISK",
                "epsilon_cadence": "EVERY_MICROBATCH_BOTH_BRANCHES",
                "epsilon_target": "STOPGRAD_NATIVE_CORRECTION",
                "lambda_rec": 1., "lambda_abs": self.config.lambda_abs,
                "lambda_eps_RR": self.config.lambda_eps, "lambda_eps_FR": self.config.lambda_eps,
                "reduction": "RR_SAMPLE_MEAN_PLUS_FR_SAMPLE_MEAN",
                "lr_indexing": "ONE_BASED_WARMUP_LINEAR_PEAK_THEN_COSINE_EXACT_ENDPOINT_MIN",
                "gradient_clipping": None, "initial_hashes": self.initial_hashes,
                "sources": asdict(self.stage.sources), "spatial": asdict(self.stage.spatial.spec),
                "pipeline": {**self.stage.pipeline_manifest(),
                    "eps": {"radius": 2., "lambda": self.config.lambda_eps,
                            "stop_reference": True, "cadence": "every_microbatch_RR_and_FR",
                            "generator": "independent_named_RR_FR_CPU"}}}

    def _batches(self, batches, branch):
        batches = tuple(batches)
        if not batches or any(not isinstance(b, ObservationBatch) for b in batches):
            raise ValueError("Each branch requires finite ObservationBatch microbatches")
        expected = getattr(self.config, f"effective_batch_{branch}")
        cap = getattr(self.config, f"microbatch_{branch}")
        if sum(len(b.pan) for b in batches) != expected:
            raise ValueError(f"{branch} microbatch sample total differs from pinned effective batch")
        if any(len(b.pan) > cap for b in batches):
            raise ValueError(f"{branch} exceeds the pinned microbatch size")
        # Fixed pixel geometry is essential for sample-weighted loss equivalence.
        if len({tuple(b.pan.shape[1:]) for b in batches}) != 1:
            raise ValueError("Mixed spatial sizes within one branch effective batch are not supported")
        return batches

    def step(self, rr_batches, fr_batches, *, data_wait_seconds=0.):
        if self.poisoned or self.updates >= self.config.horizon:
            raise RuntimeError("Failed engine needs fresh attempt, or pinned horizon is complete")
        if not math.isfinite(data_wait_seconds) or data_wait_seconds < 0:
            raise ValueError("Data wait must be a finite nonnegative duration")
        rr, fr = self._batches(rr_batches, "RR"), self._batches(fr_batches, "FR")
        rates = learning_rates(self.config, self.updates + 1)
        for group in self.optimizer.param_groups:
            group["lr"] = rates[group["role"]]
        self.stage.train()
        self.optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        before = self.objective.rng_state()
        totals, diagnostics = {}, {}
        try:
            for branch, batches in (("rr", rr), ("fr", fr)):
                count = getattr(self.config, f"effective_batch_{branch.upper()}")
                for batch in batches:
                    out = self.objective.evaluate(batch, branch)
                    fraction = len(batch.pan) / count
                    (out["loss"] * fraction).backward()
                    for prefix in ("raw", "weighted"):
                        for key, value in out[prefix].items():
                            name = f"{prefix}_{key}"
                            totals[name] = totals.get(name, 0.) + float(value.detach()) * fraction
                    for key, value in out["numeric"].items():
                        name = f"{branch}_{key}"
                        observed = float(value.detach())
                        if key.endswith('_min'):
                            diagnostics[name] = min(diagnostics.get(name, observed), observed)
                        elif key.endswith('_max'):
                            diagnostics[name] = max(diagnostics.get(name, observed), observed)
                        elif key.endswith('_rms'):
                            diagnostics[name] = diagnostics.get(name, 0.) + observed**2 * fraction
                        else:
                            diagnostics[name] = diagnostics.get(name, 0.) + observed * fraction
            parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError("Nonfinite gradient; optimizer step refused")
            self.optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError("Nonfinite updated parameter; attempt must fail")
            if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all())
                   for state in self.optimizer.state.values() for value in state.values()):
                raise FloatingPointError("Nonfinite Adam state; attempt must fail")
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            self.objective.restore_rng(before)
            self.poisoned = True  # Never imply that a failed optimizer update can be resumed.
            raise
        self.updates += 1
        diagnostics = {k: math.sqrt(v) if k.endswith('_rms') else v for k, v in diagnostics.items()}
        self.stage.forward_calls += 1
        seconds = time.perf_counter() - start
        device = next(self.stage.aligner.parameters()).device
        return {"step": self.updates, "loss": sum(v for k, v in totals.items() if k.startswith("weighted_")),
                **totals, **diagnostics, "lr_U": rates["U"], "lr_A": rates["A"],
                "effective_batch_RR": self.config.effective_batch_RR,
                "effective_batch_FR": self.config.effective_batch_FR,
                "rr_branch_microbatches": len(rr), "fr_branch_microbatches": len(fr),
                "rr_branch_samples": self.config.effective_batch_RR,
                "fr_branch_samples": self.config.effective_batch_FR,
                "rr_epsilon_calls": self.objective.calls["rr"], "fr_epsilon_calls": self.objective.calls["fr"],
                "finite_fraction": 1., "empty_roi_count": 0,
                "train_seconds": seconds, "data_wait_seconds": data_wait_seconds,
                "updates_per_second": 1. / max(seconds, 1e-12),
                "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}

    def run(self, rr_factory, fr_factory, *, after_update, should_stop):
        """Factories receive the next 1-based update and return its microbatches."""
        while self.updates < self.config.horizon and not should_stop(self):
            started = time.perf_counter()
            rr, fr = tuple(rr_factory(self.updates + 1)), tuple(fr_factory(self.updates + 1))
            trace = self.step(rr, fr, data_wait_seconds=time.perf_counter() - started)
            after_update(self, trace)
        return self.updates

    def snapshot_state(self):
        """RAM snapshot only. Caller must enforce target gate before serialization."""
        if self.poisoned:
            raise RuntimeError("Cannot snapshot a failed training attempt")
        return copy.deepcopy({"manifest": self.manifest(), "stage": self.stage.state_dict(),
                              "optimizer": self.optimizer.state_dict(), "updates": self.updates,
                              "objective_rng": self.objective.rng_state(),
                              "model_state_sha256": model_state_hash(self.stage),
                              "global_rng": _global_rng_state()})

    def restore_state(self, snapshot):
        """Exact RAM state restore; file admission/qualification belongs to caller."""
        if snapshot["manifest"] != self.manifest():
            raise ValueError("Snapshot training/source/scheduler/initialization manifest differs")
        updates = snapshot["updates"]
        if type(updates) is not int or not 0 <= updates <= self.config.horizon:
            raise ValueError("Invalid restored optimizer update")
        self.stage.load_state_dict(snapshot["stage"], strict=True)
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.objective.restore_rng(snapshot["objective_rng"])
        self.updates = updates
        if model_state_hash(self.stage) != snapshot["model_state_sha256"]:
            self.poisoned = True
            raise ValueError("Snapshot model tensor hash differs")
        _restore_global_rng(snapshot["global_rng"])
        self.poisoned = False

    @contextmanager
    def evaluation_context(self, *, gradients=False):
        """Pause at the current state, preserve all modes and RNG, detect mutation.

        ``gradients=True`` is for numeric autograd.grad probes only. Such probes
        must not call backward/optimizer.step or modify parameter .grad buffers.
        """
        if self.poisoned:
            raise RuntimeError("Cannot evaluate a failed attempt")
        state_hash, updates = model_state_hash(self.stage), self.updates
        modes = [(m, m.training) for m in self.stage.modules()]
        parameter_grads = [(p, None if p.grad is None else p.grad.detach().clone())
                           for p in self.stage.parameters()]
        optimizer_steps = self._optimizer_steps()
        rng, eps_rng = _global_rng_state(), self.objective.rng_state()
        stage_rng = copy.deepcopy(self.stage.get_extra_state())
        try:
            self.stage.eval()
            with torch.set_grad_enabled(gradients):
                yield state_hash
        finally:
            for module, mode in modes:
                module.training = mode
            for parameter, grad in parameter_grads:
                parameter.grad = grad
            _restore_global_rng(rng)
            self.objective.restore_rng(eps_rng)
            self.stage.set_extra_state(stage_rng)
            if (self.updates != updates or model_state_hash(self.stage) != state_hash
                    or self._optimizer_steps() != optimizer_steps):
                self.poisoned = True
                raise RuntimeError("A/T state changed during paused evaluation")

    def _optimizer_steps(self):
        return tuple(float(state.get("step", 0)) for state in self.optimizer.state.values())

    def gradient_diagnostics(self, rr, fr, *, epsilon_rr, epsilon_fr):
        """Four A-loss gradient norms/cosines without changing training streams."""
        with self.evaluation_context(gradients=True):
            r = self.objective.evaluate(rr, "rr", epsilon=epsilon_rr, training=False)
            f = self.objective.evaluate(fr, "fr", epsilon=epsilon_fr, training=False)
            raw = {**r["raw"], **f["raw"]}
            params = tuple(self.stage.aligner.parameters())
            vectors = {}
            for name, loss in raw.items():
                grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                vectors[name] = torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
                                           for p, g in zip(params, grads)])
            coeff = {"rec": 1., "abs_rr": self.config.lambda_abs,
                     "eps_rr": self.config.lambda_eps, "eps_fr": self.config.lambda_eps}
            norms = {k: float(v.norm()) for k, v in vectors.items()}
            cosine = {}
            names = list(vectors)
            for i, first in enumerate(names):
                for second in names[i+1:]:
                    denominator = norms[first] * norms[second]
                    cosine[f"{first}:{second}"] = (float(torch.dot(vectors[first], vectors[second])) / denominator
                                                    if denominator > 0 else None)
            return {"raw_loss": {k: float(v.detach()) for k, v in raw.items()},
                    "raw_A_grad_norm": norms,
                    "weighted_A_grad_norm": {k: norms[k] * coeff[k] for k in norms},
                    "raw_A_grad_cosine": cosine, "weighted_A_grad_cosine": dict(cosine),
                    "zero_norm_cosine": "undefined_null", "training_rng_consumed": False}


def _global_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def _restore_global_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
