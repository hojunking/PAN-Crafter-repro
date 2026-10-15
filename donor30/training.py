"""Student-only optimizer engine for donor-bound DirectShift Stage 2."""
from __future__ import annotations

import copy
import math

import torch

from .method import DirectStage2


def _norm(values):
    total = sum(float(v.detach().double().square().sum()) for v in values if v is not None)
    if not math.isfinite(total):
        raise FloatingPointError("Nonfinite Student gradient")
    return math.sqrt(total)


class StudentTrainer:
    def __init__(self, stage, optimizer, *, effective_batch_size=None, microbatch_size=None,
                 gradient_log_interval=100):
        if not isinstance(stage, DirectStage2):
            raise TypeError("DirectStage2 required")
        for value in (effective_batch_size, microbatch_size):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("Positive batch size required")
        if ((effective_batch_size is None) != (microbatch_size is None) or
                effective_batch_size is not None and effective_batch_size % microbatch_size):
            raise ValueError("Both divisible effective/micro batch sizes must be pinned")
        if type(gradient_log_interval) is not int or gradient_log_interval <= 0:
            raise ValueError("Positive diagnostic cadence required")
        self.stage, self.optimizer = stage, optimizer
        self.effective_batch_size, self.microbatch_size = effective_batch_size, microbatch_size
        self.gradient_log_interval = gradient_log_interval
        self.updates, self.poisoned = 0, False
        self._optimizer_names = self._binding()
        self.stage.assert_frozen_binding()

    def _binding(self):
        self.stage.assert_optimizer(self.optimizer)
        names = {id(p): name for name, p in self.stage.named_parameters()}
        return [[names[id(p)] for p in group["params"]] for group in self.optimizer.param_groups]

    def _check_binding(self):
        if self._binding() != self._optimizer_names:
            raise ValueError("Student optimizer parameter/group order changed")

    def step(self, rr):
        return self.step_accumulated([rr])

    def step_accumulated(self, microbatches, *, expected_effective_batch=None):
        if self.poisoned:
            raise RuntimeError("Failed optimizer mutation: restore a complete checkpoint")
        self._check_binding()
        batches = list(microbatches)
        if not batches or any(len(b.pan) < 1 for b in batches):
            raise ValueError("Nonempty RR microbatches required")
        total = sum(len(b.pan) for b in batches)
        expected = self.effective_batch_size if expected_effective_batch is None else expected_effective_batch
        if expected is not None and (type(expected) is not int or expected <= 0 or total != expected):
            raise ValueError("Effective batch size differs from pinned contract")
        if self.effective_batch_size is not None and total != self.effective_batch_size:
            raise ValueError("Cannot override pinned effective batch size")
        if self.microbatch_size is not None and any(len(b.pan) != self.microbatch_size for b in batches):
            raise ValueError("Microbatch size differs from pinned contract")
        if any(b.pan.shape[1:] != batches[0].pan.shape[1:] for b in batches):
            raise ValueError("A fixed observation shape is required")
        self.stage.train()
        self.optimizer.zero_grad(set_to_none=True)
        student = list(self.stage.student.parameters())
        frozen = list(self.stage.aligner.parameters()) + list(self.stage.teacher.parameters())
        scalars = ("loss", "fit", "hard", "soft", "spatial", "weighted_spatial")
        metrics, u, support = {}, [], {"spatial_support_ok": [], "warp_support_fraction": 0.}
        try:
            for rr in batches:
                weight = len(rr.pan) / total
                out = self.stage(rr)
                (out["loss"] * weight).backward()
                for key in scalars:
                    metrics[key] = metrics.get(key, 0.) + weight * float(out[key].detach())
                for key in ("e", "q", "w_e", "w_q"):
                    if key in out:
                        metrics[key + "_mean"] = metrics.get(key + "_mean", 0.) + weight * float(out[key].detach().mean())
                u.extend(out["correction"].detach().cpu().tolist())
                support["warp_support_fraction"] += weight * float(out["warp_support_fraction"].detach())
                support["spatial_support_ok"].extend(out["spatial_support_ok"].detach().cpu().tolist())
            if any(p.grad is not None for p in frozen):
                raise RuntimeError("Frozen donor unexpectedly received gradients")
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in student):
                raise FloatingPointError("Nonfinite Student gradients; optimizer not called")
            grad_norm = _norm(p.grad for p in student)
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        try:
            self.optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in student):
                raise FloatingPointError("Nonfinite Student parameters after optimizer update")
        except Exception:
            self.poisoned = True
            raise
        self.updates += 1
        return {"update": self.updates, "effective_batch_size": total,
                "microbatch_sizes": [len(b.pan) for b in batches], **metrics,
                "rec": metrics["fit"], "gradient_norm": {"student": grad_norm, "aligner": 0., "teacher": 0.},
                "learning_rates": [float(g["lr"]) for g in self.optimizer.param_groups],
                "alignment": {"native": {"u": u}}, "support": support}

    def state_dict(self):
        if self.poisoned:
            raise RuntimeError("Cannot save a partially failed optimizer update")
        self._check_binding()
        self.stage.assert_frozen_binding()
        return copy.deepcopy({"schema": "DONOR30_STUDENT_TRAINER_V1", "stage": self.stage.state_dict(),
            "optimizer": self.optimizer.state_dict(), "optimizer_parameter_names": self._optimizer_names,
            "optimizer_class": type(self.optimizer).__module__ + "." + type(self.optimizer).__qualname__,
            "effective_batch_size": self.effective_batch_size, "microbatch_size": self.microbatch_size,
            "gradient_log_interval": self.gradient_log_interval, "updates": self.updates})

    def load_state_dict(self, state):
        self._check_binding()
        expected = {"schema": "DONOR30_STUDENT_TRAINER_V1", "optimizer_parameter_names": self._optimizer_names,
            "optimizer_class": type(self.optimizer).__module__ + "." + type(self.optimizer).__qualname__,
            "effective_batch_size": self.effective_batch_size, "microbatch_size": self.microbatch_size,
            "gradient_log_interval": self.gradient_log_interval}
        if any(state.get(k) != v for k, v in expected.items()) or type(state.get("updates")) is not int or state["updates"] < 0:
            raise ValueError("Student optimizer checkpoint contract mismatch")
        self.stage.load_state_dict(state["stage"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self._check_binding()
        self.updates, self.poisoned = state["updates"], False
