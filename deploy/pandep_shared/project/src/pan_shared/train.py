"""Isolated FP32 mean-L1 trainer, resumable at committed optimizer boundaries."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from .checkpoints import CheckpointStore, capture_rng, restore_rng
from .losses import mean_l1
from .sampling import BalancedSampler


def learning_rate(update, shared):
    """LR used by the 1-based *next* update, never reset at block boundaries."""
    if update < 1:
        raise ValueError("optimizer update is 1-based")
    a = 3 if shared else 1
    warmup, end = 100 * a, 50000 * a
    if update <= warmup:
        return 1e-6 + (1e-4 - 1e-6) * update / warmup
    if update <= end:
        phase = (update - warmup) / (end - warmup)
        return 1e-5 + 0.5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * phase))
    return 1e-5


def events_at(step, shared):
    events = []
    if step > 0 and step % 2000 == 0:
        events.append("probe")
    if step > 0 and (step % 10000 == 0 or step % 50000 == 25000):
        events.append("validation")
    if step > 0 and (step % 50000 == 0 or (shared and step == 75000)):
        events.append("exact")
    return events


@contextmanager
def evaluation_guard(model):
    rng = capture_rng()
    modes = {name: module.training for name, module in model.named_modules()}
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
        restore_rng(rng)


def _append(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def assert_finite_optimizer_update(model, optimizer):
    """Check all weights and touched Adam tensors before committing exposure.

    Scalar reductions are collected per device and synchronized once per
    device, not once per parameter/moment. This also checks CPU step counters
    when CUDA Adam stores its moments on-device and counters on the host.
    """
    reductions = {}
    for parameter in model.parameters():
        tensors = [parameter]
        if parameter.grad is not None:
            tensors.extend(value for value in optimizer.state.get(parameter, {}).values()
                           if isinstance(value, torch.Tensor))
        for tensor in tensors:
            reductions.setdefault(tensor.device, []).append(torch.isfinite(tensor.detach()).all())
    for device, checks in reductions.items():
        if not bool(torch.stack(checks).all()):
            raise FloatingPointError("DIVERGED: nonfinite post-optimizer parameter/moment/counter on " + str(device))


class Trainer:
    def __init__(self, model, datasets, run_config, identity, run_dir, *, sample_ids=None,
                 evaluate=None, benchmark=None, device="cpu", microbatch=None):
        self.config = dict(run_config)
        self.identity = dict(identity)
        self.run_dir = Path(run_dir).resolve(); self.run_dir.mkdir(parents=True, exist_ok=True)
        self.sensors = tuple(run_config["sensors"])
        self.shared = run_config["mode"] == "SHARED"
        self.device = torch.device(device)
        self.microbatch = int(microbatch or run_config.get("microbatch", 48))
        effective_batch = run_config.get("effective_batch_size", 48)
        if effective_batch != 48 or self.microbatch not in (12, 24, 48):
            raise ValueError("plan v4 requires effective48 and preregistered microbatch12/24/48")
        if self.shared != (len(self.sensors) == 3):
            raise ValueError("SHARED requires all three sensors, SINGLE exactly one")
        seed = int(run_config["seed"])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.model = model.to(self.device).float()
        self.model.train()
        parameters = list(self.model.parameters())
        if len({id(p) for p in parameters}) != len(parameters):
            raise ValueError("duplicate parameter optimizer registration")
        self.optimizer = torch.optim.AdamW(parameters, lr=learning_rate(1, self.shared),
                                           betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        self.datasets = datasets
        if sample_ids is None:
            if run_config.get("train_fraction", 1.0) != 1.0:
                raise ValueError("C01 requires explicitly frozen subset IDs")
            sample_ids = {s: list(range(len(datasets[s]))) for s in self.sensors}
        self.sampler = BalancedSampler(seed, sample_ids, self.sensors, effective_batch)
        self.evaluate = evaluate; self.benchmark = benchmark
        self.store = CheckpointStore(self.run_dir, self.identity)
        self.state = {"step": 0, "pending_events": [], "best": None,
                      "candidate_count": 0, "validation_steps": [],
                      "times": {"training": 0.0, "data_wait": 0.0, "validation": 0.0,
                                "benchmark": 0.0, "checkpoint": 0.0},
                      "sensor_losses": {s: {"sum": 0.0, "count": 0} for s in self.sensors},
                      "status": "READY", "scheduler": {"completed_step": 0, "a": 3 if self.shared else 1}}
        self._in_update = False
        resumed = self.store.load_resume()
        if resumed is not None:
            self.model.load_state_dict(resumed["model"], strict=True)
            self.optimizer.load_state_dict(resumed["optimizer"])
            self.sampler.load_state_dict(resumed["sampler"])
            self.state = resumed["train"]
            if self.state["step"] != self.sampler.state_dict()["step"]:
                raise ValueError("resume committed counters disagree")
            if self.state["scheduler"] != {"completed_step": self.state["step"], "a": 3 if self.shared else 1}:
                raise ValueError("resume scheduler position differs")
            restore_rng(resumed["rng"])
            self.log("resume_events.jsonl", {"event": "EXACT_RESUME", "next_batch": self.sampler.peek()})

    @property
    def step(self):
        return self.state["step"]

    def log(self, name, value):
        record = {**self.identity, "schema": "PANDEP_RUN_LOG_v1", "step": self.step, **value}
        self.store._owned(self.run_dir / name)
        _append(self.run_dir / name, record)
        if name in ("train_steps.jsonl", "validation.jsonl", "errors.jsonl"):
            print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

    def full_state(self):
        if self._in_update:
            raise RuntimeError("unsafe fullstate publication during optimizer transaction")
        return {"schema": "PANDEP_FULLSTATE_v1", "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "sampler": self.sampler.state_dict(),
                "rng": capture_rng(), "train": copy.deepcopy(self.state)}

    def save(self, archive=False):
        started = time.monotonic()
        result = self.store.save_resume(self.full_state(), archive=archive)
        self.state["times"]["checkpoint"] += time.monotonic() - started
        exposure_dir = self.run_dir / "exposure_counts"
        exposure_dir.mkdir(exist_ok=True)
        # Immutable per-step counters, separate from the atomic resume authority.
        for sensor, stream in self.sampler.state_dict()["streams"].items():
            destination = exposure_dir / ("%s_step_%09d.npz" % (sensor, self.step))
            self.store._owned(destination)
            if not destination.exists():
                fd, temporary = tempfile.mkstemp(prefix=".exposure-", suffix=".npz", dir=exposure_dir)
                with os.fdopen(fd, "wb") as out:
                    np.savez_compressed(out, sample_ids=np.asarray(self.sampler.ids[sensor]),
                                        counts=np.asarray(stream["counts"], dtype=np.int64))
                    out.flush(); os.fsync(out.fileno())
                os.replace(temporary, destination)
            with np.load(destination) as verified:
                if not np.array_equal(verified["counts"], np.asarray(stream["counts"])) or not np.array_equal(verified["sample_ids"], self.sampler.ids[sensor]):
                    raise ValueError("EXPOSURE_CONFLICT: immutable per-step counts differ")
        return result

    def update(self):
        if self._in_update:
            raise RuntimeError("uncommitted optimizer transaction: reload the last verified fullstate")
        batch = self.sampler.peek(); sensor = batch["sensor"]
        start = time.monotonic()
        data = self.datasets[sensor].fetch(batch["sample_ids"], rotations=batch["rot_ids"])
        self.state["times"]["data_wait"] += time.monotonic() - start
        if len(data["pan"]) != 48:
            raise ValueError("logical optimizer batch must contain exactly48 samples")
        self.optimizer.zero_grad(set_to_none=True)
        rate = learning_rate(self.step + 1, self.shared)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        started = time.monotonic(); total_loss = 0.0
        for offset in range(0, 48, self.microbatch):
            mb = {key: data[key][offset:offset + self.microbatch].to(self.device, dtype=torch.float32)
                  for key in ("pan", "ms", "lpan", "gt")}
            prediction = self.model(sensor, mb["pan"], mb["ms"], mb["lpan"])
            loss = mean_l1(prediction, mb["gt"])
            if not torch.isfinite(loss):
                raise FloatingPointError("DIVERGED: nonfinite loss")
            weighted = loss * (len(mb["pan"]) / 48)
            weighted.backward(); total_loss += float(weighted.detach())
        norms = {"trunk": 0.0, "stem": 0.0, "head": 0.0, "other": 0.0}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("DIVERGED: nonfinite gradient: " + name)
            if name.startswith(("stems.", "heads.")) and name.split(".")[1] != sensor:
                raise RuntimeError("inactive sensor branch has a gradient: " + name)
            group = "trunk" if name.startswith("trunk.") else "stem" if name.startswith("stems.") else "head" if name.startswith("heads.") else "other"
            norms[group] += float(parameter.grad.detach().double().square().sum())
        norms = {name: math.sqrt(value) for name, value in norms.items()}
        # An OOM/exception inside optimizer.step may partially update parameters;
        # no new fullstate is published in that case. Previous verified state survives.
        self._in_update = True
        self.optimizer.step()
        assert_finite_optimizer_update(self.model, self.optimizer)
        self.sampler.commit(batch["token"])
        self.state["step"] += 1
        self._in_update = False
        self.state["scheduler"]["completed_step"] = self.step
        self.state["times"]["training"] += time.monotonic() - started
        self.state["sensor_losses"][sensor]["sum"] += total_loss
        self.state["sensor_losses"][sensor]["count"] += 1
        self.state["pending_events"] = events_at(self.step, self.shared)
        self.log("committed_batches.jsonl", {"batch": batch, "lr_used": rate})
        if self.step % 100 == 0:
            self.log("train_steps.jsonl", {"sensor": sensor, "batch_l1": total_loss,
                "sensor_loss_means": {s: (v["sum"] / v["count"] if v["count"] else None)
                                      for s, v in self.state["sensor_losses"].items()},
                "lr_used": rate, "lr_next": learning_rate(self.step + 1, self.shared),
                "grad_norms": norms, "exposures": self.sampler.exposures(),
                "times": self.state["times"], "nonfinite": False})
        return {"batch": batch, "loss": total_loss, "lr": rate, "grad_norms": norms}

    def _evaluate(self, kind):
        if self.evaluate is None:
            raise RuntimeError("BLOCKED_EVALUATOR: required " + kind + " callback missing")
        start = time.monotonic()
        with evaluation_guard(self.model):
            result = self.evaluate(self, kind)
        self.state["times"]["validation"] += time.monotonic() - start
        self.log("probes.jsonl" if kind == "probe" else "validation.jsonl", {"kind": kind, "result": result})
        return result

    def _select_best(self, result):
        sensors = result.get("sensors", {})
        if result.get("complete") is False or set(sensors) != set(self.sensors):
            raise ValueError("partial validation is not a selector candidate")
        scores = [sensors[s].get("mean_l1") for s in self.sensors]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores):
            raise ValueError("nonfinite validation is not a selector candidate")
        for sensor in self.sensors:
            if sensors[sensor].get("n_samples", 0) <= 0:
                raise ValueError("validation sample count required")
        score = sum(scores) / len(scores)
        if self.step in self.state["validation_steps"]:
            return
        self.state["candidate_count"] += 1
        self.state["validation_steps"].append(self.step)
        previous = self.state["best"]
        if previous is None or (score, self.step) < (previous["score"], previous["completed_step"]):
            checkpoint = self.store.save_weights(self.model, self.step, self.sampler.exposures(), best=True)
            self.state["best"] = dict(checkpoint, score=score)
            self.log("best/selection_history.jsonl", {"selected": self.state["best"],
                "candidate_count": self.state["candidate_count"], "sensor_validation": sensors,
                "selector": "BEST_JOINT_VAL" if self.shared else "BEST_SENSOR_VAL"})

    def _benchmark(self, checkpoint, selector):
        if self.benchmark is None:
            raise RuntimeError("BLOCKED_EVALUATOR: RR20/FR20 callback missing")
        current = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        start = time.monotonic()
        try:
            self.store.load_weights(self.model, checkpoint)
            with evaluation_guard(self.model):
                result = self.benchmark(self, checkpoint, selector)
        finally:
            self.model.load_state_dict(current, strict=True)
        self.state["times"]["benchmark"] += time.monotonic() - start
        self.log("benchmark.jsonl", {"checkpoint": checkpoint, "selector_scope": selector,
                                      "result": result})

    def process_events(self):
        while self.state["pending_events"]:
            event = self.state["pending_events"][0]
            if event == "probe":
                self._evaluate("probe")
            elif event == "validation":
                self._select_best(self._evaluate("validation"))
            elif event == "exact":
                checkpoint = self.store.save_weights(self.model, self.step, self.sampler.exposures())
                self.log("checkpoint_index.jsonl", {"checkpoint": checkpoint, "selector": "EXACT"})
                self._benchmark(checkpoint, "EXACT")
                if self.step % 50000 == 0 and self.state["best"] is not None:
                    best = self.state["best"]
                    name = "BEST_JOINT_VAL" if self.shared else "BEST_SENSOR_VAL"
                    selected = dict(best, selected_through_global_step=self.step,
                                    candidate_count=self.state["candidate_count"])
                    self._benchmark(selected, "%s_TO_B%04d" % (name, self.step // 50000))
            self.state["pending_events"].pop(0)
            # Save after each event so a transient eval failure cannot lose its queue.
            self.save()

    def run_until(self, target_step, control_check=None):
        if target_step < self.step:
            raise ValueError("target precedes committed checkpoint")
        self.state["status"] = "RUNNING"
        if not (self.run_dir / "resume" / "last.json").exists():
            self.save()
        try:
            self.process_events()
            while self.step < target_step:
                action = control_check() if control_check else "RUN"
                action = action.get("action", "RUN") if isinstance(action, dict) else action
                if action in ("STOP", "PAUSE"):
                    self.state["status"] = "STOPPED_BY_OPERATOR" if action == "STOP" else "PAUSED_SAFE"
                    self.save(); return self.state["status"]
                if action not in ("RUN", "STOP_AFTER_BLOCK"):
                    raise ValueError("unknown controller action")
                self.update()
                if self.state["pending_events"]:
                    # Include pending events in fullstate before any costly evaluation.
                    self.save()
                    self.process_events()
            self.state["status"] = "BLOCK_COMPLETE"
            self.save(archive=self.step % 50000 == 0)
            return self.state["status"]
        except Exception as error:
            text = str(error)
            status = "DIVERGED" if isinstance(error, FloatingPointError) else "PAUSED_OOM" if "out of memory" in text.lower() else "PAUSED_DISK" if text.startswith("PAUSED_DISK") else "BLOCKED_EXECUTION"
            self.state["status"] = status
            self.log("errors.jsonl", {"status": status, "error_type": type(error).__name__, "error": text,
                                      "transaction_uncommitted": self._in_update})
            if not self._in_update:
                self.save()
            raise
