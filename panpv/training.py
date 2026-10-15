"""Finite, independently resumable PANPV scene-adaptation training.

The caller supplies already verified/normalized RR data and constructs the exact
model (including a SHA-verified donor for FT). No data discovery, donor fallback,
sensor normalization, experiment selection, or sheet writes happen here.
"""
from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import threading
import time
import uuid

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from deploy.pandep_shared.project.src.pan_shared.checkpoints import tensor_state_hash
from deploy.pandep_shared.project.src.pan_shared.sampling import BalancedSampler


FIELDS = ("pan", "ms", "lpan", "gt", "mask")
DIAGNOSTIC_STEPS = (0, 500, 1000, 2000, 5000, 10000, 20000)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(_canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_dir(path.parent)


def fixed_tensor_view(tensor, rotation):
    """Tensor-equivalent of pan_shared.frontend.fixed_view, not random flips."""
    if not isinstance(rotation, int) or rotation not in range(4):
        raise ValueError("view must be an integer in 0..3")
    return torch.rot90(tensor.flip((-2, -1)), rotation, (-2, -1)).contiguous()


def _validate_item(item):
    for key in FIELDS:
        value = item[key]
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(f"{key} must be a CHW tensor")
        if value.device.type != "cpu":
            raise ValueError("dataset must expose CPU tensors, without GPU prefetch")
        if key != "mask" and value.dtype != torch.float32:
            raise ValueError(f"{key} must be normalized float32")
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite dataset {key}; repair/mask invalid input before training")
    pan, ms, lp, gt, mask = (item[key] for key in FIELDS)
    h, w = pan.shape[-2:]
    if (pan.shape[0] != 1 or h % 4 or w % 4 or
            ms.shape != (gt.shape[0], h // 4, w // 4) or
            lp.shape != (1, h // 4, w // 4) or gt.shape[-2:] != (h, w) or
            mask.shape != (1, h, w)):
        raise ValueError("inconsistent ratio-four PAN/MS/LP/target/mask geometry")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("mask must be binary valid support, not arbitrary loss weights")
    return int(mask.sum().item()) * int(gt.shape[0])


def _collate(items, device):
    return {key: torch.stack([item[key] for item in items]).to(device) for key in FIELDS}


def masked_l1_sum(prediction, target, mask):
    """Unclipped numerator and the number of *band-pixel* elements."""
    if prediction.shape != target.shape or mask.shape != (target.shape[0], 1, *target.shape[-2:]):
        raise ValueError("masked L1 geometry mismatch")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("nonfinite model output")
    support = mask.to(torch.bool).expand_as(target)
    return (prediction - target).abs().masked_select(support).sum(), int(support.sum().item())


class TrainingEngine:
    """One scene + one method; caller schedules independent 2K blocks.

    ``audit_callback(model, completed_step, run_dir) -> JSON dict`` is strictly
    diagnostic. It must not modify model weights. Its values never enter the
    validation-L1 selector. ``resume`` requires explicit local-fullstate trust.
    """

    def __init__(self, model, train_dataset, val_dataset, run_dir, *, method,
                 identity, seed=261002, device="cpu", effective_batch=48,
                 micro_batch=12, max_updates=20000, block_updates=2000,
                 val_batch_size=2, audit_callback=None, log_interval=100,
                 probe_interval=500, validation_interval=2000,
                 diagnostic_steps=DIAGNOSTIC_STEPS, sensor_id="WV3"):
        if method not in ("SC_SCENE", "FT_WV3_SCENE"):
            raise ValueError("only independent SC_SCENE / FT_WV3_SCENE training is permitted")
        if effective_batch != 48 or micro_batch not in (12, 24, 48):
            raise ValueError("freeze effective48 and microbatch48/24/12 before launching")
        if not identity or not isinstance(identity, dict):
            raise ValueError("immutable source/split/protocol identity is required")
        if getattr(train_dataset, "provisional", False) or getattr(val_dataset, "provisional", False):
            raise ValueError("BLOCKED_PROVISIONAL_DATA: diagnostic-only data cannot enter adaptation")
        if not len(train_dataset) or not len(val_dataset):
            raise ValueError("nonempty independent train and validation sets are required")
        if min(max_updates, block_updates, val_batch_size, log_interval, probe_interval, validation_interval) <= 0:
            raise ValueError("intervals, endpoint, and validation batch size must be positive")
        self.root = Path(run_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.device = torch.device(device)
        self.method, self.sensor_id, self.seed = method, sensor_id, int(seed)
        # Set before .to(cuda), which can initialize a CUDA/cuBLAS context.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.model = model.to(self.device, dtype=torch.float32)
        if any(not p.requires_grad for p in self.model.parameters()):
            raise ValueError("SC and FT require all stem/trunk/head parameters to be trainable")
        self.max_updates, self.block_updates = int(max_updates), int(block_updates)
        self.micro_batch, self.batch_size = int(micro_batch), int(effective_batch)
        self.val_batch_size = int(val_batch_size)
        self.log_interval, self.probe_interval = int(log_interval), int(probe_interval)
        self.validation_interval = int(validation_interval)
        self.diagnostic_steps = tuple(sorted(set(map(int, diagnostic_steps))))
        self.audit_callback = audit_callback
        self.lr = 1e-4 if method == "SC_SCENE" else 1e-5
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                          betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        random.seed(self.seed)
        np.random.seed(self.seed % (2 ** 32))
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.manual_seed(self.seed)

        # Exclude zero-support patches before sampler creation, never shrink a
        # logical batch after drawing it. Initial validation is CPU-only.
        self.train_ids = [i for i in range(len(train_dataset)) if _validate_item(train_dataset[i]) > 0]
        self.val_ids = [i for i in range(len(val_dataset)) if _validate_item(val_dataset[i]) > 0]
        if not self.train_ids or not self.val_ids:
            raise ValueError("no positive-support train/validation patches")
        self.sampler = BalancedSampler(self.seed, {sensor_id: self.train_ids}, effective_batch=48)
        self.config = dict(method=method, seed=self.seed, sensor_id=sensor_id, device=str(self.device),
                           effective_batch=48, micro_batch=micro_batch, max_updates=max_updates,
                           block_updates=block_updates, val_batch_size=val_batch_size,
                           learning_rate=self.lr, optimizer="AdamW", betas=[0.9, 0.999], eps=1e-8,
                           weight_decay=0.01, scheduler="constant", precision="FP32",
                           amp=False, tf32=False, augmentation="FIXED_HV_THEN_ROT4_v1",
                           log_interval=log_interval, probe_interval=probe_interval,
                           validation_interval=validation_interval, diagnostic_steps=list(self.diagnostic_steps),
                           train_length=len(train_dataset), val_length=len(val_dataset),
                           train_positive_support_ids=self.train_ids, val_positive_support_ids=self.val_ids,
                           train_probe_ids=self.train_ids[:128], val_probe_ids=self.val_ids[:128],
                           diagnostics_configured=audit_callback is not None,
                           torch_version=str(torch.__version__))
        self.identity = dict(source=copy.deepcopy(identity), config=self.config,
                             initial_tensor_sha256=tensor_state_hash(self.model.state_dict()))
        self.identity_sha = hashlib.sha256(_canonical(self.identity).encode()).hexdigest()
        manifest = self.root / "training_manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != self.identity:
                raise ValueError("BLOCKED_IDENTITY: existing run has different source/config/initial weights")
        else:
            if any(self.root.iterdir()):
                # Parent-created metadata is allowed, but not unknown training
                # state that could otherwise be silently mistaken for a new run.
                if any((self.root / p).exists() for p in ("latest.json", "checkpoints", "train_steps.jsonl")):
                    raise ValueError("unrecognized existing training artifacts")
            _atomic_json(manifest, self.identity)
        self.train = dict(step=0, samples_seen=0, adapt_seconds=0.0,
                          diagnostic_seconds=0.0, checkpoint_seconds=0.0,
                          recent_loss_sum=0.0, recent_count=0, best_val_l1=None,
                          best_step=None, best_checkpoint=None, diagnostics_done=[],
                          probe_steps=[], full_validation_steps=[], status="INITIALIZED")
        self._resume_required = (self.root / "latest.json").exists()
        self._last_checkpoint_sha = None
        self._pause_requested = False
        self._pause_signal = None
        self._in_update = False
        self._segment = uuid.uuid4().hex

    def _owned(self, path):
        path = Path(path)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("checkpoint/log path escapes the owned run directory")
        return path

    def _log(self, name, value):
        record = dict(identity_sha256=self.identity_sha, source=self.identity["source"],
                      method=self.method, seed=self.seed, segment=self._segment,
                      recorded_unix=time.time(), **value)
        with self._owned(self.root / name).open("a", encoding="utf-8") as stream:
            stream.write(_canonical(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _rng(self):
        return dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch_cpu=torch.get_rng_state(),
                    torch_cuda=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None)

    def _restore_rng(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if state["torch_cuda"] is not None:
            if self.device.type != "cuda":
                raise ValueError("CUDA fullstate cannot be resumed as a CPU run")
            torch.cuda.set_rng_state(state["torch_cuda"], self.device)

    def save_checkpoint(self, *, reason="BLOCK_BOUNDARY", publish_selectors=False):
        if self._in_update:
            raise RuntimeError("refusing a partial optimizer-update checkpoint")
        started = time.perf_counter()
        parent = self._owned(self.root / "checkpoints")
        parent.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
        destination = parent / f"step_{self.train['step']:08d}_{uuid.uuid4().hex[:12]}"
        model_state = {name: tensor.detach().cpu().contiguous() for name, tensor in self.model.state_dict().items()}
        weights = temporary / "model.safetensors"
        save_file(model_state, str(weights))
        with weights.open("rb") as stream:
            os.fsync(stream.fileno())
        if tensor_state_hash(load_file(str(weights))) != tensor_state_hash(model_state):
            raise IOError("safetensors readback mismatch")
        checkpoint_ref = dict(directory=str(destination.relative_to(self.root)),
                              completed_step=self.train["step"], model_sha256=file_sha256(weights))
        train = copy.deepcopy(self.train)
        if publish_selectors and train["best_step"] == train["step"]:
            train["best_checkpoint"] = checkpoint_ref
        state = dict(schema="PANPV_FULLSTATE_v1", identity=self.identity,
                     model=model_state, optimizer=self.optimizer.state_dict(),
                     scheduler=dict(kind="constant", completed_step=train["step"], lr=self.lr),
                     sampler=self.sampler.state_dict(), rng=self._rng(), train=train)
        state_path = temporary / "fullstate.pt"
        with state_path.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = dict(schema="PANPV_CHECKPOINT_v1", identity_sha256=self.identity_sha,
                        reason=reason, **checkpoint_ref, fullstate_sha256=file_sha256(state_path),
                        tensor_sha256=tensor_state_hash(model_state), samples_seen=train["samples_seen"],
                        checkpoint_seconds=train["checkpoint_seconds"] + time.perf_counter() - started,
                        checkpoint_cost_scope="model_fullstate_write_and_readback_excludes_manifest_and_log_publication")
        _atomic_json(temporary / "manifest.json", manifest)
        _sync_dir(temporary)
        os.replace(temporary, destination)
        _sync_dir(parent)
        _atomic_json(self._owned(self.root / "latest.json"), manifest)
        self._last_checkpoint_sha = manifest["fullstate_sha256"]
        self.train = train
        self.train["checkpoint_seconds"] = manifest["checkpoint_seconds"]
        self._log("checkpoint_index.jsonl", dict(step=train["step"], checkpoint=manifest))
        if publish_selectors:
            if train["best_step"] == train["step"]:
                _atomic_json(self._owned(self.root / "best_val_l1.json"),
                             dict(selector="MIN_RR_VALIDATION_L1", validation_l1=train["best_val_l1"], **manifest))
            if train["step"] == self.max_updates:
                exact = self._owned(self.root / "exact_endpoint.json")
                if exact.exists():
                    previous = json.loads(exact.read_text())
                    if previous["model_sha256"] != manifest["model_sha256"]:
                        raise ValueError("immutable EXACT endpoint cannot be replaced with different weights")
                else:
                    _atomic_json(exact, dict(selector="EXACT20K" if self.max_updates == 20000 else
                                            f"EXACT{self.max_updates}_TEST_ONLY", **manifest))
        return manifest

    def resume(self, *, trusted_local=False):
        """Only locally published, path-confined, SHA-verified fullstates."""
        if not trusted_local:
            raise ValueError("fullstate pickle requires explicit trusted_local=True; never use downloaded checkpoints")
        pointer = json.loads(self._owned(self.root / "latest.json").read_text())
        if pointer["identity_sha256"] != self.identity_sha:
            raise ValueError("BLOCKED_IDENTITY: resume pointer identity changed")
        directory = self._owned(self.root / pointer["directory"])
        if directory.parent.resolve() != (self.root / "checkpoints").resolve():
            raise ValueError("invalid resume directory")
        manifest = json.loads(self._owned(directory / "manifest.json").read_text())
        if manifest != pointer:
            raise ValueError("resume manifest/pointer mismatch")
        weights = self._owned(directory / "model.safetensors")
        fullstate = self._owned(directory / "fullstate.pt")
        if file_sha256(weights) != pointer["model_sha256"] or file_sha256(fullstate) != pointer["fullstate_sha256"]:
            raise ValueError("CORRUPT_CHECKPOINT: SHA256 mismatch before deserialization")
        state = torch.load(fullstate, map_location="cpu", weights_only=False)
        if state.get("schema") != "PANPV_FULLSTATE_v1" or state["identity"] != self.identity:
            raise ValueError("BLOCKED_IDENTITY: fullstate protocol changed")
        if tensor_state_hash(state["model"]) != pointer["tensor_sha256"]:
            raise ValueError("CORRUPT_CHECKPOINT: fullstate model tensor digest mismatch")
        if (state["train"]["step"] != pointer["completed_step"] or
                state["sampler"]["step"] != state["train"]["step"] or
                state["train"]["samples_seen"] != state["train"]["step"] * 48 or
                state["scheduler"] != dict(kind="constant", completed_step=state["train"]["step"], lr=self.lr)):
            raise ValueError("invalid committed step/sampler/scheduler/sample count")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        if any(group["lr"] != self.lr for group in self.optimizer.param_groups):
            raise ValueError("resume learning rate is not the frozen constant")
        self.sampler.load_state_dict(state["sampler"])
        self.train = state["train"]
        self.train["checkpoint_seconds"] = pointer["checkpoint_seconds"]
        self._restore_rng(state["rng"])
        self._resume_required = False
        self._last_checkpoint_sha = pointer["fullstate_sha256"]
        self._in_update = False
        self._pause_requested = False  # Explicit resume call is the authority.
        self._pause_signal = None
        self._log("resume_events.jsonl", dict(step=self.train["step"], checkpoint=pointer,
                  previous_log_steps_after_checkpoint="UNCOMMITTED_REPLAY_IF_PRESENT",
                  note="Earlier raw log segments are retained; only the resumed checkpoint lineage is authoritative."))
        return copy.deepcopy(self.train)

    @torch.no_grad()
    def validate(self, *, dataset=None, indices=None):
        dataset = self.val_dataset if dataset is None else dataset
        indices = self.val_ids if indices is None else list(indices)
        if not indices:
            raise ValueError("validation has no frozen positive-support indices")
        prior_mode = self.model.training
        self.model.eval()
        total, count = 0.0, 0
        try:
            for start in range(0, len(indices), self.val_batch_size):
                items = [dataset[i] for i in indices[start:start + self.val_batch_size]]
                for item in items:
                    _validate_item(item)
                batch = _collate(items, self.device)
                output = self.model(self.sensor_id, batch["pan"], batch["ms"], batch["lpan"])
                numerator, denominator = masked_l1_sum(output, batch["gt"], batch["mask"])
                total += float(numerator.item())
                count += denominator
        finally:
            self.model.train(prior_mode)
        if not count or not math.isfinite(total):
            raise FloatingPointError("empty/nonfinite validation loss")
        return dict(loss_sum=total, valid_elements=count, mean_l1=total / count, patches=len(indices))

    def _diagnostics(self, step):
        if step not in self.diagnostic_steps or step in self.train["diagnostics_done"]:
            return
        self._sync()
        started = time.perf_counter()
        mode, rng = self.model.training, self._rng()
        before = tensor_state_hash(self.model.state_dict())
        try:
            self.model.eval()
            with torch.no_grad():
                result = (self.audit_callback(self.model, step, self.root) if self.audit_callback else
                          {"status": "DIAGNOSTICS_NOT_CONFIGURED"})
            if tensor_state_hash(self.model.state_dict()) != before:
                raise RuntimeError("diagnostic callback mutated training weights/buffers")
            # Refuse JSON NaN before marking the diagnostic completed.
            _canonical(result)
            self._log("audit_diagnostics.jsonl", dict(step=step, diagnostic=result))
            # Never fabricate an FR metric for an RR-only diagnostic callback.
            def metric(name):
                if not isinstance(result, dict):
                    return "N/A"
                value = result.get(name, result.get(name.lower()))
                if value is None:
                    for part in ("rr", "fr", "RR", "FR", "metrics"):
                        nested = result.get(part)
                        if isinstance(nested, dict):
                            value = nested.get(name, nested.get(name.lower()))
                            if value is None and isinstance(nested.get("metrics"), dict):
                                inner = nested["metrics"]
                                value = inner.get(name, inner.get(name.lower()))
                            if value is not None:
                                break
                return "N/A" if value is None else str(value)
            print(f"PANPV audit step={step} HQNR={metric('HQNR')} SCC={metric('SCC')} "
                  f"ERGAS={metric('ERGAS')}", flush=True)
            self.train["diagnostics_done"].append(step)
        finally:
            self.model.train(mode)
            self._restore_rng(rng)
            self._sync()
            self.train["diagnostic_seconds"] += time.perf_counter() - started

    def _post_update_events(self):
        """Complete due events before another update, including after a pause.

        A signal can arrive exactly at update 2K/20K. Those updates are saved
        immediately, with validation still pending; resume must not skip it.
        """
        step = self.train["step"]
        if step <= 0:
            self._diagnostics(step)
            return
        if step % self.probe_interval == 0 and step not in self.train["probe_steps"]:
            self._sync()
            start = time.perf_counter()
            train_probe = self.validate(dataset=self.train_dataset, indices=self.train_ids[:128])
            val_probe = self.validate(indices=self.val_ids[:128])
            self._sync()
            self.train["diagnostic_seconds"] += time.perf_counter() - start
            self._log("probes.jsonl", dict(step=step, train=train_probe, validation=val_probe))
            self.train["probe_steps"].append(step)
        if self._pause_requested:
            return
        new_validation = False
        if ((step % self.validation_interval == 0 or step == self.max_updates)
                and step not in self.train["full_validation_steps"]):
            self._sync()
            start = time.perf_counter()
            validation = self.validate()
            self._sync()
            self.train["diagnostic_seconds"] += time.perf_counter() - start
            previous = self.train["best_val_l1"]
            improved = previous is None or validation["mean_l1"] < previous
            if improved:
                self.train.update(best_val_l1=validation["mean_l1"], best_step=step)
            self.train["full_validation_steps"].append(step)
            self._log("validation.jsonl", dict(step=step, validation=validation))
            self._log("selection_history.jsonl", dict(step=step, candidate_l1=validation["mean_l1"],
                      selected=improved, best_step=self.train["best_step"], tie_policy="EARLIER_STEP"))
            new_validation = True
        if not self._pause_requested:
            self._diagnostics(step)
        if new_validation:
            self.save_checkpoint(reason="FULL_VALIDATION", publish_selectors=True)

    def _update(self):
        self._sync()
        started = time.perf_counter()
        drawing = self.sampler.peek()
        items, raw_ids, crop_origins = [], [], []
        denominator = 0
        for index, rotation in zip(drawing["sample_ids"], drawing["rot_ids"]):
            item = self.train_dataset[index]
            valid = _validate_item(item)
            if valid <= 0:
                raise ValueError("positive-support dataset changed after sampler freeze")
            denominator += valid
            items.append({key: fixed_tensor_view(item[key], rotation) for key in FIELDS})
            raw_ids.append(int(item.get("raw_index", index)))
            origin = item.get("crop_origin")
            crop_origins.append(origin.tolist() if isinstance(origin, torch.Tensor) else origin)
        self._in_update = True
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        numerator_sum = 0.0
        # No autocast, output clipping, grad clipping, LR scheduler or batch
        # resizing. Every micro loss has the same logical-batch denominator.
        for offset in range(0, 48, self.micro_batch):
            batch = _collate(items[offset:offset + self.micro_batch], self.device)
            with torch.autocast(device_type=self.device.type, enabled=False):
                output = self.model(self.sensor_id, batch["pan"], batch["ms"], batch["lpan"])
                numerator, _ = masked_l1_sum(output, batch["gt"], batch["mask"])
                (numerator / denominator).backward()
            numerator_sum += float(numerator.detach().item())
        grad_norm_squared = 0.0
        for parameter in self.model.parameters():
            if parameter.grad is None:
                raise RuntimeError("unexpected disconnected trainable parameter")
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("nonfinite gradient before optimizer update")
            grad_norm_squared += float(parameter.grad.detach().double().square().sum().item())
        self.optimizer.step()
        for parameter in self.model.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("nonfinite updated parameter; current update is not committed")
        for parameter_state in self.optimizer.state.values():
            for value in parameter_state.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise FloatingPointError("nonfinite optimizer state; current update is not committed")
        self.sampler.commit(drawing["token"])
        self.train["step"] += 1
        self.train["samples_seen"] += 48
        self._in_update = False
        self._sync()
        elapsed = time.perf_counter() - started
        self.train["adapt_seconds"] += elapsed
        loss = numerator_sum / denominator
        self.train["recent_loss_sum"] += loss
        self.train["recent_count"] += 1
        record = dict(step=self.train["step"], samples_seen=self.train["samples_seen"],
                      effective_epochs=self.train["samples_seen"] / len(self.train_ids),
                      adapt_seconds=self.train["adapt_seconds"], update_seconds=elapsed,
                      loss_sum=numerator_sum, valid_elements=denominator, mean_l1=loss,
                      lr=self.lr, gradient_norm=math.sqrt(grad_norm_squared),
                      virtual_indices=drawing["sample_ids"], raw_indices=raw_ids,
                      crop_origins=crop_origins, view_ids=drawing["rot_ids"],
                      batch_token=drawing["token"], repeated_within_batch=48 - len(set(drawing["sample_ids"])),
                      peak_memory_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0)
        if self.train["step"] % self.log_interval == 0:
            record["recent"] = dict(loss_sum=self.train["recent_loss_sum"], count=self.train["recent_count"],
                                    mean_l1=self.train["recent_loss_sum"] / self.train["recent_count"])
            print(f"PANPV {self.method} step={self.train['step']} recent_L1={record['recent']['mean_l1']:.7f} "
                  f"LR={self.lr:g} grad_norm={record['gradient_norm']:.6f}", flush=True)
            self.train.update(recent_loss_sum=0.0, recent_count=0)
        self._log("train_steps.jsonl", record)

    def request_stop(self, signum=None):
        """Signal handlers only flag a pause; checkpointing waits for commit."""
        self._pause_requested, self._pause_signal = True, signum

    @contextlib.contextmanager
    def _signal_handlers(self):
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda signum, frame: self.request_stop(signum))
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def run_until(self, target_step=None):
        if self._resume_required:
            raise ValueError("existing run requires explicit resume(trusted_local=True), not weights-only continuation")
        if self._pause_requested:
            raise RuntimeError("paused engine needs an explicit resume before training can continue")
        target_step = min(self.train["step"] + self.block_updates, self.max_updates) if target_step is None else int(target_step)
        if not self.train["step"] <= target_step <= self.max_updates:
            raise ValueError("target must be between the committed step and the frozen endpoint")
        with self._owned(self.root / ".training.lock").open("a") as lock, self._signal_handlers():
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another trainer owns this run directory") from exc
            # A second engine may have been constructed before the first one
            # published anything. A lock alone must not let that stale engine
            # overwrite a completed block with fresh initialization afterward.
            pointer = self._owned(self.root / "latest.json")
            if pointer.exists() and json.loads(pointer.read_text())["fullstate_sha256"] != self._last_checkpoint_sha:
                raise ValueError("run state changed externally; explicit resume is required")
            try:
                self.train["status"] = "RUNNING"
                if self._last_checkpoint_sha is None:
                    # Also make failures before the first 2K checkpoint
                    # explicitly resumable; step0 is never a BEST candidate.
                    self.save_checkpoint(reason="INITIAL_STATE", publish_selectors=False)
                self._post_update_events()
                while self.train["step"] < target_step and not self._pause_requested:
                    self._update()
                    if self._pause_requested:
                        break  # Do not delay an explicit stop with expensive validation.
                    self._post_update_events()
                self.train["status"] = ("PAUSED_SIGNAL" if self._pause_requested else
                                         "COMPLETE" if self.train["step"] == self.max_updates else "BLOCK_COMPLETE")
                self.save_checkpoint(reason=self.train["status"], publish_selectors=self.train["step"] in
                                     self.train["full_validation_steps"])
                _atomic_json(self._owned(self.root / "cost_breakdown.json"),
                             dict(adapt_seconds=self.train["adapt_seconds"],
                                  diagnostic_seconds=self.train["diagnostic_seconds"],
                                  checkpoint_seconds=self.train["checkpoint_seconds"],
                                  samples_seen=self.train["samples_seen"], step=self.train["step"],
                                  status=self.train["status"], stop_signal=self._pause_signal))
            except Exception as exc:
                self.train["status"] = "ERROR_STOPPED"
                self._resume_required = True
                self._log("errors.jsonl", dict(step=self.train["step"], error_type=type(exc).__name__, error=str(exc),
                          partial_update=self._in_update, recovery="EXPLICIT_RESUME_FROM_LAST_VERIFIED_CHECKPOINT_ONLY"))
                # Never serialize a possibly half-mutated optimizer/model.
                raise
        return copy.deepcopy(self.train)
