"""Atomic, verified checkpoint publication confined to the new run directory."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        json.dump(value, out, sort_keys=True, allow_nan=False)
        out.flush(); os.fsync(out.fileno())
    os.replace(name, path); _sync_directory(path.parent)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("BLOCKED_IDENTITY: CUDA RNG cannot be restored on CPU")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def tensor_state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _load(path):
    # Only locally published, SHA-verified campaign-owned fullstates are loaded.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CheckpointStore:
    def __init__(self, run_dir, identity):
        self.root = Path(run_dir).resolve()
        self.identity = dict(identity)
        self.resume_dir = self.root / "resume"
        self._owned(self.resume_dir)
        self.resume_dir.mkdir(parents=True, exist_ok=True)

    def _owned(self, path):
        if not Path(path).resolve().is_relative_to(self.root):
            raise ValueError("BLOCKED_PATH: checkpoint path escapes owned run through symlink")

    def _check(self, state):
        if state.get("identity") != self.identity:
            raise ValueError("BLOCKED_IDENTITY: fullstate identity mismatch")
        if tensor_state_hash(state["model"]) != state["tensor_state_sha256"]:
            raise ValueError("CORRUPT_CHECKPOINT: tensor-state digest mismatch")

    def save_resume(self, state, *, archive=False):
        state = dict(state, identity=self.identity)
        state["tensor_state_sha256"] = tensor_state_hash(state["model"])
        pointer = self.resume_dir / "last.json"
        self._owned(pointer)
        previous = json.loads(pointer.read_text()) if pointer.exists() else {"generation": 0, "slots": {}}
        # Verify last valid state before replacing the other slot.
        valid_slot = None
        if previous["slots"]:
            self.load_resume()
            valid_slot = self._last_loaded_slot
        generation = previous["generation"] + 1
        slot = "B" if valid_slot == "A" else "A"
        path = self.resume_dir / ("last_" + slot + ".pt")
        self._owned(path)
        fd, temporary = tempfile.mkstemp(prefix=".resume-", suffix=".pt", dir=self.resume_dir)
        with os.fdopen(fd, "wb") as out:
            torch.save(state, out); out.flush(); os.fsync(out.fileno())
        self._check(_load(temporary))
        digest = file_sha(temporary)
        os.replace(temporary, path); _sync_directory(self.resume_dir)
        slots = dict(previous["slots"])
        slots[slot] = {"file": path.name, "sha256": digest, "generation": generation,
                       "completed_step": state["train"]["step"]}
        _json(pointer, {"schema": "PANDEP_RESUME_POINTER_v1", "generation": generation,
                        "active": slot, "slots": slots})
        if archive:
            destination = self.resume_dir / ("step_%09d.pt" % state["train"]["step"])
            self._owned(destination)
            if not destination.exists():
                # Copy rather than hard-link: A/B replacements cannot affect archive bytes.
                fd, tmp = tempfile.mkstemp(prefix=".archive-", dir=self.resume_dir)
                with os.fdopen(fd, "wb") as out, open(path, "rb") as inp:
                    for chunk in iter(lambda: inp.read(1 << 20), b""):
                        out.write(chunk)
                    out.flush(); os.fsync(out.fileno())
                if file_sha(tmp) != digest:
                    raise IOError("archive readback mismatch")
                os.replace(tmp, destination)
                _json(destination.with_suffix(".json"), {"sha256": digest, "identity": self.identity,
                                                        "completed_step": state["train"]["step"]})
        return slots[slot]

    def load_resume(self):
        pointer = self.resume_dir / "last.json"
        if not pointer.exists():
            return None
        index = json.loads(pointer.read_text())
        failures = []
        for name, slot in sorted(index["slots"].items(), key=lambda s: s[1]["generation"], reverse=True):
            path = self.resume_dir / slot["file"]
            if path.name not in ("last_A.pt", "last_B.pt") or path.parent.resolve() != self.resume_dir:
                raise ValueError("invalid resume pointer path")
            if not path.exists() or file_sha(path) != slot["sha256"]:
                failures.append(path.name); continue
            state = _load(path)
            self._check(state)  # Identity mismatch is not a reason to fall back.
            self._last_loaded_slot = name
            return state
        raise IOError("CORRUPT_CHECKPOINT: no verified resume slot: " + ",".join(failures))

    def save_weights(self, model, completed_step, exposures, *, best=False, extra=None):
        from safetensors.torch import load_file, save_file
        state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        tensor_hash = tensor_state_hash(state)
        parent = self.root / ("best" if best else "checkpoints")
        self._owned(parent)
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=parent))
        path = temporary / "model.safetensors"
        save_file(state, str(path))
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        if tensor_state_hash(load_file(str(path))) != tensor_hash:
            raise IOError("checkpoint tensor readback mismatch")
        digest = file_sha(path)
        metadata = dict(self.identity, completed_step=int(completed_step), exposures=exposures,
                        checkpoint_sha256=digest, tensor_state_sha256=tensor_hash)
        metadata.update(extra or {})
        destination = parent / (digest if best else "step_%09d" % completed_step)
        self._owned(destination)
        metadata["model_path"] = str(destination / "model.safetensors")
        _json(temporary / "identity.json", metadata)
        if destination.exists():
            previous = json.loads((destination / "identity.json").read_text())
            if previous != metadata or file_sha(destination / "model.safetensors") != digest:
                raise ValueError("CHECKPOINT_CONFLICT: immutable weight publication exists")
            # Unpublished verified temporary is retained rather than deleting user assets.
            return previous
        os.replace(temporary, destination); _sync_directory(parent)
        return metadata

    def load_weights(self, model, identity):
        from safetensors.torch import load_file
        path = Path(identity["model_path"]).resolve()
        if not path.is_relative_to(self.root) or file_sha(path) != identity["checkpoint_sha256"]:
            raise ValueError("BLOCKED_IDENTITY: benchmark weight bytes/path mismatch")
        state = load_file(str(path))
        if tensor_state_hash(state) != identity["tensor_state_sha256"]:
            raise ValueError("BLOCKED_IDENTITY: weight tensor digest mismatch")
        model.load_state_dict(state, strict=True)
