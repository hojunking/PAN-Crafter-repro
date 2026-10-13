"""Exact, training-only, checkpoint-bound calibration for MethodPlan-v1.3.

Calibration retains detached FP64 values on CPU, never on GPU.  This is an
exact population median, not a batch/sample-median approximation.  Peak host
memory is proportional to the training population; callers must budget it
before executing a real calibration.  No calibration runs on import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor, nn

from .losses import AbsoluteGradientLoss


def canonical_sha256(value: object) -> str:
    """Hash JSON-serializable provenance; nonfinite numbers are rejected."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def state_dict_sha256(module: nn.Module | Mapping[str, Tensor]) -> str:
    """Hash exact state tensors plus their names/dtypes/shapes, not pickle bytes."""
    state = module.state_dict() if isinstance(module, nn.Module) else module
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, Tensor) or value.layout != torch.strided:
            raise ValueError("calibration state must consist of dense tensors")
        x = value.detach().cpu().contiguous()
        meta = json.dumps([name, str(x.dtype), list(x.shape)], separators=(",", ":")).encode()
        raw = x.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(meta).to_bytes(8, "big"))
        digest.update(meta)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationBinding:
    aligner_sha256: str
    teacher_sha256: str
    pipeline_sha256: str
    source_manifest_sha256: str
    prior_mapping_sha256: str
    spatial_spec_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    @classmethod
    def for_modules(cls, aligner: nn.Module, teacher: nn.Module, *, pipeline: Mapping,
                    source_manifest_sha256: str, prior_mapping_sha256: str,
                    spatial: AbsoluteGradientLoss) -> CalibrationBinding:
        return cls(state_dict_sha256(aligner), state_dict_sha256(teacher),
                   canonical_sha256(pipeline), source_manifest_sha256,
                   prior_mapping_sha256, spatial.spec.fingerprint())

    def fingerprint(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class CalibrationScales:
    tau_e: float
    tau_q: float
    n_e: int
    n_q: int
    n_samples: int
    binding: CalibrationBinding
    sample_ids_sha256: str
    split: str = "train"
    median_convention: str = "midpoint_even_global_pixels"

    def __post_init__(self) -> None:
        if self.split != "train":
            raise ValueError("calibration must use the training split only")
        if self.median_convention != "midpoint_even_global_pixels":
            raise ValueError("unsupported median convention")
        if not isinstance(self.binding, CalibrationBinding):
            raise TypeError("calibration requires checkpoint/pipeline binding")
        for value in (self.tau_e, self.tau_q):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("calibration scales must be finite and nonnegative")
        if any(type(v) is not int or v <= 0 for v in (self.n_e, self.n_q, self.n_samples)):
            raise ValueError("calibration counts must be positive integers")
        if self.n_q > self.n_e:
            raise ValueError("spatial ROI count cannot exceed all-pixel error count")
        if not isinstance(self.sample_ids_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sample_ids_sha256) is None:
            raise ValueError("sample population must be identified by SHA-256")

    def assert_binding(self, expected: CalibrationBinding) -> None:
        if self.binding != expected:
            changed = [name for name in asdict(self.binding)
                       if getattr(self.binding, name) != getattr(expected, name)]
            raise ValueError("stale calibration binding: " + ", ".join(changed))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> CalibrationScales:
        payload = dict(data)
        payload["binding"] = CalibrationBinding(**payload["binding"])
        return cls(**payload)


def _global_median(chunks: list[Tensor]) -> tuple[float, int]:
    if not chunks:
        raise ValueError("cannot calibrate an empty training population")
    values = torch.cat(chunks)
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("calibration population must be finite and nonnegative")
    count = values.numel()
    # torch.median uses the lower middle for even counts.  Explicit midpoint
    # agrees with the usual mathematical/numpy population-median convention.
    lo = values.kthvalue((count + 1) // 2).values.item()
    hi = values.kthvalue(count // 2 + 1).values.item() if count % 2 == 0 else lo
    return lo / 2. + hi / 2., count


class ExactTrainCalibrator:
    def __init__(self, binding: CalibrationBinding, spatial: AbsoluteGradientLoss, *,
                 expected_sample_ids: Sequence[str]):
        if binding.spatial_spec_sha256 != spatial.spec.fingerprint():
            raise ValueError("calibration spatial definition does not match its binding")
        ids = list(expected_sample_ids)
        if isinstance(expected_sample_ids, str) or not ids:
            raise ValueError("an explicit nonempty training sample inventory is required")
        if any(not isinstance(v, str) or not v.strip() for v in ids) or len(set(ids)) != len(ids):
            raise ValueError("expected training sample IDs must be unique nonempty strings")
        self.binding = binding
        self.spatial = spatial
        self._expected_ids = frozenset(ids)
        self._e: list[Tensor] = []
        self._q: list[Tensor] = []
        self._sample_ids: set[str] = set()
        self._closed = False

    def update(self, teacher: Tensor, gt: Tensor, aligned_pan: Tensor, *, split: str,
               sample_ids: Sequence[str]) -> None:
        if self._closed:
            raise RuntimeError("calibration has already been finalized")
        if split != "train":
            raise ValueError("only explicitly identified training data may calibrate e/q")
        if self.binding.spatial_spec_sha256 != self.spatial.spec.fingerprint():
            raise ValueError("spatial definition changed during calibration")
        ids = list(sample_ids)
        if isinstance(sample_ids, str) or len(ids) != teacher.shape[0]:
            raise ValueError("sample_ids must identify each batch member")
        if any(not isinstance(v, str) or not v.strip() for v in ids):
            raise ValueError("sample IDs must be nonempty strings")
        if len(set(ids)) != len(ids) or self._sample_ids.intersection(ids):
            raise ValueError("duplicate training samples would change the calibration population")
        if not set(ids).issubset(self._expected_ids):
            raise ValueError("batch contains samples outside the registered training inventory")
        e, q = self.spatial.reference_maps(teacher, gt, aligned_pan)
        selected_q = q[self.spatial.pixel_mask(q)]
        # Clone prevents later user mutations of CPU input tensors changing
        # the already-observed calibration population.
        self._e.append(e.detach().reshape(-1).to(device="cpu", dtype=torch.float64).clone())
        self._q.append(selected_q.detach().to(device="cpu", dtype=torch.float64).clone())
        self._sample_ids.update(ids)

    def finalize(self) -> CalibrationScales:
        if self._closed:
            raise RuntimeError("calibration has already been finalized")
        if self._sample_ids != self._expected_ids:
            missing = len(self._expected_ids - self._sample_ids)
            raise ValueError(f"incomplete training calibration population: {missing} samples missing")
        tau_e, n_e = _global_median(self._e)
        tau_q, n_q = _global_median(self._q)
        result = CalibrationScales(tau_e=tau_e, tau_q=tau_q, n_e=n_e, n_q=n_q,
                                   n_samples=len(self._sample_ids), binding=self.binding,
                                   sample_ids_sha256=canonical_sha256(sorted(self._sample_ids)))
        self._closed = True
        self._e.clear()
        self._q.clear()
        return result
