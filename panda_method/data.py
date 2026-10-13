"""Explicit caller-supplied lineage; no benchmark-as-training fallback.

Hashes and declarations are a contract, not independent proof of geographic
separation. A later data adapter must validate the underlying manifests/crops.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re

import torch

from .geometry import (ScenePriorRegistry, VerifiedGridTransform,
                       convert_prior_to_grid, transform_prior)


def require_sha256(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name}: explicit SHA256 digest required")


@dataclass(frozen=True)
class BatchLineage:
    split: str
    resolution: str
    source_manifest_sha256: str
    prior_mapping_sha256: str
    scene_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    grid_name: str
    geometry_evidence: str
    native_training_evidence: str = ""

    def __post_init__(self):
        if self.split not in ("train", "validation", "test", "inference"):
            raise ValueError("Explicit train/validation/test/inference split required")
        if self.resolution not in ("RR", "FR"):
            raise ValueError("Explicit RR/FR grid required")
        for name in ("source_manifest_sha256", "prior_mapping_sha256"):
            require_sha256(getattr(self, name), name)
        if not self.grid_name.strip() or not self.geometry_evidence.strip():
            raise ValueError("Verified grid/crop/augmentation mapping evidence required")
        if (not self.scene_ids or len(self.scene_ids) != len(self.sample_ids)
                or any(not isinstance(s, str) or not s.strip()
                       for s in (*self.scene_ids, *self.sample_ids))):
            raise ValueError("Every sample needs a real source scene ID and sample ID")
        object.__setattr__(self, "scene_ids", tuple(self.scene_ids))
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))

    def require_training(self):
        if self.split != "train":
            raise ValueError("Benchmark/validation inputs cannot enter method training")
        if self.resolution == "FR" and not self.native_training_evidence.strip():
            raise ValueError("Training-only native FR source evidence is mandatory")


@dataclass(frozen=True)
class ObservationBatch:
    pan: torch.Tensor
    ms: torch.Tensor
    native_lms: torch.Tensor
    prior_dy_dx: torch.Tensor
    lineage: BatchLineage
    gt: torch.Tensor | None = None
    lpan: torch.Tensor | None = None

    def validate(self, bands, *, training=False, require_gt=False):
        p, m, lms = self.pan, self.ms, self.native_lms
        if p.ndim != 4 or p.shape[1] != 1 or p.shape[0] < 1:
            raise ValueError("PAN must be nonempty [B,1,H,W]")
        b, _, h, w = p.shape
        if h % 4 or w % 4 or min(h, w) < 8:
            raise ValueError("PAN H/W must be >=8 and divisible by four")
        if m.shape != (b, bands, h // 4, w // 4) or lms.shape != (b, bands, h, w):
            raise ValueError("Native LMS is required separately from LRMS/reconstruction base")
        if self.prior_dy_dx.shape != (b, 2) or len(self.lineage.scene_ids) != b:
            raise ValueError("One verified scene prior per sample required")
        if self.lineage.resolution == "FR" and self.gt is not None:
            raise ValueError("FR must not provide a synthetic HRMS GT")
        if require_gt and (self.lineage.resolution != "RR" or self.gt is None):
            raise ValueError("RR supervised GT required")
        if self.gt is not None and self.gt.shape != lms.shape:
            raise ValueError("RR GT must have all MS bands on the PAN grid")
        if self.lpan is not None and self.lpan.shape != (b, 1, h // 4, w // 4):
            raise ValueError("LPAN cache must be on the LRMS grid")
        tensors = [p, m, lms, self.prior_dy_dx]
        tensors += [t for t in (self.gt, self.lpan) if t is not None]
        for tensor in tensors:
            if (not tensor.is_floating_point() or tensor.device != p.device
                    or not bool(torch.isfinite(tensor).all())):
                raise ValueError("Finite floating observations on the same device required")
        if training:
            self.lineage.require_training()
        return self


class PriorBinding:
    """Bind true sample->scene identity, canonical priors and audited grid maps.

    This consumes a real external mapping, not H5 row numbers masquerading as
    acquisitions. Its fingerprint is ``TrainingSources.prior_mapping_sha256``.
    Exact canonical-grid lookup needs no invented phase conversion; any change
    of grid requires the caller's explicit matrix AND phase evidence.
    """

    def __init__(self, registry: ScenePriorRegistry, *, sample_scene_pairs,
                 grid_transforms):
        if not isinstance(registry, ScenePriorRegistry):
            raise TypeError("Accepted ScenePriorRegistry required")
        self.registry = registry
        self._samples, self._transforms = {}, {}
        for sample, scene in sample_scene_pairs:
            if not isinstance(sample, str) or not sample.strip() or sample in self._samples:
                raise ValueError("Unique, nonempty source sample IDs required")
            registry.record(scene)
            self._samples[sample] = scene
        if not self._samples:
            raise ValueError("Source sample->scene mapping is missing")
        for transform in grid_transforms:
            if not isinstance(transform, VerifiedGridTransform):
                raise TypeError("Explicit verified grid transforms required")
            key = (transform.source_grid, transform.target_grid)
            if key in self._transforms:
                raise ValueError("Conflicting/duplicate grid transforms")
            self._transforms[key] = transform

    def fingerprint(self):
        payload = {"registry": self.registry.fingerprint(), "samples": self._samples,
                   "transforms": [[*key, value.fingerprint()]
                                  for key, value in sorted(self._transforms.items())]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                              separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def resolve(self, lineage: BatchLineage, *, device, dtype):
        if lineage.prior_mapping_sha256 != self.fingerprint():
            raise ValueError("Wrong/stale sample/scene/prior/grid mapping fingerprint")
        priors = []
        for sample, scene in zip(lineage.sample_ids, lineage.scene_ids):
            if self._samples.get(sample) != scene:
                raise ValueError("Sample does not belong to the declared source scene")
            record = self.registry.record(scene)
            prior = torch.tensor(record.prior_shift_dy_dx, device=device, dtype=dtype)
            if record.prior_grid_name != lineage.grid_name:
                key = (record.prior_grid_name, lineage.grid_name)
                if key not in self._transforms:
                    raise ValueError("Missing explicit grid-scale/phase transform; no automatic x4")
                prior = convert_prior_to_grid(prior, source_grid=record.prior_grid_name,
                                              transform=self._transforms[key])
            priors.append(prior)
        return torch.stack(priors)


def bind_scene_priors(*, pan, ms, native_lms, lineage, binding: PriorBinding,
                      gt=None, lpan=None):
    """Resolve fixed priors BEFORE applying image augmentation."""
    priors = binding.resolve(lineage, device=pan.device, dtype=pan.dtype)
    batch = ObservationBatch(pan, ms, native_lms, priors, lineage, gt, lpan)
    return batch.validate(ms.shape[1])


def augment_batch(batch: ObservationBatch, *, hflip: bool, vflip: bool, rot90: int):
    """Apply the exact same spatial operation to observations and dy/dx prior.

    A single transform for this batch; mixed augmentations can be assembled by
    the future loader from individually transformed samples. No random draws,
    crops, resize, hidden phase offset, or prior-estimator rerun occurs here.
    """
    prior = transform_prior(batch.prior_dy_dx, hflip=hflip, vflip=vflip, rot90=rot90)

    def apply(tensor):
        if tensor is None:
            return None
        if hflip:
            tensor = tensor.flip(-1)
        if vflip:
            tensor = tensor.flip(-2)
        return torch.rot90(tensor, rot90 % 4, (-2, -1))

    evidence = (batch.lineage.geometry_evidence
                + f" | image+prior: hflip={hflip},vflip={vflip},rot90_ccw={rot90 % 4}")
    return replace(batch, pan=apply(batch.pan), ms=apply(batch.ms),
                   native_lms=apply(batch.native_lms), gt=apply(batch.gt), lpan=apply(batch.lpan),
                   prior_dy_dx=prior, lineage=replace(batch.lineage, geometry_evidence=evidence))
