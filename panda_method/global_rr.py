"""Explicit dataset-global, provided-LMS, RR-only method path.

The scene-conditioned v1.3 API remains unchanged. This module is not an
experiment runner: capacities, optimizer, source assets and all scientific
acceptance thresholds remain explicit caller inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from pa.offset import sample_offsets
from pa.warp import warp_support_mask
from .calibration import canonical_sha256
from .data import require_sha256
from .geometry import SceneAwarePANAligner
from .stages import Stage1, Stage2, StudentInference, _support


METHOD_REVISION = "BGlobalRR-v1"
GLOBAL_RR_METHOD_VERSION = "PANDA_BGlobal_RR_v1_2026-10-13"


@dataclass(frozen=True)
class RRTrainingSources:
    """Train-only source and immutable dataset-prior binding, with no scene IDs."""

    rr_manifest_sha256: str
    prior_mapping_sha256: str
    sensor: str
    train_source_sha256: str

    def __post_init__(self):
        for name in ("rr_manifest_sha256", "prior_mapping_sha256", "train_source_sha256"):
            require_sha256(getattr(self, name), name)
        if self.sensor not in ("WV3", "QB", "GF2"):
            raise ValueError("Explicit WV3/QB/GF2 dataset-global sensor required")

    def check(self, batch, resolution):
        batch.lineage.require_training()
        if resolution != "RR" or batch.lineage.resolution != "RR":
            raise ValueError("Dataset-global training accepts RR only; FR is evaluation-only")
        expected = {"prior_scope": "dataset_global", "sensor": self.sensor,
                    "train_source_sha256": self.train_source_sha256,
                    "source_manifest_sha256": self.rr_manifest_sha256,
                    "prior_mapping_sha256": self.prior_mapping_sha256,
                    "grid_name": "PAN_RR"}
        if any(getattr(batch.lineage, key, None) != value for key, value in expected.items()):
            raise ValueError("Batch does not belong to the pinned RR train/dataset-global prior")


class GlobalPANAligner(SceneAwarePANAligner):
    """The same v1.3 CNN/head, with an incompatible dataset-global identity.

    Magnitude limits are reviewed asset inputs, not new implicit hyperparameters.
    No fallback direction, normalization, clamping or test-time scaling occurs.
    The tensor-only identity buffer keeps existing exact calibration hashing
    usable while rejecting semantic checkpoint migration before child mutation.
    """

    CONDITIONING_REVISION = 1001

    def __init__(self, ms_bands=8, *, view_margin, sensor, train_source_sha256,
                 prior_asset_sha256, canonical_prior_dy_dx,
                 min_prior_magnitude, max_prior_magnitude):
        for name, value in (("train_source_sha256", train_source_sha256),
                            ("prior_asset_sha256", prior_asset_sha256)):
            require_sha256(value, name)
        if sensor not in ("WV3", "QB", "GF2"):
            raise ValueError("Explicit WV3/QB/GF2 dataset-global sensor required")
        if ms_bands != (8 if sensor == "WV3" else 4):
            raise ValueError("Aligner band count differs from its sensor")
        low, high = float(min_prior_magnitude), float(max_prior_magnitude)
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
            raise ValueError("Explicit positive finite ordered prior magnitude limits required")
        canonical = tuple(float(v) for v in canonical_prior_dy_dx)
        if (len(canonical) != 2 or not all(math.isfinite(v) for v in canonical)
                or not low <= math.hypot(*canonical) <= high):
            raise ValueError("Canonical prior must be finite and within reviewed magnitude limits")
        self.sensor, self.train_source_sha256 = sensor, train_source_sha256
        self.prior_asset_sha256 = prior_asset_sha256
        self.canonical_prior_dy_dx = canonical
        self.min_prior_magnitude, self.max_prior_magnitude = low, high
        super().__init__(ms_bands, view_margin=view_margin)
        self.register_buffer("_global_prior_scope_revision", torch.tensor(1, dtype=torch.int64))
        self.register_buffer("_global_prior_identity", self._identity_tensor())

    def conditioning_manifest(self):
        return {**super().conditioning_manifest(), "method": GLOBAL_RR_METHOD_VERSION,
                "prior_scope": "dataset_global", "sensor": self.sensor,
                "train_source_sha256": self.train_source_sha256,
                "prior_asset_sha256": self.prior_asset_sha256,
                "canonical_prior_dy_dx": list(self.canonical_prior_dy_dx),
                "reference": "PROVIDED_TRAIN_LMS", "canonical_prior_grid": "PAN_RR",
                "magnitude_policy": {"action": "fail_fast_no_fallback_no_clamp",
                                     "min_inclusive": self.min_prior_magnitude,
                                     "max_inclusive": self.max_prior_magnitude},
                "fr_policy": "blocked_pending_explicit_reviewed_policy"}

    def _identity_tensor(self):
        return torch.tensor(list(bytes.fromhex(canonical_sha256(self.conditioning_manifest()))),
                            dtype=torch.uint8)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        expected = {"_conditioning_revision": torch.tensor(self.CONDITIONING_REVISION),
                    "_global_prior_scope_revision": torch.tensor(1),
                    "_global_prior_identity": self._identity_tensor()}
        for name, value in expected.items():
            saved = state_dict.get(prefix + name)
            if (not isinstance(saved, torch.Tensor) or saved.dtype != value.dtype
                    or saved.shape != value.shape or not torch.equal(saved.cpu(), value)):
                raise RuntimeError("Dataset-global prior/checkpoint identity mismatch; "
                                   "scene-prior checkpoints cannot be silently reused")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, pan, native_lms, prior_dy_dx, return_details=False):
        if (not isinstance(prior_dy_dx, torch.Tensor)
                or not prior_dy_dx.is_floating_point() or prior_dy_dx.ndim != 2
                or prior_dy_dx.shape[-1] != 2 or not bool(torch.isfinite(prior_dy_dx).all())):
            raise ValueError("Finite [B,2] dataset-global signed prior required")
        # Only the frozen vector's D4 orbit is valid on this RR path. Compare
        # after the *same* dtype conversion as the data adapter, not a second
        # norm threshold: (.3,.4) has norm .5 canonically, but its rounded FP32
        # representation has FP64 norm .5000000119. Limits were already checked
        # against the canonical asset at construction; augmentation preserves
        # that physical magnitude. Exact membership also rejects patch refits.
        dy, dx = self.canonical_prior_dy_dx
        orbit = prior_dy_dx.new_tensor([(sy * y, sx * x) for y, x in ((dy, dx), (dx, dy))
                                       for sy in (-1., 1.) for sx in (-1., 1.)])
        matches = (prior_dy_dx.detach()[:, None, :] == orbit[None, :, :]).all(dim=-1).any(dim=-1)
        if not bool(matches.all()):
            raise ValueError("Prior is not the frozen canonical RR vector under a D4 augmentation; "
                             "no patch refit, near-zero fallback or magnitude change is allowed")
        return super().forward(pan, native_lms, prior_dy_dx, return_details=return_details)


def _check_global_lineage(batch, aligner):
    expected = {"prior_scope": "dataset_global", "sensor": aligner.sensor,
                "train_source_sha256": aligner.train_source_sha256,
                "prior_mapping_sha256": aligner.prior_asset_sha256}
    if any(getattr(batch.lineage, key, None) != value for key, value in expected.items()):
        raise ValueError("Observation does not match the pinned dataset-global Aligner identity")
    if tuple(getattr(batch.lineage, "canonical_prior_dy_dx", ())) != aligner.canonical_prior_dy_dx:
        raise ValueError("Observation canonical prior differs from the pinned asset vector")
    if batch.lineage.resolution == "RR":
        if batch.lineage.grid_name != "PAN_RR":
            raise ValueError("RR observations must use the explicit PAN_RR pixel grid")
    elif batch.lineage.resolution == "FR":
        raise NotImplementedError("FR prior unit/conditioning policy is unresolved; "
                                  "neither b_RR nor 4*b_RR is implicitly supplied")
    else:
        raise ValueError("Explicit RR/FR observation resolution required")


def _global_pipeline(aligner, teacher, boundary_policy, calibration_protocol):
    return {"method": GLOBAL_RR_METHOD_VERSION, "revision": METHOD_REVISION,
            "prior_scope": "dataset_global", "aligner_reference": "PROVIDED_LMS",
            "aligner_view_margin": aligner.view_margin,
            "aligner_conditioning": aligner.conditioning_manifest(),
            "teacher": teacher.spec.manifest(),
            "warp": "bicubic_border_align_corners_false_fp32",
            "boundary_policy": boundary_policy, "calibration_protocol": calibration_protocol,
            "rec_reduction": "all_pixels_and_bands_mean", "training_resolutions": ["RR"],
            "eps": {"radius": 2., "lambda": 1e-3, "stop_reference": True,
                    "cadence": "every_forward_RR_only", "generator": "dedicated_CPU"}}


class GlobalRRStage1(Stage1):
    """One native U forward and two A forwards per RR-only optimizer objective."""

    def __init__(self, aligner, teacher, spatial, *, sources, epsilon_seed,
                 boundary_policy, calibration_protocol):
        if not isinstance(aligner, GlobalPANAligner) or not isinstance(sources, RRTrainingSources):
            raise TypeError("GlobalPANAligner and RRTrainingSources required")
        if (sources.sensor != aligner.sensor
                or sources.train_source_sha256 != aligner.train_source_sha256
                or sources.prior_mapping_sha256 != aligner.prior_asset_sha256
                or spatial.spec.sensor != aligner.sensor):
            raise ValueError("Aligner, spatial loss and training source identities disagree")
        super().__init__(aligner, teacher, spatial, sources=sources, epsilon_seed=epsilon_seed,
                         boundary_policy=boundary_policy, calibration_protocol=calibration_protocol)

    def pipeline_manifest(self):
        return _global_pipeline(self.aligner, self.teacher, self.boundary_policy,
                                self.calibration_protocol)

    def get_extra_state(self):
        return {**super().get_extra_state(), "method": GLOBAL_RR_METHOD_VERSION}

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        if any(state.get(key) != expected[key] for key in ("method", "pipeline", "spatial", "sources")):
            raise ValueError("Dataset-global RR checkpoint/source contract mismatch")
        self.epsilon_seed = int(state["epsilon_seed"])
        self.forward_calls = int(state["forward_calls"])
        self._epsilon_rng.set_state(state["epsilon_rng"].cpu())

    def native_prediction(self, batch, *, return_aligner_details=False):
        _check_global_lineage(batch, self.aligner)
        return super().native_prediction(batch, return_aligner_details=return_aligner_details)

    def forward(self, rr):
        if not all(p.requires_grad for m in (self.aligner, self.teacher) for p in m.parameters()):
            raise ValueError("Stage 1 requires trainable A/T; do not reuse after transfer to Stage 2")
        rr.validate(self.teacher.spec.bands, training=True, require_gt=True)
        self.sources.check(rr, "RR")
        rng_before = self._epsilon_rng.get_state()
        try:
            with torch.autocast(device_type=rr.pan.device.type, enabled=False):
                native = self.native_prediction(rr)
                rec = (native["y"] - rr.gt.detach().float()).abs().mean()
                absolute = self.spatial.stage1(native["pan_aligned"], rr.gt)
                support = _support(rr.pan, native["correction"], self.spatial, self.boundary_policy)
                epsilon = sample_offsets(len(rr.pan), 2., self._epsilon_rng).to(rr.pan.device)
                eps_rr, c_eps = self._epsilon(rr, native["correction"], epsilon)
                con_rr = absolute + 1e-3 * eps_rr
                loss = rec + con_rr
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite dataset-global RR Stage 1 objective")
                mask = warp_support_mask(*rr.pan.shape[-2:], epsilon)
                margin = self.aligner.view_margin
                h, w = rr.pan.shape[-2:]
                epsilon_support = mask[..., margin:h-margin, margin:w-margin].float().mean()
        except Exception:
            self._epsilon_rng.set_state(rng_before)
            raise
        index = self.forward_calls
        self.forward_calls += 1
        return {"loss": loss, "rec": rec, "abs_rr": absolute, "eps_rr": eps_rr,
                "con_rr": con_rr, "weighted_eps_rr": 1e-3 * eps_rr,
                "native": native, "epsilon_rr": epsilon, "c_eps_rr": c_eps,
                "forward_index": index, "epsilon_rr_view_support_fraction": epsilon_support,
                **support}


class GlobalRRStage2(Stage2):
    """Existing frozen/shared A, e/q train median and Student loss, new identity."""

    def __init__(self, stage1, student, calibration):
        if not isinstance(stage1, GlobalRRStage1):
            raise TypeError("GlobalRRStage1 required for dataset-global Stage 2")
        super().__init__(stage1, student, calibration)

    def pipeline_manifest(self):
        return _global_pipeline(self.aligner, self.teacher, self.boundary_policy,
                                self.calibration_protocol)

    def get_extra_state(self):
        return {**super().get_extra_state(), "method": GLOBAL_RR_METHOD_VERSION}

    def forward(self, rr):
        _check_global_lineage(rr, self.aligner)
        return super().forward(rr)

    def inference_model(self):
        return GlobalStudentInference(self.aligner, self.student)


class GlobalStudentInference(StudentInference):
    """RR-only inference until a reviewed FR prior policy is implemented."""

    def forward(self, batch):
        _check_global_lineage(batch, self.aligner)
        return super().forward(batch)
