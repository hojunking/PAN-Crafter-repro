"""Edge-guided Cartesian residual PAN alignment, Stage 1 only.

An explicit new method path, not a checkpoint migration or experiment runner.
Hard per-input EdgeMatch precedes the learned CNN; its prewarped image is used
only by that CNN. Teacher reconstruction always warps the ORIGINAL input PAN.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.pancrafter_paper import ResBlock
from pa.aligner import _conv, znorm
from pa.offset import sample_offsets
from pa.warp import warp_pan, warp_support_mask, two_stage_support_mask
from .calibration import canonical_sha256
from .data import require_sha256
from .edge_match import EdgeMatcher
from .reconstruction import Reconstructor
from .stages import epsilon_loss, _disjoint, _validate_boundary


METHOD_REVISION = "EdgeGuidedResidual-S1-v1"
METHOD_VERSION = "PANDA_EdgeGuided_Residual_Stage1_v1_2026-10-13"
SENSOR_BANDS = {"WV3": 8, "QB": 4, "GF2": 4}


@dataclass(frozen=True)
class EdgeTrainingSources:
    sensor: str
    rr_manifest_sha256: str

    def __post_init__(self):
        if self.sensor not in SENSOR_BANDS:
            raise ValueError("Explicit WV3/QB/GF2 sensor required")
        require_sha256(self.rr_manifest_sha256, "RR training source manifest")

    def check(self, batch):
        if (batch.lineage.split != "train" or batch.lineage.resolution != "RR"
                or batch.lineage.sensor != self.sensor or batch.lineage.grid_name != "PAN_RR"
                or batch.lineage.source_manifest_sha256 != self.rr_manifest_sha256):
            raise ValueError("Only the pinned RR training observations may enter this objective")


class ResidualPANAligner(nn.Module):
    """GN stems/joints, common pre-LN ResBlock, and Cartesian FC64→32→2.

    ``b`` is not a head input and there is no nonzero-prior requirement or basis.
    The caller supplies the preliminary PAN warp and provided stored LMS.
    """

    CHECKPOINT_REVISION = 2001

    def __init__(self, ms_bands=8, *, view_margin):
        super().__init__()
        if ms_bands not in (4, 8):
            raise ValueError("Explicit 4/8 MS bands required")
        if type(view_margin) is not int or view_margin < 0:
            raise ValueError("An explicit nonnegative Aligner view margin is required")
        self.ms_bands, self.view_margin = ms_bands, view_margin
        self.pan_stem = _conv(1, 16, 1, 4)
        self.ms_stem = _conv(ms_bands, 16, 1, 4)
        self.joint1 = _conv(32, 32, 2, 8)
        self.joint2 = _conv(32, 64, 2, 8)
        self.res = ResBlock(64, out_channels=64, norm="ln", dropout=0.,
                            mode_mod=False, se=False)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.register_buffer("_edge_residual_revision", torch.tensor(self.CHECKPOINT_REVISION))
        self.register_buffer("_edge_residual_identity", self._identity())

    def manifest(self):
        return {"method": METHOD_VERSION, "revision": self.CHECKPOINT_REVISION,
                "ms_bands": self.ms_bands, "view_margin": self.view_margin,
                "input": ["prewarped_PAN", "provided_stored_LMS"],
                "cnn_input_normalization": "existing_sample_channel_znorm",
                "stems_joint_norm": "GN", "resblock": "common_UNet_preLN_dropout0",
                "head_dims": [64, 32, 2], "head_prior_conditioning": False,
                "output": "direct_cartesian_delta_dy_dx_current_PAN_px",
                "output_clamp": "none", "fc2_initialization": "weight_bias_zero"}

    def _identity(self):
        return torch.tensor(list(bytes.fromhex(canonical_sha256(self.manifest()))), dtype=torch.uint8)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name, expected in (("_edge_residual_revision", torch.tensor(self.CHECKPOINT_REVISION)),
                               ("_edge_residual_identity", self._identity())):
            value = state_dict.get(prefix + name)
            if (not isinstance(value, torch.Tensor) or value.dtype != expected.dtype
                    or value.shape != expected.shape or not torch.equal(value.cpu(), expected)):
                raise RuntimeError("Edge residual CNN checkpoint identity mismatch; old scene/global/"
                                   "legacy Aligner checkpoints cannot be silently reused")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, pan_pre, native_lms, *, return_details=False):
        if pan_pre.ndim != 4 or pan_pre.shape[0] < 1 or pan_pre.shape[1] != 1:
            raise ValueError("Nonempty [B,1,H,W] preliminary PAN required")
        if native_lms.shape != (len(pan_pre), self.ms_bands, *pan_pre.shape[-2:]):
            raise ValueError("Provided LMS must share the preliminary PAN grid")
        if (not pan_pre.is_floating_point() or pan_pre.device != native_lms.device
                or pan_pre.dtype != native_lms.dtype or not bool(torch.isfinite(pan_pre).all())
                or not bool(torch.isfinite(native_lms).all())):
            raise ValueError("Finite PAN/LMS floating observations with matching dtype/device required")
        margin = self.view_margin
        if min(pan_pre.shape[-2:]) - 2 * margin < 3:
            raise ValueError("CNN view is too small for reflection-padded convolutions")
        if margin:
            pan_pre = pan_pre[..., margin:-margin, margin:-margin]
            native_lms = native_lms[..., margin:-margin, margin:-margin]
        p, m = self.pan_stem(znorm(pan_pre)), self.ms_stem(znorm(native_lms))
        joint = self.joint2(self.joint1(torch.cat((p, m), dim=1)))
        features = self.res(joint, None).mean(dim=(2, 3))
        hidden = F.silu(self.fc1(features))
        delta = self.fc2(hidden)
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("Nonfinite Cartesian residual correction")
        if return_details:
            return delta, {"features": features, "head_input": features, "head_hidden": hidden}
        return delta


def _view_fraction(mask, margin):
    h, w = mask.shape[-2:]
    if 2 * margin >= min(h, w):
        raise ValueError("Declared CNN support view is empty")
    return mask[..., margin:h-margin, margin:w-margin].float().mean()


class EdgeGuidedStage1(nn.Module):
    """RR rec + Scharr absolute + .001 epsilon; RR/FR per-input inference."""

    def __init__(self, aligner, teacher, matcher, *, sources, roi_margin,
                 boundary_policy, epsilon_seed, lambda_abs=1., lambda_eps=.001):
        super().__init__()
        if (not isinstance(aligner, ResidualPANAligner) or not isinstance(teacher, Reconstructor)
                or not isinstance(matcher, EdgeMatcher) or not isinstance(sources, EdgeTrainingSources)):
            raise TypeError("Explicit Edge residual Aligner, Reconstructor, EdgeMatcher and sources required")
        if (teacher.spec.bands != SENSOR_BANDS[sources.sensor]
                or aligner.ms_bands != teacher.spec.bands or matcher.spec.sensor != sources.sensor):
            raise ValueError("Sensor/bands differ between sources, matcher, Aligner and Teacher")
        if teacher.spec.input_layout != "P0" or teacher.spec.base_reference != "BICUBIC_MS":
            raise ValueError("This Stage 1 requires the unchanged P0/BICUBIC_MS Teacher recipe")
        if type(roi_margin) is not int or roi_margin < 0:
            raise ValueError("Explicit nonnegative fixed Scharr loss ROI margin required")
        _validate_boundary(boundary_policy)
        _disjoint(aligner, teacher, matcher)
        if any(True for _ in matcher.parameters()):
            raise ValueError("Hard EdgeMatch must not contain learned parameters")
        self.aligner, self.teacher, self.matcher = aligner, teacher, matcher
        self.sources, self.roi_margin = sources, roi_margin
        self.boundary_policy = boundary_policy
        for name, value in (("lambda_abs", lambda_abs), ("lambda_eps", lambda_eps)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be an explicit finite positive full-method coefficient")
        self.lambda_abs, self.lambda_eps = float(lambda_abs), float(lambda_eps)
        self.epsilon_seed = int(epsilon_seed)
        self._epsilon_rng = torch.Generator(device="cpu").manual_seed(self.epsilon_seed)
        self.forward_calls = 0

    def pipeline_manifest(self):
        return {"method": METHOD_VERSION, "revision": METHOD_REVISION,
                "alignment": "input_specific_integer_b_plus_cartesian_CNN_residual",
                "aligner": self.aligner.manifest(), "matcher": self.matcher.manifest(),
                "teacher": self.teacher.spec.manifest(), "training_resolutions": ["RR"],
                "lms_reference": "provided_stored_LMS_not_bicubic_base",
                "warp": "bicubic_border_align_corners_false_FP32",
                "reconstruction_warp": "single_original_PAN_warp_by_b_plus_delta",
                "spatial_loss": "shared_matcher_Scharr_abs_XY_RGB_mean_first_L1",
                "roi_margin": self.roi_margin, "boundary_policy": self.boundary_policy,
                "rec_reduction": "all_pixels_bands_mean",
                "loss_coefficients": [1., self.lambda_abs, self.lambda_eps],
                "eps": {"radius": 2., "cadence": "every_RR_update", "distribution": "area_uniform_disk",
                        "generator": "dedicated_CPU", "native_target": "stop_gradient",
                        "shifted_matcher": "recomputed_from_shifted_PAN_same_LMS",
                        "score_support": self.matcher.spec.manifest()["support_mask"],
                        "history": "initial_setting_not_train_only_justified"},
                "inference_units": "current_PAN_px_no_x4_RR_or_FR"}

    def get_extra_state(self):
        return {"method": METHOD_VERSION, "pipeline": self.pipeline_manifest(),
                "sources": asdict(self.sources), "epsilon_seed": self.epsilon_seed,
                "epsilon_rng": self._epsilon_rng.get_state().clone(), "forward_calls": self.forward_calls}

    def _check_checkpoint(self, state):
        if not isinstance(state, dict):
            raise ValueError("Full Edge Stage1 checkpoint metadata required")
        expected = self.get_extra_state()
        for key in ("method", "pipeline", "sources"):
            if state.get(key) != expected[key]:
                raise ValueError(f"Edge Stage1 checkpoint {key} mismatch")
        if (type(state.get("epsilon_seed")) is not int or type(state.get("forward_calls")) is not int
                or state["forward_calls"] < 0 or not isinstance(state.get("epsilon_rng"), torch.Tensor)):
            raise ValueError("Valid Edge Stage1 RNG/progress metadata required")
        # Validate RNG on a disposable generator before mutating the live state.
        torch.Generator(device="cpu").set_state(state["epsilon_rng"].cpu())

    def set_extra_state(self, state):
        self._check_checkpoint(state)
        self.epsilon_seed, self.forward_calls = state["epsilon_seed"], state["forward_calls"]
        self._epsilon_rng.set_state(state["epsilon_rng"].cpu())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._check_checkpoint(state_dict.get(prefix + "_extra_state"))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict:
            raise ValueError("Edge Stage1 method checkpoints require strict loading")
        self._check_checkpoint(state_dict.get("_extra_state"))
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def _check_observation(self, batch):
        batch.validate(self.teacher.spec.bands)
        if (batch.lineage.sensor != self.sources.sensor
                or batch.lineage.resolution not in ("RR", "FR")
                or batch.lineage.grid_name != "PAN_" + batch.lineage.resolution):
            raise ValueError("Sensor and current PAN pixel grid must match the declared observations")
        if batch.lineage.resolution == "FR" and batch.gt is not None:
            raise ValueError("FR observations have no HRMS GT")

    def predict_shift(self, pan, native_lms, *, resolution, pan_valid_mask=None,
                      return_aligner_details=False):
        with torch.autocast(device_type=pan.device.type, enabled=False):
            with torch.no_grad():
                match = self.matcher(pan.float(), native_lms.float(), resolution=resolution,
                                     pan_valid_mask=pan_valid_mask)
                b = match.b.detach()
                if (b.shape != (len(pan), 2) or not bool(torch.isfinite(b).all())
                        or not torch.equal(b, b.round())):
                    raise ValueError("EdgeMatch must return one finite integer (dy,dx) per input")
                pre = warp_pan(pan.float(), b)
            if return_aligner_details:
                delta, details = self.aligner(pre, native_lms.float(), return_details=True)
            else:
                delta, details = self.aligner(pre, native_lms.float()), None
            u = b + delta
            if not bool(torch.isfinite(u).all()):
                raise FloatingPointError("Nonfinite total PAN correction")
        return {"u": u, "correction": u, "b": b, "delta": delta, "pan_pre": pre,
                "match_diagnostics": match.diagnostics,
                **({"aligner_details": details} if details is not None else {})}

    def native_prediction(self, batch, *, return_aligner_details=False):
        self._check_observation(batch)
        training_rr = batch.lineage.resolution == "RR" and batch.lineage.split == "train"
        # FR/held-out observations are inference-only even if the caller forgot
        # an outer no_grad scope. Native RR training still differentiates u0.
        with torch.set_grad_enabled(torch.is_grad_enabled() and training_rr), \
                torch.autocast(device_type=batch.pan.device.type, enabled=False):
            shift = self.predict_shift(batch.pan, batch.native_lms, resolution=batch.lineage.resolution,
                                       return_aligner_details=return_aligner_details)
            corrected = warp_pan(batch.pan.float(), shift["u"])
            predicted = self.teacher(corrected, batch.ms, batch.native_lms)
        return {**predicted, **shift}

    def _native_support(self, pan, correction):
        support = warp_support_mask(*pan.shape[-2:], correction.detach())
        h, w = pan.shape[-2:]
        if 2 * self.roi_margin >= min(h, w):
            raise ValueError("Fixed Scharr loss ROI is empty")
        # Scharr reads a one-pixel neighborhood. A report does not silently
        # introduce a shift-dependent loss mask or change its denominator.
        margin = max(self.roi_margin - 1, 0)
        good = support[..., margin:h-margin, margin:w-margin].flatten(1).all(1)
        if self.boundary_policy == "require_spatial_support" and not bool(good.all()):
            raise ValueError("Fixed Scharr loss ROI/stencil exceeds valid original-PAN warp support")
        return {"warp_support_fraction": support.float().mean(), "spatial_support_ok": good}

    def forward(self, rr):
        rr.validate(self.teacher.spec.bands, training=True, require_gt=True)
        self.sources.check(rr)
        if not all(p.requires_grad for m in (self.aligner, self.teacher) for p in m.parameters()):
            raise ValueError("Stage1 requires trainable residual Aligner and Teacher")
        before_rng = self._epsilon_rng.get_state().clone()
        try:
            with torch.autocast(device_type=rr.pan.device.type, enabled=False):
                native = self.native_prediction(rr)
                rec = (native["y"] - rr.gt.detach().float()).abs().mean()
                absolute = self.matcher.edges.absolute_loss(native["pan_aligned"], rr.gt.detach().float(),
                                                             roi_margin=self.roi_margin)
                support = self._native_support(rr.pan, native["u"])
                with torch.no_grad():
                    eps = sample_offsets(len(rr.pan), 2., self._epsilon_rng).to(rr.pan.device)
                    pan_eps = warp_pan(rr.pan.float(), eps)
                    eps_support = warp_support_mask(*rr.pan.shape[-2:], eps)
                shifted = self.predict_shift(pan_eps, rr.native_lms, resolution="RR",
                                             pan_valid_mask=eps_support)
                eps_rr = epsilon_loss(shifted["u"], native["u"], eps)
                loss = rec + self.lambda_abs * absolute + self.lambda_eps * eps_rr
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite Edge Stage1 objective")
                pre_support = warp_support_mask(*rr.pan.shape[-2:], native["b"])
                eps_pre_support = two_stage_support_mask(*rr.pan.shape[-2:], eps, shifted["b"])
                view = self.aligner.view_margin
                diagnostics = {"native_prewarp_view_support_fraction": _view_fraction(pre_support, view),
                    "epsilon_prewarp_view_support_fraction": _view_fraction(eps_pre_support, view),
                    "epsilon_rr_view_support_fraction": _view_fraction(eps_support, view)}
        except Exception:
            self._epsilon_rng.set_state(before_rng)
            raise
        call = self.forward_calls
        self.forward_calls += 1
        return {"loss": loss, "rec": rec, "abs_rr": absolute, "eps_rr": eps_rr,
                "weighted_abs": self.lambda_abs * absolute,
                "weighted_eps_rr": self.lambda_eps * eps_rr, "native": native, "shifted": shifted,
                "epsilon_rr": eps, "forward_index": call, **support, **diagnostics}
