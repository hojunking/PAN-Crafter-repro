"""Prior-free direct PAN translation, Scharr and RR-only R3 consistency.

This module deliberately has no dependency on a matching/prior implementation.
The existing observation adapter and reconstruction backbone remain reusable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.pancrafter_paper import ResBlock
from pa.aligner import _conv, znorm
from pa.warp import warp_pan, warp_support_mask
from panda_method.reconstruction import Reconstructor


METHOD_VERSION = "S1-DirectShift-Scharr-EpsilonR3-v1.0"
METHOD_REVISION = "DirectShiftR3-S1-v1"
SENSOR_BANDS = {"WV3": 8, "QB": 4, "GF2": 4}
RGB_INDICES = {"WV3": (4, 2, 1), "QB": (2, 1, 0), "GF2": (2, 1, 0)}
EPSILON_RADIUS = 3.0
ALIGNER_VIEW_MARGIN = 8
ABSOLUTE_ROI_MARGIN = 8


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def boundary_manifest():
    return {"epsilon_max_radius_pan_px": EPSILON_RADIUS,
            "bicubic_conservative_tap_margin": 2,
            "minimum_clean_input_margin": 5,
            "joint_output_stride": 4, "aligner_view_margin": ALIGNER_VIEW_MARGIN,
            "aligner_view_rule": "4*ceil((3+2)/4)=8_before_any_spatial_statistics",
            "local_convolution_receptive_field": 25,
            "local_convolution_rf_radius": 12,
            "local_rf_progression": [3, 5, 9, 17, 25],
            "rf_limitation": "znorm_and_GroupNorm_have_global_spatial_support",
            "support_strategy": "remove_all_synthetic_warp_extrapolation_before_znorm_stems_GN",
            "cnn_padding": "unchanged_reflection_stems_joints_and_zero_common_ResBlock",
            "pooling": "GAP_all_features_of_same_clean_native_epsilon_LMS_view",
            "same_view_RR_FR": True,
            "absolute_roi_margin": ABSOLUTE_ROI_MARGIN,
            "absolute_stencil_support_margin": ABSOLUTE_ROI_MARGIN - 1,
            "absolute_denominator": "fixed_pixels_times_2_directions_times_batch",
            "prediction_dependent_mask": False,
            "invalid_support_action": "raise_before_optimizer_update_no_clamp"}


@dataclass(frozen=True)
class DirectTrainingSources:
    sensor: str
    rr_manifest_sha256: str

    def __post_init__(self):
        if self.sensor not in SENSOR_BANDS:
            raise ValueError("Explicit WV3/QB/GF2 sensor required")
        sha = self.rr_manifest_sha256
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError("Pinned RR source SHA256 required")

    def check(self, batch):
        expected = {"split": "train", "resolution": "RR", "sensor": self.sensor,
                    "grid_name": "PAN_RR", "source_manifest_sha256": self.rr_manifest_sha256,
                    "reference": "PROVIDED_STORED_LMS"}
        if any(getattr(batch.lineage, key, None) != value for key, value in expected.items()):
            raise ValueError("Only pinned RR train observations and provided LMS may update Stage 1")


class DirectPANAligner(nn.Module):
    """Raw-intensity dual stem, common LN ResBlock and zero-init total shift."""
    CHECKPOINT_REVISION = 3001

    def __init__(self, ms_bands=8, *, view_margin=ALIGNER_VIEW_MARGIN):
        super().__init__()
        if ms_bands not in (4, 8):
            raise ValueError("Explicit 4/8 MS bands required")
        if type(view_margin) is not int or view_margin != ALIGNER_VIEW_MARGIN:
            raise ValueError("Direct R3 fixed clean input view margin must be 8")
        self.ms_bands, self.view_margin = ms_bands, view_margin
        self.pan_stem = _conv(1, 16, 1, 4)
        self.ms_stem = _conv(ms_bands, 16, 1, 4)
        self.joint1 = _conv(32, 32, 2, 8)
        self.joint2 = _conv(32, 64, 2, 8)
        self.res = ResBlock(64, out_channels=64, norm="ln", dropout=0., mode_mod=False, se=False)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.register_buffer("_direct_shift_revision", torch.tensor(self.CHECKPOINT_REVISION))
        self.register_buffer("_direct_shift_identity", self._identity())

    def manifest(self):
        return {"method": METHOD_VERSION, "revision": self.CHECKPOINT_REVISION,
                "ms_bands": self.ms_bands, "view_margin": self.view_margin,
                "input": ["original_intensity_PAN", "provided_stored_LMS"],
                "cnn_input_normalization": "sample_channel_spatial_znorm_detached_stats_after_crop",
                "stems_joint_norm": "GN4_GN8", "resblock": "common_UNet_preLN_dropout0",
                "head_dims": [64, 32, 2], "output": "total_dy_dx_current_PAN_px",
                "output_clamp": "none", "fc2_initialization": "weight_bias_zero",
                "boundary": boundary_manifest()}

    def _identity(self):
        return torch.tensor(list(bytes.fromhex(_digest(self.manifest()))), dtype=torch.uint8)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name, expected in (("_direct_shift_revision", torch.tensor(self.CHECKPOINT_REVISION)),
                               ("_direct_shift_identity", self._identity())):
            value = state_dict.get(prefix + name)
            if (not isinstance(value, torch.Tensor) or value.dtype != expected.dtype
                    or value.shape != expected.shape or not torch.equal(value.cpu(), expected)):
                raise RuntimeError("Direct-shift checkpoint identity mismatch; legacy checkpoints forbidden")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, pan, native_lms, *, return_details=False):
        if pan.ndim != 4 or pan.shape[0] < 1 or pan.shape[1] != 1:
            raise ValueError("Nonempty [B,1,H,W] original PAN required")
        if native_lms.shape != (len(pan), self.ms_bands, *pan.shape[-2:]):
            raise ValueError("Provided LMS must share the original PAN grid")
        if (pan.dtype != torch.float32 or native_lms.dtype != pan.dtype
                or pan.device != native_lms.device or not bool(torch.isfinite(pan).all())
                or not bool(torch.isfinite(native_lms).all())):
            raise ValueError("Finite FP32 PAN/LMS on one device required")
        margin = self.view_margin
        if min(pan.shape[-2:]) - 2 * margin < 8:
            raise ValueError("CNN clean view is too small for two strided reflection convolutions")
        # This MUST precede znorm and all GroupNorms. An output-only receptive-
        # field mask cannot erase border information already in spatial moments.
        p_view = pan[..., margin:-margin, margin:-margin]
        m_view = native_lms[..., margin:-margin, margin:-margin]
        with torch.autocast(device_type=pan.device.type, enabled=False):
            p, m = self.pan_stem(znorm(p_view)), self.ms_stem(znorm(m_view))
            joint = self.joint2(self.joint1(torch.cat((p, m), dim=1)))
            features = self.res(joint, None).mean(dim=(2, 3))
            hidden = F.silu(self.fc1(features))
            correction = self.fc2(hidden)
        if not bool(torch.isfinite(correction).all()):
            raise FloatingPointError("Nonfinite direct PAN correction")
        if return_details:
            return correction, {"features": features, "head_input": features, "head_hidden": hidden}
        return correction


class ScharrAbsolute(nn.Module):
    """RGB mean first; two absolute Scharr axes, fixed 1/32, no normalization."""
    def __init__(self, sensor):
        super().__init__()
        if sensor not in SENSOR_BANDS:
            raise ValueError("Explicit sensor required")
        self.sensor, self.bands = sensor, SENSOR_BANDS[sensor]
        self.rgb_indices = RGB_INDICES[sensor]

    @staticmethod
    def _check(image, channels):
        if (image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != channels
                or min(image.shape[-2:]) < 3 or image.dtype not in (torch.float32, torch.float64)
                or not bool(torch.isfinite(image).all())):
            raise ValueError("Finite FP32/FP64 image and explicit channels required")

    def descriptor(self, gray):
        self._check(gray, 1)
        with torch.autocast(device_type=gray.device.type, enabled=False):
            sx = gray.new_tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 32
            return F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"),
                            torch.stack((sx, sx.T)).unsqueeze(1)).abs()

    def rgb_luminance(self, image):
        self._check(image, self.bands)
        return image[:, self.rgb_indices].mean(dim=1, keepdim=True)

    def absolute_loss(self, aligned_pan, gt, *, roi_margin=ABSOLUTE_ROI_MARGIN):
        self._check(aligned_pan, 1)
        self._check(gt, self.bands)
        if (aligned_pan.shape[0] != gt.shape[0] or aligned_pan.shape[-2:] != gt.shape[-2:]
                or aligned_pan.dtype != gt.dtype or aligned_pan.device != gt.device):
            raise ValueError("PAN and GT must share batch/grid/dtype/device")
        if type(roi_margin) is not int or roi_margin < 1 or min(gt.shape[-2:]) <= 2 * roi_margin:
            raise ValueError("Nonempty explicit fixed Scharr interior required")
        delta = self.descriptor(aligned_pan) - self.descriptor(self.rgb_luminance(gt))
        return delta[..., roi_margin:-roi_margin, roi_margin:-roi_margin].abs().mean()

    def manifest(self):
        return {"operator": "Scharr_absolute_xy", "sensor": self.sensor,
                "rgb_indices_R_G_B": list(self.rgb_indices), "rgb_mean_before_gradient": True,
                "kernel_x_numerator": [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                "kernel_y": "transpose_x", "kernel_denominator": 32,
                "padding": "reflect1", "normalization": "none", "reduction": "mean_two_axes"}


def sample_epsilon(batch, generator):
    if type(batch) is not int or batch < 1 or generator.device.type != "cpu":
        raise ValueError("Positive batch and dedicated CPU generator required")
    draws = torch.rand(batch, 2, generator=generator, dtype=torch.float32)
    radius, angle = EPSILON_RADIUS * draws[:, 0].sqrt(), 2 * math.pi * draws[:, 1]
    return torch.stack((radius * angle.sin(), radius * angle.cos()), dim=1)


def epsilon_loss(shifted_u, native_u, epsilon):
    if shifted_u.shape != native_u.shape or epsilon.shape != native_u.shape:
        raise ValueError("Matching [B,2] correction and epsilon shapes required")
    return (shifted_u + epsilon - native_u.detach()).abs().mean()


class DirectShiftStage1(nn.Module):
    """One native U forward and two A forwards; no evaluation-set training."""
    def __init__(self, aligner, teacher, *, sources, epsilon_seed,
                 roi_margin=ABSOLUTE_ROI_MARGIN, boundary_policy="require_spatial_support",
                 lambda_abs=1., lambda_eps=.001):
        super().__init__()
        if (not isinstance(aligner, DirectPANAligner) or not isinstance(teacher, Reconstructor)
                or not isinstance(sources, DirectTrainingSources)):
            raise TypeError("Direct Aligner, unchanged Reconstructor and pinned sources required")
        if teacher.spec.bands != SENSOR_BANDS[sources.sensor] or aligner.ms_bands != teacher.spec.bands:
            raise ValueError("Sensor/architecture band identities disagree")
        if teacher.spec.input_layout != "P0" or teacher.spec.base_reference != "BICUBIC_MS":
            raise ValueError("Preserve P0/BICUBIC_MS Teacher reconstruction")
        if type(roi_margin) is not int or roi_margin != ABSOLUTE_ROI_MARGIN:
            raise ValueError("Direct method requires the fixed Scharr ROI margin8")
        if boundary_policy != "require_spatial_support":
            raise ValueError("Direct method forbids silently training on invalid Scharr support")
        if {id(p) for p in aligner.parameters()} & {id(p) for p in teacher.parameters()}:
            raise ValueError("Aligner and Teacher parameters must be disjoint")
        for name, value in (("lambda_abs", lambda_abs), ("lambda_eps", lambda_eps)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be a finite positive coefficient")
        if type(epsilon_seed) is not int:
            raise ValueError("Explicit integer epsilon seed required")
        self.aligner, self.teacher = aligner, teacher
        self.spatial = ScharrAbsolute(sources.sensor)
        self.rgb_indices = self.spatial.rgb_indices
        self.sources, self.roi_margin, self.boundary_policy = sources, roi_margin, boundary_policy
        self.lambda_abs, self.lambda_eps = float(lambda_abs), float(lambda_eps)
        self.epsilon_seed = epsilon_seed
        self._epsilon_rng = torch.Generator(device="cpu").manual_seed(epsilon_seed)
        self.forward_calls = 0

    def alignment_manifest(self):
        """Evaluation identity excludes optimizer/seed/Teacher capacity choices."""
        return {"method": METHOD_VERSION,
                "prediction": "prior_free_direct_total_dy_dx",
                "aligner_input": ["original_intensity_PAN", "provided_stored_LMS"],
                "head_dims": [64, 32, 2], "aligner_resblock": "common_UNet_preLN_dropout0",
                "native_warp": "single_original_PAN_bicubic_border_align_corners_false_FP32",
                "units": "current_PAN_px_no_x4_RR_or_FR",
                "spatial": self.spatial.manifest(), "boundary": boundary_manifest(),
                "epsilon": {"radius": 3., "distribution": "area_uniform_disk_r_equals_R_sqrt_uniform",
                            "PAN_only": True, "native_anchor": "stop_gradient", "train_split": "RR_only"}}

    def pipeline_manifest(self):
        return {"method": METHOD_VERSION, "revision": METHOD_REVISION,
                "alignment": "prior_free_direct_total_global_translation",
                "aligner": self.aligner.manifest(), "teacher": self.teacher.spec.manifest(),
                "training_resolutions": ["RR"], "lms_reference": "provided_stored_LMS_not_bicubic_base",
                "warp": "bicubic_border_align_corners_false_FP32",
                "reconstruction_warp": "single_original_PAN_warp_by_u",
                "spatial": self.spatial.manifest(), "boundary": boundary_manifest(),
                "roi_margin": self.roi_margin, "boundary_policy": self.boundary_policy,
                "rec_reduction": "all_pixels_bands_mean", "loss_coefficients": [1., self.lambda_abs, self.lambda_eps],
                "eps": {"radius": EPSILON_RADIUS, "cadence": "every_RR_update",
                        "distribution": "area_uniform_disk_r_equals_R_sqrt_uniform",
                        "generator": "dedicated_CPU", "native_target": "stop_gradient",
                        "PAN_only": True, "historical_radius_basis": "not_claimed_train_only",
                        "shifted_teacher_forward": False},
                "inference_units": "current_PAN_px_no_x4_RR_or_FR"}

    def get_extra_state(self):
        return {"method": METHOD_VERSION, "pipeline": self.pipeline_manifest(), "sources": asdict(self.sources),
                "epsilon_seed": self.epsilon_seed, "epsilon_rng": self._epsilon_rng.get_state().clone(),
                "forward_calls": self.forward_calls}

    def _check_checkpoint(self, state):
        if not isinstance(state, dict):
            raise ValueError("Full direct-shift checkpoint metadata required")
        expected = self.get_extra_state()
        for key in ("method", "pipeline", "sources"):
            if state.get(key) != expected[key]:
                raise ValueError("Direct-shift checkpoint " + key + " mismatch")
        if (type(state.get("epsilon_seed")) is not int or type(state.get("forward_calls")) is not int
                or state["forward_calls"] < 0 or not isinstance(state.get("epsilon_rng"), torch.Tensor)):
            raise ValueError("Valid epsilon RNG/progress metadata required")
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
            raise ValueError("Direct-shift checkpoints require strict loading")
        self._check_checkpoint(state_dict.get("_extra_state"))
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def _check_observation(self, batch):
        batch.validate(self.teacher.spec.bands)
        if (batch.lineage.sensor != self.sources.sensor or batch.lineage.resolution not in ("RR", "FR")
                or batch.lineage.grid_name != "PAN_" + batch.lineage.resolution
                or batch.lineage.reference != "PROVIDED_STORED_LMS"):
            raise ValueError("Pinned sensor and stored LMS on current RR/FR PAN grid required")

    def predict_shift(self, pan, native_lms, *, resolution, pan_valid_mask=None,
                      return_aligner_details=False):
        if resolution not in ("RR", "FR"):
            raise ValueError("Current RR/FR PAN pixel grid required")
        margin = self.aligner.view_margin
        if pan_valid_mask is not None:
            if (pan_valid_mask.dtype != torch.bool or pan_valid_mask.shape != pan.shape
                    or pan_valid_mask.device != pan.device
                    or not bool(pan_valid_mask[..., margin:-margin, margin:-margin].all())):
                raise ValueError("Synthetic PAN support does not contain the fixed clean Aligner view")
        with torch.autocast(device_type=pan.device.type, enabled=False):
            result = self.aligner(pan.float(), native_lms.float(), return_details=return_aligner_details)
        u, details = result if return_aligner_details else (result, None)
        return {"u": u, "correction": u,
                **({"aligner_details": details} if details is not None else {})}

    def native_prediction(self, batch, *, return_aligner_details=False):
        self._check_observation(batch)
        training_rr = batch.lineage.resolution == "RR" and batch.lineage.split == "train"
        with torch.set_grad_enabled(torch.is_grad_enabled() and training_rr), \
                torch.autocast(device_type=batch.pan.device.type, enabled=False):
            shift = self.predict_shift(batch.pan, batch.native_lms, resolution=batch.lineage.resolution,
                                       return_aligner_details=return_aligner_details)
            corrected = warp_pan(batch.pan.float(), shift["u"])
            predicted = self.teacher(corrected, batch.ms, batch.native_lms)
        return {**predicted, **shift}

    def _native_support(self, pan, correction):
        support = warp_support_mask(*pan.shape[-2:], correction.detach())
        margin = self.roi_margin - 1  # Include the entire Scharr stencil.
        if min(pan.shape[-2:]) <= 2 * self.roi_margin:
            raise ValueError("Fixed Scharr loss ROI is empty")
        good = support[..., margin:-margin, margin:-margin].flatten(1).all(1)
        if not bool(good.all()):
            evidence = {"correction_dy_dx": correction.detach().cpu().tolist(),
                        "support_fraction_per_sample": support.float().flatten(1).mean(1).cpu().tolist(),
                        "fixed_roi_margin": self.roi_margin, "valid_per_sample": good.cpu().tolist()}
            raise ValueError("SUPPORT_VIOLATION: fixed Scharr ROI/stencil reads extrapolated PAN; "
                             "no optimizer update or adaptive loss mask is permitted; " +
                             json.dumps(evidence, sort_keys=True, allow_nan=False))
        return {"warp_support_fraction": support.float().mean(), "spatial_support_ok": good}

    def forward(self, rr):
        rr.validate(self.teacher.spec.bands, training=True, require_gt=True)
        self.sources.check(rr)
        if not all(p.requires_grad for m in (self.aligner, self.teacher) for p in m.parameters()):
            raise ValueError("Stage1 jointly trains every Aligner and Teacher parameter")
        before_rng = self._epsilon_rng.get_state().clone()
        try:
            with torch.autocast(device_type=rr.pan.device.type, enabled=False):
                native = self.native_prediction(rr)
                support = self._native_support(rr.pan, native["u"])
                rec = (native["y"] - rr.gt.detach().float()).abs().mean()
                absolute = self.spatial.absolute_loss(native["pan_aligned"], rr.gt.detach().float(),
                                                       roi_margin=self.roi_margin)
                with torch.no_grad():
                    eps = sample_epsilon(len(rr.pan), self._epsilon_rng).to(rr.pan.device)
                    pan_eps = warp_pan(rr.pan.float(), eps)
                    eps_support = warp_support_mask(*rr.pan.shape[-2:], eps)
                shifted = self.predict_shift(pan_eps, rr.native_lms, resolution="RR", pan_valid_mask=eps_support)
                eps_rr = epsilon_loss(shifted["u"], native["u"], eps)
                loss = rec + self.lambda_abs * absolute + self.lambda_eps * eps_rr
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite direct-shift objective")
                margin = self.aligner.view_margin
                fraction = eps_support[..., margin:-margin, margin:-margin].float().mean()
        except Exception:
            self._epsilon_rng.set_state(before_rng)
            raise
        call = self.forward_calls
        self.forward_calls += 1
        return {"loss": loss, "rec": rec, "abs_rr": absolute, "eps_rr": eps_rr,
                "weighted_abs": self.lambda_abs * absolute, "weighted_eps_rr": self.lambda_eps * eps_rr,
                "native": native, "shifted": shifted, "epsilon_rr": eps,
                "forward_index": call, "epsilon_rr_view_support_fraction": fraction, **support}
