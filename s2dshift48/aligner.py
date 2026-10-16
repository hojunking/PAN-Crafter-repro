"""Isolated S2D4 global-translation architecture; never replaces DirectShift.

S2D rearranges both cropped, channel-normalized intensity inputs before their
stems.  It is not a flow decoder, does not warp anything, and does not change the
current-PAN-pixel (dy, dx) output convention.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from directshift48.method import (
    ABSOLUTE_ROI_MARGIN,
    ALIGNER_VIEW_MARGIN,
    EPSILON_RADIUS,
    DirectPANAligner,
)
from model.pancrafter_paper import ResBlock
from pa.aligner import _conv, znorm


METHOD_VERSION = "S1-S2DGlobalShift-Scharr-EpsilonR3-v1.0"
S2D_FACTOR = 4


def boundary_manifest():
    """Local convolution RF is 44; normalization still has global support."""
    return {
        "epsilon_max_radius_pan_px": EPSILON_RADIUS,
        "bicubic_conservative_tap_margin": 2,
        "minimum_clean_input_margin": 5,
        "joint_output_stride": 4,
        "aligner_view_margin": ALIGNER_VIEW_MARGIN,
        "aligner_view_rule": "4*ceil((3+2)/4)=8_before_any_spatial_statistics",
        "local_convolution_receptive_field": 44,
        "local_convolution_rf_radius": 21.5,
        "local_rf_progression": [4, 12, 20, 28, 36, 44],
        "local_rf_progression_layers": [
            "S2D4", "stem3x3", "joint1_3x3", "joint2_3x3",
            "ResBlock_in_3x3", "ResBlock_out_3x3",
        ],
        "rf_limitation": "znorm_and_GroupNorm_have_global_spatial_support",
        "support_strategy": "remove_all_synthetic_warp_extrapolation_before_znorm_stems_GN",
        "cnn_padding": "unchanged_reflection_stems_joints_and_zero_common_ResBlock",
        "pooling": "GAP_all_features_of_same_clean_native_epsilon_LMS_view",
        "same_view_RR_FR": True,
        "absolute_roi_margin": ABSOLUTE_ROI_MARGIN,
        "absolute_stencil_support_margin": ABSOLUTE_ROI_MARGIN - 1,
        "absolute_denominator": "fixed_pixels_times_2_directions_times_batch",
        "prediction_dependent_mask": False,
        "invalid_support_action": "raise_before_optimizer_update_no_clamp",
    }


class S2DGlobalPANAligner(DirectPANAligner):
    """Two pre-stem S2D4 paths and a zero-initialized global (dy, dx) head.

    The explicit subclass permits existing Stage 1 type guards.  Calling only
    nn.Module.__init__ avoids constructing/discarding the old architecture and
    consuming its random initialization stream.  Its checkpoint identity is
    nevertheless distinct and is never compatible with DirectShift weights.
    """

    CHECKPOINT_REVISION = 4001

    def __init__(self, ms_bands=8, *, view_margin=ALIGNER_VIEW_MARGIN):
        nn.Module.__init__(self)
        if type(ms_bands) is not int or ms_bands not in (4, 8):
            raise ValueError("Explicit integer 4/8 MS bands required")
        if type(view_margin) is not int or view_margin != ALIGNER_VIEW_MARGIN:
            raise ValueError("S2D global-shift clean input view margin must be 8")
        self.ms_bands, self.view_margin = ms_bands, view_margin
        self.pan_stem = _conv(16, 16, 1, 4)
        self.ms_stem = _conv(16 * ms_bands, 16, 1, 4)
        self.joint1 = _conv(32, 32, 1, 8)
        self.joint2 = _conv(32, 64, 1, 8)
        self.res = ResBlock(64, out_channels=64, norm="ln", dropout=0.,
                            mode_mod=False, se=False)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.register_buffer("_direct_shift_revision", torch.tensor(self.CHECKPOINT_REVISION))
        self.register_buffer("_direct_shift_identity", self._identity())

    def manifest(self):
        return {
            "method": METHOD_VERSION,
            "revision": self.CHECKPOINT_REVISION,
            "architecture": "S2DGlobalPANAligner",
            "ms_bands": self.ms_bands,
            "view_margin": self.view_margin,
            "input": ["original_intensity_PAN", "provided_stored_LMS"],
            "cnn_input_normalization": "sample_channel_spatial_znorm_detached_stats_after_crop",
            "input_order": ["crop8", "original_channel_znorm", "S2D4", "stem"],
            "s2d": {
                "operator": "torch.nn.functional.pixel_unshuffle",
                "factor": S2D_FACTOR,
                "position": "before_stem",
                "modalities": ["PAN", "provided_stored_LMS"],
                "normalized_before_S2D": True,
                "phase_order": "channel_then_row_offset_then_column_offset",
                "trainable": False,
                "PAN_channels": [1, 16],
                "LMS_channels": [self.ms_bands, 16 * self.ms_bands],
            },
            "stem_dims": [[16, 16], [16 * self.ms_bands, 16]],
            "stem_stride": [1, 1],
            "joint_dims": [[32, 32], [32, 64]],
            "joint_stride": [1, 1],
            "stems_joint_norm": "GN4_GN8",
            "resblock": "common_UNet_preLN_dropout0",
            "head_dims": [64, 32, 2],
            "output": "total_dy_dx_current_PAN_px",
            "output_clamp": "none",
            "fc2_initialization": "weight_bias_zero",
            "boundary": boundary_manifest(),
        }

    def _preflight_state(self, state_dict, prefix=""):
        """Reject incompatible identities and invalid tensors before any copy."""
        if not isinstance(state_dict, Mapping):
            raise RuntimeError("S2D checkpoint state mapping required")
        for name, expected in (
            ("_direct_shift_revision", torch.tensor(self.CHECKPOINT_REVISION)),
            ("_direct_shift_identity", self._identity()),
        ):
            value = state_dict.get(prefix + name)
            if (not isinstance(value, torch.Tensor) or value.dtype != expected.dtype
                    or value.shape != expected.shape or not torch.equal(value.cpu(), expected)):
                raise RuntimeError("S2D checkpoint identity mismatch; old/new architectures cannot cross-load")
        expected_state = self.state_dict()
        actual_keys = {key for key in state_dict if key.startswith(prefix)}
        expected_keys = {prefix + key for key in expected_state}
        if actual_keys != expected_keys:
            raise RuntimeError("S2D checkpoint requires the complete exact architecture state")
        for name, expected in expected_state.items():
            value = state_dict[prefix + name]
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                    or value.dtype != expected.dtype):
                raise RuntimeError("S2D checkpoint tensor shape/dtype mismatch: " + name)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise RuntimeError("S2D checkpoint nonfinite tensor: " + name)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._preflight_state(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict:
            raise ValueError("S2D checkpoints require strict loading")
        self._preflight_state(state_dict)
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    @staticmethod
    def _feature(value, name, *, device):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.device != device or not bool(torch.isfinite(value).all())):
            raise FloatingPointError("S2D intermediate must be finite FP32 on input device: " + name)
        return value

    def forward(self, pan, native_lms, *, return_details=False):
        if (not isinstance(pan, torch.Tensor) or pan.ndim != 4
                or pan.shape[0] < 1 or pan.shape[1] != 1):
            raise ValueError("Nonempty [B,1,H,W] original PAN required")
        if (not isinstance(native_lms, torch.Tensor)
                or native_lms.shape != (len(pan), self.ms_bands, *pan.shape[-2:])):
            raise ValueError("Provided LMS must share the original PAN grid")
        if (pan.dtype != torch.float32 or native_lms.dtype != pan.dtype
                or pan.device != native_lms.device or not bool(torch.isfinite(pan).all())
                or not bool(torch.isfinite(native_lms).all())):
            raise ValueError("Finite FP32 PAN/LMS on one device required")
        if min(pan.shape[-2:]) < 24 or any(size % S2D_FACTOR for size in pan.shape[-2:]):
            raise ValueError("S2D PAN height/width must be multiples of 4 and at least 24")
        if any(param.dtype != torch.float32 or param.device != pan.device
               for param in self.parameters()):
            raise ValueError("S2D model parameters must be FP32 on input device")
        check = lambda value, name: self._feature(value, name, device=pan.device)
        margin = self.view_margin
        with torch.autocast(device_type=pan.device.type, enabled=False):
            p = check(znorm(pan[..., margin:-margin, margin:-margin]), "PAN_znorm")
            m = check(znorm(native_lms[..., margin:-margin, margin:-margin]), "LMS_znorm")
            p = check(F.pixel_unshuffle(p, S2D_FACTOR), "PAN_S2D4")
            m = check(F.pixel_unshuffle(m, S2D_FACTOR), "LMS_S2D4")
            p = check(self.pan_stem(p), "PAN_stem")
            m = check(self.ms_stem(m), "LMS_stem")
            joint = check(torch.cat((p, m), dim=1), "concat")
            joint = check(self.joint1(joint), "joint1")
            joint = check(self.joint2(joint), "joint2")
            joint = check(self.res(joint, None), "resblock")
            features = check(joint.mean(dim=(2, 3)), "GAP")
            hidden = check(F.silu(self.fc1(features)), "head_hidden")
            correction = check(self.fc2(hidden), "dy_dx")
        if return_details:
            return correction, {"features": features, "head_input": features, "head_hidden": hidden}
        return correction
