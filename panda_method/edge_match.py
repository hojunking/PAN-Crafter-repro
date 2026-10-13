"""Nonlearned, input-specific integer Scharr alignment for EdgeGuided Stage 1.

This is deliberately NOT the legacy NCC/Census estimator, sensor-global prior,
Scharr magnitude, bandwise edge averaging, or subpixel fitting. All policies
whose values are open in the handoff are required caller arguments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


SENSOR_BANDS = {"WV3": 8, "QB": 4, "GF2": 4}
RGB_INDICES = {"WV3": (4, 2, 1), "QB": (2, 1, 0), "GF2": (2, 1, 0)}
SCHARR_X = ((-3 / 32, 0., 3 / 32), (-10 / 32, 0., 10 / 32), (-3 / 32, 0., 3 / 32))


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _image(value, *, channels, name):
    if (not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] < 1
            or value.shape[1] != channels or min(value.shape[-2:]) < 2):
        raise ValueError(f"{name} must be nonempty [B,{channels},H,W] with H,W >= 2")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} requires explicit FP32/FP64; no silent half-precision edge matching")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " contains non-finite input")


def _margin(value, height, width):
    if type(value) is not int or value < 0:
        raise ValueError("Explicit nonnegative integer ROI margin is required")
    if min(height, width) <= 2 * value:
        raise ValueError("No valid common interior remains for this ROI margin")
    return value


def _crop(value, margin):
    if margin == 0:
        return value
    return value[..., margin:-margin, margin:-margin]


class ScharrEdges(nn.Module):
    """One fixed RGB-first absolute XY operator for matching AND L_abs.

    No buffers/parameters can silently replace the fixed kernels on checkpoint
    load. Input DN/intensity scaling is the caller's existing loader contract;
    this module adds no data-dependent normalization. FP32/FP64 are preserved
    and autocast is explicitly disabled around convolution.
    """
    def __init__(self, sensor):
        super().__init__()
        if sensor not in SENSOR_BANDS:
            raise ValueError("ScharrEdges requires one explicit WV3/QB/GF2 sensor")
        self.sensor = sensor
        self.bands = SENSOR_BANDS[sensor]
        self.rgb_indices = RGB_INDICES[sensor]

    def descriptor(self, gray):
        """G(gray) = [abs(Sx * gray), abs(Sy * gray)], reflect-padding=1."""
        _image(gray, channels=1, name="Grayscale/PAN")
        with torch.autocast(device_type=gray.device.type, enabled=False):
            sx = gray.new_tensor(SCHARR_X)
            kernels = torch.stack((sx, sx.T)).unsqueeze(1)
            edges = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernels).abs()
        if not bool(torch.isfinite(edges).all()):
            raise ValueError("Scharr computation produced non-finite edge values")
        return edges

    def rgb_luminance(self, multispectral):
        _image(multispectral, channels=self.bands, name="Multispectral")
        # Intensity average first, then edge extraction. NOT an edge average.
        return multispectral[:, self.rgb_indices].mean(dim=1, keepdim=True)

    def absolute_loss(self, aligned_pan, gt, *, roi_margin):
        """Differentiable same-operator absXY L1; no U-Net involvement.

        ROI is explicit/fixed. This operator does NOT silently invent a
        shift-dependent mask or change reduction weights. The Stage-1 caller
        must separately check final-warp support against its recorded policy.
        """
        _image(aligned_pan, channels=1, name="Aligned PAN")
        _image(gt, channels=self.bands, name="RR GT")
        if (aligned_pan.shape[0] != gt.shape[0] or aligned_pan.shape[-2:] != gt.shape[-2:]
                or aligned_pan.device != gt.device or aligned_pan.dtype != gt.dtype):
            raise ValueError("Aligned PAN and RR GT need matching batch/grid/device/dtype")
        _margin(roi_margin, *aligned_pan.shape[-2:])
        difference = (self.descriptor(aligned_pan)
                      - self.descriptor(self.rgb_luminance(gt))).abs()
        return _crop(difference, roi_margin).mean()

    def forward(self, gray):
        return self.descriptor(gray)

    def manifest(self):
        return {
            "schema": "PANDA_SHARED_ABS_XY_SCHARR_V1", "sensor": self.sensor,
            "rgb_indices_R_G_B": list(self.rgb_indices), "rgb_order": "average_intensities_then_Scharr",
            "kernel_x": [list(row) for row in SCHARR_X], "kernel_y": [list(row) for row in zip(*SCHARR_X)],
            "descriptor_order": ["abs_dx", "abs_dy"], "padding": "reflect_1",
            "normalization": "none", "fixed_kernel_scale": "1/32",
            "dtype": "preserve_explicit_fp32_or_fp64_autocast_disabled",
        }

    def fingerprint(self):
        return _fingerprint(self.manifest())


@dataclass(frozen=True)
class EdgeMatchSpec:
    sensor: str
    radius_rr: int
    radius_fr: int
    tie_policy: str
    low_texture_threshold: float
    low_texture_policy: str
    boundary_policy: str
    support_policy: str = "candidate_intersection_v1"

    def __post_init__(self):
        if self.sensor not in SENSOR_BANDS:
            raise ValueError("Exactly one explicit WV3/QB/GF2 sensor is required")
        for radius in (self.radius_rr, self.radius_fr):
            if type(radius) is not int or radius < 0:
                raise ValueError("RR/FR search radii must be explicit nonnegative PAN-pixel integers")
        if self.tie_policy not in ("error", "min_norm_then_lexicographic", "min_norm_then_l1_lexicographic"):
            raise ValueError("Explicit supported tie policy required")
        threshold = self.low_texture_threshold
        if type(threshold) not in (float, int) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Low-texture threshold must be explicit finite and nonnegative")
        if self.low_texture_policy not in ("error", "zero", "report"):
            raise ValueError("Explicit supported low-texture policy required")
        if self.boundary_policy not in ("error", "report"):
            raise ValueError("Explicit supported boundary policy required")
        if self.support_policy not in ("candidate_intersection_v1", "fixed_radius_plus5_v2"):
            raise ValueError("Explicit supported score-support policy required")
        if self.support_policy == "fixed_radius_plus5_v2" and (
                self.tie_policy != "min_norm_then_l1_lexicographic"
                or self.low_texture_policy != "report" or self.boundary_policy != "report"):
            raise ValueError("Campaign v2 fixes tie ordering and diagnostic-only texture/boundary policies")

    @classmethod
    def campaign_v2(cls, sensor, *, radius=4, low_texture_threshold=0.):
        """Registered proposal, not an accuracy-calibrated radius/confidence cutoff.

        One predeclared current-PAN-pixel radius for RR and FR. Texture labels
        never alter the argmin, weight a loss, or remove an evaluation sample.
        """
        return cls(sensor, radius, radius, "min_norm_then_l1_lexicographic",
                   low_texture_threshold, "report", "report", "fixed_radius_plus5_v2")

    def radius(self, resolution):
        if resolution == "RR":
            return self.radius_rr
        if resolution == "FR":
            return self.radius_fr
        raise ValueError("Resolution must explicitly be RR or FR; no shape-based inference")

    def manifest(self):
        return {
            "schema": "PANDA_EDGE_MATCH_INTEGER_SCHARR_V1", **asdict(self),
            "candidate_grid": "square_integer_dy_dx_inclusive_[-radius,+radius]",
            "candidate_order": "lexicographic_dy_then_dx", "unit": "current_input_PAN_pixels",
            "sampling_equation": "aligned[y,x]=PAN[y+dy,x+dx]",
            "common_interior_margin": ("radius+5_fixed_native_shifted_RR_FR"
                if self.support_policy == "fixed_radius_plus5_v2" else "radius+1_Scharr_support"),
            "score": "mean_over_common_support_and_two_axes_of_raw_absolute_gradient_L1",
            "normalization": "none", "subpixel_fitting": "none",
            "tie_definition": "exact_equal_finite_minimum_no_tolerance",
            "tie_resolution_order": ("squared_L2_then_L1_then_dy_then_dx"
                if self.tie_policy == "min_norm_then_l1_lexicographic" else "squared_L2_then_dy_then_dx"),
            "tie_symmetry_note": "lexicographic_resolution_of_nonunique_ties_is_not_D4_equivariant",
            "low_texture_definition": "PAN_or_LMS_mean_absXY_over_common_support_le_threshold",
            "low_texture_precedence": ("diagnostic_only_no_shift_or_loss_override"
                if self.low_texture_policy == "report" else "explicit_zero_overrides_tie_error_for_that_sample"),
            "boundary_definition": "chosen_argmin_on_square_boundary; radius_zero_is_no_search_not_hit",
            "support_mask": ("known_geometry_mask_asserts_entire_fixed_ROI_valid_for_ALL_candidates_never_changes_ROI"
                if self.support_policy == "fixed_radius_plus5_v2" else
                "optional_known_geometry_mask_3x3_eroded_then_intersection_over_ALL_candidates"),
            "support_weighting": "binary_geometry_only_identical_support_for_every_candidate",
        }

    def fingerprint(self):
        return _fingerprint(self.manifest())


@dataclass(frozen=True)
class EdgeMatchResult:
    b: torch.Tensor
    scores: torch.Tensor
    candidates: torch.Tensor
    tie_count: torch.Tensor
    low_texture: torch.Tensor
    texture_pan: torch.Tensor
    texture_lms: torch.Tensor
    boundary_hit: torch.Tensor
    roi_margin: int
    resolution: str
    common_mask: torch.Tensor
    support_count: torch.Tensor

    @property
    def diagnostics(self):
        sorted_scores = self.scores.sort(dim=1).values
        zero_index = (self.candidates == 0).all(dim=1).nonzero().item()
        return {
            "scores": self.scores, "candidates": self.candidates,
            "tie_count": self.tie_count, "low_texture": self.low_texture,
            "texture_pan": self.texture_pan, "texture_lms": self.texture_lms,
            "boundary_hit": self.boundary_hit, "roi_margin": self.roi_margin,
            "resolution": self.resolution, "common_mask": self.common_mask,
            "support_count": self.support_count,
            "candidate_support_count": self.support_count[:, None].expand_as(self.scores),
            "best_score": sorted_scores[:, 0], "zero_score": self.scores[:, zero_index],
            "second_best_gap": (sorted_scores[:, 1] - sorted_scores[:, 0]
                                if sorted_scores.shape[1] > 1 else None),
            "score_range": sorted_scores[:, -1] - sorted_scores[:, 0],
            "flat_score": sorted_scores[:, -1].eq(sorted_scores[:, 0]),
        }


class EdgeMatcher(nn.Module):
    """Input-specific hard integer translation, intentionally nondifferentiable."""
    def __init__(self, spec, edges=None):
        super().__init__()
        if not isinstance(spec, EdgeMatchSpec):
            raise TypeError("Explicit EdgeMatchSpec required; no matcher defaults")
        if edges is None:
            edges = ScharrEdges(spec.sensor)
        if not isinstance(edges, ScharrEdges) or edges.sensor != spec.sensor:
            raise ValueError("Matching and supervision must share the correct sensor Scharr operator")
        self.spec = spec
        self.edges = edges

    def manifest(self):
        return {"matcher": self.spec.manifest(), "shared_edges": self.edges.manifest()}

    def fingerprint(self):
        return _fingerprint(self.manifest())

    @torch.no_grad()
    def forward(self, pan, lms, *, resolution, pan_valid_mask=None):
        _image(pan, channels=1, name="PAN")
        _image(lms, channels=self.edges.bands, name="Provided LMS")
        if (pan.shape[0] != lms.shape[0] or pan.shape[-2:] != lms.shape[-2:]
                or pan.device != lms.device or pan.dtype != lms.dtype):
            raise ValueError("PAN/LMS batch, grid, dtype and device must match")
        radius = self.spec.radius(resolution)
        height, width = pan.shape[-2:]
        fixed_support = self.spec.support_policy == "fixed_radius_plus5_v2"
        margin = _margin(radius + (5 if fixed_support else 1), height, width)
        offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
        candidates = torch.tensor(offsets, device=pan.device, dtype=torch.long)
        if pan_valid_mask is not None:
            if (not isinstance(pan_valid_mask, torch.Tensor) or pan_valid_mask.dtype != torch.bool
                    or pan_valid_mask.shape != pan.shape or pan_valid_mask.device != pan.device
                    or pan_valid_mask.requires_grad):
                raise ValueError("Known PAN support must be bool[B,1,H,W] on the same device")
            with torch.autocast(device_type=pan.device.type, enabled=False):
                # Zero padding makes exterior source pixels explicitly invalid.
                eroded = F.conv2d(pan_valid_mask.float(), pan.new_ones((1, 1, 3, 3), dtype=torch.float32),
                                 padding=1).eq(9.)
            common = torch.ones_like(_crop(pan_valid_mask, margin))
            for dy, dx in offsets:
                common &= eroded[..., margin + dy:height - margin + dy,
                                 margin + dx:width - margin + dx]
            if fixed_support and not bool(common.all()):
                raise ValueError("FAILED_INPUT_OR_SUPPORT: known padding intersects the fixed score ROI; "
                                 "no epsilon-dependent ROI or sample skipping is allowed")
        else:
            common = torch.ones((pan.shape[0], 1, height - 2 * margin, width - 2 * margin),
                                dtype=torch.bool, device=pan.device)
        support_count = common.sum(dim=(1, 2, 3))
        if bool((support_count == 0).any()):
            raise ValueError("No valid common candidate support remains after known-padding exclusion")
        ep = self.edges.descriptor(pan)
        em = self.edges.descriptor(self.edges.rgb_luminance(lms))
        ref = _crop(em, margin)
        weights = common.to(dtype=pan.dtype)
        denominator = (2 * support_count).to(dtype=pan.dtype)
        scores = torch.stack([
            ((ep[..., margin + dy:height - margin + dy, margin + dx:width - margin + dx]
              - ref).abs() * weights).sum(dim=(1, 2, 3)) / denominator
            for dy, dx in offsets
        ], dim=1)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("Non-finite candidate score; no arbitrary shift fallback")
        texture_pan = (_crop(ep, margin) * weights).sum(dim=(1, 2, 3)) / denominator
        texture_lms = (ref * weights).sum(dim=(1, 2, 3)) / denominator
        low_texture = ((texture_pan <= self.spec.low_texture_threshold)
                       | (texture_lms <= self.spec.low_texture_threshold))
        if self.spec.low_texture_policy == "error" and bool(low_texture.any()):
            raise ValueError("Low-texture EdgeMatch input under the explicit threshold/policy")
        minimum = scores.min(dim=1, keepdim=True).values
        minima = scores.eq(minimum)
        tie_count = minima.sum(dim=1)
        if self.spec.tie_policy == "error" and bool(((tie_count > 1) & ~low_texture).any()):
            raise ValueError("Nonunique exact EdgeMatch minimum under tie_policy=error")
        # Explicit deterministic tie handling; unlike unique winners, a chosen
        # lexicographic tie need not transform equivariantly under every D4 op.
        def tie_order(i):
            dy, dx = offsets[i]
            if self.spec.tie_policy == "min_norm_then_l1_lexicographic":
                return dy*dy+dx*dx, abs(dy)+abs(dx), dy, dx
            return dy*dy+dx*dx, dy, dx
        priority = sorted(range(len(offsets)), key=tie_order)
        order = torch.tensor(priority, device=pan.device, dtype=torch.long)
        chosen = order[minima[:, order].to(torch.int64).argmax(dim=1)]
        b = candidates[chosen].to(dtype=pan.dtype)
        if self.spec.low_texture_policy == "zero":
            b = torch.where(low_texture[:, None], torch.zeros_like(b), b)
        boundary = (b.abs() == radius).any(dim=1) if radius else torch.zeros_like(low_texture)
        if self.spec.boundary_policy == "error" and bool(boundary.any()):
            raise ValueError("Selected EdgeMatch displacement lies on the declared search boundary")
        return EdgeMatchResult(b.detach(), scores.detach(), candidates, tie_count, low_texture,
                               texture_pan, texture_lms, boundary, margin, resolution,
                               common, support_count)
