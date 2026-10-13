"""Scene-conditioned PAN geometry for MethodPlan-v1.3 (2026-10-13).

This is an additive method API, not a migration of ``pa.PANGlobalAligner``.
Vectors are sampling displacements ``(dy, dx)``: W(P,c)[y,x] = P[y+dy,x+dx].
No scene estimator, confidence cutoff, RR/FR scale factor, or crop policy is
invented here. Those unresolved choices must be supplied with provenance.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model.pancrafter_paper import ResBlock
from pa.aligner import _conv, znorm


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty provenance string")


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _finite_tuple(value, length, name):
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite numbers") from exc
    if len(result) != length or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def _shift_tensor(value: Tensor, *, name: str, nonzero: bool = True) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")
    if value.ndim < 1 or value.shape[-1] != 2 or value.numel() == 0:
        raise ValueError(f"{name} must have a nonempty (..., 2) dy/dx shape")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} has nonfinite values")
    if nonzero and bool((value.abs().amax(dim=-1) < torch.finfo(value.dtype).tiny).any()):
        raise ValueError(f"{name} has a zero/subnormal prior; fallback policy is unresolved")
    return value.detach()


@dataclass(frozen=True)
class ScenePriorRecord:
    """One accepted, immutable source-scene prior on a named canonical grid.

    ``accepted`` is a reviewed input decision, not an inferred numeric threshold.
    Its evidence must explain the scene linkage, sampling sign, reference frame,
    split, and acceptance of estimator confidence/near-zero magnitude. This
    contract verifies presence/consistency, not the truth of external evidence.
    """

    source_scene_id: str
    prior_shift_dy_dx: Tuple[float, float]
    prior_grid_name: str
    pixel_scale: Tuple[float, float]
    origin_phase: Tuple[float, float]
    n_windows: int
    n_valid: int
    inlier_fraction: float
    direction_concentration: float
    dispersion: float
    confidence: float
    estimator_version: str
    reference_type: str
    scene_input_hash: str
    mapping_rule: str
    accepted: bool
    acceptance_evidence: str

    def __post_init__(self):
        for name in ("source_scene_id", "prior_grid_name", "estimator_version",
                     "reference_type", "scene_input_hash", "mapping_rule",
                     "acceptance_evidence"):
            _nonempty(getattr(self, name), name)
        for name in ("prior_shift_dy_dx", "pixel_scale", "origin_phase"):
            object.__setattr__(self, name, _finite_tuple(getattr(self, name), 2, name))
        if not any(self.prior_shift_dy_dx):
            raise ValueError("zero scene prior: no approved fallback is defined")
        if not all(v > 0 for v in self.pixel_scale):
            raise ValueError("pixel_scale components must be positive")
        if type(self.n_windows) is not int or type(self.n_valid) is not int:
            raise ValueError("n_windows and n_valid must be integer evidence counts")
        if not (0 < self.n_valid <= self.n_windows):
            raise ValueError("require 0 < n_valid <= n_windows")
        for name in ("inlier_fraction", "direction_concentration", "confidence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0,1]")
            object.__setattr__(self, name, value)
        if not math.isfinite(float(self.dispersion)) or float(self.dispersion) < 0:
            raise ValueError("dispersion must be finite and nonnegative")
        object.__setattr__(self, "dispersion", float(self.dispersion))
        if self.accepted is not True:
            raise ValueError("scene prior was not explicitly accepted; do not invent a fallback")

    def fingerprint(self) -> str:
        """Bind vector, grid/phase and provenance to a deterministic SHA256."""
        return _fingerprint(asdict(self))


class ScenePriorRegistry:
    """Exact scene-ID lookup; unknown scenes never receive a sensor-average prior."""

    def __init__(self, records: Iterable[ScenePriorRecord]):
        self._records = {}
        for record in records:
            if not isinstance(record, ScenePriorRecord):
                raise TypeError("registry entries must be ScenePriorRecord instances")
            if record.source_scene_id in self._records:
                raise ValueError(f"duplicate source scene: {record.source_scene_id}")
            self._records[record.source_scene_id] = record
        if not self._records:
            raise ValueError("scene registry cannot be empty")

    def record(self, source_scene_id: str) -> ScenePriorRecord:
        _nonempty(source_scene_id, "source_scene_id")
        try:
            return self._records[source_scene_id]
        except KeyError as exc:
            raise KeyError(f"no accepted prior for source scene {source_scene_id!r}") from exc

    def fingerprint(self) -> str:
        """Independent of input record order, sensitive to any prior metadata change."""
        return _fingerprint({key: value.fingerprint()
                             for key, value in self._records.items()})

    def lookup(self, scene_ids: Sequence[str], *, grid_name: str,
               device=None, dtype=torch.float32) -> Tensor:
        """Return [B,2] only on the stored grid. Convert grids explicitly afterwards."""
        _nonempty(grid_name, "grid_name")
        if isinstance(scene_ids, str) or len(scene_ids) == 0:
            raise ValueError("scene_ids must be a nonempty sequence, not a string")
        records = [self.record(scene_id) for scene_id in scene_ids]
        for record in records:
            if record.prior_grid_name != grid_name:
                raise ValueError(f"{record.source_scene_id}: grid {record.prior_grid_name!r} "
                                 f"does not match {grid_name!r}; explicit verified conversion required")
        return _shift_tensor(torch.tensor([r.prior_shift_dy_dx for r in records],
                                         device=device, dtype=dtype), name="registry prior")


@dataclass(frozen=True)
class VerifiedGridTransform:
    """Audited correction transform: ``b_target = matrix @ b_source + phase``.

    The phase is the *differential PAN/MS sampling-phase correction*, not a crop
    origin. All fields are mandatory, including an explicitly measured/justified
    zero phase. The evidence string identifies the external verification; this
    type cannot independently prove same-scene/FOV/degradation correspondence.
    """

    source_grid: str
    target_grid: str
    matrix_dy_dx: Tuple[Tuple[float, float], Tuple[float, float]]
    phase_dy_dx: Tuple[float, float]
    evidence: str

    def __post_init__(self):
        for name in ("source_grid", "target_grid", "evidence"):
            _nonempty(getattr(self, name), name)
        try:
            matrix = tuple(_finite_tuple(row, 2, "matrix_dy_dx row")
                           for row in self.matrix_dy_dx)
        except TypeError as exc:
            raise ValueError("matrix_dy_dx must be a finite 2x2 matrix") from exc
        if len(matrix) != 2:
            raise ValueError("matrix_dy_dx must be a finite 2x2 matrix")
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if not math.isfinite(determinant) or determinant == 0:
            raise ValueError("matrix_dy_dx must be finite and invertible")
        object.__setattr__(self, "matrix_dy_dx", matrix)
        object.__setattr__(self, "phase_dy_dx", _finite_tuple(self.phase_dy_dx, 2, "phase_dy_dx"))

    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


def convert_prior_to_grid(prior_dy_dx: Tensor, *, source_grid: str,
                          transform: VerifiedGridTransform) -> Tensor:
    """Apply an explicit verified mapping, never infer a factor from image shapes."""
    if not isinstance(transform, VerifiedGridTransform):
        raise TypeError("transform must be VerifiedGridTransform")
    if source_grid != transform.source_grid:
        raise ValueError("source grid does not match the verified transform")
    prior = _shift_tensor(prior_dy_dx, name="source prior")
    matrix = prior.new_tensor(transform.matrix_dy_dx)
    phase = prior.new_tensor(transform.phase_dy_dx)
    return _shift_tensor(prior @ matrix.T + phase, name="converted prior")


def transform_prior(prior_dy_dx: Tensor, *, hflip: bool = False,
                    vflip: bool = False, rot90: int = 0) -> Tensor:
    """Match image hflip(-1), then vflip(-2), then torch.rot90(k,(-2,-1)).

    For a single CCW rotation in image coordinates (y down, x right),
    ``(dy,dx) -> (-dx,dy)``. Cropping needs no displacement-origin correction.
    All samples here share one augmentation; call per sample for mixed transforms.
    """
    if type(hflip) is not bool or type(vflip) is not bool or type(rot90) is not int:
        raise TypeError("hflip/vflip must be bool and rot90 must be integer")
    prior = _shift_tensor(prior_dy_dx, name="augmentation prior")
    dy, dx = prior.unbind(dim=-1)
    if hflip:
        dx = -dx
    if vflip:
        dy = -dy
    for _ in range(rot90 % 4):
        dy, dx = -dx, dy
    return torch.stack((dy, dx), dim=-1)


class SceneAwarePANAligner(nn.Module):
    """Native-LMS scene prior + patch residual, preserving the legacy stems/GN.

    ``view_margin`` is mandatory: 0 explicitly means full-field; positive values
    crop PAN and LMS equally *for regression only*. It is not an image-loss mask
    or a claim that padding artifacts are eliminated. Near-zero/low-confidence
    scientific acceptance belongs in the prior registry review, not a hidden
    magnitude threshold here. Exact zero, nonfinite and underflowed priors fail.

    The head consumes the *raw signed current-grid* prior, not its direction.
    v1.2 has the same FC shapes but different semantics and must not be loaded
    silently. The persistent revision marker protects both direct and nested
    ``load_state_dict`` calls, including ``strict=False``.
    """

    CONDITIONING_REVISION = 13
    HEAD_CONDITIONING = "raw_signed_current_pan_grid_dy_dx"

    def __init__(self, ms_bands: int = 8, *, view_margin: int):
        super().__init__()
        if type(ms_bands) is not int or ms_bands <= 0:
            raise ValueError("ms_bands must be a positive integer")
        if type(view_margin) is not int or view_margin < 0:
            raise ValueError("view_margin must be an explicitly chosen nonnegative integer")
        self.ms_bands = ms_bands
        self.view_margin = view_margin
        self.register_buffer("_conditioning_revision",
                             torch.tensor(self.CONDITIONING_REVISION, dtype=torch.int64))
        self.pan_stem = _conv(1, 16, 1, 4)
        self.ms_stem = _conv(ms_bands, 16, 1, 4)
        self.joint1 = _conv(32, 32, 2, 8)
        self.joint2 = _conv(32, 64, 2, 8)
        self.res = ResBlock(64, out_channels=64, norm="ln", dropout=0.,
                            mode_mod=False, se=False)
        self.fc1 = nn.Linear(66, 32, bias=True)
        self.fc2 = nn.Linear(32, 2, bias=True)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def conditioning_manifest(self):
        return {"revision": self.CONDITIONING_REVISION,
                "conditioning": self.HEAD_CONDITIONING,
                "input_order": ["GAP_feature_0_to_63", "prior_dy", "prior_dx"],
                "conditioning_detached": True, "head_dims": [66, 32, 2],
                "prior_normalization": "none", "prior_clamp": "none"}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        marker = state_dict.get(prefix + "_conditioning_revision")
        if (not isinstance(marker, Tensor) or marker.dtype != torch.int64
                or marker.shape != torch.Size([])
                or marker.item() != self.CONDITIONING_REVISION):
            # Raise before copying this module or traversing its child weights.
            # Appending to error_msgs alone would let the loader mutate children
            # before the eventual failure; strict=False must not bypass this.
            raise RuntimeError(
                f"{prefix or 'aligner.'}PAN Aligner conditioning revision is missing or "
                "incompatible: MethodPlan-v1.3 requires raw signed current-grid prior "
                "conditioning (revision 13). A v1.2 direction-conditioned head has "
                "identical FC shapes but cannot be reused implicitly; use fresh "
                "training or an explicitly reviewed migration outside this loader.")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, pan: Tensor, native_lms: Tensor, prior_dy_dx: Tensor,
                return_details: bool = False):
        if pan.ndim != 4 or pan.shape[1] != 1 or pan.shape[0] == 0:
            raise ValueError("pan must be a nonempty [B,1,H,W] tensor")
        if native_lms.shape != (pan.shape[0], self.ms_bands, *pan.shape[-2:]):
            raise ValueError("native_lms must be [B,ms_bands,H,W] on the PAN grid")
        if not pan.is_floating_point() or not native_lms.is_floating_point():
            raise TypeError("PAN and native LMS must be floating point")
        if pan.device != native_lms.device or pan.dtype != native_lms.dtype:
            raise ValueError("PAN and native LMS must share dtype/device")
        if not bool(torch.isfinite(pan).all()) or not bool(torch.isfinite(native_lms).all()):
            raise ValueError("PAN/native LMS has nonfinite values")
        prior = _shift_tensor(prior_dy_dx, name="scene prior")
        if prior.shape != (pan.shape[0], 2):
            raise ValueError("prior_dy_dx must be exactly [B,2]; implicit broadcasting is forbidden")
        # Validate once more after conversion so dtype underflow cannot invent d=(1,0).
        prior = _shift_tensor(prior.to(device=pan.device, dtype=pan.dtype), name="converted input prior")
        # Scale first to avoid overflow/underflow in direction normalization.
        scaled = prior / prior.abs().amax(dim=-1, keepdim=True)
        direction = scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        normal = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
        margin = self.view_margin
        if min(pan.shape[-2:]) - 2 * margin < 3:
            raise ValueError("aligner view is too small for reflection-padded stem/joint convolutions")
        if margin:
            pan = pan[..., margin:-margin, margin:-margin]
            native_lms = native_lms[..., margin:-margin, margin:-margin]
        p = self.pan_stem(znorm(pan))
        m = self.ms_stem(znorm(native_lms))
        x = self.joint2(self.joint1(torch.cat((p, m), dim=1)))
        features = self.res(x, None).mean(dim=(2, 3))
        # Magnitude and sign must reach FC1 unchanged; normalization above is
        # solely for the output residual basis. The prior was already detached.
        conditioned = torch.cat((features, prior.to(features.dtype)), dim=1)
        if not bool(torch.isfinite(conditioned).all()):
            raise ValueError("aligner head input has nonfinite values after dtype conversion")
        preactivation = self.fc1(conditioned)
        hidden = F.silu(preactivation)
        if not bool(torch.isfinite(preactivation).all()) or not bool(torch.isfinite(hidden).all()):
            raise ValueError("aligner FC1 produced nonfinite values; no implicit prior rescaling is allowed")
        alpha_beta = self.fc2(hidden).to(prior.dtype)
        alpha, beta = alpha_beta[:, :1], alpha_beta[:, 1:]
        residual = alpha * direction + beta * normal
        correction = prior + residual
        if not bool(torch.isfinite(correction).all()):
            raise ValueError("aligner produced a nonfinite correction")
        if return_details:
            return correction, {"prior": prior, "direction": direction, "normal": normal,
                                "alpha": alpha, "beta": beta, "residual": residual,
                                "features": features, "head_input": conditioned,
                                "fc1_preactivation": preactivation, "head_hidden": hidden}
        return correction
