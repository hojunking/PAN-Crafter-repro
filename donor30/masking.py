"""Training-only PAN support masking. Native inference/evaluation is unchanged."""
from __future__ import annotations

import torch

from directshift48.method import DirectShiftStage1, epsilon_loss, sample_epsilon
from directshift48.training import build_direct_stage1
from pa.warp import warp_pan, warp_support_mask
from .plan import LOSS_MASK_POLICY


def mask_manifest():
    return {"policy": LOSS_MASK_POLICY,
            "support": "all_floor_q_minus1_through_plus2_taps_inside_original_PAN",
            "coordinates": "original_PAN_sampling_grid_before_border_extension",
            "gradient": "detached_boolean_mask_no_gradient_through_selection_or_count",
            "geometry_depends_on_predicted_shift": True,
            "reduction": "mean_channels_then_valid_pixel_mean_per_sample_then_batch_mean",
            "empty_sample": "raise_before_optimizer_update",
            "spatial": "unchanged_fixed_ROI8_with_full_Scharr_stencil_support_guard",
            "evaluation": "unchanged_official_FR_full512_unmasked_RR_dim_cut21"}


def validate_mask(value, mask):
    if (value.ndim != 4 or mask.dtype != torch.bool
            or mask.shape != (len(value), 1, *value.shape[-2:])
            or mask.device != value.device):
        raise ValueError("Boolean B1HW support mask on the loss grid/device required")
    mask = mask.detach()
    counts = mask.flatten(1).sum(1)
    if len(counts) == 0 or not bool((counts > 0).all()):
        raise ValueError("EMPTY_SUPPORT: every sample must retain valid PAN pixels")
    return mask, counts


def masked_mean(value, mask):
    """Equal sample weights preserve exact microbatch accumulation semantics."""
    mask, counts = validate_mask(value, mask)
    pixels = value.float().mean(1, keepdim=True)
    if not bool(torch.isfinite(pixels).all()):
        raise FloatingPointError("Nonfinite masked loss map")
    sums = torch.where(mask, pixels, torch.zeros_like(pixels)).flatten(1).sum(1)
    return (sums / counts.to(sums.dtype)).mean()


class MaskedDonorStage1(DirectShiftStage1):
    """Same networks/epsilon/spatial objective; only reconstruction support differs."""

    @classmethod
    def from_fresh_stage(cls, stage):
        if type(stage) is not DirectShiftStage1 or stage.forward_calls != 0:
            raise ValueError("Only a fresh, untrained DirectShift Stage1 may be wrapped")
        return cls(stage.aligner, stage.teacher, sources=stage.sources,
                   epsilon_seed=stage.epsilon_seed, roi_margin=stage.roi_margin,
                   boundary_policy=stage.boundary_policy,
                   lambda_abs=stage.lambda_abs, lambda_eps=stage.lambda_eps)

    def pipeline_manifest(self):
        result = super().pipeline_manifest()
        result.update(revision="DirectShiftR3-MaskedDonor-v2",
                      rec_reduction=LOSS_MASK_POLICY, loss_mask=mask_manifest())
        result["boundary"].update(prediction_dependent_mask=True,
                                  prediction_dependent_mask_scope="reconstruction_only_spatial_ROI8_unchanged")
        return result

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
                mask = warp_support_mask(*rr.pan.shape[-2:], native["u"].detach())
                rec = masked_mean((native["y"] - rr.gt.detach().float()).abs(), mask)
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
                    raise FloatingPointError("Nonfinite masked donor objective")
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
                "forward_index": call, "epsilon_rr_view_support_fraction": fraction,
                "loss_mask": mask, **support}


def build_masked_stage1(*args, **kwargs):
    return MaskedDonorStage1.from_fresh_stage(build_direct_stage1(*args, **kwargs))
