"""GT-always Hard/advantage Soft with explicit full-image or MASK2 reduction."""
from __future__ import annotations

import math
import torch

from donor30.masking import masked_mean, validate_mask
from .spec import Stage2Spec, MASK_POLICY, MASK2_POLICY


def _image(value, name, *, shape=None, device=None):
    if (not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] < 1
            or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
        raise ValueError(name + " requires finite, nonempty FP32 NCHW input")
    if shape is not None and value.shape != shape:
        raise ValueError(name + " shape mismatch")
    if device is not None and value.device != device:
        raise ValueError(name + " device mismatch")


@torch.no_grad()
def reference_maps(scharr, teacher_y, gt, aligned_pan):
    """Full maps: e uses all bands, q/g use RGB mean before absolute Scharr."""
    _image(gt, "GT")
    _image(teacher_y, "Teacher", shape=gt.shape, device=gt.device)
    _image(aligned_pan, "Aligned PAN", shape=(len(gt), 1, *gt.shape[-2:]), device=gt.device)
    e_t = (teacher_y.detach() - gt.detach()).abs().mean(1, keepdim=True)
    gt_edges = scharr.descriptor(scharr.rgb_luminance(gt.detach()))
    pan_edges = scharr.descriptor(aligned_pan.detach())
    q = (pan_edges - gt_edges).abs().mean(1, keepdim=True)
    g = gt_edges.mean(1, keepdim=True)
    return e_t, g, q


def compute_loss(student_y, teacher_y, gt, aligned_pan, *, spec, scharr=None,
                 stats=None, return_maps=False, loss_mask=None):
    """A versioned choice applies the same reduction to both Hard and Soft.

    Calibration is read-only and detached. Only e_s and delta_ts retain the
    Student graph. Pure GT and PANDA-e do not evaluate any Scharr/CDF branch.
    """
    if not isinstance(spec, Stage2Spec):
        raise TypeError("The new immutable Stage2Spec is required")
    _image(student_y, "Student")
    _image(gt, "GT", shape=student_y.shape, device=student_y.device)
    b, c, h, w = gt.shape
    if c != spec.bands or min(h, w) <= 2 * spec.roi_margin:
        raise ValueError("Sensor bands and nonempty fixed ROI8 required")
    _image(aligned_pan, "Aligned PAN", shape=(b, 1, h, w), device=gt.device)
    if spec.mask_policy == MASK2_POLICY:
        if loss_mask is None:
            raise ValueError("MASK2 requires the detached original PAN bicubic support mask")
        loss_mask, _ = validate_mask(gt, loss_mask)
        if not bool(loss_mask[..., 7:-7, 7:-7].all()):
            raise ValueError("SUPPORT_VIOLATION: MASK2 must retain the complete ROI8 Scharr stencil")
        reduce = lambda value: masked_mean(value, loss_mask)
    else:
        if loss_mask is not None:
            raise ValueError("Full-image control forbids a loss mask")
        reduce = lambda value: value.mean()
    if stats is not None and hasattr(stats, "mask_policy") and stats.mask_policy != spec.mask_policy:
        raise ValueError("Calibration and loss mask policy mismatch")
    s, y = student_y, gt.detach()
    e_s = (s - y).abs().mean(1, keepdim=True)
    zero = e_s.new_zeros(())
    e_t = torch.zeros_like(e_s)
    d_e, advantage, r_q_bar = (torch.zeros_like(e_s) for _ in range(3))
    q, g = torch.zeros_like(e_s), torch.zeros_like(e_s)
    delta_ts = torch.zeros_like(e_s)
    sl = (..., slice(8, -8), slice(8, -8))
    if spec.variant == "gt_only":
        if teacher_y is not None or stats is not None:
            raise ValueError("gt_only must skip Teacher and calibration entirely")
        w_h = torch.ones_like(e_s)
        soft = zero
    else:
        _image(teacher_y, "Teacher", shape=gt.shape, device=gt.device)
        if stats is None:
            raise ValueError("Fresh full-image train-global tau_e calibration required")
        tau_e = stats.tau_e
        if type(tau_e) not in (int, float) or not math.isfinite(tau_e) or tau_e < 0:
            raise ValueError("Calibrated tau_e must be finite and nonnegative")
        t = teacher_y.detach()
        with torch.no_grad():
            e_t = (t - y).abs().mean(1, keepdim=True)
            tau = e_t.new_tensor(tau_e)
            d_e = e_t / (e_t + tau).clamp_min(torch.finfo(torch.float32).tiny)
            if spec.beta > 0:
                advantage = (e_s.detach() - e_t).clamp_min(0) / (e_s.detach() + spec.delta)
            if spec.variant == "panda_condq":
                if scharr is None or scharr.sensor != spec.sensor:
                    raise ValueError("Sensor-bound Stage 1 Scharr operator required")
                _, g, q = reference_maps(scharr, t, y, aligned_pan)
                ranks = stats.lookup_exact(g[sl], q[sl])
                if (not isinstance(ranks, torch.Tensor) or ranks.shape != q[sl].shape
                        or ranks.dtype != torch.float32 or ranks.device != q.device
                        or ranks.requires_grad or not bool(torch.isfinite(ranks).all())
                        or not bool(((ranks >= 0) & (ranks <= 1)).all())):
                    raise ValueError("Conditional ranks must be detached finite FP32 ROI maps in [0,1]")
                r_q_bar[sl] = ranks
            w_h = 1 + spec.alpha * d_e + spec.gamma * d_e * r_q_bar
        if spec.beta > 0:
            delta_ts = (s - t).abs().mean(1, keepdim=True)
            soft = reduce((1 - d_e) * advantage * delta_ts)
        else:
            # No S->T graph or advantage gate in hard-only experiments.
            # T is still needed for e_T/d_e and its calibration.
            soft = zero
    hard = reduce(w_h * e_s)
    loss = hard if spec.beta == 0 else hard + spec.beta * soft
    output = {"loss": loss, "hard_weighted": hard, "soft_unscaled": soft,
              "soft_weighted": spec.beta * soft, "gt_raw": reduce(e_s),
              "e_extra_raw": reduce(d_e * e_s),
              "q_extra_raw": reduce(d_e * r_q_bar * e_s),
              "d_e_mean": reduce(d_e), "a_T_mean": reduce(advantage),
              "soft_active_fraction": reduce((advantage > 0).float()),
              "w_h_mean": reduce(w_h),
              "w_h_max": (w_h if loss_mask is None else w_h[loss_mask]).max(),
              "w_h_p95": torch.quantile((w_h if loss_mask is None else w_h[loss_mask]).flatten(), .95),
              "q_roi_mean": q[sl].mean(), "g_roi_mean": g[sl].mean(),
              "r_q_roi_mean": r_q_bar[sl].mean(),
              "r_q_roi_quantiles": torch.quantile(r_q_bar[sl].reshape(-1),
                                                  r_q_bar.new_tensor([0., .25, .5, .75, 1.]))}
    for name, value in output.items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite new Stage 2 loss/diagnostic: " + name)
    if not bool((w_h >= 1).all()):
        raise FloatingPointError("GT Hard weight must be at least one")
    # Detached maps permit correct effective-batch diagnostic aggregation.
    output.update(w_h=w_h.detach(), r_q_bar=r_q_bar.detach())
    if loss_mask is not None:
        output["loss_mask"] = loss_mask
    if spec.variant == "panda_condq" and hasattr(stats, "group_counts"):
        output["bin_counts"] = stats.group_counts(g[sl])
    if return_maps:
        output["maps"] = {"e_s": e_s, "e_t": e_t, "delta_ts": delta_ts,
                          "d_e": d_e, "a_T": advantage, "r_q_bar": r_q_bar,
                          "w_h": w_h, "q": q, "g": g}
    return output


compute_s2_loss = compute_loss
