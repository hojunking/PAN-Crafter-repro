"""Differentiable TA2 structure/epsilon losses, independent of U predictions.

All public image arguments are normalized [-1,1] observations. Structural
descriptors convert to DN/maxDN exactly once, without clipping. Official image
quality evaluation must NOT reuse this module's fixed training support.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from pa.warp import warp_pan


class StructSupportViolation(RuntimeError):
    pass


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"NONFINITE_{name}")


def gaussian_scharr(image, sigma):
    """Normalized separable Gaussian followed by normalized Scharr (dy,dx)."""
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Positive finite Gaussian sigma is mandatory")
    radius = math.ceil(3 * sigma)
    if min(image.shape[-2:]) <= radius:
        raise ValueError("Image too small for Gaussian reflect support")
    axis = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-axis.square() / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    bands = image.shape[1]
    smooth = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="reflect"),
                      kernel.view(1, 1, 1, -1).expand(bands, 1, 1, -1), groups=bands)
    smooth = F.conv2d(F.pad(smooth, (0, 0, radius, radius), mode="reflect"),
                      kernel.view(1, 1, -1, 1).expand(bands, 1, -1, 1), groups=bands)
    kx = image.new_tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 32
    kernels = torch.stack((kx.t(), kx))[:, None].repeat(bands, 1, 1, 1)
    grad = F.conv2d(F.pad(smooth, (1, 1, 1, 1), mode="reflect"), kernels, groups=bands)
    return grad.reshape(image.shape[0], bands, 2, *image.shape[-2:])


def assert_struct_support(height, width, correction, scales=(0.8, 1.6), margin=16,
                          resize_scale=1):
    """Check the SAME fixed region plus Gaussian, Scharr AND all bicubic taps.

    Checks clamp-before-source coordinates, including zero-weight integer taps.
    No adaptive/shrinking mask and no all-zero loss fallback is allowed.
    """
    _finite(correction, "CORRECTION")
    radius = max(math.ceil(3 * float(sigma)) for sigma in scales) + 1
    if margin <= radius or min(height, width) <= 2 * margin:
        raise StructSupportViolation("STRUCT_SUPPORT_VIOLATION: empty/insufficient fixed interior")
    low = correction.detach() + correction.new_tensor([margin - radius, margin - radius])
    high = correction.detach() + correction.new_tensor(
        [height - margin - 1 + radius, width - margin - 1 + radius])
    ok = ((torch.floor(low) - 1 >= 0)
          & (torch.floor(high) + 2 <= correction.new_tensor([height - 1, width - 1])))
    if not bool(ok.all()):
        bad = (~ok.all(dim=1)).nonzero().flatten().cpu().tolist()
        raise StructSupportViolation(f"STRUCT_SUPPORT_VIOLATION: samples={bad}, margin={margin}, filter_radius={radius}")
    if resize_scale != 1:
        # X scale source is itself a bicubic HR resize. Trace both footprints
        # back to the original64 PAN rather than merely trusting enlarged edges.
        k = int(resize_scale)
        if k not in (2, 4, 8) or height % k or width % k:
            raise ValueError("Invalid auxiliary source resize geometry")
        original_low = (torch.floor(low) - 1 + .5) / k - .5
        original_high = (torch.floor(high) + 2 + .5) / k - .5
        ok = ((torch.floor(original_low) - 1 >= 0)
              & (torch.floor(original_high) + 2 <= correction.new_tensor([height // k - 1, width // k - 1])))
        if not bool(ok.all()):
            raise StructSupportViolation("STRUCT_SUPPORT_VIOLATION: auxiliary resize source footprint")


def structure_loss(warped_pan, target, *, scales=(0.8, 1.6), descriptor="NCC",
                   margin=16, band_weights=None):
    """Sample-mean, then scale-mean, then target-valid band-weighted loss.

    Caller must run assert_struct_support on the source PAN and actual total
    correction before warping. Target masks/weights have no gradient path.
    The returned raw rho/rho2 arrays include validity flags for honest logging.
    """
    if (warped_pan.ndim != 4 or warped_pan.shape[1] != 1 or target.ndim != 4
            or warped_pan.shape[0] != target.shape[0]
            or warped_pan.shape[-2:] != target.shape[-2:]):
        raise ValueError("Structure needs PAN[B,1,H,W], target[B,C,H,W]")
    if descriptor not in ("NCC", "NGF") or not scales:
        raise ValueError("Unregistered structural descriptor/scales")
    if min(target.shape[-2:]) <= 2 * margin:
        raise StructSupportViolation("STRUCT_SUPPORT_VIOLATION: empty fixed interior")
    _finite(warped_pan, "STRUCT_PAN")
    _finite(target, "STRUCT_TARGET")
    p, y = (warped_pan + 1) * .5, (target.detach() + 1) * .5
    bsz, bands = y.shape[:2]
    if band_weights is not None:
        weights = torch.as_tensor(band_weights, device=y.device, dtype=y.dtype).detach()
        if weights.shape == (bands,):
            weights = weights[None].expand(len(scales), -1)
        if weights.shape != (len(scales), bands) or not bool(torch.isfinite(weights).all()):
            raise ValueError("CAL frozen weights must be [bands] or [scales,bands]")
        if bool((weights < 0).any()) or not bool((weights.sum(-1) > 0).all()):
            raise ValueError("CAL weights must be nonnegative, nonempty per scale")
    else:
        weights = y.new_ones((len(scales), bands))
    values, rhos, valids, rms_values = [], [], [], []
    for index, sigma in enumerate(scales):
        pg = gaussian_scharr(p, sigma)[..., margin:-margin, margin:-margin]
        yg = gaussian_scharr(y, sigma)[..., margin:-margin, margin:-margin]
        rms = yg.square().mean(dim=(2, 3, 4)).sqrt()
        valid = (rms >= 1e-4).detach()
        # Center x/y separately, THEN concatenate the two maps.
        pc = (pg - pg.mean(dim=(-2, -1), keepdim=True)).flatten(2)
        yc = (yg - yg.mean(dim=(-2, -1), keepdim=True)).flatten(2)
        rho = (pc * yc).sum(-1) / ((pc.square().sum(-1) + 1e-12).sqrt()
                                            * (yc.square().sum(-1) + 1e-12).sqrt())
        if descriptor == "NCC":
            band_loss = 1 - rho.square()
        else:
            pn, yn = pg.square().sum(2), yg.square().sum(2)
            dot = (pg * yg).sum(2)
            pixel_loss = 1 - dot.square() / ((pn + 1e-6) * (yn + 1e-6))
            edge = yn.sqrt().detach()
            band_loss = (pixel_loss * edge).sum((-2, -1)) / edge.sum((-2, -1)).clamp_min(1e-12)
        active = valid.to(y.dtype) * weights[index][None]
        denominator = active.sum(-1, keepdim=True)
        # Do not silently choose uniform weights when CAL assigns zero mass to
        # all remaining valid bands. That changes the preregistered estimator.
        if bool(((denominator[:, 0] == 0) & valid.any(-1)).any()):
            raise ValueError("CAL_WEIGHTS_NO_VALID_MASS")
        active = active / denominator.clamp_min(1e-12)
        values.append((active * band_loss).sum(-1))
        rhos.append(rho.detach())
        valids.append(valid)
        rms_values.append(rms.detach())
    per_sample = torch.stack(values, dim=1).mean(1)
    valid = torch.stack(valids, dim=1)
    return per_sample.mean(), dict(
        rho=torch.stack(rhos, 1), rho2=torch.stack(rhos, 1).square(),
        target_gradient_rms=torch.stack(rms_values, 1), valid=valid,
        valid_band_count=valid.sum(-1).to(y.dtype).mean(),
        low_texture_fraction=(~valid.any(-1)).to(y.dtype).mean(),
        low_texture_sample_fraction=(~valid.flatten(1).any(-1)).to(y.dtype).mean(),
        per_sample=per_sample.detach(), scales=list(scales), descriptor=descriptor,
        margin=int(margin), normalization="(normalized_plus_one)/2_no_clip")


def _target_loss(pan, correction, gt, lms, config, *, margin=16, band_weights=None,
                 resize_scale=1):
    scales = tuple(config.get("struct_scales", (0.8, 1.6)))
    assert_struct_support(*pan.shape[-2:], correction, scales, margin, resize_scale)
    warped = warp_pan(pan, correction)
    target_name = config["struct_target"]
    options = dict(scales=scales, descriptor=config.get("struct_descriptor", "NCC"),
                   margin=margin, band_weights=band_weights)
    if target_name == "GT":
        return structure_loss(warped, gt, **options)
    if target_name == "NATIVE_LMS":
        if lms is None:
            raise ValueError("Native LMS structural target is required")
        return structure_loss(warped, lms, **options)
    if target_name == "HALF_GT_HALF_LMS":
        if lms is None:
            raise ValueError("Mixed structural target requires native LMS")
        a, da = structure_loss(warped, gt, **options)
        b, db = structure_loss(warped, lms, **options)
        return (a + b) * .5, {"GT": da, "NATIVE_LMS": db}
    raise ValueError(f"Unregistered structure target {target_name}")


def _geometry(field, auxiliary, choice):
    if field is None:
        return None
    if auxiliary == "CROP":
        size = int(choice)
        if size not in (48, 56, 64) or field.shape[-2:] != (64, 64):
            raise ValueError("X00 requires native64 and crop48/56/64")
        top = left = (64 - size) // 2
        return field[..., top:top + size, left:left + size]
    scale = int(choice)
    if scale not in (1, 2, 4, 8) or field.shape[-2:] != (64, 64):
        raise ValueError("X scale requires native64 and factor1/2/4/8")
    return F.interpolate(field, scale_factor=scale, mode="bicubic", align_corners=False)


def compute_objective(model, batch, config, step, epsilon=None, auxiliary_choice=None,
                      fixed_shift=None, band_weights=None, pseudo_shift=None):
    """Build exactly one update's mean loss; no RNG or optimizer side effects.

    ``step`` is the 1-based update being executed. Every2 epsilon applies on
    even updates. Caller supplies epsilon from a separate area-uniform-disk
    stream and X geometry from another stream. X branches are checkpointed
    sample by sample: native U still sees the unmodified native batch.
    """
    if int(step) != step or step < 1:
        raise ValueError("step must be the 1-based update number")
    pan, ms, lms, gt = (batch.get(key) for key in ("pan", "ms", "lms", "gt"))
    if pan is None or ms is None or gt is None:
        raise ValueError("Training objective requires PAN, MS and GT")
    if gt.shape != (pan.shape[0], model.bands, *pan.shape[-2:]):
        raise ValueError("GT shape differs from native U output")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            _finite(value, key.upper())
    reference, _ = model.references(pan, ms, lms)
    correction = model.predict_correction(pan, reference, fixed_shift)
    output = model.reconstruction(pan, ms, lms, correction,
                                  detach_correction=not bool(config.get("a_receives_rec", True)))
    rec = F.l1_loss(output["prediction"], gt)
    zero = correction.sum() * 0
    lambda_epsilon = float(config.get("lambda_epsilon", 0))
    lambda_struct = float(config.get("lambda_struct", 0))
    every = int(config.get("epsilon_every", 1))
    if every < 1:
        raise ValueError("epsilon_every must be positive")
    epsilon_active = lambda_epsilon > 0 and int(step) % every == 0
    struct_active = lambda_struct > 0
    auxiliary = config.get("auxiliary", "NONE")
    if auxiliary not in ("NONE", "CROP", "SCALE", "SCALE_CONSISTENCY"):
        raise ValueError("Unregistered auxiliary branch")
    descriptor = config.get("struct_descriptor", "NCC")
    if descriptor not in ("NCC", "NGF", "PSEUDO_HUBER"):
        raise ValueError("Unregistered structure descriptor")
    branch = config.get("struct_branch", "NATIVE")
    if branch not in ("NATIVE", "NATIVE_CORRECTED"):
        raise ValueError("Unregistered structure branch")
    if config.get("band_weights", "UNIFORM") not in ("UNIFORM", "CAL_FROZEN"):
        raise ValueError("Unregistered band weight policy")
    if config.get("band_weights", "UNIFORM") == "CAL_FROZEN" and band_weights is None:
        raise ValueError("CAL_FROZEN requires train-only immutable weights")
    if config.get("band_weights", "UNIFORM") == "UNIFORM" and band_weights is not None:
        raise ValueError("Uniform case cannot accept unregistered band weights")
    need_epsilon = epsilon_active or (struct_active and branch == "NATIVE_CORRECTED") or auxiliary != "NONE"
    pe = ce = None
    eps_loss, struct, scale_loss, diag = zero, zero, zero, {}
    if need_epsilon:
        if model.policy != "LEARNED" or epsilon is None or epsilon.shape != correction.shape:
            raise ValueError("Epsilon branch needs learned A and explicit [B,2] epsilon")
        _finite(epsilon, "EPSILON")
        if bool((epsilon.norm(dim=1) > 2 + 1e-6).any()):
            raise ValueError("Training epsilon must lie within radius2 disk")
        pe = warp_pan(pan, epsilon.detach()).detach()
        ce = model.predict_correction(pe, reference.detach())
        anchor = correction.detach() if config.get("epsilon_detach_anchor", True) else correction
        if epsilon_active:
            eps_loss = (ce + epsilon.detach() - anchor).abs().mean()
    if struct_active:
        if descriptor == "PSEUDO_HUBER":
            if pseudo_shift is None or pseudo_shift.shape != correction.shape:
                raise ValueError("S07 requires explicit train GT-proxy shifts")
            _finite(pseudo_shift, "PSEUDO_SHIFT")
            # The registry key means pseudo-label Huber, NOT smooth pseudo-Huber.
            struct = F.huber_loss(correction, pseudo_shift.detach(), delta=.25)
            diag["structure"] = {"descriptor": "HUBER_DELTA_0.25", "unit": "PAN_pixel"}
        else:
            struct, diag["structure"] = _target_loss(
                pan, correction, gt, lms, config, band_weights=band_weights)
            if branch == "NATIVE_CORRECTED":
                shifted, shifted_diag = _target_loss(
                    pan, epsilon.detach() + ce, gt, lms, config, band_weights=band_weights)
                struct = .5 * (struct + shifted)
                diag["corrected_structure"] = shifted_diag
    if auxiliary != "NONE":
        if auxiliary_choice is None:
            raise ValueError("X auxiliary geometry must come from its separate RNG")
        if descriptor == "PSEUDO_HUBER" or branch != "NATIVE":
            raise ValueError("No pseudo-shift/corrected X combinations are registered")
        scale = 1 if auxiliary == "CROP" else int(auxiliary_choice)
        margin = 16 * scale
        aux_eps, aux_struct, aux_scale = [], [], []
        # Closed-over parameters still receive exact gradients with non-reentrant
        # checkpoint. Only one auxiliary sample's large graph is live at a time.
        for i in range(pan.shape[0]):
            def sample_loss(p, shifted_p, ref, truth, low_ref, known_epsilon, native_delta):
                ap = _geometry(p, auxiliary, auxiliary_choice)
                ae = _geometry(shifted_p, auxiliary, auxiliary_choice)
                ar = _geometry(ref, auxiliary, auxiliary_choice)
                ay = _geometry(truth, auxiliary, auxiliary_choice)
                al = _geometry(low_ref, auxiliary, auxiliary_choice)
                ac = model.predict_correction(ap, ar)
                ace = model.predict_correction(ae, ar)
                aa = ac.detach() if config.get("epsilon_detach_anchor", True) else ac
                le = ((ace + known_epsilon * scale - aa) / scale).abs().mean()
                ls = ac.sum() * 0
                if struct_active:
                    ls, _ = _target_loss(ap, ac, ay, al, config,
                                         margin=margin, band_weights=band_weights,
                                         resize_scale=scale)
                lc = ((ace - ac) / scale - native_delta.detach()).abs().mean()
                return le, ls, lc
            # X cases all have native LMS. Do not substitute it for B: ref is
            # the already selected A observation; each field uses one HR mapping.
            if lms is None:
                raise ValueError("X cases require native LMS")
            values = checkpoint(sample_loss, pan[i:i+1], pe[i:i+1], reference[i:i+1],
                                gt[i:i+1], lms[i:i+1], epsilon[i:i+1].detach(),
                                (ce - correction)[i:i+1], use_reentrant=False)
            aux_eps.append(values[0]); aux_struct.append(values[1]); aux_scale.append(values[2])
        if epsilon_active:
            eps_loss = .5 * (eps_loss + torch.stack(aux_eps).mean())
        if struct_active:
            struct = .5 * (struct + torch.stack(aux_struct).mean())
        if auxiliary == "SCALE_CONSISTENCY":
            scale_loss = torch.stack(aux_scale).mean()
        diag["auxiliary"] = dict(kind=auxiliary, choice=int(auxiliary_choice), margin=margin,
                                 descriptor_sigma_units="current_grid_pixels", native_weight=.5,
                                 auxiliary_weight=.5, microbatch=1, scale=scale,
                                 crop_position="center" if auxiliary == "CROP" else None,
                                 measurement_domain="SYNTHETIC_SCALE" if scale != 1 else "TRAIN64")
    ramp = int(config.get("struct_ramp_updates", 0))
    effective_struct = lambda_struct * (min(float(step) / ramp, 1.) if ramp > 0 else 1.)
    weighted_epsilon = lambda_epsilon * eps_loss
    weighted_struct = effective_struct * struct
    lambda_scale = float(config.get("lambda_scale", 0))
    if lambda_scale != 0 and auxiliary != "SCALE_CONSISTENCY":
        raise ValueError("Scale consistency coefficient requires X03 branch")
    weighted_scale = lambda_scale * scale_loss
    total = rec + weighted_epsilon + weighted_struct + weighted_scale
    _finite(total, "TOTAL_LOSS")
    diag.update(epsilon_active=epsilon_active, epsilon_effective_count=pan.shape[0] if epsilon_active else 0,
                lambda_struct_effective=effective_struct, step=int(step))
    return dict(total=total, rec=rec, epsilon=eps_loss, struct=struct, scale=scale_loss,
                weighted_epsilon=weighted_epsilon, weighted_struct=weighted_struct,
                weighted_scale=weighted_scale, output=output, diagnostics=diag)
