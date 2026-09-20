"""Bounded A-only size/context probes; no reconstruction or official metric edits.

Offsets and AXIS16 response are model diagnostics, not displacement ground truth.
The confidence score describes z-score epsilon dominance, not alignment accuracy.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from pa.warp import warp_pan
from qg40.calibration import AXIS16, CONSTANT_ALIGNER_Q
from qg40.common import check_deadline

ZNORM_EPSILON = 1e-6
PROXY_RATIO = 4
PROXY_LR_GRID = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))


def _proxy_band_report(scores, variances, valid):
    values = [float(score) if ok else None for score, ok in zip(scores, valid)]
    usable = [i for i, score in enumerate(values) if score is not None]
    result = dict(correlations=values, ms_variance_by_grid=[float(value) for value in variances],
        low_texture_by_grid=[not bool(ok) for ok in valid], low_texture=not usable,
        zero_offset_correlation=values[4], negative_zero_offset_correlation=values[4] is not None and values[4] < 0,
        available=bool(usable), best_signal_offset_lr=None, best_signal_offset_hr=None,
        best_correlation=None, peak_margin=None, tied_peak_count=0, local_peak_count=0,
        repeated_pattern_or_aperture_ambiguity=None, best_on_search_boundary=None,
        strongest_absolute_signal_offset_lr=None, strongest_absolute_correlation=None)
    if not usable:
        return result
    best = max(values[i] for i in usable)
    # Numerical equality tolerance is derived from float64 arithmetic, not data
    # or a performance-calibrated confidence threshold. Margins remain raw.
    tolerance = 64 * torch.finfo(torch.float64).eps
    tied = [i for i in usable if abs(values[i] - best) <= tolerance]
    preference = lambda i: (sum(v * v for v in PROXY_LR_GRID[i]), *PROXY_LR_GRID[i])
    preferred = min(tied, key=preference)
    ranked = sorted((values[i] for i in usable), reverse=True)
    local_peaks = []
    for index in usable:
        dy, dx = PROXY_LR_GRID[index]
        neighbors = [j for j in usable if j != index and
                     max(abs(PROXY_LR_GRID[j][0] - dy), abs(PROXY_LR_GRID[j][1] - dx)) <= 1]
        if neighbors and all(values[index] >= values[j] - tolerance for j in neighbors):
            local_peaks.append(index)
    absolute = max(abs(values[i]) for i in usable)
    absolute_index = min([i for i in usable if abs(abs(values[i]) - absolute) <= tolerance], key=preference)
    offset = PROXY_LR_GRID[preferred]
    result.update(best_signal_offset_lr=list(offset), best_signal_offset_hr=[v * PROXY_RATIO for v in offset],
        best_correlation=best, peak_margin=ranked[0] - ranked[1] if len(ranked) > 1 else None,
        tied_peak_count=len(tied), tied_signal_offsets_lr=[list(PROXY_LR_GRID[i]) for i in tied],
        local_peak_count=len(local_peaks), repeated_pattern_or_aperture_ambiguity=len(tied) > 1 or len(local_peaks) > 1,
        best_on_search_boundary=any(abs(value) == 1 for value in offset),
        strongest_absolute_signal_offset_lr=list(PROXY_LR_GRID[absolute_index]),
        strongest_absolute_correlation=values[absolute_index])
    return result


@torch.no_grad()
def native_pan_ms_proxy(pan, ms_base, *, deadline_utc=None):
    """Fixed common-band, per-band signal correlation; never displacement GT.

    Both HR inputs receive the same 4×4 normalized box filter and stride-4
    decimation. Each of nine fixed LR offsets compares PAN(y,x) against an MS
    band(y+dy,x+dx), using the same one-LR-pixel-cropped PAN support throughout.
    No spectral averaging, fitted SRF, adaptive radius, FFT, GT or test fitting
    is used. The filter is an algorithmic common-band proxy, not sensor MTF.
    """
    if (pan.ndim != 4 or pan.shape[1] != 1 or ms_base.shape != (len(pan), 4, *pan.shape[-2:])
            or min(pan.shape[-2:]) < 16 or any(size % 4 for size in pan.shape[-2:])):
        raise ValueError("Native proxy requires corresponding PAN1/MS4 HR images divisible by four")
    if not bool(torch.isfinite(pan).all() & torch.isfinite(ms_base).all()):
        raise FloatingPointError("Native proxy inputs must be finite")
    check_deadline(deadline_utc)
    with torch.autocast(device_type=pan.device.type, enabled=False):
        # Float64 filtering/reductions keep ambiguity comparisons tied to numeric
        # precision, without adding a fitted stabilization constant to NCC.
        common = F.avg_pool2d(torch.cat((pan, ms_base), dim=1).double(), kernel_size=4, stride=4)
        p = common[:, :1, 1:-1, 1:-1]
        centered_pan = p - p.mean((2, 3), keepdim=True)
        pan_variance = centered_pan.square().mean((2, 3))
        height, width = common.shape[-2:]
        scores, variances, validity = [], [], []
        for dy, dx in PROXY_LR_GRID:
            check_deadline(deadline_utc)
            ms = common[:, 1:, 1 + dy:height - 1 + dy, 1 + dx:width - 1 + dx]
            centered_ms = ms - ms.mean((2, 3), keepdim=True)
            variance = centered_ms.square().mean((2, 3))
            valid = (pan_variance > ZNORM_EPSILON) & (variance > ZNORM_EPSILON)
            denominator = torch.sqrt(pan_variance * variance)
            covariance = (centered_pan * centered_ms).mean((2, 3))
            correlation = covariance / torch.where(valid, denominator, torch.ones_like(denominator))
            scores.append(correlation.cpu()); variances.append(variance.cpu()); validity.append(valid.cpu())
        scores = torch.stack(scores, dim=2).tolist()
        variances = torch.stack(variances, dim=2).tolist()
        validity = torch.stack(validity, dim=2).tolist()
    scenes = []
    for i in range(len(pan)):
        bands = [_proxy_band_report(scores[i][band], variances[i][band], validity[i][band])
                 for band in range(4)]
        preferred = {tuple(row["best_signal_offset_lr"]) for row in bands if row["available"]}
        scenes.append(dict(pan_variance=float(pan_variance[i, 0]),
            pan_epsilon_fraction=ZNORM_EPSILON / (float(pan_variance[i, 0]) + ZNORM_EPSILON),
            low_pan_texture=float(pan_variance[i, 0]) <= ZNORM_EPSILON, bands=bands,
            per_band_preferred_offsets_disagree=len(preferred) > 1,
            spectral_polarity_warning=any(row["negative_zero_offset_correlation"] for row in bands),
            repeated_pattern_or_aperture_warning=any(row["repeated_pattern_or_aperture_ambiguity"] is True for row in bands)))
    return dict(schema="QG40_NATIVE_PAN_MS_PROXY_v1", diagnostic_only=True,
        alignment_ground_truth=False, used_for_model_or_official_metrics=False, scenes=scenes,
        method=dict(common_filter="normalized 4x4 box applied identically to native PAN and HR MS base",
            computation_dtype="float64", decimation_stride_hr=4, padding="none", first_sample_center_hr=[1.5, 1.5],
            grid_dydx_lr=[list(shift) for shift in PROXY_LR_GRID], grid_radius_lr=1, hr_pixels_per_lr=4,
            support_lr=[height - 2, width - 2], support_pixels=(height - 2) * (width - 2),
            support_policy="same PAN interior for every candidate and band",
            score="signed zero-mean normalized crosscorrelation, independently per MS band",
            sampling_convention="PAN(y,x) versus MS_band(y+dy,x+dx); this samples MS, not the model PAN correction",
            low_texture_policy="NCC is unavailable when PAN or MS variance<=existing z-score epsilon1e-6",
            ambiguity_policy="raw best/runner-up margin, local peaks and float64-equal maxima; no calibrated confidence threshold",
            numerical_tie_tolerance=64 * torch.finfo(torch.float64).eps,
            radius_adaptation=False, spectral_response_fitted=False, sensor_mtf_claim=False),
        caveat="Best grid correlation is a signal-comparison proxy, never physical misalignment or registration ground truth. "
               "PAN spectral response is not a mean MS band or a fitted SRF; polarity, band disagreement, low texture, "
               "repeated patterns, aliasing and the bounded search can confound the reported peaks.")


@torch.no_grad()
def fit_axis16_response(native, shifted, epsilon=None):
    """Fit delta-c against epsilon through the origin, output-axis × input-axis.

    Ideal cancellation has diagonal -1 and cross-axis 0. A constant predictor
    has all slopes 0, q=.46875, and gain=0, irrespective of its native bias.
    """
    native, shifted = native.float(), shifted.float()
    epsilon = (native.new_tensor(AXIS16) if epsilon is None else epsilon.to(native).float())
    if native.ndim != 2 or native.shape[1] != 2 or shifted.shape != (len(native), 16, 2) or epsilon.shape != (16, 2):
        raise ValueError("AXIS16 needs native[B,2], shifted[B,16,2], epsilon[16,2]")
    if not all(bool(torch.isfinite(value).all()) for value in (native, shifted, epsilon)):
        raise FloatingPointError("Nonfinite AXIS16 response")
    difference = shifted - native[:, None]
    denominator = epsilon.square().sum(0)
    if bool((denominator <= 0).any()):
        raise ValueError("AXIS16 must excite both dy and dx axes")
    matrix = torch.einsum("bko,ki->boi", difference, epsilon) / denominator[None, None]
    residual = (shifted + epsilon[None] - native[:, None]).abs().mean(2)
    gain = -(difference * epsilon[None]).sum((1, 2)) / epsilon.square().sum()
    return dict(q=residual.mean(1), per_radius_q=residual.reshape(len(native), 4, 4).mean(2),
                native_delta=native, probe_delta=shifted, response_matrix=matrix, response_gain=gain,
                dy_slope=matrix[:, 0, 0], dx_slope=matrix[:, 1, 1],
                dy_from_dx=matrix[:, 0, 1], dx_from_dy=matrix[:, 1, 0])


@torch.no_grad()
def alignment_response(model, pan, ms_base, *, deadline_utc=None):
    """Native plus sixteen A-only predictions on already corresponding HR inputs."""
    if pan.ndim != 4 or pan.shape[1] != 1 or ms_base.shape != (len(pan), 4, *pan.shape[-2:]):
        raise ValueError("Alignment diagnostics require corresponding native PAN1/MS4")
    with torch.autocast(device_type=pan.device.type, enabled=False):
        pan, ms_base = pan.float(), ms_base.float()
        check_deadline(deadline_utc)
        native = model.predict_delta(pan, ms_base)
        probes = []
        for shift in AXIS16:
            check_deadline(deadline_utc)
            epsilon = pan.new_tensor(shift).expand(len(pan), 2)
            probes.append(model.predict_delta(warp_pan(pan, epsilon), ms_base))
    return fit_axis16_response(native, torch.stack(probes, dim=1))


@torch.no_grad()
def border_sampling(delta, height, width):
    """Unclamped source-coordinate and conservative bicubic support fractions."""
    if delta.ndim != 2 or delta.shape[1] != 2 or min(height, width) < 4:
        raise ValueError("Border sampling needs [B,2] offsets and a valid image size")
    y = torch.arange(height, device=delta.device, dtype=torch.float32)[None] + delta[:, :1].float()
    x = torch.arange(width, device=delta.device, dtype=torch.float32)[None] + delta[:, 1:].float()
    interior_y = ((y >= 0) & (y <= height - 1)).sum(1)
    interior_x = ((x >= 0) & (x <= width - 1)).sum(1)
    support_y = ((torch.floor(y) - 1 >= 0) & (torch.floor(y) + 2 <= height - 1)).sum(1)
    support_x = ((torch.floor(x) - 1 >= 0) & (torch.floor(x) + 2 <= width - 1)).sum(1)
    return dict(border_sampling_fraction=1. - (interior_y * interior_x).float() / (height * width),
                bicubic_border_support_fraction=1. - (support_y * support_x).float() / (height * width))


@torch.no_grad()
def normalization_context(pan, ms_base, margin=4):
    """Statistics on the same margin-4 A views used by the unchanged model."""
    if min(pan.shape[-2:]) <= 2 * margin or ms_base.shape != (len(pan), 4, *pan.shape[-2:]):
        raise ValueError("Invalid native PAN/MS views for z-score diagnostics")
    view = torch.cat((pan.float(), ms_base.float()), 1)[..., margin:-margin, margin:-margin] if margin else torch.cat((pan.float(), ms_base.float()), 1)
    variance = view.var((2, 3), unbiased=False)
    if not bool(torch.isfinite(variance).all()):
        raise FloatingPointError("Nonfinite A-view variance in context diagnostic")
    epsilon_fraction = ZNORM_EPSILON / (variance + ZNORM_EPSILON)
    confidence = (variance / (variance + ZNORM_EPSILON)).amin(1)
    return dict(zscore_variance=variance, zscore_variance_over_epsilon=variance / ZNORM_EPSILON,
                zscore_epsilon_fraction=epsilon_fraction,
                low_texture_channels=variance <= ZNORM_EPSILON,
                low_texture_confidence=confidence)


def _plain(values):
    return {key: value.detach().cpu().numpy().tolist() for key, value in values.items()}


def _crop_positions(height, width, size, names):
    positions = {"top_left": (0, 0), "center": ((height - size) // 2, (width - size) // 2),
                 "bottom_right": (height - size, width - size)}
    for name in names:
        if name not in positions:
            raise ValueError("Only the fixed top_left/center/bottom_right diagnostic crops are defined")
        yield name, positions[name]


@torch.no_grad()
def capture_alignment_sizes(model, datasets, *, device, deadline_utc=None, max_scenes=2,
                            crop_sizes=(64, 128), crop_positions=("top_left", "center", "bottom_right")):
    """Fixed scene IDs 0..max_scenes-1: crops64/128 and full RR256/FR512.

    M is upsampled once at full-frame support, then PAN/M are cropped together
    only for these A probes. No U forward, GT read, LP regeneration, image warp
    in-place, reconstruction reference change, or official metric call occurs.
    At defaults there are at most 28 views, each with 17 A predictions.
    """
    from qg40.training import restore_rng, rng_state
    if isinstance(max_scenes, bool) or not isinstance(max_scenes, int) or not 1 <= max_scenes <= 2:
        raise ValueError("P1 size diagnostics are bounded to at most two fixed scenes per split")
    if tuple(crop_sizes) != (64, 128):
        raise ValueError("The fixed P1 crop sizes are 64 and 128")
    if not crop_positions or len(crop_positions) != len(set(crop_positions)) or len(crop_positions) > 3:
        raise ValueError("Use at most three distinct predefined crop positions")
    modes, rng, records = {module: module.training for module in model.modules()}, rng_state(), []
    try:
        model.eval()
        for split, full_size in (("rr", 256), ("fr", 512)):
            dataset = datasets[split]
            count = getattr(dataset, "base_count", len(dataset))
            if count < 1:
                raise ValueError("Size diagnostics need at least one native scene per split")
            for index in range(min(max_scenes, count)):
                check_deadline(deadline_utc)
                # Native arrays avoid reading or presuming FR GT; readers keep
                # these original tensors untouched for official evaluation.
                if hasattr(dataset, "arrays") and hasattr(dataset, "max_pixel"):
                    ms = torch.as_tensor(dataset.arrays["ms"][index], dtype=torch.float32).unsqueeze(0)
                    pan = torch.as_tensor(dataset.arrays["pan"][index], dtype=torch.float32).unsqueeze(0)
                    ms, pan = ms * (2. / dataset.max_pixel) - 1., pan * (2. / dataset.max_pixel) - 1.
                else:
                    raise TypeError("A-only size diagnostics require explicit native PAN/MS arrays and max_pixel")
                ms, pan = ms.to(device), pan.to(device)
                if pan.shape != (1, 1, full_size, full_size) or ms.shape != (1, 4, full_size // 4, full_size // 4):
                    raise ValueError(f"P1 {split} must retain its actual native {full_size}px full frame")
                base = F.interpolate(ms.float(), scale_factor=4, mode="bicubic", align_corners=False)
                views = [("full", full_size, 0, 0)]
                for size in crop_sizes:
                    views.extend((name, size, top, left) for name, (top, left) in
                                 _crop_positions(full_size, full_size, size, crop_positions))
                for name, size, top, left in views:
                    p, m = pan[..., top:top + size, left:left + size], base[..., top:top + size, left:left + size]
                    response = alignment_response(model, p, m, deadline_utc=deadline_utc)
                    context = normalization_context(p, m, getattr(model, "aligner_margin", 4))
                    border = border_sampling(response["native_delta"], size, size)
                    proxy = native_pan_ms_proxy(p, m, deadline_utc=deadline_utc)
                    records.append(dict(split=split, source_sample_id=int(index), view=name, size=size,
                        crop_top_left_hr=[top, left], native_full_size=full_size,
                        response=_plain(response), context=_plain(context), border=_plain(border),
                        native_pan_ms_proxy=proxy))
        return dict(schema="QG40_ALIGNMENT_SIZE_DIAGNOSTICS_v1", status="MEASURED", diagnostic_only=True,
            records=records, n_views=len(records), max_scenes=max_scenes, fixed_sample_selection="first base IDs, no performance selection",
            ms_upsampling="once on native full frame before paired diagnostic crops", reconstruction_forward_calls=0,
            zscore_epsilon=ZNORM_EPSILON, ideal_axis_slopes=-1., constant_axis_slopes=0.,
            constant_aligner_q=CONSTANT_ALIGNER_Q, alignment_ground_truth=False,
            confidence_definition="minimum channel variance/(variance+epsilon) over native PAN1/MS4 A views; heuristic, not accuracy",
            warning="c is a model correction, q is consistency, and low-texture confidence is not physical alignment accuracy")
    finally:
        for module, mode in modes.items():
            module.training = mode
        restore_rng(rng)
