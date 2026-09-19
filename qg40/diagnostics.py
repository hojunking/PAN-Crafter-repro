"""Fixed train128 distributions and train24 gradient evidence for QG40.

These are diagnostic measurements, never alignment ground truth or input to a
normal inference forward. All sampling is independent of performance and test.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F

from fh12.calibration import _batch
from pa.losses import scharr
from qg40.calibration import axis16_q, select_calibration_indices
from qg40.common import check_deadline, object_sha
from qg40.losses import STUDENT_PROFILES, student_losses, teacher_loss
from qg40.model import state_hash
from qg40.training import restore_rng, rng_state
from qg40.alignment_diagnostics import alignment_response, border_sampling, normalization_context

DIAGNOSTIC_SEED = 5678


def diagnostic_indices(base_count):
    if base_count < 128:
        raise ValueError("Diagnostics require 128 distinct train base IDs")
    return np.random.default_rng(DIAGNOSTIC_SEED).choice(base_count, 128, replace=False)


def gradient_statistics(left, right):
    """Norm ratio and cosine on matching parameter sets, without clipping."""
    left_sq = sum(float(g.detach().double().square().sum()) for g in left if g is not None)
    right_sq = sum(float(g.detach().double().square().sum()) for g in right if g is not None)
    dot = sum(float(a.detach().double().mul(b.detach().double()).sum())
              for a, b in zip(left, right) if a is not None and b is not None)
    norm_left, norm_right = math.sqrt(left_sq), math.sqrt(right_sq)
    return dict(left_norm=norm_left, right_norm=norm_right,
                norm_ratio=norm_left / norm_right if norm_right > 0 else None,
                cosine=dot / (norm_left * norm_right) if norm_left > 0 and norm_right > 0 else None)


@torch.no_grad()
def response_gain(model, pan, ms):
    base = F.interpolate(ms.float(), scale_factor=4, mode="bicubic", align_corners=False)
    return alignment_response(model, pan, base)["response_gain"]


def _median(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.median(values)) if values else None


def _gradient_norm(values):
    return math.sqrt(sum(float(value.detach().double().square().sum()) for value in values if value is not None))


def _distribution(values):
    levels = [.01, .05, .25, .5, .75, .95, .99]
    return dict(mean=float(np.mean(values)), quantile_levels=levels,
                quantiles=np.quantile(values, levels).tolist(), min=float(np.min(values)), max=float(np.max(values)))


def _texture_strata(arrays, error_key):
    texture = arrays["edge_energy"]
    threshold = float(np.median(texture))
    result = dict(definition="GT signed-Scharr absolute energy per train patch; low<=fixed-sample median, high>median",
                  threshold=threshold, affects_sampling_or_loss=False)
    for name, selected in (("low", texture <= threshold), ("high", texture > threshold)):
        count = int(selected.sum())
        result[name] = dict(count=count, residual_mean=float(arrays[error_key][selected].mean()) if count else None,
            per_band_l1=arrays["band_l1"][selected].mean(0).tolist() if count else None,
            per_band_mse=arrays["band_mse"][selected].mean(0).tolist() if count else None,
            per_band_bias=arrays["band_bias"][selected].mean(0).tolist() if count else None)
        if "e_student" in arrays:
            result[name]["teacher_residual_mean"] = float(arrays["e_teacher"][selected].mean()) if count else None
    return result


def capture_diagnostics(model, teacher, dataset, *, device, profile="BASE", tau_R=None,
                         q_cache=None, q_ref=None, consistency_weight=1e-4,
                         deadline_utc=None, batch_size=8, indices=None, synthetic_test=False):
    """Return a JSON summary and raw arrays; successful return means all 128/24.

    Production IDs are fixed by dataset count. Each gradient statistic is first
    measured for one of 24 samples, then summarized by its sample median.
    """
    count = getattr(dataset, "base_count", len(dataset))
    fixed = diagnostic_indices(count) if indices is None else np.asarray(indices, dtype=np.int64)
    if not synthetic_test and not np.array_equal(fixed, diagnostic_indices(count)):
        raise ValueError("Production diagnostics must use fixed train128 IDs")
    if len(fixed) == 0 or len(np.unique(fixed)) != len(fixed) or fixed.min() < 0 or fixed.max() >= count:
        raise ValueError("Invalid diagnostic IDs")
    n_gradient = min(24, len(fixed)) if synthetic_test else 24
    if teacher is not None and (q_cache is None or tau_R is None or q_ref is None):
        raise ValueError("Student diagnostics require its measured frozen reference")
    modes = {module: module.training for root in (model, teacher) if root is not None for module in root.modules()}
    teacher_hash = state_hash(teacher.state_dict()) if teacher is not None else None
    rng = rng_state()
    arrays = {}
    def append(name, value):
        arrays.setdefault(name, []).append(value.detach().cpu().numpy())
    gradient_rows = []
    try:
        model.eval()
        if teacher is not None:
            teacher.eval()
        with torch.no_grad():
            for start in range(0, len(fixed), batch_size):
                check_deadline(deadline_utc)
                ids = fixed[start:start + batch_size]
                gt, _, ms, lp, pan = _batch(dataset, ids, device, rot=0)
                out = model(pan, ms, lp)
                error = (out["y"] - gt).abs().mean(1)
                base = F.interpolate(ms.float(), scale_factor=4, mode="bicubic", align_corners=False)
                response = alignment_response(model, pan, base, deadline_utc=deadline_utc)
                delta = response["native_delta"]
                gx, gy = scharr(gt)
                append("e_student", error)
                append("reconstruction_loss", error.mean((1, 2)))
                for name in ("native_delta", "q", "per_radius_q", "response_gain", "response_matrix",
                             "dy_slope", "dx_slope", "dy_from_dx", "dx_from_dy"):
                    append(name, response[name])
                for name, value in border_sampling(delta, *pan.shape[-2:]).items():
                    append(name, value)
                for name, value in normalization_context(pan, base, getattr(model, "aligner_margin", 4)).items():
                    append(name, value)
                residual = out["y"] - gt
                append("band_bias", residual.mean((2, 3)))
                append("band_l1", residual.abs().mean((2, 3)))
                append("band_mse", residual.square().mean((2, 3)))
                append("edge_energy", (.5 * (gx.abs() + gy.abs())).mean((1, 2, 3)))
                append("highpass_energy", out["H"].abs().mean((1, 2, 3)))
                if teacher is not None:
                    tout = teacher(pan, ms, lp)
                    e_teacher = (tout["y"] - gt).abs().mean(1)
                    tq = torch.as_tensor(q_cache, dtype=torch.float32, device=pan.device)[torch.as_tensor(ids, device=pan.device), 0]
                    weights = q_ref / (q_ref + tq)
                    losses = student_losses(out, tout, gt, tau_R, weights, profile)
                    teacher_residual = tout["y"] - gt
                    online_q, _, _ = axis16_q(teacher, pan, ms)
                    append("e_teacher", e_teacher)
                    append("e_student_minus_teacher", error - e_teacher)
                    append("teacher_band_l1", teacher_residual.abs().mean((2, 3)))
                    append("teacher_band_mse", teacher_residual.square().mean((2, 3)))
                    append("teacher_band_bias", teacher_residual.mean((2, 3)))
                    append("teacher_q", tq)
                    append("teacher_q_online", online_q)
                    append("s", weights)
                    append("advantage_fraction", (error > e_teacher).float().mean((1, 2)))
                    append("advantage", losses["advantage"].squeeze(1))
                    append("difficulty", losses["difficulty"].squeeze(1))
                    _, beta, edge_coefficient = STUDENT_PROFILES[profile]
                    append("soft_weight", (beta * (1. - losses["difficulty"]) * losses["advantage"]).squeeze(1))
                    append("epsilon_dominated_error_fraction", (error <= 1e-6).float().mean((1, 2)))
                    append("hard_loss", losses["hard_i"])
                    append("soft_loss", losses["soft_i"])
                    append("weighted_edge_loss", edge_coefficient * weights * losses["edge_i"])
                    append("aligner_loss", weights * losses["hard_i"])
        params = list((model.backbone if teacher is not None else model.aligner).parameters())
        for index in fixed[:n_gradient]:
            check_deadline(deadline_utc)
            gt, _, ms, lp, pan = _batch(dataset, [index], device, rot=0)
            out = model(pan, ms, lp)
            if teacher is not None:
                with torch.no_grad():
                    tout = teacher(pan, ms, lp)
                raw_q = float(torch.as_tensor(q_cache)[int(index), 0])
                losses = student_losses(out, tout, gt, tau_R, [q_ref / (q_ref + raw_q)], profile)
                hard = torch.autograd.grad(losses["hard"], params, retain_graph=True)
                soft = torch.autograd.grad(losses["soft"], params, retain_graph=True)
                edge = torch.autograd.grad(losses["edge_weighted"], params, retain_graph=True)
                total_u = torch.autograd.grad(losses["L_U"], params, retain_graph=True)
                aligner = torch.autograd.grad(losses["L_A"], list(model.aligner.parameters()))
                gradient_rows.append(dict(index=int(index), soft_hard=gradient_statistics(soft, hard),
                    edge_hard=gradient_statistics(edge, hard), u_gradient_norm=_gradient_norm(total_u),
                    a_la_gradient_norm=_gradient_norm(aligner), hard=float(losses["hard"].detach()),
                    soft=float(losses["soft"].detach()), weighted_edge=float(losses["edge_weighted"].detach()),
                    aligner_loss=float(losses["L_A"].detach())))
            else:
                corruption = torch.Generator(device="cpu").manual_seed(DIAGNOSTIC_SEED + 100000 + int(index))
                losses = teacher_loss(model, out, gt, pan, ms, 1, corruption, consistency_weight)
                rec_u = torch.autograd.grad(losses["rec"], list(model.backbone.parameters()), retain_graph=True)
                rec = torch.autograd.grad(losses["rec"], params, retain_graph=True)
                con = torch.autograd.grad(losses["off"] * consistency_weight, params)
                gradient_rows.append(dict(index=int(index), consistency_rec=gradient_statistics(con, rec),
                    reconstruction=float(losses["rec"].detach()), weighted_consistency=float((losses["off"] * consistency_weight).detach()),
                    a_gradient_norm=_gradient_norm([a + b for a, b in zip(rec, con)]),
                    u_gradient_norm=_gradient_norm(rec_u)))
        result = {name: np.concatenate(parts) for name, parts in arrays.items() if parts}
        error_key = "e_student" if teacher is not None else "e_teacher"
        if teacher is None:
            result["e_teacher"] = result.pop("e_student")
        result["diagnostic_indices"] = fixed
        result["gradient_indices"] = fixed[:n_gradient]
        for name, value in result.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite diagnostic array: {name}")
        calibration_overlap = (int(np.isin(fixed, select_calibration_indices(count)).sum()) if count >= 3072 else None)
        summary = dict(schema="QG40_DIAGNOSTICS_v1", status="MEASURED", synthetic_test=synthetic_test,
            role="S" if teacher is not None else "T",
            diagnostic_seed=DIAGNOSTIC_SEED, distribution_samples=len(fixed), gradient_samples=n_gradient,
            diagnostic_indices_sha256=object_sha(fixed.tolist()), calibration3072_overlap=calibration_overlap,
            view=dict(fixed_hflip=True, fixed_vflip=True, rotation=0), gradient_rows=gradient_rows,
            response_gain_median=float(np.median(result["response_gain"])),
            native_delta_mean=result["native_delta"].mean(0).tolist(),
            native_delta_abs_max=float(np.abs(result["native_delta"]).max()),
            native_delta_quantile_levels=[0., .01, .05, .5, .95, .99, 1.],
            native_delta_quantiles=np.quantile(result["native_delta"], [0., .01, .05, .5, .95, .99, 1.], axis=0).tolist(),
            native_delta_norm_distribution=_distribution(np.linalg.norm(result["native_delta"], axis=1)),
            border_sampling_fraction=_distribution(result["border_sampling_fraction"]),
            bicubic_border_support_fraction=_distribution(result["bicubic_border_support_fraction"]),
            axis_response_matrix_mean=result["response_matrix"].mean(0).tolist(),
            axis_response=dict(ideal_diagonal=-1., constant_diagonal=0., ideal_cross_axis=0.,
                **{name: _distribution(result[name]) for name in ("dy_slope", "dx_slope", "dy_from_dx", "dx_from_dy")}),
            per_band_l1=result["band_l1"].mean(0).tolist(), per_band_mse=result["band_mse"].mean(0).tolist(),
            per_band_bias=result["band_bias"].mean(0).tolist(),
            texture_strata=_texture_strata(result, error_key),
            error_distribution=_distribution(result[error_key]),
            low_texture_channel_fraction=float(result["low_texture_channels"].mean()),
            zscore_epsilon=1e-6, low_texture_confidence=_distribution(result["low_texture_confidence"]),
            q_median=float(np.median(result["q"])),
            error_quantiles=np.quantile(result[error_key], [0, .5, .9, .95, .99, 1]).tolist(),
            warning="c is not native displacement GT; q is consistency; e is not pure misalignment")
        if teacher is not None:
            if state_hash(teacher.state_dict()) != teacher_hash:
                raise ValueError("Frozen Teacher changed during diagnostics")
            summary.update(advantage_fraction=float(result["advantage_fraction"].mean()),
                soft_hard_gradient_ratio_median=_median([row["soft_hard"]["norm_ratio"] for row in gradient_rows]),
                weighted_edge_hard_gradient_ratio=_median([row["edge_hard"]["norm_ratio"] for row in gradient_rows]),
                edge_hard_gradient_cosine=_median([row["edge_hard"]["cosine"] for row in gradient_rows]),
                u_gradient_norm_median=_median([row["u_gradient_norm"] for row in gradient_rows]),
                a_la_gradient_norm_median=_median([row["a_la_gradient_norm"] for row in gradient_rows]),
                scalar_losses={name: float(result[name].mean()) for name in
                               ("hard_loss", "soft_loss", "weighted_edge_loss", "aligner_loss")},
                difficulty_distribution=_distribution(result["difficulty"]),
                q_weight_distribution=_distribution(result["s"]),
                soft_weight_mean=float(result["soft_weight"].mean()),
                epsilon_dominated_error_fraction=float(result["epsilon_dominated_error_fraction"].mean()),
                epsilon_dominated_definition="fraction of pixels with e_student<=1e-6 in the advantage denominator",
                teacher_per_band_l1=result["teacher_band_l1"].mean(0).tolist(),
                teacher_per_band_mse=result["teacher_band_mse"].mean(0).tolist(),
                teacher_per_band_bias=result["teacher_band_bias"].mean(0).tolist(),
                frozen_teacher_state_hash=teacher_hash,
                cache_online_q_max_abs_error=float(np.abs(result["teacher_q_online"] - result["teacher_q"]).max()),
                cache_online_q_matches=bool(np.allclose(result["teacher_q_online"], result["teacher_q"], atol=3e-6, rtol=2e-5)))
        else:
            summary["weighted_consistency_rec_a_gradient_ratio"] = _median([
                row["consistency_rec"]["norm_ratio"] for row in gradient_rows])
            summary["scalar_losses"] = dict(reconstruction_loss=float(result["reconstruction_loss"].mean()),
                weighted_consistency_gradient_subset_mean=float(np.mean([row["weighted_consistency"] for row in gradient_rows])))
            summary["a_gradient_norm_median"] = _median([row["a_gradient_norm"] for row in gradient_rows])
            summary["u_gradient_norm_median"] = _median([row["u_gradient_norm"] for row in gradient_rows])
        return summary, result
    finally:
        for module, mode in modes.items():
            module.training = mode
        restore_rng(rng)
