"""Fixed numeric-only EdgeRes probes; never checkpoint qualification.

Every intervention is DIAG_ONLY. Input IDs and budgets are pinned in advance;
no data-dependent sample skipping, training update, image dump or live RNG
consumption is permitted. Normal and cropped-CNN clean-support domains remain
separate, with identical denominator for edge/residual/total response gains.
"""
from __future__ import annotations

import copy
import json
import math
import time

import numpy as np
import torch

from pa.offset import sample_offsets
from pa.warp import warp_pan, warp_support_mask, two_stage_support_mask
from .evaluation import (metric_functions_for, metric_values, model_state_hash,
                         paused_model)
from panda_method.calibration import canonical_sha256


PROTOCOL = "EDGERES48_NUMERIC_PROBES_V2"
PROBE_SEED = 261013900


def axis16():
    return [{"radius": r, "axis": axis, "epsilon_dy_dx": [r * y, r * x],
             "offset_type": "integer" if r in (1., 2.) else "fractional"}
            for r in (.25, .5, 1., 2.)
            for axis, y, x in (("+y", 1, 0), ("-y", -1, 0), ("+x", 0, 1), ("-x", 0, -1))]


def response_statistics(rows):
    if not rows:
        raise ValueError("Response statistics need observed rows, never fabricated zeros")
    eps = np.asarray([r["epsilon_dy_dx"] for r in rows], dtype=np.float64)
    db = np.asarray([r["db"] for r in rows], dtype=np.float64)
    dr = np.asarray([r["dr"] for r in rows], dtype=np.float64)
    du = np.asarray([r["du"] for r in rows], dtype=np.float64)
    if any(a.shape != (len(rows), 2) or not np.isfinite(a).all() for a in (eps, db, dr, du)):
        raise ValueError("Finite equal Nx2 epsilon/edge/residual/total responses required")
    if not np.allclose(db + dr, du, atol=2e-5, rtol=2e-6):
        raise ValueError("Total response is not the edge+residual decomposition")
    denominator = float(np.square(eps).sum())
    if denominator <= 0:
        raise ValueError("Response gain needs a nonzero fixed epsilon bundle")
    edge_error, total_error = db + eps, du + eps
    out = {"n": len(rows), "gain_denominator": denominator,
           "g_edge": float(-(db * eps).sum() / denominator),
           "g_residual": float(-(dr * eps).sum() / denominator),
           "g_total": float(-(du * eps).sum() / denominator),
           "EPE_edge": float(np.linalg.norm(edge_error, axis=1).mean()),
           "EPE_total": float(np.linalg.norm(total_error, axis=1).mean()),
           "MAE_edge": float(np.abs(edge_error).mean()), "MAE_total": float(np.abs(total_error).mean()),
           "MAE_edge_dy_dx": np.abs(edge_error).mean(0).tolist(),
           "MAE_total_dy_dx": np.abs(total_error).mean(0).tolist()}
    out["EPE_improvement_residual"] = out["EPE_edge"] - out["EPE_total"]
    out["gain_additivity_error"] = out["g_total"] - out["g_edge"] - out["g_residual"]
    return out


def response_summary(rows):
    return {"all": response_statistics(rows),
            "by_offset_type": {kind: response_statistics([r for r in rows if r["offset_type"] == kind])
                               for kind in ("integer", "fractional")
                               if any(r["offset_type"] == kind for r in rows)},
            "by_radius": {str(radius): response_statistics([r for r in rows if r["radius"] == radius])
                          for radius in (.25, .5, 1., 2.) if any(r["radius"] == radius for r in rows)}}


def vector_statistics(rows):
    result = {}
    for key in ("b", "delta", "u"):
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        norms = np.linalg.norm(values, axis=1)
        result[key] = {"n": len(values), "mean_dy_dx": values.mean(0).tolist(),
            "sample_sd_dy_dx": values.std(0, ddof=1).tolist() if len(values) > 1 else None,
            "norm_mean": float(norms.mean()),
            "norm_sample_sd": float(norms.std(ddof=1)) if len(norms) > 1 else None,
            "norm_p50": float(np.quantile(norms, .5)), "norm_p95": float(np.quantile(norms, .95))}
    cosine = [r["cos_b_delta"] for r in rows if r["cos_b_delta"] is not None]
    result["cos_b_delta"] = {"mean": float(np.mean(cosine)) if cosine else None,
                             "defined_count": len(cosine), "zero_vector_NA": len(rows) - len(cosine)}
    result["boundary_hit_rate"] = float(np.mean([r["boundary_hit"] for r in rows]))
    result["tie_rate"] = float(np.mean([r["tie_count"] > 1 for r in rows]))
    result["flat_score_rate"] = float(np.mean([r["score_range"][0] == r["score_range"][1] for r in rows]))
    result["low_texture_rate"] = float(np.mean([r["low_texture"] for r in rows]))
    result["mean_warp_valid_fraction"] = float(np.mean([r["warp_valid_fraction"] for r in rows]))
    result["residual_opposes_prior_rate"] = float(np.mean(np.asarray(cosine) < 0)) if cosine else None
    result["discrepancy_roles"] = {}
    for reference in ("RGB_LMS", "RGB_GT"):
        observed = [r for r in rows if reference in r["discrepancy"]["ZERO"]]
        if not observed:
            continue
        harmed = [r for r in observed if r["discrepancy"]["PRIOR_ONLY"][reference] > r["discrepancy"]["ZERO"][reference]]
        recovered = [r for r in harmed if r["discrepancy"]["CURRENT"][reference] < r["discrepancy"]["PRIOR_ONLY"][reference]]
        result["discrepancy_roles"][reference] = {"n": len(observed),
            "prior_worse_than_zero_count": len(harmed), "prior_worse_than_zero_rate": len(harmed)/len(observed),
            "current_improves_harmful_prior_count": len(recovered),
            "current_improves_harmful_prior_rate": len(recovered)/len(harmed) if harmed else None,
            "comparison": "strict measured Scharr inequality, not physical displacement correctness"}
    return result


def validate_probe_spec(spec):
    required = {"rows", "gradient_rows", "gradient_microbatch", "probe_seed", "response_domains",
                "clean_cnn_view_margin", "counterfactual_metrics", "max_wall_seconds"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("Explicit fixed probe specification required: " + ",".join(sorted(required)))
    if set(spec["rows"]) != {"train", "val", "rr", "fr"}:
        raise ValueError("Train/validation/RR/FR probe IDs must be separately fixed")
    for group in list(spec["rows"].values()) + list(spec["gradient_rows"]):
        if (not isinstance(group, list) or not group or len(set(group)) != len(group)
                or any(type(i) is not int or i < 0 for i in group)):
            raise ValueError("Each probe/effective gradient batch needs unique explicit H5 rows")
    for split in ("rr", "fr"):
        if max(spec["rows"][split]) >= 20:
            raise ValueError("Evaluation probe is a fixed subset of the official20")
    if (type(spec["gradient_microbatch"]) is not int or spec["gradient_microbatch"] < 1
            or type(spec["probe_seed"]) is not int or type(spec["clean_cnn_view_margin"]) is not int
            or spec["clean_cnn_view_margin"] < 0):
        raise ValueError("Explicit finite probe seed/microbatch/crop required")
    if (not spec["response_domains"] or len(set(spec["response_domains"])) != len(spec["response_domains"])
            or set(spec["response_domains"]) - {"normal", "clean_support"}
            or spec["counterfactual_metrics"] != "official_subset"):
        raise ValueError("Pinned response domains and official-subset DIAG_ONLY policy required")
    if type(spec["max_wall_seconds"]) not in (int, float) or not math.isfinite(spec["max_wall_seconds"]) or spec["max_wall_seconds"] <= 0:
        raise ValueError("An explicit positive diagnostic wall budget is required")
    return copy.deepcopy(spec)


def _check_deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("DIAGNOSTIC_BUDGET_EXHAUSTED")


def _clean_shift(stage, aligner, pan, lms, resolution, *, epsilon=None):
    mask = None if epsilon is None else warp_support_mask(*pan.shape[-2:], epsilon)
    match = stage.matcher(pan, lms, resolution=resolution, pan_valid_mask=mask)
    b = match.b
    support = (warp_support_mask(*pan.shape[-2:], b) if epsilon is None
               else two_stage_support_mask(*pan.shape[-2:], epsilon, b))
    margin = aligner.view_margin
    h, w = pan.shape[-2:]
    if min(h, w) <= 2 * margin or not bool(support[..., margin:h-margin, margin:w-margin].all()):
        raise ValueError("CLEAN_SUPPORT_UNAVAILABLE: fixed diagnostic CNN view exceeds original-pixel support")
    delta = aligner(warp_pan(pan, b), lms)
    return {"b": b, "delta": delta, "u": b + delta}


def _gradient_vectors(stage, batch, epsilon):
    batch.validate(stage.teacher.spec.bands, training=True, require_gt=True)
    stage.sources.check(batch)
    native = stage.native_prediction(batch)
    rec = (native["y"] - batch.gt.detach()).abs().mean()
    absolute = stage.matcher.edges.absolute_loss(native["pan_aligned"], batch.gt.detach(), roi_margin=stage.roi_margin)
    stage._native_support(batch.pan, native["u"])
    with torch.no_grad():
        shifted_pan = warp_pan(batch.pan, epsilon)
        mask = warp_support_mask(*batch.pan.shape[-2:], epsilon)
    shifted = stage.predict_shift(shifted_pan, batch.native_lms, resolution="RR", pan_valid_mask=mask)
    eps_loss = (shifted["u"] + epsilon - native["u"].detach()).abs().mean()
    params = list(stage.aligner.parameters())
    vectors, losses = {}, {"rec": rec, "abs": absolute, "eps": eps_loss}
    for name, loss in losses.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
                                   for p, g in zip(params, grads)]).double()
    return vectors, {k: float(v.detach()) for k, v in losses.items()}


def _gradient_summary(vectors, stage):
    scales = {"rec": 1., "abs": stage.lambda_abs, "eps": stage.lambda_eps}
    groups = {"whole": [], "fc2": [], "fc1": [], "body": [], "stem": []}
    start = 0
    for name, p in stage.aligner.named_parameters():
        idx = torch.arange(start, start+p.numel(), device=vectors["rec"].device)
        groups["whole"].append(idx)
        key = "fc2" if name.startswith("fc2.") else "fc1" if name.startswith("fc1.") else "stem" if "stem." in name else "body"
        groups[key].append(idx); start += p.numel()
    output = {}
    for group, parts in groups.items():
        indices = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=vectors["rec"].device)
        sliced = {key: value[indices] for key, value in vectors.items()}
        raw = {key: float(value.norm()) for key, value in sliced.items()}
        weighted = {key: value * scales[key] for key, value in raw.items()}
        cosine = {}
        for first, second in (("rec", "abs"), ("rec", "eps"), ("abs", "eps")):
            den = raw[first] * raw[second]
            cosine[first+":"+second] = float(torch.dot(sliced[first], sliced[second]) / den) if den > 0 else None
        output[group] = {"n_parameters": len(indices), "raw_norm": raw, "weighted_norm": weighted,
                         "weighted_to_rec_ratio": {k: v / weighted["rec"] if weighted["rec"] > 0 else None
                                                   for k, v in weighted.items()}, "cosine": cosine}
    return output


def gradient_probe(trainer, sources, *, rows, microbatch, seed, device="cpu", deadline_unix=None):
    """Autograd.grad on pinned RR training observations; no .backward or step."""
    stage = trainer.stage
    before = model_state_hash(stage)
    if type(microbatch) is not int or microbatch < 1 or not rows or any(not row for row in rows):
        raise ValueError("Explicit nonempty effective rows and positive microbatch required")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    completed, combined, total_samples = [], None, 0
    with paused_model(trainer, gradients=True):
        for effective_rows in rows:
            _check_deadline(deadline_unix)
            eps = sample_offsets(len(effective_rows), 2., generator).to(device)
            aggregate, loss_sum = None, {"rec": 0., "abs": 0., "eps": 0.}
            for start in range(0, len(effective_rows), microbatch):
                _check_deadline(deadline_unix)
                selected = effective_rows[start:start+microbatch]
                batch = sources.batch("train", selected, device=device)
                vectors, losses = _gradient_vectors(stage, batch, eps[start:start+len(selected)])
                weight = len(selected) / len(effective_rows)
                if aggregate is None:
                    aggregate = {k: v * weight for k, v in vectors.items()}
                else:
                    for k, value in vectors.items():
                        aggregate[k] += value * weight
                for k, value in losses.items():
                    loss_sum[k] += value * weight
            n = len(effective_rows)
            if combined is None:
                combined = {k: v * n for k, v in aggregate.items()}
            else:
                for k, value in aggregate.items():
                    combined[k] += value * n
            total_samples += n
            completed.append({"rows": effective_rows, "n_samples": n, "raw_loss": loss_sum,
                              "epsilon_dy_dx": eps.cpu().tolist(), "scopes": _gradient_summary(aggregate, stage)})
    if combined is None:
        raise ValueError("At least one full effective RR gradient probe batch is required")
    combined = {k: v / total_samples for k, v in combined.items()}
    return {"measurement_domain": "DIAG_ONLY_RR_TRAIN_FIXED_EFFECTIVE_BATCHES",
            "model_state_sha256": before, "completed_update": trainer.updates,
            "train_source_manifest_sha256": sources.manifest_sha("train"),
            "completed_effective_batches": len(completed), "n_samples": total_samples,
            "microbatch": microbatch, "probe_seed": seed,
            "reduction": "gradient_of_sample_mean_over_all_registered_effective_batches",
            "coefficients": {"rec": 1., "abs": stage.lambda_abs, "eps": stage.lambda_eps},
            "scopes": _gradient_summary(combined, stage), "batches": completed,
            "optimizer_updates": 0, "uses_evaluation_inputs": False}


def run_probes(trainer, sources, probe_spec, *, device="cpu", evaluator_pin=None,
               metric_functions=None, deadline_unix=None):
    spec = validate_probe_spec(probe_spec)
    functions = metric_functions or metric_functions_for(sources, evaluator_pin)
    started = time.monotonic()
    deadline = min(time.time()+spec["max_wall_seconds"], deadline_unix or float("inf"))
    stage = trainer.stage
    clean_aligner = None
    if "clean_support" in spec["response_domains"]:
        clean_aligner = copy.deepcopy(stage.aligner).eval()
        clean_aligner.view_margin = spec["clean_cnn_view_margin"]
        for parameter in clean_aligner.parameters():
            parameter.requires_grad_(False)
    result = {"protocol": PROTOCOL, "record_type": "DIAGNOSTICS_NUMERIC", "status": "RUNNING",
              "completed_update": trainer.updates, "model_state_sha256": model_state_hash(stage),
              "fixed_probe_spec": spec, "measurement_domain": "DIAG_ONLY",
              "official_selection_input": False, "full_method_eligible": False,
              "images_written": False, "response": {}, "geometry": {}, "counterfactual": [],
              "edge_protocol_digest": canonical_sha256(stage.matcher.manifest()),
              "source_manifest_sha256": {split: sources.manifest_sha(split) for split in spec["rows"]},
              "metric_function_injection": metric_functions is not None,
              "physical_displacement_ground_truth": False,
              "clean_support_definition": "same EdgeMatch; frozen copied CNN uses declared fixed inner view only; diagnostic objective not official/native pipeline"}
    try:
        # Gradient/initialization evidence is collected first, so expensive FR
        # response or official-formula subsets cannot consume its entire budget.
        result["gradient"] = gradient_probe(trainer, sources, rows=spec["gradient_rows"],
            microbatch=spec["gradient_microbatch"], seed=spec["probe_seed"], device=device, deadline_unix=deadline)
        with paused_model(trainer):
            for split, indices in spec["rows"].items():
                per_geometry, per_response = [], {domain: [] for domain in spec["response_domains"]}
                result["response"][split] = per_response
                result["geometry"][split] = {"rows": per_geometry}
                for index in indices:
                    _check_deadline(deadline)
                    batch = sources.batch(split, [index], device=device)
                    native = stage.predict_shift(batch.pan, batch.native_lms, resolution=batch.lineage.resolution)
                    b, delta, u = (native[k][0].cpu().tolist() for k in ("b", "delta", "u"))
                    den = float(np.linalg.norm(b) * np.linalg.norm(delta))
                    cosine = float(np.dot(b, delta) / den) if den > 0 else None
                    scores = native["match_diagnostics"]["scores"][0]
                    sorted_scores = scores.sort().values
                    candidate0 = (native["match_diagnostics"]["candidates"] == 0).all(1)
                    geom = {"sample_id": batch.lineage.sample_ids[0], "b": b, "delta": delta, "u": u,
                            "cos_b_delta": cosine, "best_score": float(sorted_scores[0]),
                            "zero_score": float(scores[candidate0][0]),
                            "score_gap": float(sorted_scores[1]-sorted_scores[0]) if len(scores)>1 else None,
                            "score_range": [float(scores.min()), float(scores.max())],
                            "tie_count": int(native["match_diagnostics"]["tie_count"][0]),
                            "boundary_hit": bool(native["match_diagnostics"]["boundary_hit"][0]),
                            "low_texture": bool(native["match_diagnostics"]["low_texture"][0]),
                            "edge_amplitude_pan": float(native["match_diagnostics"]["texture_pan"][0]),
                            "edge_amplitude_lms": float(native["match_diagnostics"]["texture_lms"][0]),
                            "warp_valid_fraction": float(warp_support_mask(*batch.pan.shape[-2:], native["u"]).float().mean()),
                            "fixed_discrepancy_roi_margin": stage.roi_margin, "discrepancy": {}}
                    if batch.gt is not None:
                        gt_edges = stage.matcher.edges.descriptor(stage.matcher.edges.rgb_luminance(batch.gt))
                        margin = stage.roi_margin
                        geom["edge_amplitude_gt_fixed_roi"] = float(gt_edges[..., margin:gt_edges.shape[-2]-margin,
                                                                                margin:gt_edges.shape[-1]-margin].mean())
                    for mode, shift in (("ZERO", torch.zeros_like(native["u"])), ("PRIOR_ONLY", native["b"]), ("CURRENT", native["u"])):
                        corrected = warp_pan(batch.pan, shift)
                        values = {"RGB_LMS": float(stage.matcher.edges.absolute_loss(corrected, batch.native_lms, roi_margin=stage.roi_margin))}
                        if batch.gt is not None:
                            values["RGB_GT"] = float(stage.matcher.edges.absolute_loss(corrected, batch.gt, roi_margin=stage.roi_margin))
                        geom["discrepancy"][mode] = values
                        if split in ("rr", "fr"):
                            prediction = stage.teacher(corrected, batch.ms, batch.native_lms)["y"]
                            raw = ((prediction[0]+1)*(sources.peak/2)).cpu().numpy()
                            arrays = {k: v[0] for k, v in sources.raw(split, [index]).items()}
                            metrics = metric_values(functions[split](raw, arrays), split, sources.bands)
                            result["counterfactual"].append({"sample_id": batch.lineage.sample_ids[0],
                                "split": split, "mode": mode, "metrics": metrics,
                                "measurement_domain": "DIAG_ONLY_OFFICIAL_FORMULA_SUBSET_NOT_FULL20",
                                "full_method_eligible": False, "retrained_ablation": False,
                                "teacher_state_fixed": True})
                    per_geometry.append(geom)
                    base = {"normal": native}
                    if clean_aligner is not None:
                        base["clean_support"] = _clean_shift(stage, clean_aligner, batch.pan, batch.native_lms, batch.lineage.resolution)
                    for point in axis16():
                        _check_deadline(deadline)
                        eps = batch.pan.new_tensor(point["epsilon_dy_dx"])[None]
                        pan_eps = warp_pan(batch.pan, eps)
                        support = warp_support_mask(*batch.pan.shape[-2:], eps)
                        for domain in spec["response_domains"]:
                            shifted = (stage.predict_shift(pan_eps, batch.native_lms,
                                resolution=batch.lineage.resolution, pan_valid_mask=support) if domain == "normal"
                                else _clean_shift(stage, clean_aligner, pan_eps, batch.native_lms, batch.lineage.resolution, epsilon=eps))
                            changes = {key: (shifted[key]-base[domain][key])[0].cpu().tolist() for key in ("b", "delta", "u")}
                            per_response[domain].append({**point, "sample_id": batch.lineage.sample_ids[0],
                                "db": changes["b"], "dr": changes["delta"], "du": changes["u"],
                                "measurement_domain": domain,
                                "synthetic_warp_valid_fraction": float(support.float().mean())})
                result["geometry"][split]["summary"] = vector_statistics(per_geometry)
                result["response"][split] = {domain: {"rows": rows, "summary": response_summary(rows),
                    "n_inputs": len(indices), "n_offsets_per_input": 16} for domain, rows in per_response.items()}
        result["status"] = "COMPLETE"
    except TimeoutError as exc:
        result.update(status="PARTIAL_BUDGET", incomplete_reason=str(exc))
    result["elapsed_seconds"] = time.monotonic() - started
    result["model_state_sha256_after"] = model_state_hash(stage)
    json.dumps(result, allow_nan=False)
    return result


def benchmark_inference(trainer, sources, *, device, warmup, repeats, rows,
                        deadline_unix=None):
    """Synchronized batch1 native timing; no image writes or summed component total.

    The stored-LMS acquisition/H2D path is reported separately. Infer(ms) is
    the measured resident-observation full pipeline, including a fresh uncached
    EdgeMatch on every call, prewarp, CNN, final warp, MS-base formation and T.
    No FLOPs value is invented for integer search or copied from an old model.
    """
    if (type(warmup) is not int or warmup < 0 or type(repeats) is not int or repeats < 1
            or set(rows) != {"rr", "fr"} or any(type(v) is not int or not 0 <= v < 20 for v in rows.values())):
        raise ValueError("Pin nonnegative warmup, positive repeats and one official row per grid")
    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
    def measure(call):
        for _ in range(warmup):
            _check_deadline(deadline_unix); call()
        values = []
        for _ in range(repeats):
            _check_deadline(deadline_unix); sync(); tick = time.perf_counter(); call(); sync()
            values.append((time.perf_counter()-tick)*1000)
        return {"median_ms": float(np.median(values)), "mean_ms": float(np.mean(values)),
                "p90_ms": float(np.quantile(values, .9)), "samples_ms": values}
    stage, measurements = trainer.stage, {}
    with paused_model(trainer):
        for split, row in rows.items():
            sync(); tick = time.perf_counter()
            batch = sources.batch(split, [row], device=device)
            sync(); load_ms = (time.perf_counter()-tick)*1000
            sync(); tick = time.perf_counter()
            native = stage.native_prediction(batch)
            sync(); first_native_ms = (time.perf_counter()-tick)*1000
            corrected = warp_pan(batch.pan, native["u"])
            measurements[split] = {
                "sample_id": batch.lineage.sample_ids[0], "shape": list(batch.pan.shape),
                "observation_read_normalize_H2D_ms": load_ms,
                "first_call_in_probe_ms": first_native_ms,
                "full_pipeline": measure(lambda: stage.native_prediction(batch)),
                "EdgeMatch_only": measure(lambda: stage.matcher(batch.pan, batch.native_lms,
                                                       resolution=batch.lineage.resolution)),
                "A_only_prepared_inputs": measure(lambda: stage.aligner(native["pan_pre"], batch.native_lms)),
                "T_only_prepared_pan": measure(lambda: stage.teacher(corrected, batch.ms, batch.native_lms)),
                "prewarp_only": measure(lambda: warp_pan(batch.pan, native["b"])),
                "final_warp_only": measure(lambda: warp_pan(batch.pan, native["u"])),
            }
    return {"record_type": "COST", "measurement_domain": "DIAG_ONLY_NATIVE_INFERENCE_TIMING",
            "model_state_sha256": model_state_hash(stage), "device": str(device),
            "torch": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
            "precision": "FP32", "warmup": warmup, "repeats": repeats, "batch": 1,
            "cache": "no_edge_match_result_cache; every call recomputes matching",
            "cold_start_claim": "none: first_call_in_probe is reported, not a fresh process/device cold start",
            "Infer_ms_scope": "resident stored-LMS observations; acquisition+H2D separately reported; total measured not component sum",
            "observations": measurements,
            "trainable_params": sum(p.numel() for m in (stage.aligner, stage.teacher) for p in m.parameters() if p.requires_grad),
            "edge_match_params": sum(p.numel() for p in stage.matcher.parameters()),
            "Mem_MB": None,
            "Mem_reason": "isolated inference allocation not measured in paused live training; do not substitute process high-water memory",
            "FLOPs": None, "FLOPs_reason": "not measured; integer search/reduction requires explicit non-convolution coverage"}
