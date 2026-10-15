"""Offline, review-only native-scene PAN sampling-prior proposals.

No GT, training, files, or approval side effects. The local estimator is reused
from :mod:`align.estimator`; scene aggregation is a new, explicitly versioned
deterministic translation-consensus implementation, not a claim of bitwise
reproduction of the historical dense-field audit. Every threshold is supplied
and retained. A numerically good proposal still has ``accepted=False``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from align import estimator


ESTIMATOR_VERSION = "native_lms_scharr_zncc_census+weighted_translation_consensus/v1"
LOCAL_GATE_KEYS = frozenset(estimator.GATES)
CONFIG_KEYS = LOCAL_GATE_KEYS | frozenset({
    "window_size", "stride", "min_reference_std", "min_pan_std",
    "min_peak_zncc", "min_valid_windows", "min_valid_fraction",
    "consensus_tolerance", "min_inlier_fraction", "min_direction_concentration",
    "max_dispersion", "min_confidence", "min_prior_magnitude", "rationale",
})


def audited_hr_local_gates():
    """Return *partial* existing HR-audit settings, not a scene acceptance policy.

    The old audit used PAN as reference and often MTF-blurred PAN / bicubic MS;
    this helper reverses reference/moving and uses provided native LMS. These
    settings therefore require explicit review before use on each data source.
    Missing structural/window/aggregation decisions are deliberately not filled.
    """
    return {
        "gates": dict(estimator.GATES, search_int=4, max_magnitude=4.0),
        "evidence": ["align/estimator.py:GATES",
                     "tools/align_after_training_diag.py:HR_GATES"],
        "status": "historical_local_diagnostic_settings_not_scene_approval",
        "reference_difference": "provided native LMS mean is reference; raw native PAN is moving",
    }


def _json_copy(value, name):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON data") from exc


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_hash(array):
    h = hashlib.sha256()
    h.update(json.dumps({"shape": list(array.shape), "dtype": array.dtype.str},
                        sort_keys=True).encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric value")
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} is outside the finite permitted range")
    return float(value)


def validate_prior_config(config):
    """Validate and copy a complete explicit policy; no implicit numeric defaults."""
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    missing, unknown = CONFIG_KEYS - set(config), set(config) - CONFIG_KEYS
    if missing or unknown:
        raise ValueError(f"prior config missing={sorted(missing)}, unknown={sorted(unknown)}")
    c = _json_copy(dict(config), "config")
    for key in ("search_int", "window_size", "stride", "min_valid_windows"):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    # Census radius is two, and ZNCC needs at least sixteen masked interior pixels.
    if c["window_size"] < 2 * max(c["search_int"], 2) + 5:
        raise ValueError("window_size too small for estimator search/census support")
    if c["stride"] > c["window_size"]:
        raise ValueError("stride must not leave unexamined gaps between windows")
    for key in ("max_magnitude", "min_peak_margin", "consensus_tolerance",
                "max_dispersion", "min_prior_magnitude", "min_reference_std", "min_pan_std"):
        _finite_number(c[key], key, positive=True)
    _finite_number(c["max_primary_secondary_diff"], "max_primary_secondary_diff", nonnegative=True)
    for key in ("edge_top_frac", "min_peak_zncc", "min_valid_fraction",
                "min_inlier_fraction", "min_direction_concentration", "min_confidence"):
        if not 0 < _finite_number(c[key], key) <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if c["min_prior_magnitude"] > c["max_magnitude"]:
        raise ValueError("min_prior_magnitude exceeds max_magnitude")
    if not isinstance(c["rationale"], str) or not c["rationale"].strip():
        raise ValueError("rationale must document why this explicit policy is being proposed")
    return c


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    index = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2, side="left")
    return float(values[order[index]])


def _median2(vectors, weights):
    return np.asarray([_weighted_median(vectors[:, j], weights) for j in range(2)])


def _aggregate(vectors, weights, tolerance):
    """Exhaust all observed translation hypotheses plus the weighted median.

    This is the deterministic exhaustive analogue of translation RANSAC, with
    no random sampling or affine/dense-flow fit. Weighted support is primary;
    count, weighted error, then original hypothesis order break exact ties.
    Refit the winning support by coordinate weighted median and report the
    actual consensus distances to that final vector (not the seed distances).
    """
    hypotheses = np.vstack((_median2(vectors, weights), vectors))
    best = None
    for i, center in enumerate(hypotheses):
        distances = np.linalg.norm(vectors - center, axis=1)
        inliers = distances <= tolerance
        key = (float(weights[inliers].sum()), int(inliers.sum()),
               -float((weights[inliers] * distances[inliers]).sum()), -i)
        if best is None or key > best[0]:
            best = key, center, inliers
    _, seed, initial_inliers = best
    prior = _median2(vectors[initial_inliers], weights[initial_inliers])
    distances = np.linalg.norm(vectors - prior, axis=1)
    inliers = distances <= tolerance
    return prior, distances, inliers, seed


def _positions(size, window, stride):
    positions = list(range(0, size - window + 1, stride))
    if positions[-1] != size - window:
        positions.append(size - window)
    return positions


def _pair(value, name, *, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two dy/dx values")
    return [_finite_number(v, name, positive=positive) for v in value]


def propose_scene_prior(pan_hw, native_lms_chw, *, source_scene_id,
                        prior_grid_name, pixel_scale, origin_phase,
                        config, provenance):
    """Propose a canonical-grid native scene prior; never authorize it.

    Input arrays are observed PAN ``[H,W]`` and provided native LMS ``[C,H,W]``
    on the same native PAN grid. No resampling, GT, generated bicubic LMS, MTF
    modification, range normalization, or sign reversal happens here. Callers
    must document true source identity and LMS provenance independently.

    Sign: ``aligned_pan[y,x] = pan[y + dy, x + dx]``. Each estimator call uses
    ``ref=LMS.mean(axis=0), mov=PAN`` to obtain this sign directly. Its legacy
    ``*_lr_raw`` names are historical; all numbers here are native PAN pixels.
    """
    c = validate_prior_config(config)
    for value, name in ((source_scene_id, "source_scene_id"), (prior_grid_name, "prior_grid_name")):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty explicit identity")
    scale = _pair(pixel_scale, "pixel_scale", positive=True)
    phase = _pair(origin_phase, "origin_phase")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("provenance must document source identity and observed LMS origin")
    provenance = _json_copy(dict(provenance), "provenance")
    pan, lms = np.asarray(pan_hw), np.asarray(native_lms_chw)
    if pan.ndim != 2 or lms.ndim != 3 or not lms.shape[0] or pan.shape != lms.shape[1:]:
        raise ValueError("require native PAN [H,W] and provided native LMS [C,H,W] on identical grids")
    if min(pan.shape) < c["window_size"]:
        raise ValueError("native scene is smaller than window_size")
    for array, name in ((pan, "PAN"), (lms, "native LMS")):
        if array.dtype.kind not in "uif" or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite real numeric array")
    inputs = {"pan_sha256": _array_hash(pan), "native_lms_sha256": _array_hash(lms),
              "pan_shape": list(pan.shape), "native_lms_shape": list(lms.shape),
              "pan_dtype": pan.dtype.str, "native_lms_dtype": lms.dtype.str}
    reference = lms.astype(np.float64).mean(axis=0).astype(np.float32)
    moving = pan.astype(np.float32)
    if not np.isfinite(reference).all() or not np.isfinite(moving).all():
        raise ValueError("estimator float32 conversion produced nonfinite input")
    gates = {key: c[key] for key in sorted(LOCAL_GATE_KEYS)}
    windows = []
    size = c["window_size"]
    for y in _positions(pan.shape[0], size, c["stride"]):
        for x in _positions(pan.shape[1], size, c["stride"]):
            ref, mov = reference[y:y + size, x:x + size], moving[y:y + size, x:x + size]
            ref_std, pan_std = float(ref.std(dtype=np.float64)), float(mov.std(dtype=np.float64))
            reasons = []
            if ref_std < c["min_reference_std"]:
                reasons.append("reference_texture_below_minimum")
            if pan_std < c["min_pan_std"]:
                reasons.append("pan_texture_below_minimum")
            raw = None
            weight = 0.0
            if not reasons:
                raw = estimator.estimate_shift(ref, mov, gates)
                raw = _json_copy(raw, "local estimator output")
                if raw["boundary_hit"]:
                    reasons.append("search_boundary_hit")
                if raw["primary_secondary_diff"] > c["max_primary_secondary_diff"]:
                    reasons.append("primary_census_disagreement")
                if raw["peak_margin"] < c["min_peak_margin"]:
                    reasons.append("ambiguous_zncc_peak")
                if raw["magnitude_raw"] > c["max_magnitude"]:
                    reasons.append("shift_exceeds_expected_range")
                if raw["peak_zncc"] < c["min_peak_zncc"]:
                    reasons.append("peak_zncc_below_minimum")
                if not raw["accepted"] and not reasons:
                    reasons.append("local_estimator_rejected")
                if not reasons:
                    weight = float(raw["peak_margin"] * max(raw["peak_zncc"], 0.0))
                    if not math.isfinite(weight) or weight <= 0:
                        reasons.append("nonpositive_or_nonfinite_confidence_weight")
                        weight = 0.0
            windows.append({"index": len(windows), "crop_yxhw": [y, x, size, size],
                            "reference_std": ref_std, "pan_std": pan_std,
                            "reference_range": [float(ref.min()), float(ref.max())],
                            "pan_range": [float(mov.min()), float(mov.max())],
                            "estimator": raw, "valid": not reasons, "gate_reasons": reasons,
                            "weight": weight, "consensus_inlier": False,
                            "consensus_gate_reasons": [], "distance_to_proposal": None})
    valid = [w for w in windows if w["valid"]]
    valid_fraction = len(valid) / len(windows)
    reasons = []
    if len(valid) < c["min_valid_windows"]:
        reasons.append("too_few_valid_windows")
    if valid_fraction < c["min_valid_fraction"]:
        reasons.append("valid_window_fraction_below_minimum")
    prior, seed, dispersion = None, None, None
    inlier_fraction = concentration = confidence = 0.0
    if valid:
        vectors = np.asarray([[w["estimator"]["dy_lr_raw"], w["estimator"]["dx_lr_raw"]] for w in valid])
        weights = np.asarray([w["weight"] for w in valid])
        prior, distances, inliers, seed = _aggregate(vectors, weights, c["consensus_tolerance"])
        for window, distance, inlier in zip(valid, distances, inliers):
            window["distance_to_proposal"] = float(distance)
            window["consensus_inlier"] = bool(inlier)
            if not inlier:
                window["consensus_gate_reasons"].append("translation_consensus_outlier")
        inlier_fraction = float(inliers.mean())
        active = vectors[inliers]
        active_weights = weights[inliers]
        if not inliers.any():
            reasons.append("no_inliers_after_weighted_median_refit")
        else:
            norms = np.linalg.norm(active, axis=1)
            units = np.divide(active, norms[:, None], out=np.zeros_like(active),
                              where=norms[:, None] > np.finfo(np.float64).tiny)
            concentration = float(np.clip(np.linalg.norm((units * active_weights[:, None]).sum(axis=0)
                                                         / active_weights.sum()), 0, 1))
            dispersion = _weighted_median(distances[inliers], active_weights)
            peaks = np.asarray([w["estimator"]["peak_zncc"] for w in valid])[inliers]
            confidence = float(np.clip(valid_fraction * inlier_fraction * concentration
                                       * np.average(np.clip(peaks, 0, 1), weights=active_weights), 0, 1))
        if float(np.linalg.norm(prior)) < c["min_prior_magnitude"]:
            reasons.append("zero_or_near_zero_prior_no_fallback")
        if float(np.linalg.norm(prior)) > c["max_magnitude"]:
            reasons.append("aggregate_shift_exceeds_expected_range")
        if inlier_fraction < c["min_inlier_fraction"]:
            reasons.append("insufficient_translation_consensus")
        if concentration < c["min_direction_concentration"]:
            reasons.append("direction_concentration_below_minimum")
        if dispersion is not None and dispersion > c["max_dispersion"]:
            reasons.append("dispersion_above_maximum")
    if confidence < c["min_confidence"]:
        reasons.append("confidence_below_minimum")
    result = {
        "schema": "teacher48_scene_prior_proposal_v1", "source_scene_id": source_scene_id,
        "prior_shift_dy_dx": None if prior is None else prior.tolist(),
        "prior_grid_name": prior_grid_name, "pixel_scale": scale, "origin_phase": phase,
        "n_windows": len(windows), "n_valid": len(valid), "valid_fraction": valid_fraction,
        "inlier_fraction": inlier_fraction, "direction_concentration": concentration,
        "dispersion": dispersion, "confidence": confidence,
        "confidence_definition": "valid_fraction * count_inlier_fraction * weighted_direction_concentration * weighted_mean_clipped_peak_zncc; diagnostic, not calibrated probability",
        "weight_definition": "positive local peak_margin * max(peak_zncc, 0)",
        "estimator_version": ESTIMATOR_VERSION,
        "estimator_source_sha256": hashlib.sha256(Path(estimator.__file__).read_bytes()).hexdigest(),
        "proposal_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_versions": {"numpy": np.__version__, "opencv": cv2.__version__},
        "reference_type": "provided_native_lms_bandmean", "scene_input_hash": _hash_json(inputs),
        "input_evidence": inputs,
        "mapping_rule": "ref=native LMS bandmean; mov=native PAN; aligned[y,x]=PAN[y+dy,x+dx]; raw native PAN pixel units; no inversion or scale conversion",
        "accepted": False, "acceptance_evidence": "not reviewed; offline proposal only",
        "ready_for_review": not reasons, "gate_reasons": reasons,
        "config": c, "config_sha256": _hash_json(c), "provenance": provenance,
        "consensus_seed_dy_dx": None if seed is None else seed.tolist(),
        "aggregation": "exhaustive local-vector hypotheses plus coordinate weighted median; maximum weighted translation consensus; coordinate weighted-median refit; final-distance inlier check",
        "windows": windows,
    }
    return _json_copy(result, "scene proposal")
