"""Train-only 64-square provided-LMS dataset prior preprocessing.

This module does not load GT/MS, run an optimizer, guess source-scene IDs, fit
test inputs, or provide unapproved sensor thresholds. ``propose_train_prior``
visits EVERY registered training row. The O(N**2) consensus intentionally has
no hidden subsampling cap; large real scans are explicit offline work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import cv2
import h5py
import numpy as np

from align import estimator
from panda_method.global_prior import (
    AGGREGATION_METHOD, DIRECTION, GlobalPriorAsset, LOCAL_GATE_KEYS,
    SAMPLING_EQUATION, SCHEMA, SENSOR_BANDS, canonical_sha256, json_copy,
    require_sha256, validate_global_settings,
)
from teacher48 import prepare_prior


SOURCE_SCHEMA = "PANDA_B_GLOBAL_RR_TRAIN_SOURCE_V1"
PROPOSAL_SCHEMA = "PANDA_B_GLOBAL_RR_PROPOSAL_V1"
REVIEW_SCHEMA = "PANDA_B_GLOBAL_RR_POLICY_REVIEW_V1"
SOURCE_KEYS = frozenset({"schema", "sensor", "path", "sha256", "split", "rows",
                         "pan_key", "lms_key", "band_order", "band_order_evidence",
                         "lms_provenance", "radiometry", "normalization", "split_evidence"})


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stamp(path):
    st = Path(path).stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " requires explicit nonempty provenance")


def validate_train_source(source):
    s = json_copy(source)
    if not isinstance(s, dict) or set(s) != SOURCE_KEYS or s.get("schema") != SOURCE_SCHEMA:
        raise ValueError("Explicit PANDA_B_GLOBAL_RR_TRAIN_SOURCE_V1 registration required; keys: "
                         + ", ".join(sorted(SOURCE_KEYS)))
    if s["sensor"] not in SENSOR_BANDS or s["split"] != "train":
        raise ValueError("Only one registered WV3/QB/GF2 train split is permitted")
    if type(s["rows"]) is not int or s["rows"] < 1:
        raise ValueError("Actual full training row count is required")
    if s["pan_key"] != "pan" or s["lms_key"] != "lms":
        raise ValueError("This adapter accesses only the actual pan/lms datasets")
    if s["radiometry"] != "stored_raw_dn" or s["normalization"] != "none":
        raise ValueError("No unrecorded normalization or radiometric conversion is permitted")
    for key in ("path", "band_order_evidence", "lms_provenance", "split_evidence"):
        _nonempty(s[key], key)
    require_sha256(s["sha256"], "train source hash")
    bands = s["band_order"]
    if (not isinstance(bands, list) or len(bands) != SENSOR_BANDS[s["sensor"]]
            or not all(isinstance(b, str) and b.strip() for b in bands)
            or len(set(bands)) != len(bands)):
        raise ValueError("Explicit unique sensor band order is required")
    # Filename is only a leakage tripwire; source registration/hash are still
    # required, and declarations cannot establish geographic independence.
    name = Path(s["path"]).name.lower()
    if not name.startswith("train_") or any(word in name for word in ("valid", "test", "origscale")):
        raise ValueError("Registered source must identify a train_ H5, never val/RR/FR evaluation")
    return s


def match_patch(pan_hw, provided_lms_chw, settings):
    """Signed raw PAN->provided-LMS matching on the unchanged RR PAN grid.

    The historical estimator calls its fields *_lr_raw. Here both inputs are
    PAN-grid 64-square arrays, so those fields are RR-PAN pixels: no x4 factor,
    inverse sign, bicubic LMS, interpolation phase, or applied-zero is used.
    """
    c = validate_global_settings(settings)
    pan, lms = np.asarray(pan_hw), np.asarray(provided_lms_chw)
    if pan.shape != (64, 64) or lms.ndim != 3 or lms.shape[1:] != (64, 64) or lms.shape[0] not in (4, 8):
        raise ValueError("PAN [64,64] and supplied LMS [4|8,64,64] are required")
    for array in (pan, lms):
        if array.dtype.kind not in "uif" or not np.isfinite(array).all():
            raise ValueError("Finite real PAN and provided LMS required")
    reference = lms.astype(np.float64).mean(axis=0).astype(np.float32)
    moving = pan.astype(np.float32)
    if not np.isfinite(reference).all() or not np.isfinite(moving).all():
        raise ValueError("Float32 estimator conversion overflowed")
    ref_std = float(reference.std(dtype=np.float64))
    pan_std = float(moving.std(dtype=np.float64))
    reasons, raw, weight = [], None, 0.
    if ref_std < c["min_reference_std"]:
        reasons.append("reference_texture_below_minimum")
    if pan_std < c["min_pan_std"]:
        reasons.append("pan_texture_below_minimum")
    if not reasons:
        raw = json_copy(estimator.estimate_shift(reference, moving,
                        {key: c[key] for key in sorted(LOCAL_GATE_KEYS)}))
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
            weight = float(raw["peak_margin"] * max(raw["peak_zncc"], 0.))
            if not math.isfinite(weight) or weight <= 0:
                reasons.append("nonpositive_or_nonfinite_confidence_weight")
                weight = 0.
    return json_copy({
        "candidate_dy_dx": None if raw is None else [raw["dy_lr_raw"], raw["dx_lr_raw"]],
        "candidate_direction": DIRECTION, "output_direction": DIRECTION,
        "candidate_grid": "PAN_RR", "sampling_equation": SAMPLING_EQUATION,
        "reference_std": ref_std, "pan_std": pan_std,
        "pan_sha256": prepare_prior._array_hash(pan),
        "provided_lms_sha256": prepare_prior._array_hash(lms),
        "raw_estimator": raw, "confidence_weight": weight,
        "weight_definition": "peak_margin * max(peak_zncc,0); not a calibrated probability",
        "valid": not reasons, "rejection_reasons": reasons,
        "consensus_inlier": False, "consensus_rejection_reasons": [],
        "distance_to_prior": None,
    })


def aggregate_candidates(candidates, settings):
    """All candidates retained, coordinate weighted median explicitly named.

    Imported consensus tests all observed vectors plus their coordinate median;
    highest weighted support wins, then count/error/order tie breaks. It refits
    once and recomputes inliers around the final vector. The reported final
    weight/count may therefore differ from the seed support; both are retained.
    """
    c = validate_global_settings(settings)
    rows = json_copy(candidates)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Every registered training row must supply one candidate record")
    if [r.get("row") for r in rows] != list(range(len(rows))):
        raise ValueError("Candidate report must cover every row exactly once, in H5 order")
    valid = [r for r in rows if r["valid"]]
    for r in rows:
        if (type(r["valid"]) is not bool or r.get("candidate_grid") != "PAN_RR"
                or r.get("candidate_direction") != DIRECTION
                or r.get("output_direction") != DIRECTION
                or r.get("sampling_equation") != SAMPLING_EQUATION):
            raise ValueError("Ambiguous candidate coordinate/sign contract")
        r["consensus_inlier"] = False
        r["consensus_rejection_reasons"] = []
        r["distance_to_prior"] = None
        if r["valid"]:
            v = r["candidate_dy_dx"]
            w = r["confidence_weight"]
            if (not isinstance(v, list) or len(v) != 2
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
                    or type(w) not in (int, float) or not math.isfinite(w) or w <= 0
                    or r["rejection_reasons"]):
                raise ValueError("Invalid vector/confidence cannot enter signed aggregation")
            raw = r.get("raw_estimator", {})
            if (raw.get("accepted") is not True or raw.get("boundary_hit") is not False
                    or v != [raw.get("dy_lr_raw"), raw.get("dx_lr_raw")]
                    or not math.isclose(math.hypot(*v), raw["magnitude_raw"], rel_tol=1e-12, abs_tol=1e-12)
                    or math.hypot(*v) > c["max_magnitude"]
                    or raw["primary_secondary_diff"] > c["max_primary_secondary_diff"]
                    or raw["peak_margin"] < c["min_peak_margin"]
                    or raw["peak_zncc"] < c["min_peak_zncc"]
                    or w != raw["peak_margin"] * max(raw["peak_zncc"], 0.)):
                raise ValueError("Candidate vector/confidence disagrees with raw matcher gates")
        elif r["confidence_weight"] != 0 or not r["rejection_reasons"]:
            raise ValueError("Rejected candidates need reasons and zero aggregation weight")
    reasons = []
    fraction = len(valid) / len(rows)
    if len(valid) < c["min_valid_candidates"]:
        reasons.append("too_few_valid_candidates")
    if fraction < c["min_valid_fraction"]:
        reasons.append("valid_candidate_fraction_below_minimum")
    prior = seed = None
    inlier_count = seed_count = 0
    pre_weight = post_weight = seed_weight = 0.
    concentration = confidence = inlier_fraction = 0.
    dispersion = None
    if valid:
        vectors = np.asarray([r["candidate_dy_dx"] for r in valid], dtype=np.float64)
        weights = np.asarray([r["confidence_weight"] for r in valid], dtype=np.float64)
        prior, distances, inliers, seed = prepare_prior._aggregate(vectors, weights, c["consensus_tolerance"])
        seed_inliers = np.linalg.norm(vectors - seed, axis=1) <= c["consensus_tolerance"]
        pre_weight, post_weight = float(weights.sum()), float(weights[inliers].sum())
        seed_weight, seed_count = float(weights[seed_inliers].sum()), int(seed_inliers.sum())
        inlier_count = int(inliers.sum())
        inlier_fraction = inlier_count / len(valid)
        for row, distance, inlier in zip(valid, distances, inliers):
            row["distance_to_prior"] = float(distance)
            row["consensus_inlier"] = bool(inlier)
            if not inlier:
                row["consensus_rejection_reasons"] = ["translation_consensus_outlier"]
        if not inliers.any():
            reasons.append("no_inliers_after_component_weighted_median_refit")
        else:
            active, active_weights = vectors[inliers], weights[inliers]
            norms = np.linalg.norm(active, axis=1)
            units = np.divide(active, norms[:, None], out=np.zeros_like(active),
                              where=norms[:, None] > np.finfo(np.float64).tiny)
            concentration = float(np.clip(np.linalg.norm(
                (units * active_weights[:, None]).sum(axis=0) / active_weights.sum()), 0, 1))
            dispersion = prepare_prior._weighted_median(distances[inliers], active_weights)
            peaks = np.asarray([r["raw_estimator"]["peak_zncc"] for r in valid])[inliers]
            confidence = float(np.clip(fraction * inlier_fraction * concentration
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
    return json_copy({
        "prior_dy_dx": None if prior is None else prior.tolist(),
        "ready_for_review": not reasons, "gate_reasons": reasons,
        "aggregation": {
            "method": AGGREGATION_METHOD,
            "implementation": "teacher48.prepare_prior._aggregate + _weighted_median; teacher48.prepare_global_prior.aggregate_candidates",
            "code_sha256": _file_hash(prepare_prior.__file__),
            "adapter_code_sha256": _file_hash(__file__),
            "valid_count": len(valid), "rejected_count": len(rows) - len(valid),
            "inlier_count": inlier_count, "outlier_count": len(valid) - inlier_count,
            "pre_consensus_weight_sum": pre_weight, "post_consensus_weight_sum": post_weight,
            "seed_inlier_count": seed_count, "seed_weight_sum": seed_weight,
            "consensus_seed_dy_dx": None if seed is None else seed.tolist(),
            "valid_fraction": fraction, "inlier_fraction": inlier_fraction,
            "direction_concentration": concentration, "dispersion": dispersion,
            "confidence": confidence,
            "confidence_definition": "valid_fraction * final_count_inlier_fraction * weighted_direction_concentration * weighted_mean_clipped_peak_zncc; not a probability",
            "settings": c,
        }, "candidates": rows,
    })


def propose_train_prior(source, settings, *, root="."):
    """Read registered pan/lms only; proposal remains unfrozen/unaccepted."""
    s, c = validate_train_source(source), validate_global_settings(settings)
    path = (Path(root) / s["path"]).resolve(strict=True)
    before = _stamp(path)
    if not path.is_file() or _file_hash(path) != s["sha256"] or before != _stamp(path):
        raise ValueError("Training file hash changed or does not match registration")
    candidates = []
    with h5py.File(path, "r") as handle:
        arrays = []
        for key in ("pan", "lms"):
            if not isinstance(handle.get(key, getlink=True), h5py.HardLink):
                raise ValueError("PAN/LMS must be materialized datasets, not external/soft links")
            dataset = handle[key]
            if (not isinstance(dataset, h5py.Dataset) or dataset.is_virtual or dataset.external
                    or Path(dataset.file.filename).resolve() != path
                    or dataset.dtype.kind not in "uif"):
                raise ValueError("External/virtual/non-numeric PAN/LMS is not admissible")
            arrays.append(dataset)
        pan, lms = arrays
        if (pan.shape != (s["rows"], 1, 64, 64)
                or lms.shape != (s["rows"], SENSOR_BANDS[s["sensor"]], 64, 64)):
            raise ValueError("Registered full train rows/sensor/band/64-grid contract mismatch")
        for i in range(s["rows"]):
            candidate = match_patch(pan[i, 0], lms[i], c)
            candidate["row"] = i
            candidates.append(candidate)
    if before != _stamp(path) or _file_hash(path) != s["sha256"] or before != _stamp(path):
        raise ValueError("Training source changed during preprocessing; proposal discarded")
    result = aggregate_candidates(candidates, c)
    result.update({
        "schema": PROPOSAL_SCHEMA, "sensor": s["sensor"], "accepted": False, "frozen": False,
        "train_source_path": str(path), "train_source_sha256": s["sha256"],
        "train_registration_sha256": canonical_sha256(s), "source_registration": s,
        "train_rows": s["rows"], "created_from_splits": ["train"],
        "source_scene_mapping_used": False, "validation_policy": c,
        "settings_sha256": canonical_sha256(c),
        "estimator": {
            "implementation": "align.estimator.estimate_shift(ref=provided_lms_bandmean,mov=PAN)",
            "code_sha256": _file_hash(estimator.__file__), "adapter_code_sha256": _file_hash(__file__),
            "settings": {key: c[key] for key in sorted(LOCAL_GATE_KEYS)},
            "candidate_direction": DIRECTION, "output_direction": DIRECTION,
            "candidate_grid": "PAN_RR", "prior_grid": "PAN_RR", "sampling_equation": SAMPLING_EQUATION,
            "descriptor": "Scharr magnitude median/MAD normalized ZNCC primary; 5x5 census secondary",
            "scale_and_sign_conversion": "none: both input grids already RR PAN, LMS reference/PAN moving",
        },
        "normalization_and_band_handling": {
            "lms_input": "provided_h5_lms", "radiometry": s["radiometry"], "normalization": s["normalization"],
            "reference": "float64_bandmean_then_float32", "pan": "float32_no_resampling",
            "band_order": s["band_order"], "band_order_evidence": s["band_order_evidence"],
            "lms_provenance": s["lms_provenance"],
        },
        "basis_policy": "fail_outside_explicit_magnitude_bounds",
        "runtime_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "opencv": cv2.__version__, "h5py": h5py.__version__},
        "scope": "stored training RR distribution initialization/reference; not sensor calibration GT",
    })
    return json_copy(result)


def freeze_reviewed_proposal(proposal, review):
    """Turn an explicitly reviewed complete proposal into a loadable asset.

    This is a provenance record, not an additional authorization service. It
    requires the numerical policy/near-zero choice to be consciously supplied.
    Candidate aggregation is reproduced to reject changed vectors or statistics.
    """
    p, r = json_copy(proposal), json_copy(review)
    if (p.get("schema") != PROPOSAL_SCHEMA or p.get("accepted") is not False
            or p.get("frozen") is not False or not p.get("ready_for_review")
            or p.get("gate_reasons") != []):
        raise ValueError("Only an unresolved-gate-free offline proposal can be frozen")
    if (r.get("schema") != REVIEW_SCHEMA or r.get("accepted") is not True
            or r.get("proposal_sha256") != canonical_sha256(p)
            or r.get("settings_sha256") != canonical_sha256(p["validation_policy"])):
        raise ValueError("Explicit review must bind this exact proposal and numerical policy")
    source = validate_train_source(p["source_registration"])
    if (p["train_rows"] != source["rows"] or p["train_rows"] != len(p["candidates"])
            or p["train_source_sha256"] != source["sha256"]
            or p["sensor"] != source["sensor"]
            or p["train_registration_sha256"] != canonical_sha256(source)):
        raise ValueError("Source registration and candidate coverage changed")
    current = aggregate_candidates(p["candidates"], p["validation_policy"])
    for key in ("prior_dy_dx", "ready_for_review", "gate_reasons", "aggregation", "candidates"):
        if p[key] != current[key]:
            raise ValueError("Proposal consensus or implementation changed; regenerate/review")
    payload = {key: value for key, value in p.items() if key != "candidates"}
    payload.update(schema=SCHEMA, accepted=True, frozen=True, policy_review=r,
                   candidate_report_sha256=canonical_sha256(p))
    return GlobalPriorAsset.from_dict(payload)


def _write_new_json(path, value):
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise ValueError("Refusing to overwrite an existing proposal/asset")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".bglobal-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    propose = sub.add_parser("propose", help="Explicit all-row train-only provided-LMS scan; no defaults")
    propose.add_argument("--source", required=True, help="Registered train-source JSON")
    propose.add_argument("--settings", required=True, help="Complete explicit numerical policy JSON")
    propose.add_argument("--root", default=".", help="Root for registered source path, default CWD")
    propose.add_argument("--output", required=True, help="New proposal JSON, never overwrite")
    freeze = sub.add_parser("freeze", help="Freeze only a numerically valid, explicitly reviewed proposal")
    freeze.add_argument("--proposal", required=True)
    freeze.add_argument("--review", required=True, help="Explicit policy/proposal review record")
    freeze.add_argument("--output", required=True, help="New frozen asset JSON, never overwrite")
    args = parser.parse_args(argv)
    def load(path):
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if args.command == "propose":
        result = propose_train_prior(load(args.source), load(args.settings), root=args.root)
        _write_new_json(args.output, result)
        print(json.dumps({"output": args.output, "rows": result["train_rows"],
                          "ready_for_review": result["ready_for_review"],
                          "gate_reasons": result["gate_reasons"],
                          "proposal_sha256": canonical_sha256(result)}, sort_keys=True))
    else:
        asset = freeze_reviewed_proposal(load(args.proposal), load(args.review))
        _write_new_json(args.output, asset.manifest)
        Path(args.output).chmod(0o444)
        print(json.dumps({"output": args.output, "sensor": asset.sensor,
                          "prior_dy_dx": asset.vector, "fingerprint": asset.fingerprint()}, sort_keys=True))


if __name__ == "__main__":
    main()
