"""Frozen, train-only dataset prior contract; no source-scene surrogate IDs.

The JSON is a reviewed measurement asset, not an assertion of physical sensor
calibration.  Source/report hashes bind evidence but do not make user declarations
of split or spectral order independently true.  Training adapters must compare
``train_source_sha256`` with the actual training file before consuming it.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA = "PANDA_B_GLOBAL_RR_V1"
SENSOR_BANDS = {"WV3": 8, "QB": 4, "GF2": 4}
DIRECTION = "PAN_TO_MS_WARP_CONVENTION"
SAMPLING_EQUATION = "aligned[y,x]=PAN[y+dy,x+dx]"
AGGREGATION_METHOD = (
    "exhaustive_weighted_translation_consensus_then_component_weighted_median_v1"
)
LOCAL_GATE_KEYS = frozenset({"search_int", "max_magnitude", "min_peak_margin",
                             "max_primary_secondary_diff", "edge_top_frac"})
SETTINGS_KEYS = LOCAL_GATE_KEYS | frozenset({
    "match_window", "min_reference_std", "min_pan_std", "min_peak_zncc",
    "min_valid_candidates", "min_valid_fraction", "consensus_tolerance",
    "min_inlier_fraction", "min_direction_concentration", "max_dispersion",
    "min_confidence", "min_prior_magnitude", "rationale",
})


def json_copy(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Prior metadata must be finite JSON data") from exc


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require_sha256(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(name + " requires an explicit SHA256")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " requires nonempty text")
    return value


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (positive and value <= 0)):
        raise ValueError(name + " must be a finite" + (" positive" if positive else "") + " number")
    return float(value)


def _count(value, name, *, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError(name + " must be an explicit nonnegative integer")
    return value


def validate_global_settings(settings):
    """Complete explicit numeric policy, with no approved sensor defaults."""
    c = json_copy(settings)
    if not isinstance(c, dict) or set(c) != SETTINGS_KEYS:
        raise ValueError("Global prior settings require exactly: " + ", ".join(sorted(SETTINGS_KEYS)))
    if type(c["match_window"]) is not int or c["match_window"] != 64:
        raise ValueError("This adapter matches one full stored 64x64 training patch")
    radius = _count(c["search_int"], "search_int", positive=True)
    _count(c["min_valid_candidates"], "min_valid_candidates", positive=True)
    if c["match_window"] < 2 * max(radius, 2) + 5:
        raise ValueError("Search/census support exceeds the stored training patch")
    for key in ("max_magnitude", "min_peak_margin", "consensus_tolerance", "max_dispersion",
                "min_prior_magnitude", "min_reference_std", "min_pan_std"):
        _number(c[key], key, positive=True)
    if _number(c["max_primary_secondary_diff"], "max_primary_secondary_diff") < 0:
        raise ValueError("Matcher agreement tolerance must be nonnegative")
    for key in ("edge_top_frac", "min_peak_zncc", "min_valid_fraction", "min_inlier_fraction",
                "min_direction_concentration", "min_confidence"):
        if not 0 < _number(c[key], key) <= 1:
            raise ValueError(key + " must be in (0,1]")
    if c["min_prior_magnitude"] > c["max_magnitude"]:
        raise ValueError("Near-zero threshold exceeds maximum prior magnitude")
    _text(c["rationale"], "settings rationale")
    return c


def _validate(payload):
    p = json_copy(payload)
    if not isinstance(p, dict) or p.get("schema") != SCHEMA:
        raise ValueError("A PANDA_B_GLOBAL_RR_V1 asset is required")
    if p.get("sensor") not in SENSOR_BANDS:
        raise ValueError("Exactly one WV3/QB/GF2 sensor is required")
    if (p.get("created_from_splits") != ["train"]
            or p.get("source_scene_mapping_used") is not False
            or p.get("frozen") is not True or p.get("accepted") is not True):
        raise ValueError("Only accepted, frozen, train-only dataset priors may be loaded")
    if p.get("ready_for_review") is not True or p.get("gate_reasons") != []:
        raise ValueError("Unresolved prior gates cannot enter training")
    _text(p.get("train_source_path"), "train_source_path")
    require_sha256(p.get("train_source_sha256"), "train_source_sha256")
    require_sha256(p.get("train_registration_sha256"), "train_registration_sha256")
    require_sha256(p.get("candidate_report_sha256"), "candidate_report_sha256")
    n = _count(p.get("train_rows"), "train_rows", positive=True)
    registration = p.get("source_registration")
    if (not isinstance(registration, dict)
            or registration.get("schema") != "PANDA_B_GLOBAL_RR_TRAIN_SOURCE_V1"
            or canonical_sha256(registration) != p["train_registration_sha256"]
            or registration.get("sha256") != p["train_source_sha256"]
            or registration.get("sensor") != p["sensor"]
            or registration.get("rows") != n or registration.get("split") != "train"):
        raise ValueError("Frozen source registration is missing or inconsistent")
    if p.get("normalization_and_band_handling", {}).get("lms_input") != "provided_h5_lms":
        raise ValueError("Prior estimation must use the provided LMS")
    normalization = p["normalization_and_band_handling"]
    if (normalization.get("radiometry") != "stored_raw_dn"
            or normalization.get("normalization") != "none"
            or normalization.get("reference") != "float64_bandmean_then_float32"
            or normalization.get("pan") != "float32_no_resampling"
            or not isinstance(normalization.get("band_order"), list)
            or len(normalization.get("band_order", [])) != SENSOR_BANDS[p["sensor"]]):
        raise ValueError("Explicit provided-LMS band order and unmodified DN input are required")
    if (not all(isinstance(x, str) and x.strip() for x in normalization["band_order"])
            or len(set(normalization["band_order"])) != len(normalization["band_order"])):
        raise ValueError("Unique explicit spectral band names are required")
    _text(normalization.get("band_order_evidence"), "band_order_evidence")
    _text(normalization.get("lms_provenance"), "lms_provenance")
    e = p.get("estimator", {})
    _text(e.get("implementation"), "estimator implementation")
    require_sha256(e.get("code_sha256"), "estimator code_sha256")
    require_sha256(e.get("adapter_code_sha256"), "estimator adapter_code_sha256")
    if (e.get("candidate_direction") != DIRECTION or e.get("output_direction") != DIRECTION
            or e.get("candidate_grid") != "PAN_RR" or e.get("prior_grid") != "PAN_RR"
            or e.get("sampling_equation") != SAMPLING_EQUATION):
        raise ValueError("Prior must directly use signed RR-PAN sampling coordinates")
    if not isinstance(e.get("settings"), dict) or not e["settings"]:
        raise ValueError("Exact local matcher settings are mandatory")
    agg = p.get("aggregation", {})
    if agg.get("method") != AGGREGATION_METHOD:
        raise ValueError("Unsupported/unidentified vector aggregation")
    require_sha256(agg.get("code_sha256"), "aggregation code_sha256")
    valid = _count(agg.get("valid_count"), "valid_count", positive=True)
    rejected = _count(agg.get("rejected_count"), "rejected_count")
    inliers = _count(agg.get("inlier_count"), "inlier_count", positive=True)
    outliers = _count(agg.get("outlier_count"), "outlier_count")
    if valid + rejected != n or inliers + outliers != valid:
        raise ValueError("Counts must cover every training row and every valid candidate")
    w0 = _number(agg.get("pre_consensus_weight_sum"), "pre consensus weights", positive=True)
    w1 = _number(agg.get("post_consensus_weight_sum"), "post consensus weights", positive=True)
    if w1 > w0:
        raise ValueError("Consensus cannot increase summed candidate weights")
    settings = validate_global_settings(p.get("validation_policy", {}))
    if e["settings"] != {key: settings[key] for key in sorted(LOCAL_GATE_KEYS)}:
        raise ValueError("Local matcher settings disagree with the reviewed policy")
    # No threshold is silently supplied, including the small-vector policy.
    low = _number(settings.get("min_prior_magnitude"), "min_prior_magnitude", positive=True)
    high = _number(settings.get("max_magnitude"), "max_magnitude", positive=True)
    if low > high:
        raise ValueError("Invalid prior magnitude bounds")
    if valid < _count(settings.get("min_valid_candidates"), "min_valid_candidates", positive=True):
        raise ValueError("Too few valid train candidates")
    comparisons = ((valid / n, "min_valid_fraction"),
                   (inliers / valid, "min_inlier_fraction"),
                   (_number(agg.get("direction_concentration"), "direction_concentration"),
                    "min_direction_concentration"),
                   (_number(agg.get("confidence"), "confidence"), "min_confidence"))
    for value, key in comparisons:
        threshold = _number(settings.get(key), key, positive=True)
        if not 0 <= value <= 1 or not 0 < threshold <= 1 or value < threshold:
            raise ValueError(key + " gate not satisfied")
    dispersion = _number(agg.get("dispersion"), "dispersion")
    if dispersion < 0 or dispersion > _number(settings.get("max_dispersion"), "max_dispersion", positive=True):
        raise ValueError("Dispersion exceeds explicit policy")
    vector = p.get("prior_dy_dx")
    if not isinstance(vector, list) or len(vector) != 2:
        raise ValueError("Exactly two signed (dy, dx) components are required")
    vector = [_number(v, "prior component") for v in vector]
    if not low <= math.hypot(*vector) <= high:
        raise ValueError("Zero/near-zero/large prior requires review; no basis fallback")
    if (p.get("basis_policy") != "fail_outside_explicit_magnitude_bounds"
            or agg.get("settings") != settings):
        raise ValueError("Explicit aggregation/basis settings are inconsistent")
    _text(settings.get("rationale"), "validation rationale")
    review = p.get("policy_review", {})
    if (review.get("schema") != "PANDA_B_GLOBAL_RR_POLICY_REVIEW_V1"
            or review.get("accepted") is not True):
        raise ValueError("Numerical settings and proposal must be explicitly reviewed")
    _text(review.get("reviewed_by"), "reviewed_by")
    _text(review.get("rationale"), "review rationale")
    if review.get("settings_sha256") != canonical_sha256(settings):
        raise ValueError("Review belongs to a different numerical policy")
    require_sha256(review.get("proposal_sha256"), "review proposal_sha256")
    if review["proposal_sha256"] != p["candidate_report_sha256"]:
        raise ValueError("Review belongs to a different prior proposal")
    return p


class GlobalPriorAsset:
    """Immutable canonical metadata with tuple vector and stable fingerprint.

    ``from_dict`` uses the same validation as disk loading (primarily useful for
    tests). The fingerprint is canonical JSON identity, not the pretty-printed
    file-byte hash; both remain independently recordable in run manifests.
    """
    def __init__(self, payload):
        try:
            self._payload = _validate(payload)
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError("Malformed frozen prior contract") from exc
        self._fingerprint = canonical_sha256(self._payload)

    @classmethod
    def from_dict(cls, payload):
        return cls(payload)

    @classmethod
    def from_file(cls, path, *, sensor=None, expected_train_sha256=None):
        with Path(path).open("r", encoding="utf-8") as handle:
            asset = cls(json.load(handle))
        if sensor is not None and asset.sensor != sensor:
            raise ValueError("Frozen prior sensor mismatch")
        if (expected_train_sha256 is not None
                and asset.train_source_sha256 != require_sha256(expected_train_sha256, "expected train hash")):
            raise ValueError("Frozen prior was built from a different training file")
        return asset

    @property
    def vector(self):
        return tuple(float(v) for v in self._payload["prior_dy_dx"])

    @property
    def sensor(self):
        return self._payload["sensor"]

    @property
    def train_source_sha256(self):
        return self._payload["train_source_sha256"]

    @property
    def min_magnitude(self):
        return float(self._payload["validation_policy"]["min_prior_magnitude"])

    @property
    def max_magnitude(self):
        return float(self._payload["validation_policy"]["max_magnitude"])

    @property
    def manifest(self):
        return json_copy(self._payload)

    def fingerprint(self):
        return self._fingerprint
