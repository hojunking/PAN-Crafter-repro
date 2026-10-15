"""Pure, fail-closed policy for the MethodPlan-v1.3 Stage-1 search.

No data, network, training, or weight-file operations belong in this module.
``resolve_case`` only resolves explicitly pinned baseline values.  A recipe is
configuration, never a donor checkpoint.  Evaluation qualification accepts
complete scene records and four independently captured inference identities;
an aggregate HQNR/ERGAS pair alone cannot authorize weight persistence.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import combinations
import hashlib
import math
import numbers
import re
from typing import Any, Mapping, Sequence

from panda_method import METHOD_REVISION


PARENT_CAMPAIGN_ID = "PANDA_S1_M12_TEACHER48H_S123_20261013_v1"
CAMPAIGN_ID = "PANDA_S1_M13_TEACHER48H_S123_20261013_v1"
METHOD_FILE_NAME = "PANDA_Method_Implementation_Validation_Plan_2026-10-13_v1_3.md"
METHOD_SHA256 = "122472a5c89b2b1e827bbb5c46d19d02821efdae264fb3b311ec2bd5f07f64c0"
HEAD_CONDITIONING = "raw_signed_current_pan_grid_dy_dx"
METHOD_CONTRACT = {"method_revision": METHOD_REVISION,
                   "method_sha256": METHOD_SHA256,
                   "head_conditioning": HEAD_CONDITIONING}
RETENTION_POLICY = "STRICT_TARGET_ONLY"
PROBE_SEED = 261013900
SEED_DERIVATION_PROTOCOL = "teacher48.sha256.seed-label.v1"
STREAM_SEED_NAMESPACE = "teacher48"
LANES = {
    "s1": {"dataset": "WV3", "hqnr_min": 0.958, "ergas_max": 2.040,
           "seeds": (261013101, 261013102, 261013103)},
    "s2": {"dataset": "QB", "hqnr_min": 0.923, "ergas_max": 3.500,
           "seeds": (261013201, 261013202, 261013203)},
    "s3": {"dataset": "GF2", "hqnr_min": 0.965, "ergas_max": 0.522,
           "seeds": (261013301, 261013302, 261013303)},
}
# Tuple order: lambda_epsilon, lambda_absolute.
LOSS_CASES = {
    "F00": (0.001, 1.), "F01": (0.0003, 1.), "F02": (0.0001, 1.),
    "F03": (0.003, 1.), "F04": (0.001, 0.3), "F05": (0.001, 0.1),
    "F06": (0.001, 3.), "F07": (0.0003, 0.3), "F08": (0.0001, 0.3),
    "F09": (0.003, 0.3), "F10": (0.0003, 0.1), "F11": (0.0001, 0.1),
    "F12": (0.003, 0.1), "F13": (0.0003, 3.), "F14": (0.0001, 3.),
    "F15": (0.003, 3.),
}
# Tuple order: lr_A / lr_A0, lr_U / lr_U0, WD / WD0.
OPTIMIZER_CASES = {
    "T00": (0.1, 1., 1.), "T01": (0.3, 1., 1.), "T02": (3., 1., 1.),
    "T03": (1., 0.5, 1.), "T04": (1., 1.5, 1.), "T05": (1., 1., 0.1),
    "T06": (1., 1., 0.), "T07": (0.3, 1.5, 1.), "T08": (0.1, 0.5, 1.),
}
LOSS_ORDER = {
    "s1": ("F00", "F01", "F04", "F07", "F02", "F06", "F08", "F05",
           "F10", "F11", "F13", "F14", "F03", "F09", "F12", "F15"),
    "s2": ("F00", "F02", "F08", "F11", "F01", "F07", "F04", "F10",
           "F05", "F06", "F14", "F13", "F03", "F09", "F12", "F15"),
    "s3": ("F00", "F06", "F13", "F01", "F14", "F04", "F07", "F02",
           "F08", "F05", "F10", "F11", "F03", "F15", "F09", "F12"),
}
OPTIMIZER_ORDER = {
    "s1": ("T01", "T03", "T00", "T07"),
    "s2": ("T00", "T01", "T07", "T03"),
    "s3": ("T04", "T07", "T05", "T01"),
}
HORIZONS = {"H50": 50_000, "H100": 100_000, "H150": 150_000}
BASELINE_REQUIRED = (
    "method_revision", "method_sha256", "head_conditioning",
    "teacher_architecture", "aligner_architecture", "teacher_input_layout",
    "teacher_base", "optimizer", "optimizer_betas", "optimizer_eps",
    "lr_U0", "lr_A0", "weight_decay0", "warmup_updates0", "min_lr",
    "scheduler_family", "effective_batch_RR", "effective_batch_FR",
    "microbatch_RR", "microbatch_FR", "accumulation_RR", "accumulation_FR",
    "RR_train_HW", "FR_train_HW", "precision", "augmentation",
    "evaluator_revision", "gradient_kernel_roi_normalization",
)
IDENTITY_FIELDS = (
    "run_id", "attempt", "seed", "completed_update", "model_state_sha256",
    "inference_policy_digest", "evaluator_revision", "prior_rule_digest",
    "method_revision", "method_sha256", "head_conditioning",
)
FR_METRICS = ("HQNR", "D_s", "D_lambda", "JQM")
RR_METRICS = ("ERGAS", "SCC", "SAM", "PSNR", "SSIM", "RMSE", "CC")


def _finite(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= minimum


def _unresolved(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or any(word in value.upper() for word in
                                       ("MUST_PIN", "TODO", "UNRESOLVED"))
    if isinstance(value, Mapping):
        return not value or any(_unresolved(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return not value or any(_unresolved(v) for v in value)
    return False


def validate_baseline(baseline: Mapping[str, Any]) -> None:
    """Raise on missing/placeholder values; never infer the approved recipe.

    ``RR_train_HW`` and ``FR_train_HW`` are PAN-grid (height, width).
    Batch fields describe a single device's effective batch, including its
    branch-specific gradient accumulation.  Other method/data admission gates
    (source-scene mapping, prior validity, architecture approval) are separate.
    """
    errors = [f"{key}: missing or unresolved" for key in BASELINE_REQUIRED
              if key not in baseline or _unresolved(baseline[key])]
    for key, expected in METHOD_CONTRACT.items():
        if baseline.get(key) != expected:
            errors.append(f"{key}: explicit v1.3 method contract required; no implicit migration")
    for key in ("lr_U0", "lr_A0", "optimizer_eps"):
        if not _finite(baseline.get(key)) or baseline[key] <= 0:
            errors.append(f"{key}: positive finite number required")
    for key in ("weight_decay0", "min_lr"):
        if not _finite(baseline.get(key)) or baseline[key] < 0:
            errors.append(f"{key}: nonnegative finite number required")
    if not _integer(baseline.get("warmup_updates0")) or baseline.get("warmup_updates0", 50_000) >= 50_000:
        errors.append("warmup_updates0: integer in [0,50000) required")
    betas = baseline.get("optimizer_betas")
    if (not isinstance(betas, (list, tuple)) or len(betas) != 2
            or any(not _finite(v) or not 0 <= v < 1 for v in betas)):
        errors.append("optimizer_betas: two finite values in [0,1) required")
    for branch in ("RR", "FR"):
        keys = (f"effective_batch_{branch}", f"microbatch_{branch}", f"accumulation_{branch}")
        if any(not _integer(baseline.get(key), 1) for key in keys):
            errors.append(f"{branch}: positive integer batch/accumulation required")
        elif baseline[keys[0]] != baseline[keys[1]] * baseline[keys[2]]:
            errors.append(f"{branch}: effective batch must equal microbatch * accumulation")
        hw = baseline.get(f"{branch}_train_HW")
        if (not isinstance(hw, (list, tuple)) or len(hw) != 2
                or any(not _integer(v, 4) or v % 4 for v in hw)):
            errors.append(f"{branch}_train_HW: two positive multiples of four required")
    if errors:
        raise ValueError("Unpinned/invalid baseline: " + "; ".join(errors))


def validate_method_config(config: Mapping[str, Any]) -> None:
    """Reject old or relabelled run envelopes, even with identical FC shapes."""
    if any(config.get(k) != v for k, v in METHOD_CONTRACT.items()):
        raise ValueError("Run method contract is not MethodPlan-v1.3; explicit fresh training required")
    if config.get("campaign_id") != CAMPAIGN_ID or config.get("parent_campaign_id") != PARENT_CAMPAIGN_ID:
        raise ValueError("Run campaign identity is not the separate v1.3 campaign")
    if not str(config.get("run_id", "")).startswith("S1M13_48H_"):
        raise ValueError("v1.3 run requires the new S1M13_48H namespace")
    if config.get("initialization") != "FRESH" or config.get("donor_checkpoint") is not None:
        raise ValueError("v1.3 campaign requires fresh initialization, not v1.2 checkpoint migration")
    baseline = config.get("baseline_resolved", {})
    if any(baseline.get(k) != v for k, v in METHOD_CONTRACT.items()):
        raise ValueError("Baseline method contract differs from v1.3 run")


def evaluation_steps(schedule: str | int) -> tuple[int, ...]:
    """Exactly 27/52/77 official points; endpoints are not duplicated."""
    horizon = HORIZONS.get(schedule, schedule)
    if not _integer(horizon) or horizon not in HORIZONS.values():
        raise ValueError(f"Unsupported schedule: {schedule!r}")
    return tuple(sorted({1000, 25_000, horizon, *range(2000, horizon + 1, 2000)}))


def stream_seeds(seed: int) -> dict[str, int]:
    """Stable independent streams: seed+label only, never the F/T case ID."""
    if not _integer(seed):
        raise ValueError("seed must be a nonnegative integer")
    labels = ("A_init", "T_init", "rr_data", "fr_data", "rr_augmentation",
              "fr_augmentation", "rr_epsilon", "fr_epsilon")
    result = {label: 1 + int.from_bytes(hashlib.sha256(
        f"{STREAM_SEED_NAMESPACE}:{seed}:{label}".encode()).digest()[:8], "big") % (2**63 - 1)
        for label in labels}
    result["diagnostic_probe"] = PROBE_SEED
    return result


def resolve_case(server: str, *, baseline: Mapping[str, Any], loss_case: str = "F00",
                 optimizer_case: str | None = None, schedule: str = "H50",
                 replica: str = "R1", attempt: int = 1) -> dict[str, Any]:
    """Resolve a fresh full-method run; T multipliers never compound.

    A technical retry retains run_id/seed and changes only ``attempt``.  A T
    recipe inherits the named F coefficients, not an anchor's LR or weights.
    T05/T06 with WD0=0 return a non-runnable SKIP_DUPLICATE_CONFIG record.
    """
    validate_baseline(baseline)
    if server not in LANES or loss_case not in LOSS_CASES or schedule not in HORIZONS:
        raise ValueError("Unknown server, loss case, or schedule")
    if optimizer_case is not None and optimizer_case not in OPTIMIZER_CASES:
        raise ValueError("Unknown optimizer case")
    if replica not in ("R1", "R2", "R3") or not _integer(attempt, 1):
        raise ValueError("replica must be R1/R2/R3; attempt must be positive")
    lane = LANES[server]
    seed = lane["seeds"][int(replica[1]) - 1]
    eps, absolute = LOSS_CASES[loss_case]
    ma, mu, mw = OPTIMIZER_CASES.get(optimizer_case, (1., 1., 1.))
    lr_a, lr_u = baseline["lr_A0"] * ma, baseline["lr_U0"] * mu
    if baseline["min_lr"] > min(lr_a, lr_u):
        raise ValueError("Pinned min_lr exceeds a resolved learning rate")
    recipe = loss_case + (f"_{optimizer_case}" if optimizer_case else "")
    duplicate = optimizer_case in ("T05", "T06") and baseline["weight_decay0"] == 0
    return {
        "campaign_id": CAMPAIGN_ID, "parent_campaign_id": PARENT_CAMPAIGN_ID,
        **METHOD_CONTRACT,
        "server": server, "dataset": lane["dataset"], "loss_case": loss_case,
        "optimizer_case": optimizer_case, "case": recipe,
        "parent_recipe": loss_case if optimizer_case else None,
        "run_id": f"S1M13_48H_{lane['dataset']}_{server.upper()}_{recipe}_{schedule}_{replica}_S{seed}_v1",
        "attempt": attempt, "replica": replica, "seed": seed,
        "initialization": "FRESH", "donor_checkpoint": None,
        "schedule": schedule, "planned_horizon": HORIZONS[schedule],
        "evaluation_steps": list(evaluation_steps(schedule)),
        "confirmation": replica != "R1", "baseline_resolved": deepcopy(dict(baseline)),
        "lambda_rec": 1., "lambda_abs": absolute,
        "lambda_epsilon": eps, "lambda_eps_RR": eps, "lambda_eps_FR": eps,
        "epsilon_radius_rr": 2., "epsilon_radius_fr": 2., "epsilon_every_update": True,
        "epsilon_distribution": "UNIFORM_AREA_DISK", "epsilon_reduction": "RR_MEAN_PLUS_FR_MEAN",
        "lr_A": lr_a, "lr_U": lr_u, "weight_decay": baseline["weight_decay0"] * mw,
        "lr_A_multiplier": ma, "lr_U_multiplier": mu, "weight_decay_multiplier": mw,
        "stream_seeds": stream_seeds(seed),
        "stream_seed_protocol": SEED_DERIVATION_PROTOCOL,
        "stream_seed_derivation": "1 + sha256('teacher48:<seed>:<label>')[:8] big-endian modulo (2**63-1)",
        "hqnr_min": lane["hqnr_min"], "ergas_max": lane["ergas_max"],
        "retention_policy": RETENTION_POLICY, "stage2": False,
        "status": "SKIP_DUPLICATE_CONFIG" if duplicate else "PLANNED",
        "runnable": not duplicate,
    }


def target_gap(hqnr: float, ergas: float, *, server: str) -> dict[str, Any]:
    """Full-precision, inclusive AND gate.  Invalid numbers never pass."""
    lane = LANES[server]
    valid = _finite(hqnr) and _finite(ergas) and 0 <= hqnr <= 1 and ergas >= 0
    h_pass = bool(valid and hqnr >= lane["hqnr_min"])
    e_pass = bool(valid and ergas <= lane["ergas_max"])
    vh = max(0., (lane["hqnr_min"] - hqnr) / lane["hqnr_min"]) if valid else None
    ve = max(0., (ergas - lane["ergas_max"]) / lane["ergas_max"]) if valid else None
    return {"H_target": lane["hqnr_min"], "E_target": lane["ergas_max"],
            "h_pass": h_pass, "e_pass": e_pass, "target_pair_pass": h_pass and e_pass,
            "v_H": vh, "v_E": ve, "D_inf": max(vh, ve) if valid else None,
            "D1": vh + ve if valid else None}


def qualify_evaluation(record: Mapping[str, Any], *, server: str,
                       expected_rr_ids: Sequence[str], expected_fr_ids: Sequence[str]) -> dict[str, Any]:
    """Return an auditable gate result without writing artifacts.

    Schema: rr/fr={metrics:{...}, per_scene:[{scene_id, metrics:{...}}]}.
    Identity fields are required in state_before, rr_state, fr_state,
    state_after.  Their *rule* digests must match; numerical RR/FR priors may
    differ legitimately.  No round-off tolerance is used anywhere in the gate.
    """
    lane = LANES[server]
    reasons: list[str] = []
    complete, finite = True, True
    counts: dict[str, Any] = {}
    for branch, expected in (("rr", expected_rr_ids), ("fr", expected_fr_ids)):
        expected = list(expected)
        expected_valid = bool(expected) and all(isinstance(x, str) and x for x in expected)
        expected_valid = expected_valid and len(expected) == len(set(expected))
        if not expected_valid:
            complete = False
            reasons.append(f"{branch.upper()}_INVALID_EXPECTED_IDS")
        section = record.get(branch, {})
        section = section if isinstance(section, Mapping) else {}
        rows = section.get("per_scene", [])
        if not isinstance(rows, (tuple, list)):
            rows = []
        ids = [row.get("scene_id") if isinstance(row, Mapping) else None for row in rows]
        valid_ids = [x for x in ids if isinstance(x, str) and x]
        duplicates = sorted(x for x, n in Counter(valid_ids).items() if n > 1)
        missing = sorted(set(x for x in expected if isinstance(x, str)) - set(valid_ids))
        unexpected = sorted(set(valid_ids) - set(x for x in expected if isinstance(x, str)))
        if missing or unexpected or duplicates or len(valid_ids) != len(ids) or len(rows) != len(expected):
            complete = False
            reasons.append(f"{branch.upper()}_SCENE_SET_INCOMPLETE")
        counts[branch] = {"expected": len(expected), "completed": len(rows),
                          "missing": missing, "unexpected": unexpected, "duplicates": duplicates}
        required = FR_METRICS if branch == "fr" else RR_METRICS + (("Q8" if lane["dataset"] == "WV3" else "Q4"),)
        for label, metrics in [("aggregate", section.get("metrics"))] + [
                (f"scene:{i}", row.get("metrics") if isinstance(row, Mapping) else None)
                for i, row in enumerate(rows)]:
            if not isinstance(metrics, Mapping) or any(key not in metrics for key in required):
                complete = False
                reasons.append(f"{branch.upper()}_MISSING_METRIC:{label}")
            if not isinstance(metrics, Mapping) or any(not _finite(metrics.get(key)) for key in required):
                finite = False
                reasons.append(f"{branch.upper()}_NONFINITE_METRIC:{label}")
        aggregate = section.get("metrics")
        if rows and isinstance(aggregate, Mapping):
            for key in required:
                values = [row.get("metrics", {}).get(key) if isinstance(row, Mapping)
                          and isinstance(row.get("metrics"), Mapping) else None for row in rows]
                if _finite(aggregate.get(key)) and all(_finite(value) for value in values):
                    # Means must come from these exact scene records, not a
                    # different checkpoint's best or mean(D_lambda)*mean(D_s).
                    if not math.isclose(aggregate[key], math.fsum(values) / len(values),
                                        rel_tol=0., abs_tol=1e-12):
                        complete = False
                        reasons.append(f"{branch.upper()}_AGGREGATE_MISMATCH:{key}")
    identities = [record.get(key) for key in ("state_before", "rr_state", "fr_state", "state_after")]
    identity_valid = all(isinstance(identity, Mapping)
                         and all(key in identity and not _unresolved(identity[key]) for key in IDENTITY_FIELDS)
                         for identity in identities)
    if identity_valid:
        for identity in identities:
            identity_valid &= all(identity.get(key) == expected for key, expected in METHOD_CONTRACT.items())
            identity_valid &= bool(_integer(identity["attempt"], 1)
                                   and _integer(identity["seed"])
                                   and _integer(identity["completed_update"], 1)
                                   and all(isinstance(identity[key], str) and identity[key].strip()
                                           for key in ("run_id", "inference_policy_digest",
                                                       "evaluator_revision", "prior_rule_digest"))
                                   and isinstance(identity["model_state_sha256"], str)
                                   and re.fullmatch(r"[a-f0-9]{64}", identity["model_state_sha256"]))
        identity_valid &= all(all(identity[key] == identities[0][key] for key in IDENTITY_FIELDS)
                              for identity in identities[1:])
    if not identity_valid:
        reasons.append("SAME_STATE_IDENTITY_NOT_VERIFIED")
    method_eligible = record.get("full_method_eligible") is True
    if not method_eligible:
        reasons.append("FULL_METHOD_NOT_ELIGIBLE")
    if record.get("dataset", lane["dataset"]) != lane["dataset"] or record.get("server", server) != server:
        method_eligible = False
        reasons.append("LANE_MISMATCH")
    # Cross-check outer provenance if provided: do not accept identity from a
    # different run attached to otherwise valid aggregates.
    if identity_valid and any(key in record and record[key] != identities[0][key]
                              for key in IDENTITY_FIELDS):
        identity_valid = False
        reasons.append("OUTER_IDENTITY_MISMATCH")
    rr = record.get("rr", {})
    fr = record.get("fr", {})
    rr = rr if isinstance(rr, Mapping) else {}
    fr = fr if isinstance(fr, Mapping) else {}
    rr_metrics = rr.get("metrics", {})
    fr_metrics = fr.get("metrics", {})
    ergas = rr_metrics.get("ERGAS") if isinstance(rr_metrics, Mapping) else None
    hqnr = fr_metrics.get("HQNR") if isinstance(fr_metrics, Mapping) else None
    gap = target_gap(hqnr, ergas, server=server)
    if gap["D_inf"] is None:
        reasons.append("INVALID_TARGET_METRIC_RANGE")
        finite = False
    joint_pass = bool(method_eligible and complete and finite and identity_valid and gap["target_pair_pass"])
    return {**gap, "HQNR": hqnr, "ERGAS": ergas, "joint_pass": joint_pass,
            "full_method_eligible": method_eligible, "metrics_complete": complete,
            "metrics_finite": finite, "same_state_verified": bool(identity_valid),
            "scene_completeness": counts, "reasons": reasons,
            "selection": "TARGET_QUALIFIED" if joint_pass else "BEST_OBSERVED_NOT_QUALIFIED"}


def _pair(record: Mapping[str, Any]) -> tuple[float, float]:
    h, e = record.get("HQNR"), record.get("ERGAS")
    if not _finite(h) or not _finite(e) or not 0 <= h <= 1 or e < 0:
        raise ValueError("Ranking requires finite, valid same-state HQNR/ERGAS")
    return h, e


def representative_rank(record: Mapping[str, Any]) -> tuple:
    h, e = _pair(record)
    return (-h, e, record["completed_update"], record["run_id"])


def gap_rank(record: Mapping[str, Any], *, server: str) -> tuple:
    h, e = _pair(record)
    gap = target_gap(h, e, server=server)
    return (gap["D_inf"], gap["D1"], -h, e, record["completed_update"], record["run_id"])


def representative(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Only records qualified by the full retention gate are representatives."""
    passed = [r for r in records if r.get("joint_pass") is True]
    return min(passed, key=representative_rank) if passed else None


def rr_reserve(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    passed = [r for r in records if r.get("joint_pass") is True]
    return min(passed, key=lambda r: (_pair(r)[1], -_pair(r)[0], r["completed_update"], r["run_id"])) if passed else None


def pareto_front(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Numeric-only frontier; Pareto membership does NOT authorize saving."""
    for record in records:
        _pair(record)
    return sorted([r for r in records if not any(
        _pair(other)[0] >= _pair(r)[0] and _pair(other)[1] <= _pair(r)[1]
        and _pair(other) != _pair(r) for other in records)], key=representative_rank)


def should_prune(current: Mapping[str, Any], previous: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Conservative §10.4 heuristic on numeric histories, never saved weights.

    Run schema: server,dataset,schedule,seed,replica,run_id,loss_case,
    optimizer_case,completed_update,evaluations (flat HQNR/ERGAS/step records).
    At least two *different recipes*, not technical retries, must weakly
    dominate at the same three most recent common points.  Any lexicographic
    (D_inf,D1) improvement after the exact 10K baseline blocks pruning.
    """
    no = lambda reason: {"prune": False, "reason": reason, "comparators": [], "common_steps": []}
    if current.get("replica") != "R1" or current.get("confirmation"):
        return no("CONFIRMATION_PROTECTED")
    if current.get("loss_case") == "F00" and current.get("optimizer_case") is None:
        return no("F00_PROTECTED")
    if not _integer(current.get("completed_update"), 25_000):
        return no("BEFORE_25K")
    server = current["server"]
    if server not in LANES or current.get("dataset") != LANES[server]["dataset"]:
        return no("LANE_MISMATCH")
    if current.get("seed") != LANES[server]["seeds"][0]:
        return no("NOT_REGISTERED_R1_SEED")
    def history(run):
        result = {}
        for record in run.get("evaluations", ()):
            step = record.get("completed_update")
            if not _integer(step, 1):
                return None
            if step > current["completed_update"]:
                continue
            if step in result or any(record.get(key) is False for key in
                                     ("metrics_complete", "metrics_finite", "same_state_verified")):
                return None
            try:
                _pair(record)
            except ValueError:
                return None
            result[step] = record
        return result
    own = history(current)
    if not own or 10_000 not in own:
        return no("MISSING_10K_REFERENCE")
    trend = [gap_rank(own[s], server=server)[:2] for s in sorted(own) if s >= 10_000]
    if len(trend) < 2 or any(b < a for a, b in zip(trend, trend[1:])):
        return no("TARGET_GAP_IMPROVING_OR_INSUFFICIENT")
    comparators = []
    recipe_ids = set()
    for run in previous:
        recipe = (run.get("loss_case"), run.get("optimizer_case"))
        if recipe in recipe_ids or recipe == (current.get("loss_case"), current.get("optimizer_case")):
            continue
        if run.get("run_id") == current.get("run_id"):
            continue
        if any(run.get(key) != current.get(key) for key in ("dataset", "schedule", "seed", "server", "replica")):
            continue
        hist = history(run)
        if hist:
            comparators.append((run, hist))
            recipe_ids.add(recipe)
    for (left, lh), (right, rh) in combinations(comparators, 2):
        common = sorted(set(own) & set(lh) & set(rh))[-3:]
        if len(common) < 3:
            continue
        if all(_pair(hist[step])[0] >= _pair(own[step])[0]
               and _pair(hist[step])[1] <= _pair(own[step])[1]
               for hist in (lh, rh) for step in common):
            return {"prune": True, "reason": "PRUNED_DOMINATED",
                    "comparators": [left["run_id"], right["run_id"]], "common_steps": common}
    return no("INSUFFICIENT_CONSISTENT_DOMINATION")


def estimate_remaining_seconds(*, remaining_updates: int, update_p90_seconds: float,
                               remaining_evaluations: int, rr_fr_eval_seconds: float,
                               artifact_seconds: float) -> float:
    if not _integer(remaining_updates) or not _integer(remaining_evaluations):
        raise ValueError("update/evaluation counts must be nonnegative integers")
    if any(not _finite(v) or v < 0 for v in (update_p90_seconds, rr_fr_eval_seconds, artifact_seconds)):
        raise ValueError("timings must be observed finite nonnegative seconds")
    if remaining_updates and update_p90_seconds == 0:
        raise ValueError("positive measured update P90 required")
    return remaining_updates * update_p90_seconds + remaining_evaluations * rr_fr_eval_seconds + artifact_seconds


def admit_new_run(*, elapsed_seconds: float, estimated_to_25k_seconds: float,
                  estimated_full_seconds: float, closing_reserve_seconds: float = 0.) -> dict[str, Any]:
    """Local 48h clock, no starts from 46h, 20% ETA safety, minimum 25K.

    ETA inputs include official RR/FR evaluations and possible artifact writes.
    The closing reserve is additional explicitly measured cleanup time.  This
    conservative admission checks both the minimum judgment point and the
    planned endpoint; it never lengthens the wall-clock budget.
    """
    vals = (elapsed_seconds, estimated_to_25k_seconds, estimated_full_seconds, closing_reserve_seconds)
    if any(not _finite(v) or v < 0 for v in vals) or estimated_full_seconds < estimated_to_25k_seconds:
        raise ValueError("Invalid measured time budget/ETA")
    if estimated_to_25k_seconds <= 0:
        raise ValueError("A positive measured estimate to 25K is required")
    deadline = 48 * 3600
    full_finish = elapsed_seconds + 1.2 * estimated_full_seconds + closing_reserve_seconds
    min_finish = elapsed_seconds + 1.2 * estimated_to_25k_seconds + closing_reserve_seconds
    if elapsed_seconds >= 46 * 3600:
        reason = "NOT_STARTED_NO_NEW_AFTER_46H"
    elif min_finish > deadline:
        reason = "NOT_STARTED_BUDGET_25K"
    elif full_finish > deadline:
        reason = "NOT_STARTED_BUDGET_ENDPOINT"
    else:
        reason = "ADMITTED"
    return {"admitted": reason == "ADMITTED", "reason": reason,
            "safety_multiplier": 1.2, "estimated_finish_seconds": full_finish,
            "estimated_25k_seconds": min_finish, "deadline_seconds": deadline}
