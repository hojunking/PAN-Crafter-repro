"""Pure policy for independent, bounded 30-hour donor -> Student campaigns.

No mutation of DirectShift48's pinned campaign and no imports of Torch/network.
The donor recipe is inherited, but run identity, seed selection, retention and
Stage 2 are owned by this separate campaign.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from directshift48 import plan as direct_plan

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CAMPAIGNS = {"PANDA_DONOR_STAGE2_S13_30H_20261015_v1": "DirectShift-Scharr-Donor-Stage2-v1"}
CAMPAIGN_ID = "PANDA_DONOR_STAGE2_S13_30H_20261015_v2_MASK"
METHOD_REVISION = "DirectShift-Scharr-Donor-Stage2-v2-MaskedLoss"
LOSS_MASK_POLICY = "DETACHED_BICUBIC4_SUPPORT_PER_SAMPLE_MEAN_V1"
PLAN_RELATIVE_PATH = "donor30/EXPERIMENT_PLAN.txt"
LANES = {
    "s1": {"dataset": "WV3", "donor_loss_case": "DF07", "hqnr_min": .959,
           "donor_seeds": (261015101, 261015102), "student_seeds": (261015111, 261015112)},
    "s3": {"dataset": "GF2", "donor_loss_case": "DF01", "hqnr_min": .965,
           "donor_seeds": (261015301, 261015302), "student_seeds": (261015311, 261015312)},
}
STAGE2_MODES = ("gt", "e", "eq", "uniform")
HORIZON = 50000
RETENTION_POLICY = "BEST_HQNR_AND_HQNR_GATE_MIN_ERGAS_PLUS_RESUME"
QUALIFIED_DONOR = "HQNR_GATE_ERGAS_BEST"
EXPLORATORY_DONOR = "BEST_HQNR_EXPLORATORY"
SEED_PROTOCOL = "donor30.sha256.seed-label.v1"


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive_int(value):
    return type(value) is int and value > 0


def stream_seeds(seed):
    if not _positive_int(seed):
        raise ValueError("Positive independent seed required")
    return {label: 1 + int.from_bytes(hashlib.sha256(
        f"donor30:{seed}:{label}".encode()).digest()[:8], "big") % (2 ** 63 - 1)
        for label in ("A_init", "T_init", "S_init", "rr_data", "rr_augmentation", "rr_epsilon")}


def default_config(server="s1"):
    """Load the versioned lane, resolving its unchanged DirectShift baseline."""
    if server not in LANES:
        raise ValueError("Only s1/WV3 and s3/GF2 are registered")
    return load_config(ROOT / "config" / "donor30" / (server + ".json"))


def load_config(path):
    c = json.loads(Path(path).read_text())
    relative = c.get("baseline_path")
    if not isinstance(relative, str) or relative != "config/directshift48/" + str(c.get("server")) + ".baseline.json":
        raise ValueError("Registered read-only baseline path required")
    baseline = json.loads((ROOT / relative).read_text())
    if "baseline_resolved" in c and c["baseline_resolved"] != baseline:
        raise ValueError("Embedded baseline differs from referenced baseline")
    c["baseline_resolved"] = baseline
    validate_config(c)
    return c


def validate_config(c):
    if c.get("campaign_id") != CAMPAIGN_ID or c.get("method_revision") != METHOD_REVISION:
        raise ValueError("Donor30 campaign/method identity required")
    if c.get("loss_mask_policy") != LOSS_MASK_POLICY:
        raise ValueError("Explicit v2 training support mask required")
    server = c.get("server")
    lane = LANES.get(server)
    if not lane or c.get("dataset") != lane["dataset"]:
        raise ValueError("Lane must remain s1/WV3 or s3/GF2")
    b = c.get("baseline_resolved", {})
    direct_plan.validate_baseline(b)
    if b.get("server") != server or b.get("dataset") != c["dataset"]:
        raise ValueError("Baseline lane mismatch")
    if (b["lr_U0"], b["lr_A0"], b["weight_decay0"], b["warmup_updates0"], b["min_lr"]) != (1e-4, 1e-5, .01, 100, 0):
        raise ValueError("Preserve original donor learning-rate schedule")
    if (b["effective_batch_RR"], b["microbatch_RR"], b["accumulation_RR"]) != (48, 12, 4):
        raise ValueError("Preserve 48=12x4 FP32 training batch")
    donor = c.get("donor", {})
    if donor.get("loss_case") != lane["donor_loss_case"] or donor.get("seeds") != list(lane["donor_seeds"]):
        raise ValueError("Two registered fresh donor seeds and sensor-specific recipe required")
    if donor.get("hqnr_min") != lane["hqnr_min"] or type(donor.get("allow_exploratory_fallback")) is not bool:
        raise ValueError("Explicit donor gate and fallback policy required")
    s = c.get("stage2", {})
    expected_order = ["gt", "e", "eq"] if server == "s1" else ["eq", "gt", "e"]
    expected_second = ["gt", "eq"] if server == "s1" else ["eq", "gt"]
    if (s.get("core_order") != expected_order or s.get("second_seed_order") != expected_second
            or s.get("optional_order") != ["uniform"]):
        raise ValueError("Registered core/paired/optional queue required")
    expected_arch = {"bands": 8 if server == "s1" else 4, "width": 104, "depth": [1, 2, 2]}
    if s.get("student_architecture") != expected_arch or s.get("seeds") != list(lane["student_seeds"]):
        raise ValueError("Fixed sensor-matched W104/d122 and paired Student seeds required")
    if (s.get("input_layout") != "P0" or s.get("base_reference") != "BICUBIC_MS"
            or not _finite(s.get("lambda_edge")) or s.get("lambda_edge") != 1.):
        raise ValueError("Pin P0/bicubic base and lambda_edge=1")
    if (s.get("calibration_split") != "train" or s.get("calibration_aggregation") != "GLOBAL_PIXEL_MEDIAN"
            or s.get("shared_frozen_donor") is not True):
        raise ValueError("One frozen donor and train-only global-pixel calibration required")
    budget = c.get("budget", {})
    for k in ("total_seconds", "closeout_seconds", "smoke_estimate_seconds", "calibration_estimate_seconds",
              "donor_initial_estimate_seconds", "stage2_initial_estimate_seconds"):
        if not _positive_int(budget.get(k)):
            raise ValueError("Positive budget field required: " + k)
    if budget["total_seconds"] != 30 * 3600 or budget["closeout_seconds"] != 3600 or budget.get("safety_factor") != 1.2:
        raise ValueError("30h total, 1h closeout and 1.2 admission safety required")
    if c.get("retention_policy") != RETENTION_POLICY or c.get("pruning") != "DISABLED_COMPLETE_ADMITTED_H50":
        raise ValueError("Donor retention and complete admitted H50 policy required")


def evaluation_steps():
    return direct_plan.evaluation_steps("H50")


def _case_base(config, *, phase, replica, attempt, seed, case):
    validate_config(config)
    if replica not in ("R1", "R2") or not _positive_int(attempt):
        raise ValueError("Registered replica and positive attempt required")
    b = config["baseline_resolved"]
    return {"campaign_id": CAMPAIGN_ID, "method_revision": METHOD_REVISION,
            "parent_campaign_id": direct_plan.CAMPAIGN_ID,
            "parent_method_revision": direct_plan.METHOD_REVISION,
            "parent_method_sha256": direct_plan.METHOD_SHA256,
            "server": config["server"], "dataset": config["dataset"], "phase": phase,
            "case": case, "loss_case": case, "replica": replica, "attempt": attempt, "seed": seed,
            "run_id": f"D30_M2_{config['dataset']}_{config['server'].upper()}_{phase.upper()}_{case}_H50_{replica}_S{seed}",
            "loss_mask_policy": LOSS_MASK_POLICY,
            "schedule": "H50", "planned_horizon": HORIZON, "evaluation_steps": list(evaluation_steps()),
            "baseline_resolved": deepcopy(b), "stream_seeds": stream_seeds(seed),
            "stream_seed_protocol": SEED_PROTOCOL, "lr_U": b["lr_U0"], "lr_A": b["lr_A0"],
            "weight_decay": b["weight_decay0"], "retention_policy": RETENTION_POLICY,
            "hqnr_min": config["donor"]["hqnr_min"], "ergas_max": direct_plan.LANES[config["server"]]["ergas_max"],
            "sheet_targets": deepcopy(direct_plan.SHEET_TARGETS[config["server"]]),
            "alignment_protocol_digest": direct_plan.alignment_protocol_digest(b),
            "config_digest": canonical_sha256(config), "requires_fr_train": False,
            "requires_source_scene_map": False, "pruning": "DISABLED_COMPLETE_ADMITTED_H50"}


def resolve_donor_case(config, replica="R1", attempt=1):
    if replica not in ("R1", "R2"):
        raise ValueError("Only two donor replicas registered")
    d = config["donor"]
    case = d["loss_case"]
    eps, absolute = direct_plan.LOSS_CASES[case]
    out = _case_base(config, phase="donor", replica=replica, attempt=attempt,
                     seed=d["seeds"][int(replica[1]) - 1], case=case)
    out.update(initialization="FRESH", donor_checkpoint=None, stage2=False,
               lambda_rec=1., lambda_abs=absolute, lambda_epsilon=eps, lambda_eps_RR=eps,
               epsilon_radius_rr=3., epsilon_every_update=True,
               epsilon_distribution="UNIFORM_AREA_DISK", epsilon_reduction="RR_SAMPLE_MEAN")
    validate_case(out, config=config)
    return out


def _record_valid(record):
    h, e, s = record.get("HQNR"), record.get("ERGAS"), record.get("SCC")
    return (all(record.get(k) is True for k in ("checkpoint_verified", "metrics_complete", "metrics_finite", "same_state_verified"))
            and all(_finite(v) for v in (h, e, s)) and 0 <= h <= 1 and e >= 0 and -1 <= s <= 1
            and _positive_int(record.get("completed_update"))
            and isinstance(record.get("run_id"), str) and bool(record["run_id"])
            and isinstance(record.get("checkpoint_path"), str) and bool(record["checkpoint_path"])
            and isinstance(record.get("state_hash"), str)
            and re.fullmatch(r"[a-f0-9]{64}", record["state_hash"]) is not None)


def selection_key(record, threshold):
    """Gate first, then ERGAS; ties HQNR, SCC, earlier step, stable run ID."""
    if not _record_valid(record) or not _finite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Verified same-state checkpoint and valid HQNR threshold required")
    h, e, s = record["HQNR"], record["ERGAS"], record["SCC"]
    tail = (record["completed_update"], record["run_id"], record["checkpoint_path"])
    return ((0, e, -h, -s, *tail) if h >= threshold else (1, -h, -s, e, *tail))


def best_hqnr_key(record):
    if not _record_valid(record):
        raise ValueError("Verified same-state checkpoint required")
    return (-record["HQNR"], -record["SCC"], record["ERGAS"], record["completed_update"], record["run_id"])


def select_donor(records, threshold, allow_fallback=True):
    """Never fabricate a gate pass or combine RR/FR from different states.

    Caller verifies file content/reload and same-state full scene evaluation;
    this pure policy requires explicit evidence flags, not filesystem access.
    """
    if type(allow_fallback) is not bool or not _finite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Valid threshold and explicit fallback decision required")
    valid = [r for r in records if _record_valid(r)]
    if not valid:
        return None
    chosen = min(valid, key=lambda r: selection_key(r, threshold))
    passed = chosen["HQNR"] >= threshold
    if not passed and not allow_fallback:
        return None
    out = deepcopy(chosen)
    out.update(donor_selection_status=QUALIFIED_DONOR if passed else EXPLORATORY_DONOR,
               donor_hqnr_qualified=passed, donor_hqnr_threshold=threshold,
               donor_final_target_pass=None)
    # Final joint-target status is deliberately NOT inferred from the donor
    # gate: publication's existing final thresholds remain a separate result.
    return out


def select_stage2(records, threshold, allow_fallback=True):
    """Same threshold-first research selection for a trained Student.

    The donor lineage already present on records is retained, not replaced by
    Student qualification; final joint-target success remains external.
    """
    selected = select_donor(records, threshold, allow_fallback)
    if selected is None:
        return None
    source = next(r for r in records if r.get("run_id") == selected["run_id"]
                  and r.get("completed_update") == selected["completed_update"]
                  and r.get("checkpoint_path") == selected["checkpoint_path"])
    out = deepcopy(source)
    out.update(model_selection_status=selected["donor_selection_status"],
               model_hqnr_qualified=selected["donor_hqnr_qualified"],
               model_hqnr_threshold=threshold)
    return out


def resolve_stage2_case(config, mode="gt", replica="R1", donor=None, *, attempt=1, optional=False):
    if mode not in STAGE2_MODES or replica not in ("R1", "R2"):
        raise ValueError("Unknown Stage2 mode/replica")
    if not isinstance(donor, Mapping) or not _record_valid(donor):
        raise ValueError("Verified local donor checkpoint required")
    lane = LANES[config["server"]]
    if donor.get("server") != config["server"] or donor.get("dataset") != lane["dataset"]:
        raise ValueError("Stage2 must consume this server's own sensor-matched donor")
    if donor.get("donor_selection_status") not in (QUALIFIED_DONOR, EXPLORATORY_DONOR):
        raise ValueError("Explicit donor selection status required")
    passed = donor["HQNR"] >= config["donor"]["hqnr_min"]
    expected_status = QUALIFIED_DONOR if passed else EXPLORATORY_DONOR
    if donor.get("donor_hqnr_qualified") is not passed or donor["donor_selection_status"] != expected_status:
        raise ValueError("Donor HQNR qualification mismatch")
    if donor.get("phase", "donor") != "donor":
        raise ValueError("A Student checkpoint cannot be substituted for a Stage1 donor")
    if not passed and not config["donor"]["allow_exploratory_fallback"]:
        raise ValueError("Exploratory fallback is disabled")
    out = _case_base(config, phase="stage2", replica=replica, attempt=attempt,
                     seed=config["stage2"]["seeds"][int(replica[1]) - 1], case=mode.upper())
    out.update(stage2=True, stage2_mode=mode, initialization="FRESH_STUDENT_FROZEN_DONOR",
               donor_checkpoint=donor["checkpoint_path"], donor_state_hash=donor["state_hash"],
               donor_run_id=donor["run_id"], donor_completed_update=donor["completed_update"],
               donor_selection_status=donor["donor_selection_status"], donor_hqnr_qualified=passed,
               shared_frozen_aligner=True, frozen_teacher=True, optional=bool(optional),
               student_architecture=deepcopy(config["stage2"]["student_architecture"]),
               lambda_edge=0. if mode in ("gt", "e") else config["stage2"]["lambda_edge"],
               calibration_split="train", calibration_aggregation="GLOBAL_PIXEL_MEDIAN")
    validate_case(out, config=config)
    return out


def stage2_cases(config, donor, include_optional=True):
    out = [resolve_stage2_case(config, mode, "R1", donor) for mode in config["stage2"]["core_order"]]
    out += [resolve_stage2_case(config, mode, "R2", donor) for mode in config["stage2"]["second_seed_order"]]
    if include_optional:
        out += [resolve_stage2_case(config, mode, "R1", donor, optional=True)
                for mode in config["stage2"]["optional_order"]]
    return out


def validate_case(c, *, config):
    validate_config(config)
    phase = c.get("phase")
    if phase not in ("donor", "stage2") or c.get("campaign_id") != CAMPAIGN_ID or c.get("method_revision") != METHOD_REVISION:
        raise ValueError("New campaign and explicit phase required")
    if c.get("loss_mask_policy") != LOSS_MASK_POLICY:
        raise ValueError("Case must use the masked-loss contract")
    for key in ("server", "dataset", "baseline_resolved"):
        if c.get(key) != config[key]:
            raise ValueError("Case changed pinned " + key)
    if c.get("config_digest") != canonical_sha256(config):
        raise ValueError("Configuration digest mismatch")
    if c.get("planned_horizon") != HORIZON or c.get("evaluation_steps") != list(evaluation_steps()):
        raise ValueError("Never shorten cosine schedule to an early donor selection step")
    replica = c.get("replica")
    if replica not in ("R1", "R2") or not _positive_int(c.get("attempt")):
        raise ValueError("Invalid replica/attempt")
    seed = config["donor" if phase == "donor" else "stage2"]["seeds"][int(replica[1]) - 1]
    if (c.get("seed") != seed or c.get("stream_seeds") != stream_seeds(seed)
            or c.get("stream_seed_protocol") != SEED_PROTOCOL):
        raise ValueError("RNG streams differ from registered paired seed")
    expected_run = f"D30_M2_{config['dataset']}_{config['server'].upper()}_{phase.upper()}_{c['case']}_H50_{replica}_S{seed}"
    if c.get("run_id") != expected_run or c.get("stage2") is not (phase == "stage2"):
        raise ValueError("Run identity/phase mismatch")
    b = config["baseline_resolved"]
    if (c.get("lr_U"), c.get("lr_A"), c.get("weight_decay")) != (b["lr_U0"], b["lr_A0"], b["weight_decay0"]):
        raise ValueError("Pinned optimizer changed")
    if c.get("requires_fr_train") is not False or c.get("requires_source_scene_map") is not False:
        raise ValueError("RR-only prior-free training required")
    if c.get("retention_policy") != RETENTION_POLICY or c.get("hqnr_min") != config["donor"]["hqnr_min"]:
        raise ValueError("Donor storage gate mismatch")
    if c.get("alignment_protocol_digest") != direct_plan.alignment_protocol_digest(b):
        raise ValueError("Prior-free native-PAN protocol identity mismatch")
    if c.get("pruning") != "DISABLED_COMPLETE_ADMITTED_H50":
        raise ValueError("Admitted runs cannot gain silent metric pruning")
    if phase == "donor":
        loss = config["donor"]["loss_case"]
        eps, absolute = direct_plan.LOSS_CASES[loss]
        if c.get("case") != loss or c.get("loss_case") != loss or c.get("donor_checkpoint") is not None:
            raise ValueError("Registered fresh donor recipe required")
        coefficient_keys = ("lambda_rec", "lambda_abs", "lambda_epsilon", "lambda_eps_RR", "epsilon_radius_rr")
        if (any(not _finite(c.get(k)) for k in coefficient_keys)
                or tuple(c.get(k) for k in coefficient_keys) != (1., absolute, eps, eps, 3.)):
            raise ValueError("Donor losses/radius differ from sensor-specific selection")
        if (c.get("initialization") != "FRESH" or c.get("epsilon_every_update") is not True
                or c.get("epsilon_distribution") != "UNIFORM_AREA_DISK" or c.get("epsilon_reduction") != "RR_SAMPLE_MEAN"):
            raise ValueError("Fresh full-method donor contract changed")
    else:
        mode = c.get("stage2_mode")
        if mode not in STAGE2_MODES or c.get("case") != mode.upper():
            raise ValueError("Stage2 mode mismatch")
        if (c.get("shared_frozen_aligner") is not True or c.get("frozen_teacher") is not True
                or c.get("student_architecture") != config["stage2"]["student_architecture"]):
            raise ValueError("Fixed donor and identical Student architecture required")
        if not _finite(c.get("lambda_edge")) or c.get("lambda_edge") != (0. if mode in ("gt", "e") else 1.):
            raise ValueError("Stage2 edge ablation coefficient changed")
        if (c.get("initialization") != "FRESH_STUDENT_FROZEN_DONOR"
                or c.get("calibration_split") != "train" or c.get("calibration_aggregation") != "GLOBAL_PIXEL_MEDIAN"):
            raise ValueError("Fresh Student and train-only calibration required")
        if (not isinstance(c.get("donor_checkpoint"), str) or not c["donor_checkpoint"]
                or not isinstance(c.get("donor_state_hash"), str)
                or re.fullmatch(r"[a-f0-9]{64}", c["donor_state_hash"]) is None
                or not _positive_int(c.get("donor_completed_update"))):
            raise ValueError("Complete checkpoint lineage required")
        qualified = c.get("donor_hqnr_qualified")
        if type(qualified) is not bool or c.get("donor_selection_status") != (QUALIFIED_DONOR if qualified else EXPLORATORY_DONOR):
            raise ValueError("Donor qualification/status inconsistent")
        if not qualified and not config["donor"]["allow_exploratory_fallback"]:
            raise ValueError("Exploratory lineage is not authorized by config")


def admission_decision(elapsed_seconds, estimated_seconds, *, budget=None):
    """Measured whole-task duration includes updates, evaluation and saving.

    Call before smoke/calibration/runs and re-check deadline during each task.
    This policy cannot itself interrupt a kernel or guarantee completion time.
    """
    b = budget or {"total_seconds": 108000, "closeout_seconds": 3600, "safety_factor": 1.2}
    if not all(_finite(v) and v >= 0 for v in (elapsed_seconds, estimated_seconds)) or estimated_seconds == 0:
        raise ValueError("Nonnegative elapsed time and positive whole-task estimate required")
    total, reserve, factor = b["total_seconds"], b["closeout_seconds"], b["safety_factor"]
    if not all(_finite(v) for v in (total, reserve, factor)) or not 0 <= reserve < total or factor < 1:
        raise ValueError("Invalid campaign budget")
    remaining = max(0., total - elapsed_seconds)
    guarded = estimated_seconds * factor
    admitted = elapsed_seconds + guarded + reserve <= total
    return {"admitted": admitted, "remaining_seconds": remaining,
            "estimated_seconds": estimated_seconds, "guarded_seconds": guarded,
            "closeout_seconds": reserve, "deadline_seconds": total,
            "reason": "MEASURED_ESTIMATE_FITS" if admitted else "SKIPPED_BUDGET"}


def estimated_run_seconds(update_seconds, evaluation_seconds, *, saving_seconds=0., diagnostics_seconds=0., horizon=HORIZON):
    if horizon != HORIZON or not _finite(update_seconds) or update_seconds <= 0:
        raise ValueError("Positive measured update timing for full H50 required")
    if any(not _finite(x) or x < 0 for x in (evaluation_seconds, saving_seconds, diagnostics_seconds)):
        raise ValueError("Nonnegative measured evaluation/save/diagnostic duration required")
    return horizon * update_seconds + len(evaluation_steps()) * evaluation_seconds + saving_seconds + diagnostics_seconds
