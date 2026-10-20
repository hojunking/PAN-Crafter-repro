"""Registered local s1/s3 queues; pending groups may be skipped, never mutated."""
from copy import deepcopy
from dataclasses import asdict
import csv
import io
from pathlib import Path

from donor24.plan import evaluation_steps
from donor30.plan import stream_seeds, SEED_PROTOCOL, LOSS_MASK_POLICY, canonical_sha256
from directshift48.plan import alignment_protocol_digest
from panda30r.plan import baseline as old_baseline
from .contract import CAMPAIGN_ID, METHOD_REVISION, BUDGET, POLICY, STAGE1_SELECTION

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = Path(__file__).resolve()
SOURCE_SEED = 261016101
SEEDS = {"R1": 261018101, "R2": 261018102}
HORIZON = 50000
DONORS = ("G8_MATURE_U23",)
LEGACY_SOURCE = ("work_dir/panda_band30/"
    "PANDA_S1_ALLBAND_WARP_BANDWISE_30H_20261019_v1/s1/WV3/runs/"
    "PB30_WV3_S1_G8_S1_U23_R1_S261016101/attempt_01")


def baseline(server):
    if server not in ("s1", "s3"):
        raise ValueError("Mature30 is registered only for s1/s3 WV3")
    return deepcopy(old_baseline(server))


def groups(server):
    baseline(server)
    result = []
    if server == "s3":
        result.append({"name": "mature_donor", "optional": False,
                       "runs": [("G8_MATURE_U23", "R1")]})
    for policy in ("FULL", "HALF", "DOWN", "LOW"):
        result.append({"name": policy.lower(), "optional": False,
                       "runs": [(policy, "R1")]})
    result.extend([
        {"name": "paired_repeat", "optional": True,
         "runs": [("HALF", "R2"), ("DOWN", "R2")]},
        {"name": "cap_screen", "optional": True, "runs": [("CAP085", "R1")]},
    ])
    if server == "s3":
        result.append({"name": "cap_repeat", "optional": True, "runs": [("CAP085", "R2")]})
    return [dict(g, admission="WHOLE_GROUP_OR_SKIP") for g in result]


def default_config(server):
    return {
        "campaign_id": CAMPAIGN_ID, "method_revision": METHOD_REVISION,
        "server": server, "dataset": "WV3", "baseline_resolved": baseline(server),
        "budget": deepcopy(BUDGET), "donors": list(DONORS), "groups": groups(server),
        "donor_policy": "LOCAL_MEAN8_INTERP23_COMPLETE_50K_VAL_RRE_SELECTION",
        "donor_source_seed": SOURCE_SEED, "allow_cross_server_donor": False,
        "allow_old_donor_reuse": server == "s1",
        "legacy_source": LEGACY_SOURCE if server == "s1" else None,
        "checkpoint_selection": POLICY.manifest(),
        "stage1_selection": deepcopy(STAGE1_SELECTION),
        "sheet_selection": "STAGE1_RRE_SELECTED_STAGE2_JOINT_PASS_OR_BEST_PLUS_FINAL50K",
        "evaluation": "ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20",
        "fixed_loss": {"alpha": .25, "beta": 0., "gamma": .5},
        "soft_distillation": "REMOVED_USER_DECISION_20261020",
        "optional_policy": "MEASURED_WHOLE_GROUP_ADMISSION",
        "time_prior_hours": {"stage1": 2.606 if server == "s1" else 4.,
                             "stage2": 3.610 if server == "s1" else 2.1},
        "priority_policy": "ORDERED_ATTEMPTS_NOT_GUARANTEED_COMPLETION",
        "pruning": "EXPLICIT_PENDING_GROUP_SKIP_ONLY",
    }


def validate_config(cfg):
    if canonical_sha256(cfg) != canonical_sha256(default_config(cfg.get("server"))):
        raise ValueError("Unregistered mature30 campaign configuration")


def resolve_case(server, case, replica="R1", *, attempt=1):
    import hashlib
    from panda_stage2_condq.training import TrainingSpec
    from .spec import MatureSpec, policy_manifest
    campaign = default_config(server)
    registered = [r for g in groups(server) for r in g["runs"]]
    # s1 reselects an existing complete local source, it does not retrain it.
    if ((case, replica) not in registered and not
            (case == DONORS[0] and replica == "R1")) or type(attempt) is not int or attempt < 1:
        raise ValueError("Unregistered case/replica/attempt")
    donor = case in DONORS
    seed = SOURCE_SEED if donor else SEEDS[replica]
    seeds = stream_seeds(seed)
    seeds["rr_band_augmentation"] = 1 + int.from_bytes(hashlib.sha256(
        f"donor30:{seed}:rr_band_augmentation".encode()).digest()[:8], "big") % (2**63 - 1)
    spec = MatureSpec("WV3", "panda_condq", seeds["S_init"], .25, 0., .5, 1e-6,
                      mask_policy=LOSS_MASK_POLICY, policy_id="FULL" if donor else case)
    b = campaign["baseline_resolved"]
    return {
        "campaign_id": CAMPAIGN_ID, "method_revision": METHOD_REVISION,
        "server": server, "dataset": "WV3", "case": case, "loss_case": case,
        "replica": replica, "attempt": attempt, "seed": seed,
        "run_id": f"PM30_WV3_{server.upper()}_{case}_NOSOFT_{replica}_S{seed}",
        "run_display": f"MATURE30 NOSOFT {case} {replica}",
        "stream_seeds": seeds, "stream_seed_protocol": SEED_PROTOCOL + "+rr_band_augmentation",
        "baseline_resolved": deepcopy(b), "schedule": "H50", "start_update": 0,
        "planned_horizon": HORIZON, "evaluation_steps": list(evaluation_steps()),
        "retention_steps": [HORIZON], "phase": "donor" if donor else "stage2",
        "stage2": not donor, "engine_kind": "stage1_mature" if donor else "stage2_mature",
        "initialization": "FRESH", "aligner_kind": "direct", "upsampler": "interp23",
        "frame": "pan", "shift_scale": 1., "aligner_outputs": 2, "pan_channels": 1,
        "alignment_target": "SCHARR_OF_MEAN8_GT", "policy_id": spec.policy_id,
        "geometry": policy_manifest(spec), "band_augmentation": "NONE",
        "band_augmentation_spec": None, "lr_U": 1e-4, "lr_A": 1e-5 if donor else 0.,
        "weight_decay": .01, "lambda_rec": 1., "lambda_abs": 2. if donor else 0.,
        "lambda_epsilon": .003 if donor else 0., "lambda_eps_RR": .003 if donor else 0.,
        "epsilon_radius_rr": 3., "epsilon_distribution": "UNIFORM_AREA_DISK",
        "hqnr_min": .960, "ergas_max": 2.045,
        "checkpoint_selection": deepcopy(STAGE1_SELECTION) if donor else POLICY.manifest(),
        "loss_mask_policy": LOSS_MASK_POLICY, "mask_policy": "MASK2",
        "baseline_alignment_protocol_digest": alignment_protocol_digest(b),
        "config_digest": canonical_sha256(campaign),
        "evidence_level": "EXPLORATORY_TEST_AWARE_PAIRED_SCREEN_NOT_CONFIRMATORY",
        "student_architecture": {"bands": 8, "width": 112, "depth": [1, 2, 3]},
        "stage2_spec": asdict(spec),
        "augmentation": {"hflip_probability": 1., "vflip_probability": 1., "rot90": True},
        "training_spec": asdict(TrainingSpec(lr=1e-4, total_updates=HORIZON,
                                           warmup_updates=100, full_audit_interval=100)),
        "donor_version": DONORS[0],
        "soft_distillation": "REMOVED; Teacher retained for e_T/d_e only",
        "question": "Mature all-band donor; hard-only e+q, PAN-only applied-geometry control",
    }


def rank(record):
    return POLICY.rank(record["HQNR"], record["ERGAS"], record["completed_update"])


def catalog_csv(server):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["order", "server", "case", "replica", "seed", "phase", "policy", "optional", "updates"])
    for order, (group, case, replica) in enumerate(
            ((g, c, r) for g in groups(server) for c, r in g["runs"]), 1):
        cfg = resolve_case(server, case, replica)
        writer.writerow([order, server, case, replica, cfg["seed"], cfg["phase"],
                         cfg["policy_id"], group["optional"], HORIZON])
    return output.getvalue()
