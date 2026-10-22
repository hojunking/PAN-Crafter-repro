"""Immutable Scale30 fixed64 evidence. No cloud access on import or emission.

emit_evaluation(..., selector_receipt=None) accepts original unchanged full
evaluations for reused Stage1 donors. Receipt.selected binds original files.
Stage2 evaluations carry new campaign identity and deployment_manifest.
"""
from __future__ import annotations

from copy import deepcopy
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import re

from directshift48.reporting import (HEADERS, INDEX, METRICS, atomic_json,
    append_jsonl, canonical_json, serialize_row, utc_now)
from .contract import CAMPAIGN_ID, METHOD_REVISION, CASES, SCALE_POLICIES

RECORD_NAMESPACE = "PAN_FIXED64_V1:2026-10-22:PANDA_SCALE30_HARDTUNE_V1"
RECORD_PREFIX = "PS30-"


class ReportingError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def scientific_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def compact_receipt(root, receipt):
    """Persist complete per-patch validation evidence, embed only selected summary.

    The compact envelope has its own digest, never the full receipt's digest.
    """
    selected_source(receipt)
    if receipt.get("source_receipt_path"):
        return deepcopy(receipt)
    if root is None:
        raise ReportingError("Receipt storage requires explicit campaign root")
    path = Path(root) / "selection_receipts" / (receipt["receipt_sha256"] + ".json")
    if path.exists():
        if scientific_digest(json.loads(path.read_text())) != scientific_digest(receipt):
            raise ReportingError("Immutable selection receipt collision")
    else:
        atomic_json(path, receipt)
    selected = deepcopy(receipt["selected"])
    selected["validation"] = {k: v for k, v in selected["validation"].items()
                             if k not in ("per_sample", "protocol")}
    compact = {"schema": "PANDA_SCALE30_COMPACT_SELECTION_V1",
        "selector": "BEST_VAL_RRE", "campaign_id": receipt.get("campaign_id"),
        "server": receipt.get("server"), "selected": selected,
        "source_root": receipt.get("source_root", selected.get("source_root")),
        "source_receipt_path": str(path.resolve()),
        "source_receipt_sha256": receipt["receipt_sha256"]}
    compact["receipt_sha256"] = scientific_digest(compact)
    return compact


def deployment(evaluation):
    value = evaluation.get("mature30_deployment", evaluation.get("deployment_manifest"))
    if ("deployment_manifest" in evaluation and "mature30_deployment" in evaluation
            and evaluation["deployment_manifest"] != evaluation["mature30_deployment"]):
        raise ReportingError("Contradictory deployment manifests")
    return value


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def destination(server, dataset="WV3"):
    if server not in ("s1", "s3") or dataset != "WV3":
        raise ReportingError("Scale30 requires WV3 on s1 or s3")
    return {"spreadsheet_id": "1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE",
            "sheet_title": "WV3-ablations", "sheet_id": 261001010}


def phase(cfg):
    result = cfg.get("phase", cfg.get("stage"))
    if result not in ("donor", "stage2"):
        raise ReportingError("Explicit donor/stage2 phase required")
    return result


def cost_config(cfg):
    """Key Stage 2 costs on deployed A+S, never on its training-only Teacher."""
    value = deepcopy(cfg)
    if phase(cfg) == "stage2":
        baseline = value["baseline_resolved"]
        student = cfg.get("student_architecture", baseline.get("student_architecture"))
        if not isinstance(student, dict):
            raise ReportingError("Stage 2 cost needs explicit student_architecture")
        baseline["teacher_architecture"] = deepcopy(student)
    return value


def _config(cfg):
    destination(cfg["server"], cfg["dataset"])
    phase(cfg)
    if cfg.get("campaign_id") != CAMPAIGN_ID or cfg.get("method_revision") != METHOD_REVISION:
        raise ReportingError("Wrong Scale30 campaign or method revision")
    if cfg.get("upsampler") != "interp23" or cfg.get("frame") != "pan":
        raise ReportingError("Scale30 requires interp23 and original MS output coordinates")
    if cfg.get("aligner_outputs") != 2 or cfg.get("pan_channels") != 1:
        raise ReportingError("Scale30 requires two Aligner outputs and one PAN")
    if cfg.get("alignment_target") != "SCHARR_OF_MEAN8_GT":
        raise ReportingError("Scale30 requires the G8 alignment target")
    if phase(cfg) == "donor":
        if cfg.get("policy_id") != "FULL":
            raise ReportingError("Reused Stage1 retains its original full-u policy")
    else:
        _stage2_config(cfg)
    for key in ("run_id", "case", "engine_kind", "run_display"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ReportingError("Missing identity: " + key)
    run = cfg["run_id"]
    if Path(run).name != run or not run.startswith("PS30_WV3_" + cfg["server"].upper() + "_"):
        raise ReportingError("Distinct server-bound PS30 run namespace required")
    for key, minimum in (("attempt", 1), ("seed", 0), ("planned_horizon", 1)):
        if type(cfg.get(key)) is not int or cfg[key] < minimum:
            raise ReportingError("Invalid integer config: " + key)
    return cfg


def _stage2_config(cfg):
    """Validate the registered knobs without importing GPU/training packages."""
    case = CASES.get(cfg.get("case"))
    if case is None or cfg.get("policy_id") != case["policy_id"]:
        raise ReportingError("Unregistered Scale30 case/policy")
    spec = cfg.get("stage2_spec", {})
    for field, expected in (("alpha", case["alpha"]), ("gamma", case["gamma"]),
                            ("beta", 0.), ("delta", 1e-6),
                            ("policy_id", case["policy_id"])):
        if spec.get(field) != expected:
            raise ReportingError("Registered hard-only objective differs: " + field)
    if (cfg.get("shift_scale") != case["scale"]
            or not (cfg.get("lr_S") == cfg.get("lr_U")
                    == cfg.get("training_spec", {}).get("lr") == case["lr_S"])
            or cfg.get("tuning_parameters") != case
            or SCALE_POLICIES[case["policy_id"]] != case["scale"]):
        raise ReportingError("Registered positive scale/learning rate differs")
    if (cfg.get("student_views") != 1 or cfg.get("rotation_consistency_weight") != 0.
            or cfg.get("pair_transform") != "NONE" or cfg.get("inference_tta") is not False):
        raise ReportingError("Scale30 requires one view, no consistency loss and no TTA")
    policy = cfg.get("geometry", {})
    phases = policy.get("phases", [])
    key = case["policy_id"].lower()
    if (policy.get("schema") != "PANDA_SCALE30_POLICY_V1"
            or policy.get("policy_id") != case["policy_id"]
            or policy.get("total_updates") != 50000 or len(phases) != 1
            or phases[0].get("first_update") != 1 or phases[0].get("last_update") != 50000
            or phases[0].get("geometry_key") != key):
        raise ReportingError("Registered fixed-scale schedule differs")
    geometry = phases[0].get("geometry", {})
    for field, expected in (("schema", "PANDA_SCALE30_APPLIED_GEOMETRY_V1"),
            ("geometry_key", key), ("operator", "SCALE"), ("scale", case["scale"]),
            ("cap_px", None), ("upsampler", "interp23"), ("frame", "pan"),
            ("output_coordinates", "MS"), ("ms_shift_factor", 0.), ("output_shift_factor", 0.),
            ("shift_unit", "CURRENT_PAN_PIXEL_NO_X4"),
            ("evaluation_references", "ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20")):
        if geometry.get(field) != expected:
            raise ReportingError("Registered applied geometry differs: " + field)


def selected_source(receipt):
    if not isinstance(receipt, dict) or receipt.get("selector") != "BEST_VAL_RRE":
        raise ReportingError("Stage1 requires an explicit BEST_VAL_RRE receipt")
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if receipt.get("receipt_sha256") != scientific_digest(unsigned):
        raise ReportingError("RRE selection receipt digest mismatch")
    selected = receipt.get("selected")
    if not isinstance(selected, dict):
        raise ReportingError("RRE receipt lacks selected source")
    for key in ("source_config", "source_state", "validation"):
        if not isinstance(selected.get(key), dict):
            raise ReportingError("RRE selected source lacks " + key)
    for key in ("source_evaluation_path", "checkpoint_path"):
        if not isinstance(selected.get(key), str) or not selected[key]:
            raise ReportingError("RRE selected source lacks " + key)
    if not _sha(selected.get("file_sha256")):
        raise ReportingError("RRE source checkpoint SHA required")
    import math
    for key in ("RRE50", "RRE90"):
        value = selected["validation"].get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ReportingError("Finite nonnegative validation " + key + " required")
    if selected["validation"]["RRE90"] < selected["validation"]["RRE50"]:
        raise ReportingError("RRE90 cannot be smaller than RRE50")
    return selected

def update_lineage(cfg, step):
    """Selected_step/Updates are LOCAL; lifetime includes inherited updates.

    Never label a ten-thousand-update tail as a fresh fifty-thousand-update run.
    Imported donor training is lineage, not charged to this run's Train(h).
    """
    return {"local_updates": step, "start_update": cfg.get("start_update", 0),
            "lifetime_updates": cfg.get("start_update", 0) + step,
            "parent_run": cfg.get("parent_run", cfg.get("parent_run_id", "")),
            "parent_binding": deepcopy(cfg.get("parent_binding")),
            "timing_scope": "local_run_only_excludes_parent_and_donor_training"}


def build_record(cfg, *, scope, selector, step, metrics=None, identity=None,
                 extra=None, artifact=None, status="OBSERVED", timestamp=None):
    _config(cfg)
    if type(step) is not int or not 0 <= step <= cfg["planned_horizon"]:
        raise ReportingError("Selected step outside registered horizon")
    timestamp = timestamp or utc_now()
    parsed = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReportingError("Timestamp requires timezone")
    identity, metrics, extra = dict(identity or {}), dict(metrics or {}), dict(extra or {})
    receipt = extra.get("selector_receipt")
    imported = selected_source(receipt) if receipt is not None else None
    reused = bool(imported and (imported["source_config"]["campaign_id"] != cfg["campaign_id"]
                  or imported["source_config"]["run_id"] != cfg["run_id"]))
    if metrics and phase(cfg) == "donor" and imported is None:
        raise ReportingError("Stage1 numerical publication needs a selection receipt")
    expected_cfg = imported["source_config"] if imported else cfg
    if imported and identity != imported["source_state"]:
        raise ReportingError("Imported evaluation must preserve original source identity")
    model_sha = identity.get("model_state_sha256")
    if metrics and not _sha(model_sha):
        raise ReportingError("Numerical observation requires model-state SHA")
    if identity and any(identity.get(k) != v for k, v in
        (("run_id", expected_cfg["run_id"]), ("attempt", expected_cfg["attempt"]),
         ("seed", expected_cfg["seed"]), ("completed_update", step),
         ("campaign_id", expected_cfg["campaign_id"]), ("method_revision", expected_cfg["method_revision"]),
         ("server", expected_cfg["server"]), ("dataset", expected_cfg["dataset"]), ("phase", expected_cfg["phase"]))):
        raise ReportingError("Evaluation identity/config mismatch")
    if identity and not all(_sha(identity.get(k)) for k in
            ("model_state_sha256", "inference_policy_digest", "alignment_protocol_digest")):
        raise ReportingError("Evaluation state/policy digests required")
    if identity and (identity.get("engine_kind") != expected_cfg["engine_kind"] or
                     identity.get("lifetime_update") != expected_cfg.get("start_update", 0) + step):
        raise ReportingError("Evaluation engine/lifetime lineage mismatch")
    if "HQNR" in metrics and "ERGAS" in metrics and extra.get("same_state_verified") is not True:
        raise ReportingError("Mixed RR/FR states cannot be reported")
    artifact = dict(artifact or {})
    saved = artifact.get("weights_saved") is True
    checkpoint = artifact.get("file_sha256") if saved else None
    if saved and (not _sha(checkpoint) or artifact.get("model_state_sha256") != model_sha
                  or artifact.get("identity") != identity or not artifact.get("checkpoint_path")):
        raise ReportingError("Saved file must identify this exact evaluated state")
    if saved and imported and checkpoint != imported["file_sha256"]:
        raise ReportingError("Selected source checkpoint and saved artifact SHA differ")
    if not saved and artifact.get("file_sha256") is not None:
        raise ReportingError("Unsaved artifact cannot claim a file SHA")
    lineage = update_lineage(cfg, step)
    if reused:
        lineage.update(local_updates=0, lifetime_updates=identity["lifetime_update"],
                       parent_run=identity["run_id"], timing_scope="REUSED_SOURCE_NO_NEW_TRAINING")
    extra.update(configuration=deepcopy(cfg), identity=identity,
                 artifact=artifact, weights_saved=saved, update_lineage=lineage)
    ident = {"namespace": RECORD_NAMESPACE, "campaign": cfg["campaign_id"],
        "dataset": cfg["dataset"], "server": cfg["server"], "run": cfg["run_id"],
        "attempt": cfg["attempt"], "selected_step": step, "selector": selector,
        "eval_scope": scope, "eval_protocol": identity.get("evaluator_revision", "STATUS_ONLY"),
        "model_state_sha256": model_sha, "checkpoint": checkpoint}
    record_id = RECORD_PREFIX + digest(ident)
    repeat = cfg.get("replica", "R1")
    note = {"schema_version": "PAN_FIXED64_V1", "reset_epoch": "2026-10-13",
        "model_state_sha256": model_sha, "eval_protocol_id": ident["eval_protocol"],
        "config_sha256": digest(cfg), "extra_parameters": extra,
        "method_revision": cfg.get("method_revision"),
        "retention_is_not_joint_quality_pass": True,
        "output_coordinate_system": "MIDPOINT" if cfg["frame"] == "mid" else "MS",
        "official_references_unchanged": True,
        "geometry": deepcopy(extra.get("deployment_manifest", cfg.get("geometry", {}))), "upsampler": cfg["upsampler"],
        "aligner_outputs": cfg['aligner_outputs'], 'pan_channels': cfg['pan_channels'],
        'alignment_target': cfg['alignment_target'],
        'band_augmentation': cfg.get('band_augmentation', 'NONE'),
        "checkpoint_selection": ("BEST_VAL_RRE" if phase(cfg) == "donor" else "HQNR_DESC_ERGAS_ASC_STEP_ASC"),
        "publication_policy": "RRE_SELECTED_STAGE1_STAGE2_RUNNING_BEST_HQNR_ONE_ROW",
        "imported_source_not_new_training": reused,
        "date_semantics": "original_evaluation_time" if imported else "evaluation_time"}
    if phase(cfg) == "stage2":
        knobs = CASES[cfg["case"]]
        approach = ("PANDA-SCALE30 / Hard-only / interp23 / single view; "
                    f"s={knobs['scale']:g}; alpha={knobs['alpha']:g}; "
                    f"gamma={knobs['gamma']:g}; lr={knobs['lr_S']:g}")
    else:
        approach = "PANDA-SCALE30 / reused Stage1 / original full-u / interp23"
    fields = {"Experiment": cfg["run_display"],
        "Approach": approach,
        "Role": ("Reused Teacher+Aligner (RRE selection)" if reused else
                 "Teacher+Aligner" if phase(cfg) == "donor" else "Student+FrozenAligner"),
        "Server": cfg["server"], "Dataset": cfg["dataset"], "Bucket": "ablations",
        "Campaign_ID": cfg["campaign_id"], "Run_ID": cfg["run_id"], "Case_ID": cfg["case"],
        "Repeat": int(repeat[1:]) if isinstance(repeat, str) and repeat.startswith("R") else int(repeat),
        "Seed": cfg["seed"], "Attempt": cfg["attempt"], "Selected_step": step,
        "Updates": lineage["local_updates"], "Lifetime_updates": lineage["lifetime_updates"], "Status": status,
        "Parent_run": lineage["parent_run"],
        "Teacher / reference": cfg.get("donor_key", cfg.get("donor_id", "")),
        "Checkpoint_SHA": checkpoint, "Selector": selector, "Readback": "PENDING",
        "Eval_scope": scope, "Test_aware": True, "Result_ID": record_id,
        "JQM_variant": extra.get("jqm_variant", ""),
        "Selection@Step": selector + "@" + str(step), "Date": timestamp,
        "U_peak_lr": cfg.get("lr_U", cfg.get("lr_S")),
        "A_peak_lr": cfg.get("lr_A", 0.0),
        "alpha": cfg.get("stage2_spec", {}).get("alpha", ""),
        "beta": cfg.get("stage2_spec", {}).get("beta", ""),
        "lambda_E": cfg.get("lambda_epsilon", 0.),
        "Original description / notes": canonical_json(note)}
    for key, value in metrics.items():
        if key not in METRICS:
            raise ReportingError("Unknown metric: " + key)
        fields[METRICS[key]] = value
    row = serialize_row(fields)
    if len(row[INDEX["Original description / notes"]]) >= 45000:
        raise ReportingError("Provenance exceeds safe sheet-cell limit")
    return json.loads(canonical_json({"record_id": record_id, "identity": ident,
        "destination": destination(cfg["server"], cfg["dataset"]), "row": row,
        "payload_sha256": digest(row), "record_type": scope, "created_at": timestamp,
        "source": {"metrics": metrics, "fields": fields, "extra": extra, "provenance": {}}}))


def validate_record(record):
    source, ident = record["source"], record["identity"]
    extra = source["extra"]
    rebuilt = build_record(extra["configuration"], scope=record["record_type"],
        selector=ident["selector"], step=ident["selected_step"], metrics=source["metrics"],
        identity=extra["identity"], extra=extra, artifact=extra["artifact"],
        status=source["fields"]["Status"], timestamp=record["created_at"])
    if canonical_json(rebuilt) != canonical_json(record):
        raise ReportingError("Immutable envelope/source/row mismatch")


class LocalOutbox:
    """Idempotent immutable evidence, durable ledger before atomic outbox file."""
    def __init__(self, root):
        self.root = Path(root)
        self.outbox = self.root / "outbox"
        self.outbox.mkdir(parents=True, exist_ok=True)

    def enqueue(self, record):
        validate_record(record)
        target = self.outbox / (record["record_id"] + ".json")
        with (self.root / ".enqueue.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists():
                if json.loads(target.read_text()) != record:
                    raise ReportingError("Immutable source ID has conflicting payload")
                return record["record_id"]
            append_jsonl(self.root / "records.jsonl", record)
            atomic_json(target, record)
        return record["record_id"]


def make_record(cfg, evaluation, artifact=None, timestamp=None, *, root=None, selector_receipt=None):
    identity = evaluation["state_before"]
    if any(evaluation.get(k) != identity for k in ("rr_state", "fr_state", "state_after")):
        raise ReportingError("Mixed RR/FR evaluation state")
    gate = deepcopy(evaluation["qualification"])
    gate["jqm_variant"] = evaluation.get("jqm_variant", "")
    gate["deployment_manifest"] = deepcopy(deployment(evaluation))
    if selector_receipt is not None:
        selector_receipt = compact_receipt(root, selector_receipt)
        gate["selector_receipt"] = deepcopy(selector_receipt)
    metrics = {**evaluation["rr"]["metrics"], **evaluation["fr"]["metrics"]}
    step = identity["completed_update"]
    suffix = Path("runs") / cfg["run_id"] / ("attempt_%02d" % cfg["attempt"]) / ("evaluation_%07d.json" % step)
    gate["source_evaluation"] = (selected_source(selector_receipt)["source_evaluation_path"]
        if selector_receipt is not None else str(Path(root) / suffix if root is not None else suffix))
    return build_record(cfg, scope="EVAL_SUMMARY", selector="BEST_VAL_RRE" if selector_receipt is not None else "OBSERVED_%07d" % step,
        step=step, metrics=metrics, identity=identity, extra=gate,
        artifact=artifact, timestamp=timestamp)


def emit_evaluation(root, cfg, evaluation, artifact=None, timestamp=None, selector_receipt=None):
    record = make_record(cfg, evaluation, artifact, timestamp, root=root, selector_receipt=selector_receipt)
    LocalOutbox(root).enqueue(record)
    return record


def emit_status(root, cfg, status, step=0, extra=None, timestamp=None):
    timestamp = timestamp or utc_now()
    record = build_record(cfg, scope="STATUS", selector=status + ":" + timestamp,
        step=step, status=status, extra=extra, timestamp=timestamp)
    LocalOutbox(root).enqueue(record)
    return record

