"""S4-only immutable fixed64 evidence for fresh Direct donors and Stage 2.

Cloud display qualification and research checkpoint retention are intentionally
independent. A saved donor can be below the published joint-quality threshold.
No function in this module authenticates with or writes to Google Sheets.
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
CAMPAIGN_ID = "PANDA_S4_DONOR_STAGE2_40H_20261017_v1"
METHOD_REVISION = "PANDA-S4-DIRECT-MASK2-CONDQ-40H-v1"

RECORD_NAMESPACE = "PAN_FIXED64_V1:2026-10-17:PANDAS4"
RECORD_PREFIX = "P4-"


class ReportingError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def destination(server, dataset="WV3"):
    if server not in ("s4",) or dataset != "WV3":
        raise ReportingError("PandaS4 requires WV3 on s4")
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
        raise ReportingError("Wrong PandaS4 campaign or method revision")
    if (cfg.get("aligner_kind") != "direct" or
            cfg["baseline_resolved"]["aligner_architecture"]["name"] != "DirectPANAligner"):
        raise ReportingError("S4 campaign requires the existing DirectPANAligner")
    if not isinstance(cfg.get("loss_mask_policy"), str) or not cfg["loss_mask_policy"]:
        raise ReportingError("Explicit loss support policy required")
    for key in ("run_id", "case", "campaign_id"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise ReportingError("Missing identity: " + key)
    if Path(cfg["run_id"]).name != cfg["run_id"] or cfg["run_id"] in (".", ".."):
        raise ReportingError("Run identity must be a safe path component")
    if not cfg["run_id"].startswith("P4_"):
        raise ReportingError("Distinct P4_ run namespace required")
    for key, minimum in (("attempt", 1), ("seed", 0), ("planned_horizon", 1)):
        if type(cfg.get(key)) is not int or cfg[key] < minimum:
            raise ReportingError("Invalid integer config: " + key)
    return cfg


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
    model_sha = identity.get("model_state_sha256")
    if metrics and not _sha(model_sha):
        raise ReportingError("Numerical observation requires model-state SHA")
    if identity and any(identity.get(k) != v for k, v in
        (("run_id", cfg["run_id"]), ("attempt", cfg["attempt"]),
         ("seed", cfg["seed"]), ("completed_update", step),
         ("campaign_id", cfg["campaign_id"]), ("method_revision", cfg["method_revision"]),
         ("server", cfg["server"]), ("dataset", cfg["dataset"]), ("phase", cfg["phase"]))):
        raise ReportingError("Evaluation identity/config mismatch")
    if identity and not all(_sha(identity.get(k)) for k in
            ("model_state_sha256", "inference_policy_digest", "alignment_protocol_digest")):
        raise ReportingError("Evaluation state/policy digests required")
    if "HQNR" in metrics and "ERGAS" in metrics and extra.get("same_state_verified") is not True:
        raise ReportingError("Mixed RR/FR states cannot be reported")
    artifact = dict(artifact or {})
    saved = artifact.get("weights_saved") is True
    checkpoint = artifact.get("file_sha256") if saved else None
    if saved and (not _sha(checkpoint) or artifact.get("model_state_sha256") != model_sha
                  or artifact.get("identity") != identity or not artifact.get("checkpoint_path")):
        raise ReportingError("Saved file must identify this exact evaluated state")
    if not saved and artifact.get("file_sha256") is not None:
        raise ReportingError("Unsaved artifact cannot claim a file SHA")
    extra.update(configuration=deepcopy(cfg), identity=identity,
                 artifact=artifact, weights_saved=saved)
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
        "retention_is_not_joint_quality_pass": True}
    fields = {"Experiment": cfg["run_display"], "Approach": "PANDAS4 / " + phase(cfg) + " / " + cfg["loss_mask_policy"],
        "Role": "Teacher+Aligner" if phase(cfg) == "donor" else "Student+FrozenAligner",
        "Server": cfg["server"], "Dataset": cfg["dataset"], "Bucket": "ablations",
        "Campaign_ID": cfg["campaign_id"], "Run_ID": cfg["run_id"], "Case_ID": cfg["case"],
        "Repeat": int(repeat[1:]) if isinstance(repeat, str) and repeat.startswith("R") else int(repeat),
        "Seed": cfg["seed"], "Attempt": cfg["attempt"], "Selected_step": step,
        "Updates": step, "Lifetime_updates": step, "Status": status,
        "Checkpoint_SHA": checkpoint, "Selector": selector, "Readback": "PENDING",
        "Eval_scope": scope, "Test_aware": True, "Result_ID": record_id,
        "Selection@Step": selector + "@" + str(step), "Date": timestamp,
        "U_peak_lr": cfg.get("lr_U", cfg.get("lr_S")),
        "A_peak_lr": cfg.get("lr_A", 0.0) if phase(cfg) == "donor" else 0.0,
        "Original description / notes": canonical_json(note)}
    for key, value in metrics.items():
        if key not in METRICS:
            raise ReportingError("Unknown metric: " + key)
        fields[METRICS[key]] = value
    row = serialize_row(fields)
    if len(row[INDEX["Original description / notes"]]) >= 45000:
        raise ReportingError("Provenance exceeds safe sheet-cell limit")
    return {"record_id": record_id, "identity": ident,
        "destination": destination(cfg["server"], cfg["dataset"]), "row": row,
        "payload_sha256": digest(row), "record_type": scope, "created_at": timestamp,
        "source": {"metrics": metrics, "fields": fields, "extra": extra, "provenance": {}}}


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


def make_record(cfg, evaluation, artifact=None, timestamp=None, *, root=None):
    identity = evaluation["state_before"]
    if any(evaluation.get(k) != identity for k in ("rr_state", "fr_state", "state_after")):
        raise ReportingError("Mixed RR/FR evaluation state")
    gate = deepcopy(evaluation["qualification"])
    metrics = {**evaluation["rr"]["metrics"], **evaluation["fr"]["metrics"]}
    step = identity["completed_update"]
    suffix = Path("runs") / cfg["run_id"] / ("attempt_%02d" % cfg["attempt"]) / ("evaluation_%07d.json" % step)
    gate["source_evaluation"] = str(Path(root) / suffix if root is not None else suffix)
    return build_record(cfg, scope="EVAL_SUMMARY", selector="OBSERVED_%07d" % step,
        step=step, metrics=metrics, identity=identity, extra=gate,
        artifact=artifact, timestamp=timestamp)


def emit_evaluation(root, cfg, evaluation, artifact=None, timestamp=None):
    record = make_record(cfg, evaluation, artifact, timestamp, root=root)
    LocalOutbox(root).enqueue(record)
    return record


def emit_status(root, cfg, status, step=0, extra=None, timestamp=None):
    timestamp = timestamp or utc_now()
    record = build_record(cfg, scope="STATUS", selector=status + ":" + timestamp,
        step=step, status=status, extra=extra, timestamp=timestamp)
    LocalOutbox(root).enqueue(record)
    return record
