"""Local-only immutable fixed64 evidence for DirectShift R3.

Training only appends JSONL/outbox envelopes. Raw scene/trace/status events are
never uploaded. The independent publication module derives qualified-step or
best-HQNR views while preserving authenticated historical campaigns.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Dict, Mapping, Optional

from .plan import (CAMPAIGN_ID, PARENT_CAMPAIGN_ID, METHOD_REVISION,
                   METHOD_CONTRACT, validate_method_config)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "research_log/gspread_rules/PAN_FIXED64_V1.schema.json"
SPREADSHEET_ID = "1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE"
HEADER_SHA256 = "bfc7f73fd406f447edb9721823e2c1df85540f7068e15647947cd0d2e83a11b1"
SCHEMA_VERSION = "PAN_FIXED64_V1"
RESET_EPOCH = "2026-10-13"
RECORD_NAMESPACE = "PAN_FIXED64_V1:2026-10-14:DIRECTSHIFT_R3"
DESTINATIONS = {"s1": ("WV3", "WV3-ablations", 261001010),
                "s3": ("GF2", "GF2-ablations", 261001012)}


class ReportingError(ValueError):
    pass


class RecordConflict(ReportingError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _schema() -> Dict[str, Any]:
    obj = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    headers = [c["header"] for c in obj["columns"]]
    if (len(headers) != 64 or _sha(headers) != HEADER_SHA256 or
            obj["spreadsheet_id"] != SPREADSHEET_ID or
            obj["schema_version"] != SCHEMA_VERSION or obj["reset_epoch"] != RESET_EPOCH):
        raise ReportingError("Local fixed64 schema differs from the pinned contract")
    return obj


SCHEMA = _schema()
HEADERS = tuple(c["header"] for c in SCHEMA["columns"])
INDEX = {name: index for index, name in enumerate(HEADERS)}
METRICS = {"HQNR": "HQNR↑", "D_s": "D_s↓", "Ds": "D_s↓",
           "D_lambda": "D_lambda↓", "Dl": "D_lambda↓", "JQM": "JQM↑",
           "ERGAS": "ERGAS↓", "SCC": "SCC↑", "SAM": "SAM↓",
           "PSNR": "PSNR↑", "SSIM": "SSIM↑", "Q4/Q8": "Q4/Q8↑",
           "Q4": "Q4/Q8↑", "Q8": "Q4/Q8↑", "Q2n": "Q4/Q8↑",
           "RMSE": "RMSE↓", "CC": "CC↑"}
OBSOLETE_PARAMETERS = ("q_ref", "tau_R", "alpha", "beta", "lambda_E")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def destination(server: str, dataset: str) -> Dict[str, Any]:
    expected = DESTINATIONS.get(server)
    if expected is None or dataset != expected[0]:
        raise ReportingError("Only s1/WV3 or s3/GF2 may publish this campaign")
    return {"spreadsheet_id": SPREADSHEET_ID, "sheet_title": expected[1],
            "sheet_id": expected[2]}


def serialize_row(fields: Mapping[str, Any]) -> list:
    unknown = set(fields) - set(HEADERS)
    if unknown:
        raise ReportingError("Unknown top-level sheet fields: " + str(sorted(unknown)))
    row = []
    for spec in SCHEMA["columns"]:
        value = fields.get(spec["header"])
        if value is None or value == "":
            row.append("")
            continue
        kind = spec["type"]
        if kind == "number":
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ReportingError("Finite number required: " + spec["header"])
        elif kind == "boolean":
            if type(value) is not bool:
                raise ReportingError("Actual boolean required: " + spec["header"])
        elif not isinstance(value, str):
            raise ReportingError("String required: " + spec["header"])
        row.append(value)
    return row


def build_record(*, server: str, dataset: str, run_id: str, case_id: str,
                 selector: str, eval_scope: str, eval_protocol_id: str,
                 checkpoint_sha: Optional[str] = None,
                 model_state_sha: Optional[str] = None,
                 metrics: Optional[Mapping[str, Any]] = None,
                 fields: Optional[Mapping[str, Any]] = None,
                 extra: Optional[Mapping[str, Any]] = None,
                 provenance: Optional[Mapping[str, Any]] = None,
                 timestamp: Optional[str] = None) -> Dict[str, Any]:
    """Build a numeric/status/diagnostic row; never filter on target pass.

    ``fields`` accepts exact fixed64 header names; method-specific settings and
    full evidence go into ``extra`` (BI JSON). One scene is one scope, e.g.
    ``RR_SCENE:scene001``. Status selector/scope must identify each event, and a
    correction needs a new selector plus supersedes_record_id/reason in extra.
    Evaluations carrying both HQNR and ERGAS must explicitly attest the same
    checkpoint via extra.same_state_verified; checkpoint SHA is the saved file SHA only. A separate model_state_sha
    identifies every numeric observation, including unsaved failed candidates.
    """
    dest = destination(server, dataset)
    for name, value in (("run_id", run_id), ("case_id", case_id),
                        ("selector", selector), ("eval_scope", eval_scope),
                        ("eval_protocol_id", eval_protocol_id)):
        if not isinstance(value, str) or not value.strip():
            raise ReportingError(name + " must be a nonempty string")
    if checkpoint_sha is not None and (not isinstance(checkpoint_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha)):
        raise ReportingError("checkpoint_sha must be an actual saved file SHA256 or null")
    metrics, extra, provenance = dict(metrics or {}), dict(extra or {}), dict(provenance or {})
    if not run_id.startswith('DS3_'):
        raise ReportingError('Only DS3_ run identities may enter this campaign')
    for key, value in METHOD_CONTRACT.items():
        if key in extra and extra[key] != value:
            raise ReportingError('Conflicting method semantics: ' + key)
    for key in ('lambda_eps_FR', 'lambda_epsilon_FR', 'requires_fr_train', 'requires_source_scene_map'):
        if extra.get(key) not in (None, False, 0):
            raise ReportingError('Obsolete FR-training/scene dependency is forbidden: ' + key)
    for key in ('b_s', 'b_global', 'b_i', 'scene_prior', 'global_prior_asset_sha256', 'edge_protocol_digest', 'edge_match'):
        if extra.get(key) is not None:
            raise ReportingError('Obsolete fixed-prior semantics: ' + key)
    if 'configuration' in extra:
        validate_method_config(extra['configuration'])
    for key in ('evaluation_identity', 'identity'):
        if key in extra and (not isinstance(extra[key], Mapping)
                            or any(extra[key].get(k) != v for k,v in METHOD_CONTRACT.items())):
            raise ReportingError('Observation identity has obsolete method semantics: ' + key)
        if key in extra:
            identity = extra[key]
            if ('edge_protocol_digest' in identity or
                    not isinstance(identity.get('alignment_protocol_digest'), str) or
                    not re.fullmatch(r'[0-9a-f]{64}', identity['alignment_protocol_digest'])):
                raise ReportingError('Direct alignment protocol SHA required; no matcher identity alias')
    data = dict(fields or {})
    unknown = set(provenance) - {"config_sha256", "code_commit", "dataset_manifest_sha256",
                                "evaluator_revision", "artifact_manifest_url"}
    if unknown:
        raise ReportingError("Unknown provenance fields: " + str(sorted(unknown)))
    for key in OBSOLETE_PARAMETERS:
        if data.get(key) not in (None, ""):
            raise ReportingError(key + " belongs to the old method; use extra_parameters")
    for key, value in metrics.items():
        if key not in METRICS:
            raise ReportingError("Unknown metric: " + key)
        target = METRICS[key]
        if target in data and data[target] != value:
            raise ReportingError("Conflicting metric values: " + key)
        data[target] = value
    metric_present = any(data.get(name) is not None and data.get(name) != "" for name in METRICS.values())
    if model_state_sha is not None and (not isinstance(model_state_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", model_state_sha)):
        raise ReportingError("model_state_sha must be a full logical SHA256")
    aggregate = extra.get('aggregate_no_single_model') is True
    if aggregate and (eval_scope != 'CAMPAIGN_AGGREGATE' or metric_present
                      or model_state_sha is not None or checkpoint_sha is not None):
        raise ReportingError('Multi-state campaign aggregates require their own scope, blank metric columns and no single model/file SHA')
    if (metric_present or extra.get('numeric_observations')) and model_state_sha is None and not aggregate:
        raise ReportingError("Metric observations require the model-state SHA")
    artifact = extra.get('artifact', {})
    if not isinstance(artifact, Mapping):
        raise ReportingError('Artifact evidence must be an object')
    if checkpoint_sha is not None:
        if (extra.get('joint_pass') is not True or artifact.get('weights_saved') is not True
                or artifact.get('file_sha256') != checkpoint_sha
                or artifact.get('model_state_sha256') != model_state_sha
                or model_state_sha is None or extra.get('weights_saved') is False):
            raise ReportingError('Checkpoint_SHA requires a qualified saved artifact/file-SHA and same model-state evidence')
    elif artifact.get('weights_saved') is True or extra.get('weights_saved') is True:
        raise ReportingError('Saved artifacts must supply their actual checkpoint file SHA')
    if data.get("HQNR↑") not in (None, "") and data.get("ERGAS↓") not in (None, ""):
        if extra.get("same_state_verified") is not True:
            raise ReportingError("HQNR and ERGAS require explicit same-state verification")
    for name in ("rr_model_state_sha256", "fr_model_state_sha256", "model_state_sha256"):
        if name in extra and extra[name] != model_state_sha:
            raise ReportingError("Mixed checkpoint identity: " + name)
    if "supersedes_record_id" in extra and not extra.get("correction_reason"):
        raise ReportingError("A correction requires a reason")
    timestamp = timestamp or utc_now()
    try:
        parsed = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone missing")
    except (TypeError, ValueError) as exc:
        raise ReportingError("Date must be a timezone-aware ISO 8601 string") from exc
    identity = {"namespace": RECORD_NAMESPACE,
                "campaign": CAMPAIGN_ID, "dataset": dataset, "server": server,
                "run": run_id, "selector": selector, "checkpoint": checkpoint_sha,
                "model_state_sha256": model_state_sha,
                "eval_protocol": eval_protocol_id, "eval_scope": eval_scope,
                "attempt": data.get("Attempt"), "selected_step": data.get("Selected_step")}
    record_id = "DS3-" + _sha(identity)
    note = {"schema_version": SCHEMA_VERSION, "reset_epoch": RESET_EPOCH,
            **METHOD_CONTRACT, "parent_campaign_id": PARENT_CAMPAIGN_ID,
            "model_state_sha256": model_state_sha,
            "config_sha256": None,
            "code_commit": None, "dataset_manifest_sha256": None,
            "evaluator_revision": None, "artifact_manifest_url": None,
            **provenance, "eval_protocol_id": eval_protocol_id,
            "extra_parameters": extra}
    required = {"Server": server, "Dataset": dataset, "Bucket": "ablations",
                "Campaign_ID": CAMPAIGN_ID, "Run_ID": run_id, "Case_ID": case_id,
                "Checkpoint_SHA": checkpoint_sha, "Selector": selector,
                "Readback": "PENDING", "Eval_scope": eval_scope, "Test_aware": True,
                "Result_ID": record_id, "Original description / notes": canonical_json(note),
                "Date": timestamp}
    for key, value in required.items():
        if key in data and data[key] != value:
            raise ReportingError("Reserved field conflict: " + key)
        data[key] = value
    data.setdefault("Experiment", run_id)
    data.setdefault("Approach", "Stage 1 Teacher — " + METHOD_REVISION)
    data.setdefault("Role", "Teacher+Aligner")
    data.setdefault("Selection@Step", selector + "@" + str(data.get("Selected_step", "")))
    row = serialize_row(data)
    if len(row[INDEX["Original description / notes"]]) > 20000:
        raise ReportingError("BI JSON exceeds safe cell limit; split numeric scopes")
    record = {"record_id": record_id, "identity": identity, "destination": dest,
              "row": row, "payload_sha256": _sha(row), "record_type": eval_scope,
              "source": {"metrics": metrics, "fields": data, "extra": extra,
                         "provenance": provenance}, "created_at": timestamp}
    canonical_json(record)  # fail on NaN/tensors or otherwise non-JSON evidence
    return record


def build_records_split(**kwargs) -> list:
    """Build one row, or deterministic bounded inline-numeric continuation rows.

    No observation is replaced by a URL or silently truncated. Each part keeps
    the complete configuration/provenance and model-state identity. A single
    overlong leaf or base configuration fails explicitly rather than publishing
    a misleading partial record. This helper performs no IO/network operations.
    """
    try:
        return [build_record(**kwargs)]
    except ReportingError as exc:
        if 'exceeds safe cell limit' not in str(exc):
            raise
    base = dict(kwargs)
    extra = dict(base.get('extra') or {})
    observations = extra.pop('numeric_observations', None)
    if not isinstance(observations, Mapping) or not observations:
        raise ReportingError('Oversized non-numeric metadata cannot be silently truncated')
    if any(not isinstance(key, str) for key in observations):
        raise ReportingError('Inline numeric observation paths must be strings')
    numeric_sha = _sha(observations)
    selector = base['selector']
    timestamp = base.get('timestamp') or utc_now()
    def make(values, part, count):
        return build_record(**{**base, 'timestamp': timestamp,
            'selector': selector + ':NUMERIC_PART:' + str(part) + '/' + str(count),
            'extra': {**extra, 'numeric_observations': values,
                      'numeric_part_index': part, 'numeric_part_count': count,
                      'numeric_payload_sha256': numeric_sha}})
    chunks, current = [], {}
    for key, value in sorted(observations.items()):
        trial = {**current, key: value}
        try:
            make(trial, 999999999, 999999999)  # Reserve metadata length before partitioning.
        except ReportingError as exc:
            if 'exceeds safe cell limit' not in str(exc):
                raise
            if not current:
                raise ReportingError('One inline numeric leaf/base metadata exceeds the safe cell limit') from exc
            chunks.append(current)
            current = {key: value}
            make(current, 999999999, 999999999)  # A single leaf must itself fit.
        else:
            current = trial
    if current:
        chunks.append(current)
    return [make(values, part, len(chunks)) for part, values in enumerate(chunks, start=1)]


def validate_record(record: Mapping[str, Any]) -> None:
    identity, row = record["identity"], record["row"]
    dest = destination(identity["server"], identity["dataset"])
    if record["destination"] != dest or identity["campaign"] != CAMPAIGN_ID:
        raise ReportingError("Historical or misrouted record is forbidden")
    if identity["namespace"] != RECORD_NAMESPACE:
        raise ReportingError("Wrong reporting namespace")
    if len(row) != 64 or serialize_row(dict(zip(HEADERS, row))) != row:
        raise ReportingError("Invalid fixed64 payload")
    if record["record_id"] != "DS3-" + _sha(identity) or record["payload_sha256"] != _sha(row):
        raise ReportingError("Record identity or payload digest mismatch")
    for name, expected in (("Result_ID", record["record_id"]), ("Campaign_ID", CAMPAIGN_ID),
                           ("Run_ID", identity["run"]), ("Dataset", identity["dataset"]),
                           ("Server", identity["server"]), ("Selector", identity["selector"]),
                           ("Eval_scope", identity["eval_scope"]), ("Bucket", "ablations"),
                           ("Checkpoint_SHA", identity["checkpoint"] or ""),
                           ("Readback", "PENDING"), ("Test_aware", True)):
        if row[INDEX[name]] != expected or (name == "Test_aware" and type(row[INDEX[name]]) is not bool):
            raise ReportingError("Record field identity mismatch: " + name)
    if any(row[INDEX[name]] != "" for name in OBSOLETE_PARAMETERS):
        raise ReportingError("Obsolete method parameters are forbidden")
    note = json.loads(row[INDEX['Original description / notes']])
    if any(note.get(k) != v for k,v in METHOD_CONTRACT.items()):
        raise ReportingError('Stored envelope has obsolete or missing DirectShift-R3 method semantics')
    if note.get('parent_campaign_id') != PARENT_CAMPAIGN_ID:
        raise ReportingError('Stored envelope lacks original campaign provenance')
    extra = note.get('extra_parameters', {})
    if 'configuration' in extra:
        validate_method_config(extra['configuration'])
    for key in ('evaluation_identity', 'identity'):
        if key in extra and (not isinstance(extra[key], Mapping)
                            or any(extra[key].get(k) != v for k,v in METHOD_CONTRACT.items())):
            raise ReportingError('Stored observation has obsolete method semantics: ' + key)
    # Reconstruct the immutable envelope through the same admission checks.
    # A forged/torn local payload with freshly recomputed digests must not bypass
    # model-state, saved-file, method, row/source or same-state consistency.
    source = record['source']
    rebuilt = build_record(server=identity['server'], dataset=identity['dataset'],
        run_id=identity['run'], case_id=row[INDEX['Case_ID']], selector=identity['selector'],
        eval_scope=identity['eval_scope'], eval_protocol_id=identity['eval_protocol'],
        checkpoint_sha=identity['checkpoint'], model_state_sha=identity.get('model_state_sha256'),
        metrics=source['metrics'], fields=source['fields'], extra=source['extra'],
        provenance=source['provenance'], timestamp=record['created_at'])
    if canonical_json(rebuilt) != canonical_json(record):
        raise ReportingError('Immutable record/source/envelope reconstruction mismatch')


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


@contextlib.contextmanager
def _lock(path: Path, *, nonblocking: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as exc:
            raise ReportingError("A publisher is already active for this destination") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)



class LocalOutbox:
    """Append-only raw evidence; there is deliberately no raw cloud transport."""

    def __init__(self, root):
        self.root = Path(root)
        self.outbox = self.root / "outbox"
        self.receipts = self.root / "receipts"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.receipts.mkdir(parents=True, exist_ok=True)
        self._recovery_offset = 0

    def enqueue(self, record, *, journal="records.jsonl"):
        validate_record(record)
        relative = Path(journal)
        if (relative.is_absolute() or ".." in relative.parts
                or self.root.resolve() not in (self.root / relative).resolve().parents):
            raise ReportingError("Journal must remain inside campaign root")
        target = self.outbox / (record["record_id"] + ".json")
        with _lock(self.root / ".enqueue.lock"):
            if target.exists():
                if canonical_json(json.loads(target.read_text())) != canonical_json(record):
                    raise RecordConflict("Immutable source ID has different payload")
                return record["record_id"]
            append_jsonl(self.root / "records.jsonl", record)
            if journal != "records.jsonl":
                append_jsonl(self.root / journal, record)
            atomic_json(target, record)
        return record["record_id"]

    def recover(self):
        ledger = self.root / "records.jsonl"
        if not ledger.exists():
            return 0
        count = 0
        with _lock(self.root / ".enqueue.lock"):
            if ledger.stat().st_size < self._recovery_offset:
                raise ReportingError("Append-only ledger was truncated")
            with ledger.open("rb") as stream:
                stream.seek(self._recovery_offset)
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        raise ReportingError("Incomplete ledger tail; preserve for repair")
                    record = json.loads(line)
                    validate_record(record)
                    path = self.outbox / (record["record_id"] + ".json")
                    if path.exists():
                        if canonical_json(json.loads(path.read_text())) != canonical_json(record):
                            raise RecordConflict("Ledger/outbox mismatch")
                    else:
                        atomic_json(path, record)
                        count += 1
                    self._recovery_offset = stream.tell()
        return count

    def pending(self):
        """Compatibility inventory of RAW LOCAL events, not remote upload debt."""
        records = []
        for path in sorted(self.outbox.glob("DS3-*.json")):
            record = json.loads(path.read_text())
            validate_record(record)
            records.append(record)
        return sorted(records, key=lambda r: (r["created_at"], r["record_id"]))
