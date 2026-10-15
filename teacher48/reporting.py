"""Local-first reporting for the Stage-1 campaign's immutable fixed-64 contract.

No import or enqueue operation contacts Google. Only ``publish_once`` does, via
an explicitly supplied transport. The training process must never call it.
Each dataset has one owning server; a host-local publisher lock is independent
of learning and never makes another server wait for a model or checkpoint.
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
DESTINATIONS = {"s1": ("WV3", "WV3-ablations", 261001010),
                "s2": ("QB", "QB-ablations", 261001011),
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
        raise ReportingError("Only s1/WV3, s2/QB, s3/GF2 may publish this campaign")
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
    checkpoint via extra.same_state_verified; checkpoint SHA is the logical
    model-state SHA, not a hash of a nonexistent failed-model file.
    """
    dest = destination(server, dataset)
    for name, value in (("run_id", run_id), ("case_id", case_id),
                        ("selector", selector), ("eval_scope", eval_scope),
                        ("eval_protocol_id", eval_protocol_id)):
        if not isinstance(value, str) or not value.strip():
            raise ReportingError(name + " must be a nonempty string")
    if checkpoint_sha is not None and not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha):
        raise ReportingError("checkpoint_sha must be a full logical SHA256 or null")
    metrics, extra, provenance = dict(metrics or {}), dict(extra or {}), dict(provenance or {})
    if run_id.startswith('S1M12_') or PARENT_CAMPAIGN_ID in run_id:
        raise ReportingError('v1.2 run cannot be relabelled as the v1.3 campaign')
    for key, value in METHOD_CONTRACT.items():
        if key in extra and extra[key] != value:
            raise ReportingError('Conflicting method semantics: ' + key)
    if 'configuration' in extra:
        validate_method_config(extra['configuration'])
    for key in ('evaluation_identity', 'identity'):
        if key in extra and any(extra[key].get(k) != v for k,v in METHOD_CONTRACT.items()):
            raise ReportingError('Observation identity has obsolete method semantics: ' + key)
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
    if metric_present and checkpoint_sha is None:
        raise ReportingError("Metric observations require the model-state SHA")
    if data.get("HQNR↑") not in (None, "") and data.get("ERGAS↓") not in (None, ""):
        if extra.get("same_state_verified") is not True:
            raise ReportingError("HQNR and ERGAS require explicit same-state verification")
    for name in ("rr_model_state_sha256", "fr_model_state_sha256", "model_state_sha256"):
        if name in extra and extra[name] != checkpoint_sha:
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
    identity = {"namespace": SCHEMA_VERSION + ":" + RESET_EPOCH,
                "campaign": CAMPAIGN_ID, "dataset": dataset, "server": server,
                "run": run_id, "selector": selector, "checkpoint": checkpoint_sha,
                "eval_protocol": eval_protocol_id, "eval_scope": eval_scope,
                "attempt": data.get("Attempt"), "selected_step": data.get("Selected_step")}
    record_id = "T48-" + _sha(identity)
    note = {"schema_version": SCHEMA_VERSION, "reset_epoch": RESET_EPOCH,
            **METHOD_CONTRACT, "parent_campaign_id": PARENT_CAMPAIGN_ID,
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
    if len(row[INDEX["Original description / notes"]]) > 45000:
        raise ReportingError("BI JSON exceeds safe cell limit; split numeric scopes")
    record = {"record_id": record_id, "identity": identity, "destination": dest,
              "row": row, "payload_sha256": _sha(row), "record_type": eval_scope,
              "source": {"metrics": metrics, "fields": data, "extra": extra,
                         "provenance": provenance}, "created_at": timestamp}
    canonical_json(record)  # fail on NaN/tensors or otherwise non-JSON evidence
    return record


def validate_record(record: Mapping[str, Any]) -> None:
    identity, row = record["identity"], record["row"]
    dest = destination(identity["server"], identity["dataset"])
    if record["destination"] != dest or identity["campaign"] != CAMPAIGN_ID:
        raise ReportingError("Historical or misrouted record is forbidden")
    if identity["namespace"] != SCHEMA_VERSION + ":" + RESET_EPOCH:
        raise ReportingError("Wrong reporting namespace")
    if len(row) != 64 or serialize_row(dict(zip(HEADERS, row))) != row:
        raise ReportingError("Invalid fixed64 payload")
    if record["record_id"] != "T48-" + _sha(identity) or record["payload_sha256"] != _sha(row):
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
        raise ReportingError('Stored envelope has obsolete or missing v1.3 method semantics')
    if note.get('parent_campaign_id') != PARENT_CAMPAIGN_ID:
        raise ReportingError('Stored envelope lacks original campaign provenance')
    extra = note.get('extra_parameters', {})
    if 'configuration' in extra:
        validate_method_config(extra['configuration'])
    for key in ('evaluation_identity', 'identity'):
        if key in extra and any(extra[key].get(k) != v for k,v in METHOD_CONTRACT.items()):
            raise ReportingError('Stored observation has obsolete method semantics: ' + key)


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
    """Append-only numeric ledger plus immutable outbound envelopes and receipts."""

    def __init__(self, root):
        self.root = Path(root)
        self.outbox = self.root / "outbox"
        self.receipts = self.root / "receipts"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.receipts.mkdir(parents=True, exist_ok=True)
        self._recovery_offset = 0

    def enqueue(self, record: Mapping[str, Any], *, journal: str = "records.jsonl") -> str:
        validate_record(record)
        relative = Path(journal)
        if relative.is_absolute() or ".." in relative.parts:
            raise ReportingError("Journal must stay inside the campaign root")
        record_id = record["record_id"]
        target = self.outbox / (record_id + ".json")
        with _lock(self.root / ".enqueue.lock"):
            if target.exists():
                if canonical_json(json.loads(target.read_text())) != canonical_json(record):
                    raise RecordConflict("Same Result_ID has different local payload: " + record_id)
                return record_id
            # Canonical ledger first, so interrupted envelope publication is recoverable.
            append_jsonl(self.root / "records.jsonl", record)
            if journal != "records.jsonl":
                append_jsonl(self.root / journal, record)
            atomic_json(target, record)
        return record_id

    def recover(self) -> int:
        """Rebuild missing envelopes after a crash between ledger and outbox writes."""
        ledger = self.root / "records.jsonl"
        if not ledger.exists():
            return 0
        count = 0
        with _lock(self.root / ".enqueue.lock"):
            if ledger.stat().st_size < self._recovery_offset:
                raise ReportingError('Append-only ledger was truncated')
            # Full replay once per process; subsequent passes inspect only new
            # appends while holding the brief local enqueue mutex. No long
            # rescan of thousands of old rows stalls a live training enqueue.
            with ledger.open('rb') as stream:
                stream.seek(self._recovery_offset)
                for line in stream:
                    if not line.endswith(b'\n'):
                        raise ReportingError("Incomplete final ledger line; preserve and repair before publishing")
                    record = json.loads(line.decode('utf-8'))
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

    def pending(self) -> list:
        records = []
        for path in sorted(self.outbox.glob("T48-*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            validate_record(record)
            receipt = self.receipts / path.name
            if receipt.exists():
                got = json.loads(receipt.read_text(encoding="utf-8"))
                if got.get("status") == "READBACK_VERIFIED" and got.get("payload_sha256") == record["payload_sha256"]:
                    continue
            records.append(record)
        return sorted(records, key=lambda record: (record["created_at"], record["record_id"]))

    def publish_once(self, transport, *, server: str, dataset: str, limit: Optional[int] = None) -> dict:
        """One publisher pass; failed records persist with a retry timestamp.

        This never sleeps, retries appends blindly, clears the outbox, or writes
        model artifacts. Call repeatedly from a separate process/CLI worker.
        """
        dest = destination(server, dataset)
        lockpath = Path(tempfile.gettempdir()) / ("panda_teacher48_" + str(dest["sheet_id"]) + ".publisher.lock")
        report = {"verified": 0, "failed": 0, "deferred": 0, "errors": []}
        with _lock(lockpath, nonblocking=True):
            self.recover()
            validate_remote_schema(transport.schema_snapshot(), server=server, dataset=dataset)
            if hasattr(transport, 'publish_records'):
                return self._publish_batches(transport, dest, limit=limit)
            attempted = 0
            for record in self.pending():
                if record["destination"] != dest:
                    continue
                receipt_path = self.receipts / (record["record_id"] + ".json")
                previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                if previous.get("retry_after_unix", 0) > time.time():
                    report["deferred"] += 1
                    continue
                if limit is not None and attempted >= limit:
                    break
                attempted += 1
                attempt = int(previous.get("attempt", 0)) + 1
                try:
                    receipt = publish_record(transport, record)
                    receipt.update(attempt=attempt, last_attempt=utc_now())
                    atomic_json(receipt_path, receipt)
                    append_jsonl(self.root / "upload_events.jsonl", receipt)
                    report["verified"] += 1
                except Exception as exc:
                    kind = _failure_kind(exc)
                    # Permission/schema errors are not reinterpreted as rate limits.
                    delay = min(3600, (60 if kind == "PERMISSION" else 5) * (2 ** min(attempt - 1, 9)))
                    receipt = {"record_id": record["record_id"], "payload_sha256": record["payload_sha256"],
                               "status": "FAILED_RETRY", "error_kind": kind,
                               "error": str(exc), "attempt": attempt, "last_attempt": utc_now(),
                               "retry_after_unix": time.time() + delay, "destination": dest}
                    atomic_json(receipt_path, receipt)
                    append_jsonl(self.root / "upload_events.jsonl", receipt)
                    report["failed"] += 1
                    report["errors"].append({"record_id": record["record_id"], "kind": kind, "error": str(exc)})
                    if kind in {"PERMISSION", "RATE_LIMIT", "NETWORK_OR_IO"}:
                        # Do not hammer the same service with every other pending
                        # record after a shared permission/quota/network failure.
                        break
        report["pending"] = len(self.pending())
        return report

    def _publish_batches(self, transport, dest, *, limit):
        """Bounded API fast path: 32 rows/pass by default, no in-process sleeps."""
        report = {"verified": 0, "failed": 0, "deferred": 0, "errors": []}
        records, previous = [], {}
        for record in self.pending():
            if record['destination'] != dest:
                continue
            path = self.receipts / (record['record_id'] + '.json')
            old = json.loads(path.read_text()) if path.exists() else {}
            previous[record['record_id']] = old
            if old.get('retry_after_unix', 0) > time.time():
                report['deferred'] += 1
            else:
                records.append(record)
        records = records[:32 if limit is None else limit]
        for offset in range(0, len(records), 32):
            batch = records[offset:offset+32]
            try:
                receipts = transport.publish_records(batch)
                if {r['record_id'] for r in receipts} != {r['record_id'] for r in batch} or len(receipts) != len(batch):
                    raise ReportingError('Batch did not return exactly one receipt per record')
                for receipt in receipts:
                    receipt.update(attempt=int(previous[receipt['record_id']].get('attempt', 0))+1, last_attempt=utc_now())
                    atomic_json(self.receipts / (receipt['record_id'] + '.json'), receipt)
                    append_jsonl(self.root / 'upload_events.jsonl', receipt)
                    report['verified'] += 1
            except Exception as exc:
                kind = _failure_kind(exc)
                for record in batch:
                    attempt = int(previous[record['record_id']].get('attempt', 0))+1
                    delay = min(3600, (60 if kind == 'PERMISSION' else 5)*(2**min(attempt-1, 9)))
                    receipt = {'record_id': record['record_id'], 'payload_sha256': record['payload_sha256'],
                        'status': 'FAILED_RETRY', 'error_kind': kind, 'error': str(exc),
                        'attempt': attempt, 'last_attempt': utc_now(), 'retry_after_unix': time.time()+delay,
                        'destination': dest}
                    atomic_json(self.receipts / (record['record_id']+'.json'), receipt)
                    append_jsonl(self.root / 'upload_events.jsonl', receipt)
                    report['failed'] += 1
                report['errors'].append({'record_ids': [r['record_id'] for r in batch], 'kind': kind, 'error': str(exc)})
                break
        report['pending'] = len(self.pending())
        return report


def _failure_kind(exc: Exception) -> str:
    if isinstance(exc, RecordConflict):
        return "CONFLICT"
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if code in (401, 403):
        return "PERMISSION"
    if code == 429:
        return "RATE_LIMIT"
    if isinstance(exc, ReportingError):
        return "VALIDATION"
    return "NETWORK_OR_IO"


def validate_remote_schema(snapshot: Mapping[str, Any], *, server: str, dataset: str) -> None:
    dest = destination(server, dataset)
    for key, value in dest.items():
        if snapshot.get(key) != value:
            raise ReportingError("Unexpected remote " + key)
    if snapshot.get("column_count") != 64 or snapshot.get("headers") != list(HEADERS):
        raise ReportingError("Remote fixed64 header/column mismatch; auto-repair forbidden")
    if _sha(snapshot["headers"]) != HEADER_SHA256:
        raise ReportingError("Remote header digest mismatch")
    note = snapshot.get("a2_note")
    if not isinstance(note, str) or not note.strip():
        raise ReportingError("A2 NOTE is required (A2 displayed value is not metadata)")
    try:
        metadata = json.loads(note)
    except json.JSONDecodeError:
        # Existing notes may use prose key=value rather than a JSON object.
        metadata = {key: match.group(1) if match else None for key, match in (
            ("schema_version", re.search(r"\bschema_version\s*[:=]\s*[\"']?(PAN_FIXED64_V1)\b", note)),
            ("reset_epoch", re.search(r"\breset_epoch\s*[:=]\s*[\"']?(\d{4}-\d{2}-\d{2})\b", note)))}
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("reset_epoch") != RESET_EPOCH:
        raise ReportingError("A2 note schema_version/reset_epoch mismatch")


def _row_equal(actual: list, expected: list, *, readback_any: bool = False) -> bool:
    if len(actual) > 64:
        return False
    # Sheets omits empty trailing cells. The actual A:BL request is still fixed-width.
    actual = list(actual) + [""] * (64 - len(actual))
    if len(expected) != 64:
        return False
    for index, (got, want) in enumerate(zip(actual, expected)):
        if readback_any and index == INDEX["Readback"]:
            if got not in ("PENDING", "VERIFIED"):
                return False
            continue
        if isinstance(want, bool):
            if type(got) is not bool or got != want:
                return False
        elif type(want) in (float, int):
            if (type(got) not in (float, int) or not math.isfinite(got) or
                    (got != want if type(want) is int else not math.isclose(got, want, rel_tol=1e-14, abs_tol=1e-15))):
                return False
        elif got != want:
            return False
    return True


def publish_record(transport, record: Mapping[str, Any]) -> Dict[str, Any]:
    validate_record(record)
    dest, identity = record["destination"], record["identity"]
    # Also validate per record: callers cannot bypass the admission helper.
    validate_remote_schema(transport.schema_snapshot(), server=identity["server"], dataset=identity["dataset"])
    matches = transport.find_result_ids(record["record_id"])
    if len(matches) > 1:
        raise RecordConflict("Duplicate Result_ID already exists on sheet")
    if matches:
        row_number = matches[0]
        actual = transport.read_row(row_number)
        if not _row_equal(actual, record["row"], readback_any=True):
            raise RecordConflict("Same Result_ID has a different remote payload")
    else:
        # Do not catch and repeat here: ambiguous timeout is recovered next pass
        # by Result_ID lookup. append_row must return its actual updated row.
        row_number = transport.append_row(record["row"])
        if type(row_number) is not int or row_number < 6:
            raise ReportingError("Append returned an invalid actual data row")
        actual = transport.read_row(row_number)
        if not _row_equal(actual, record["row"]):
            raise ReportingError("Initial fixed64 readback mismatch")
    if row_number < 6:
        raise ReportingError("Header-region writes are forbidden")
    if actual[INDEX["Readback"]] != "VERIFIED":
        transport.set_verified(row_number)
    verified = list(record["row"])
    verified[INDEX["Readback"]] = "VERIFIED"
    actual = transport.read_row(row_number)
    if not _row_equal(actual, verified):
        raise ReportingError("Final VERIFIED readback mismatch")
    # Display only: RAW numbers in the cells and local row remain unchanged.
    transport.format_row(row_number, record)
    return {"record_id": record["record_id"], "payload_sha256": record["payload_sha256"],
            "status": "READBACK_VERIFIED", "destination": dest, "row_number": row_number,
            "actual_range": "'" + dest["sheet_title"] + "'!A" + str(row_number) + ":BL" + str(row_number),
            "row": actual, "readback_sha256": _sha(actual), "verified_at": utc_now()}


class GSpreadTransport:
    """Narrow real transport. No workbook-title lookup or automatic schema repair."""

    def __init__(self, spreadsheet, worksheet, *, server: str, dataset: str):
        self.server, self.dataset = server, dataset
        self.dest = destination(server, dataset)
        self.spreadsheet = spreadsheet
        self.worksheet = worksheet
        if str(spreadsheet.id) != SPREADSHEET_ID or int(worksheet.id) != self.dest["sheet_id"] or worksheet.title != self.dest["sheet_title"]:
            raise ReportingError("Explicit spreadsheet/sheet identity mismatch")

    @classmethod
    def from_credentials(cls, filename=None, *, server: str, dataset: str):
        import gspread
        dest = destination(server, dataset)
        filename = Path(filename) if filename is not None else ROOT / "gspread/account.json"
        client = gspread.service_account(filename=str(filename), scopes=["https://www.googleapis.com/auth/spreadsheets"])
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
        worksheet = spreadsheet.get_worksheet_by_id(dest["sheet_id"])
        return cls(spreadsheet, worksheet, server=server, dataset=dataset)

    def schema_snapshot(self):
        title = self.dest["sheet_title"]
        metadata = self.spreadsheet.fetch_sheet_metadata(params={
            "includeGridData": "true", "ranges": "'" + title + "'!A2",
            "fields": "spreadsheetId,sheets(properties(sheetId,title,gridProperties),data(startRow,startColumn,rowData(values(note))))"})
        sheets = [sheet for sheet in metadata.get("sheets", []) if sheet["properties"]["sheetId"] == self.dest["sheet_id"]]
        if len(sheets) != 1:
            raise ReportingError("Expected one destination sheet in metadata")
        sheet, note = sheets[0], ""
        # gspread.append_rows increments its cached rowCount even with OVERWRITE.
        # Only the fresh server metadata is reliable for capacity planning.
        self._actual_row_count = sheet["properties"]["gridProperties"]["rowCount"]
        for grid in sheet.get("data", []):
            if grid.get("startRow", 0) == 1 and grid.get("startColumn", 0) == 0:
                rows = grid.get("rowData", [])
                values = rows[0].get("values", []) if rows else []
                note = values[0].get("note", "") if values else ""
        headers = self.worksheet.get("A5:BL5", value_render_option="FORMULA")
        return {"spreadsheet_id": metadata.get("spreadsheetId"),
                "sheet_id": sheet["properties"]["sheetId"], "sheet_title": sheet["properties"]["title"],
                "column_count": sheet["properties"]["gridProperties"]["columnCount"],
                "headers": headers[0] if headers else [], "a2_note": note}

    def find_result_ids(self, result_id):
        values = self.worksheet.get("BH6:BH", value_render_option="UNFORMATTED_VALUE")
        return [index + 6 for index, row in enumerate(values) if row and row[0] == result_id]

    def read_row(self, row_number):
        values = self.worksheet.get("A" + str(row_number) + ":BL" + str(row_number), value_render_option="UNFORMATTED_VALUE")
        return (values[0] if values else []) + [""] * (64 - len(values[0] if values else []))

    def append_row(self, row):
        # The writer owns only this dataset tab. Use a bounded table, not whole-
        # worksheet automatic detection that could select the title/header.
        used = self.worksheet.get("BH6:BH", value_render_option="UNFORMATTED_VALUE")
        return self._append_rows([row], len(used))[0]

    def _append_rows(self, rows, used_count):
        needed = 5 + used_count + len(rows)
        row_count = self._actual_row_count
        if needed > row_count:
            row_count = max(needed, row_count + 1000)
            self.worksheet.resize(rows=row_count)
            self._actual_row_count = row_count
            self.spreadsheet.batch_update({"requests": [{"setBasicFilter": {"filter": {"range": {
                "sheetId": self.dest["sheet_id"], "startRowIndex": 4, "endRowIndex": row_count,
                "startColumnIndex": 0, "endColumnIndex": 64}}}}]})
        reply = self.worksheet.append_rows(rows, value_input_option="RAW", insert_data_option="OVERWRITE",
                                           table_range="A5:BL" + str(row_count))
        updated = reply.get("updates", {}).get("updatedRange", "")
        match = re.fullmatch(r"(?:'[^']+'|[^!]+)!A(\d+):BL(\d+)", updated)
        if not match or int(match.group(2))-int(match.group(1))+1 != len(rows) or int(match.group(1)) < 6:
            raise ReportingError("Unexpected append range: " + updated)
        return list(range(int(match.group(1)), int(match.group(2))+1))

    def _read_rows(self, numbers):
        areas = ['A' + str(n) + ':BL' + str(n) for n in numbers]
        result = self.worksheet.batch_get(areas, value_render_option='UNFORMATTED_VALUE')
        if len(result) != len(numbers):
            raise ReportingError('Batch readback returned incomplete ranges')
        return [(block[0] if block else []) + ['']*(64-len(block[0] if block else [])) for block in result]

    def publish_records(self, records):
        """One guarded batch with a shared Result_ID index and two readbacks.

        An ambiguous append is never retried here. The next pass resolves all
        Result_IDs again and only appends those genuinely absent from the sheet.
        """
        if not 1 <= len(records) <= 32:
            raise ReportingError('Publish batches must contain 1..32 records')
        for record in records:
            validate_record(record)
            if record['destination'] != self.dest:
                raise ReportingError('Cross-destination batch forbidden')
        if len({r['record_id'] for r in records}) != len(records):
            raise ReportingError('Duplicate local Result_ID in batch')
        validate_remote_schema(self.schema_snapshot(), server=self.server, dataset=self.dataset)
        index = self.worksheet.get('BH6:BH', value_render_option='UNFORMATTED_VALUE')
        positions = {}
        for number, row in enumerate(index, start=6):
            if row and row[0]:
                positions.setdefault(row[0], []).append(number)
        numbers, missing = {}, []
        for record in records:
            hits = positions.get(record['record_id'], [])
            if len(hits) > 1:
                raise RecordConflict('Duplicate remote Result_ID: ' + record['record_id'])
            if hits:
                numbers[record['record_id']] = hits[0]
            else:
                missing.append(record)
        # Existing IDs are checked before adding anything. A conflicting row
        # cannot be hidden behind an otherwise valid batch.
        existing = [record for record in records if record['record_id'] in numbers]
        if existing:
            rows = self._read_rows([numbers[r['record_id']] for r in existing])
            for got, record in zip(rows, existing):
                if not _row_equal(got, record['row'], readback_any=True):
                    raise RecordConflict('Remote payload conflict: ' + record['record_id'])
        if missing:
            added = self._append_rows([r['row'] for r in missing], len(index))
            numbers.update({r['record_id']: number for r, number in zip(missing, added)})
        ordered = [numbers[r['record_id']] for r in records]
        actual = self._read_rows(ordered)
        for got, record in zip(actual, records):
            if not _row_equal(got, record['row'], readback_any=True):
                raise ReportingError('Batch initial readback mismatch: ' + record['record_id'])
        markers = [{'range': 'AZ' + str(n), 'values': [['VERIFIED']]}
                   for n, row in zip(ordered, actual) if row[INDEX['Readback']] != 'VERIFIED']
        if markers:
            self.worksheet.batch_update(markers, value_input_option='RAW')
        actual = self._read_rows(ordered)
        receipts, formatting = [], []
        for number, got, record in zip(ordered, actual, records):
            expected = list(record['row']); expected[INDEX['Readback']] = 'VERIFIED'
            if not _row_equal(got, expected):
                raise ReportingError('Batch final VERIFIED readback mismatch')
            receipts.append({'record_id': record['record_id'], 'payload_sha256': record['payload_sha256'],
                'status': 'READBACK_VERIFIED', 'destination': self.dest, 'row_number': number,
                'actual_range': "'"+self.dest['sheet_title']+"'!A"+str(number)+':BL'+str(number),
                'row': got, 'readback_sha256': _sha(got), 'verified_at': utc_now()})
            formatting.extend(self._format_requests(number, record))
        self.spreadsheet.batch_update({'requests': formatting})
        return receipts

    def set_verified(self, row_number):
        self.worksheet.update(values=[["VERIFIED"]], range_name="AZ" + str(row_number), value_input_option="RAW")

    def format_row(self, row_number, record):
        self.spreadsheet.batch_update({'requests': self._format_requests(row_number, record)})

    def _format_requests(self, row_number, record):
        extra = record["source"]["extra"]
        status = record["row"][INDEX["Status"]]
        scope = record["row"][INDEX["Eval_scope"]]
        color = {"red": 0.95, "green": 0.95, "blue": 0.95}
        if extra.get("joint_pass") is True:
            color = {"red": 0.85, "green": 0.95, "blue": 0.85}
        elif status in {"FAILED", "BLOCKED", "BLOCKED_DATA_PROTOCOL"}:
            color = {"red": 1.0, "green": 0.86, "blue": 0.86}
        elif scope.startswith("DIAGNOSTIC"):
            color = {"red": 0.92, "green": 0.87, "blue": 0.98}
        elif status in {"STARTED", "RUNNING", "PLANNED"}:
            color = {"red": 0.86, "green": 0.92, "blue": 1.0}
        elif extra.get("h_pass") or extra.get("e_pass"):
            color = {"red": 1.0, "green": 0.96, "blue": 0.78}
        grid = {"sheetId": self.dest["sheet_id"], "startRowIndex": row_number - 1,
                "endRowIndex": row_number, "startColumnIndex": 0, "endColumnIndex": 64}
        requests = [{"repeatCell": {"range": grid, "cell": {"userEnteredFormat": {"backgroundColor": color}},
                                     "fields": "userEnteredFormat.backgroundColor"}}]
        for first, last, pattern in ((3, 15, "0.0000"), (15, 25, "0.0000"), (31, 34, "0"),
                                     (38, 41, "0"), (48, 50, "0.000000"), (62, 63, "0")):
            requests.append({"repeatCell": {"range": {**grid, "startColumnIndex": first, "endColumnIndex": last},
                "cell": {"userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": pattern}}},
                "fields": "userEnteredFormat.numberFormat"}})
        return requests
