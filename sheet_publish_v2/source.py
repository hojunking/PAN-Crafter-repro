"""Read ALL append-only local evidence, including acknowledged raw records."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .selection import digest, run_key, validate_envelope


def scan_campaign(root):
    """Read a bounded ledger snapshot plus every outbox file; never mutate either.

    A concurrently incomplete final JSONL line is reported and blocks publication
    until a later poll reads it complete. Duplicate source IDs must be identical.
    Large scene/trace envelopes are validated and fingerprinted but not retained.
    """
    root = Path(root).resolve(strict=True)
    kept, seen, errors, excluded, owners, files = {}, {}, [], Counter(), set(), []
    latest = {}

    def consume(record, path):
        try:
            validate_envelope(record)
            rid = record["record_id"]
            sha = digest(record)
            if rid in seen:
                if seen[rid] != sha:
                    errors.append({"reason": "SOURCE_ID_PAYLOAD_CONFLICT", "source_result_id": rid, "path": str(path)})
                return
            seen[rid] = sha
            key = run_key(record)
            owners.add(key)
            scope = record["record_type"]
            if scope == "EVAL_SUMMARY" or scope.startswith("QUAL_ARTIFACT"):
                kept[rid] = record
            else:
                excluded[scope] += 1
            if key not in latest or (record["created_at"], rid) > (latest[key]["created_at"], latest[key]["record_id"]):
                latest[key] = record
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"reason": str(exc), "source_result_id": record.get("record_id"), "path": str(path)})

    ledger = root / "records.jsonl"
    if ledger.is_file():
        # Do not lock the trainer for a multi-megabyte scan. Read only the initial
        # byte length; writer atomic append boundaries are checked explicitly.
        size = ledger.stat().st_size
        h = hashlib.sha256()
        with ledger.open("rb") as stream:
            number, read = 0, 0
            while read < size:
                line = stream.readline(size - read)
                read += len(line); h.update(line); number += 1
                if not line.endswith(b"\n"):
                    errors.append({"reason": "INCOMPLETE_LEDGER_TAIL_RETRY", "path": str(ledger), "line": number})
                    break
                try:
                    consume(json.loads(line), str(ledger) + ":" + str(number))
                except (ValueError, UnicodeError) as exc:
                    errors.append({"reason": "INVALID_LEDGER_JSON:" + str(exc), "path": str(ledger), "line": number})
        files.append({"path": str(ledger), "snapshot_bytes": size, "sha256": h.hexdigest()})
    for path in sorted((root / "outbox").glob("EGR2-*.json")):
        try:
            raw = path.read_bytes()
            record = json.loads(raw)
            if path.stem != record.get("record_id"):
                raise ValueError("OUTBOX_FILENAME_ID_MISMATCH")
            consume(record, path)
        except (ValueError, OSError) as exc:
            errors.append({"reason": str(exc), "path": str(path)})
    for record in latest.values():
        kept[record["record_id"]] = record
    # Receipt presence is irrelevant for candidate selection, but keep counts in
    # the audit so that an accidental pending-only scan is readily detectable.
    receipt_count = sum(1 for p in (root / "receipts").glob("EGR2-*.json") if p.stem in seen)
    return {"root": str(root), "records": list(kept.values()), "source_digest": digest(sorted(seen.items())),
            "raw_count": len(seen), "summary_count": sum(r["record_type"] == "EVAL_SUMMARY" for r in kept.values()),
            "raw_ids": sorted(seen), "owned_run_keys": sorted(owners), "receipt_file_count": receipt_count,
            "errors": errors, "excluded_counts": dict(excluded), "source_manifest": files}
