"""Read-only local completeness audit; no result repair, Google, or training.

An evaluation is not a missing upload merely because another step won the
qualified-or-best selection. This module separately checks physical evidence,
authenticated summaries, all forty scene events, and intentional selection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from directshift48 import publication as direct
from sheet_publish_v2 import selection as legacy


def _key(identity):
    return (identity["run"], identity["attempt"], identity["selected_step"])


def _scan(root, server, api):
    summaries, scenes, seen = defaultdict(list), defaultdict(list), {}
    counts, errors, manifests = Counter(), [], []

    def consume(record, source):
        try:
            api.validate_envelope(record)
            if record["identity"]["server"] != server:
                raise ValueError("FOREIGN_SERVER")
            rid, sha = record["record_id"], direct.digest(record)
            if rid in seen:
                if seen[rid] != sha:
                    raise ValueError("SOURCE_ID_PAYLOAD_CONFLICT")
                return
            seen[rid] = sha
            scope = record["record_type"]
            counts[scope.split(":")[0]] += 1
            key = _key(record["identity"])
            if scope == "EVAL_SUMMARY":
                summaries[key].append(record)
            elif scope.startswith(("RR_SCENE:", "FR_SCENE:")):
                extra = record["source"]["extra"]
                scenes[key].append({"scope": scope, "summary_id": extra.get("summary_record_id"),
                    "identity": extra.get("identity"), "metrics": record["source"]["metrics"]})
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"path": str(source), "reason": str(exc)})

    ledger = root / "records.jsonl"
    if ledger.is_file():
        size = ledger.stat().st_size
        manifests.append({"path": str(ledger), "snapshot_bytes": size})
        with ledger.open("rb") as stream:
            line_number = 0
            while stream.tell() < size:
                line = stream.readline(size - stream.tell()); line_number += 1
                location = str(ledger) + ":" + str(line_number)
                if not line.endswith(b"\n"):
                    errors.append({"path": location, "reason": "INCOMPLETE_LEDGER_TAIL_RETRY"})
                    break
                try:
                    consume(json.loads(line), location)
                except (ValueError, UnicodeError) as exc:
                    errors.append({"path": location, "reason": "INVALID_LEDGER_JSON:" + str(exc)})
    prefix = "DS3-" if api is direct else "EGR2-"
    for path in sorted((root / "outbox").glob(prefix + "*.json")):
        try:
            record = json.loads(path.read_bytes())
            if path.stem != record.get("record_id"):
                raise ValueError("OUTBOX_FILENAME_ID_MISMATCH")
            consume(record, path)
        except (ValueError, OSError) as exc:
            errors.append({"path": str(path), "reason": str(exc)})

    formal, missing, orphan, rejected, incomplete = {}, [], [], [], []
    for path in sorted(root.glob("runs/*/attempt_*/evaluation_*.json")):
        try:
            value = json.loads(path.read_bytes())
            if value.get("record_type") != "EVAL_SUMMARY" or value.get("full_method_eligible") is not True:
                continue
            identity = value["state_before"]
            key = (identity["run_id"], identity["attempt"], identity["completed_update"])
            expected = root / "runs" / key[0] / ("attempt_%02d" % key[1]) / ("evaluation_%07d.json" % key[2])
            if path.resolve() != expected.resolve() or key in formal:
                raise ValueError("EVALUATION_PATH_OR_DUPLICATE_IDENTITY_MISMATCH")
            formal[key] = (path, value)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append({"path": str(path), "reason": "EVALUATION_UNREADABLE_OR_INVALID:" + str(exc)})
    authenticated, authenticated_records = set(), 0
    for key, records in summaries.items():
        if key not in formal:
            orphan.append({"key": key, "source_ids": [r["record_id"] for r in records]})
        for record in records:
            try:
                api.validate_candidate(record, server=server, evidence_root=root)
                authenticated.add(key); authenticated_records += 1
            except (ValueError, TypeError, KeyError, OSError) as exc:
                rejected.append({"key": key, "source_id": record["record_id"], "reason": str(exc)})
    for key, (path, evaluation) in formal.items():
        if key not in authenticated:
            missing.append({"key": key, "path": str(path), "reason": "NO_AUTHENTICATED_SUMMARY"})
        expected_scenes = {}
        for split in ("rr", "fr"):
            for scene in evaluation.get(split, {}).get("per_scene", []):
                expected_scenes[split.upper() + "_SCENE:" + str(scene.get("scene_id"))] = scene.get("metrics")
        observed = Counter(item["scope"] for item in scenes[key])
        missing_ids = sorted(set(expected_scenes) - set(observed))
        unexpected = sorted(set(observed) - set(expected_scenes))
        duplicate = sorted(k for k, count in observed.items() if count != 1)
        summary_ids = {r["record_id"] for r in summaries[key]}
        mismatched = sorted({item["scope"] for item in scenes[key]
            if (item["summary_id"] not in summary_ids or item["identity"] != evaluation["state_before"]
                or item["metrics"] != expected_scenes.get(item["scope"]))})
        if len(expected_scenes) != 40 or missing_ids or unexpected or duplicate or mismatched:
            incomplete.append({"key": key, "expected": 40, "observed": sum(observed.values()),
                "missing": missing_ids, "unexpected": unexpected, "duplicates": duplicate,
                "lineage_or_metric_mismatch": mismatched})
    return {"root": str(root), "formal_evaluation_count": len(formal),
        "summary_record_count": sum(map(len, summaries.values())),
        "authenticated_summary_record_count": authenticated_records,
        "authenticated_evaluation_count": len(authenticated),
        "missing_summaries": missing, "orphan_summaries": orphan, "rejected_summaries": rejected,
        "incomplete_scene_events": incomplete, "source_errors": errors, "source_manifest": manifests,
        "scope_counts": dict(counts), "unique_raw_record_count": len(seen)}


def audit_campaign(root, server, history_roots=None):
    """Return JSON-ready local proof, with status INCOMPLETE on any real gap.

    No local file or remote sheet is written. Live evaluation writes may be in
    progress: INCOMPLETE must be retried, never repaired with fabricated values.
    Actual remote sheet completeness requires separate transport readback.
    """
    root = Path(root).resolve(strict=True)
    direct.destination(server)
    if history_roots is None:
        candidate = direct.ROOT / "work_dir/edgeres48" / legacy.CAMPAIGN_ID / server / direct.LANES[server][0]
        history_roots = [candidate] if candidate.is_dir() else []
    if isinstance(history_roots, (str, Path)):
        history_roots = [history_roots]
    roots = [Path(p).resolve(strict=True) for p in history_roots]
    if len(set(roots)) != len(roots) or root in roots:
        raise ValueError("DUPLICATE_HISTORY_SOURCE_MISMATCH")
    scans = [_scan(p, server, legacy) for p in roots] + [_scan(root, server, direct)]
    try:
        projection = direct.combined_projection(root, server, history_roots=roots)
        selection = {"selected_row_count": len(projection["rows"]), "selected": projection["selected"],
            "runs": projection["runs"], "conflicts": projection["conflicts"],
            "excluded_reason_counts": dict(Counter(item["reason"] for item in projection["excluded"]))}
    except (ValueError, TypeError, KeyError, OSError) as exc:
        selection = {"selected_row_count": 0, "selected": [], "runs": [],
            "conflicts": [{"reason": "PROJECTION_FAILED:" + str(exc)}], "excluded_reason_counts": {}}
    issues = ("missing_summaries", "orphan_summaries", "rejected_summaries", "incomplete_scene_events", "source_errors")
    counts = {name: sum(scan[name] for scan in scans) for name in
        ("formal_evaluation_count", "summary_record_count", "authenticated_summary_record_count", "authenticated_evaluation_count")}
    counts.update({name + "_count": sum(len(scan[name]) for scan in scans) for name in issues})
    complete = not selection["conflicts"] and not any(counts[name + "_count"] for name in issues)
    return {"schema_version": "LOCAL_RESULT_COMPLETENESS_V1", "status": "COMPLETE" if complete else "INCOMPLETE",
        "server": server, "read_only": True, "remote_sheet_verified": False, "counts": counts,
        "campaigns": scans, "selection": selection,
        "intentional_exclusions": {"preflight_evaluations": sum(s["scope_counts"].get("PREFLIGHT_EVAL", 0) for s in scans),
            "valid_steps_not_selected": sum(max(0, r["candidate_count"] - r["kept_count"]) for r in selection["runs"]),
            "rule": "Qualified steps all retained; otherwise one best HQNR step per run. Preflight and scene/diagnostic rows stay local."},
        "on_incomplete": "Retry after ongoing evaluation emission; investigate persisted gaps. Never synthesize metrics or upload unverified evidence."}
