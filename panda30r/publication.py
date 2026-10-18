"""Row-scoped WV3 publisher. Never replaces the tab or another server's rows.

The display rule remains qualified-all-steps or best-HQNR fallback. It never
uses the distinct donor selection gate to promote a row to a paper target pass.
Import/projection is offline; Google access requires an explicit publish flag.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from directshift48 import publication as legacy_direct
from . import reporting as R

ROOT = Path(__file__).resolve().parents[1]
INDEX, HEADERS = R.INDEX, R.HEADERS
POLICY = legacy_direct.POLICY
LANES = {s: ("WV3", "WV3-ablations", 261001010, .960, 2.045) for s in ("s1", "s3", "s4")}
canonical_json, digest = R.canonical_json, R.digest
destination = R.destination


def _require(test, reason):
    if not test:
        raise ValueError(reason)


def run_key(record):
    return tuple(record["identity"][key] for key in ("campaign", "dataset", "server", "run", "attempt"))


def scan_campaign(root):
    """Read all immutable evidence, including ledger-only crash-recovery events."""
    root = Path(root).resolve(strict=True)
    seen, errors, files = {}, [], []

    def consume(record, path):
        try:
            R.validate_record(record)
            rid = record["record_id"]
            _require(rid not in seen or seen[rid] == record, "SOURCE_ID_PAYLOAD_CONFLICT")
            seen[rid] = record
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"path": str(path), "reason": str(exc)})

    ledger = root / "records.jsonl"
    if ledger.exists():
        size, hashed = ledger.stat().st_size, hashlib.sha256()
        with ledger.open("rb") as stream:
            read, number = 0, 0
            while read < size:
                line = stream.readline(size - read)
                read += len(line); number += 1; hashed.update(line)
                if not line.endswith(b"\n"):
                    errors.append({"path": str(ledger), "reason": "INCOMPLETE_LEDGER_TAIL_RETRY"})
                    break
                try:
                    consume(json.loads(line), f"{ledger}:{number}")
                except (ValueError, UnicodeError) as exc:
                    errors.append({"path": str(ledger), "reason": str(exc)})
        files.append({"path": str(ledger), "snapshot_bytes": size, "sha256": hashed.hexdigest()})
    for path in sorted((root / "outbox").glob(R.RECORD_PREFIX + "*.json")):
        try:
            record = json.loads(path.read_bytes())
            _require(path.stem == record.get("record_id"), "OUTBOX_ID_MISMATCH")
            consume(record, path)
        except (ValueError, OSError) as exc:
            errors.append({"path": str(path), "reason": str(exc)})
    return {"records": list(seen.values()), "conflicts": errors, "raw_ids": sorted(seen),
        "raw_count": len(seen), "source_manifest": files,
        "source_digest": digest(sorted((key, digest(value)) for key, value in seen.items()))}


def _candidate(record, root, server):
    R.validate_record(record)
    R._config(record["source"]["extra"]["configuration"])
    ident, extra = record["identity"], record["source"]["extra"]
    cfg, state = extra["configuration"], extra["identity"]
    _require(cfg.get("engine_kind") != "donor_replay", "IMPORTED_REPLAY_IS_NOT_NEW_RUN_RESULT")
    _require(ident["server"] == server, "FOREIGN_SERVER")
    _require(ident["eval_scope"] == "EVAL_SUMMARY", "NOT_EVAL_SUMMARY")
    legacy_direct._check_complete(extra)
    _require(R._sha(state.get("model_state_sha256")) and
             R._sha(state.get("evaluator_revision")), "INVALID_STATE_OR_EVALUATOR_SHA")
    _require(state["evaluator_revision"] == ident["eval_protocol"], "EVALUATOR_IDENTITY_MISMATCH")
    metrics = record["source"]["metrics"]
    proof = legacy_direct._verify_full_evaluation(record, cfg, state, metrics, root)
    _require(0 <= metrics["HQNR"] <= 1 and metrics["ERGAS"] >= 0, "INVALID_METRIC_RANGE")
    return {"record": record, "key": run_key(record), "step": ident["selected_step"],
        "hqnr": metrics["HQNR"], "ergas": metrics["ERGAS"], "metrics": metrics,
        "state": state["model_state_sha256"], "protocol": ident["eval_protocol"],
        "config_sha": digest(cfg), "inference_policy": state.get("inference_policy_digest"),
        "source_result_id": record["record_id"], "evaluation_evidence": proof}


def _artifact(candidate, root):
    record = candidate["record"]
    artifact = record["source"]["extra"].get("artifact", {})
    if artifact.get("weights_saved") is not True:
        return {"weights_saved": False, "file_sha256": None,
                "model_state_sha256": candidate["state"]}
    # Mapping container paths by run/attempt suffix avoids trusting arbitrary
    # user-declared paths or another server's same-named checkpoint.
    cfg = record["source"]["extra"]["configuration"]
    raw = Path(artifact["checkpoint_path"])
    marker = ("runs", cfg["run_id"], "attempt_%02d" % cfg["attempt"])
    matches = [i for i in range(len(raw.parts) - 2) if raw.parts[i:i+3] == marker]
    _require(len(matches) == 1, "ARTIFACT_PATH_LINEAGE_MISMATCH")
    base = Path(root).resolve()
    path = base.joinpath(*raw.parts[matches[0]:]).resolve()
    run = base.joinpath(*marker).resolve()
    _require(base in run.parents and run in path.parents, "ARTIFACT_PATH_ESCAPE")
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hashed.update(block)
    _require(hashed.hexdigest() == artifact["file_sha256"], "ARTIFACT_FILE_DIGEST_MISMATCH")
    return {**artifact, "checkpoint_path": str(path), "verification": "SAME_STATE_FILE_SHA256_VERIFIED"}


def project_campaign(root, server):
    destination(server)
    scan = scan_campaign(root)
    groups, latest, owners, excluded = defaultdict(list), {}, set(), []
    conflicts = list(scan["conflicts"])
    for record in scan["records"]:
        key = run_key(record)
        if record["identity"]["server"] != server:
            conflicts.append({"source_result_id": record["record_id"], "reason": "FOREIGN_SERVER"})
            continue
        owners.add(key)
        if key not in latest or (record["created_at"], record["record_id"]) > (latest[key]["created_at"], latest[key]["record_id"]):
            latest[key] = record
        if record["record_type"] != "EVAL_SUMMARY":
            excluded.append({"source_result_id": record["record_id"], "reason": "STATUS_NOT_NUMERICAL_VIEW"})
            continue
        try:
            groups[key].append(_candidate(record, root, server))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            conflicts.append({"source_result_id": record["record_id"], "reason": str(exc)})
    rows, notes, selected, runs = [], [], [], []
    for key in sorted(owners):
        candidates = groups[key]
        by_step = defaultdict(list)
        for candidate in candidates:
            by_step[candidate["step"]].append(candidate)
        bad = len({(c["config_sha"], c["protocol"], c["inference_policy"]) for c in candidates}) > 1
        if bad:
            conflicts.append({"run_key": key, "reason": "CONFIG_OR_PROTOCOL_CHANGED_WITHIN_RUN"})
        state_metrics = defaultdict(set)
        for candidate in candidates:
            state_metrics[candidate["state"]].add(digest(candidate["metrics"]))
        if any(len(values) > 1 for values in state_metrics.values()):
            conflicts.append({"run_key": key, "reason": "SAME_MODEL_STATE_DIFFERENT_METRICS"})
            bad = True
        unique = []
        for step, values in sorted(by_step.items()):
            if len({(c["state"], digest(c["metrics"])) for c in values}) > 1:
                conflicts.append({"run_key": key, "step": step, "reason": "CONFLICTING_STATE_OR_METRICS"})
                bad = True
            unique.append(min(values, key=lambda c: c["source_result_id"]))
        h, e = LANES[server][3:]
        passed = [c for c in unique if c["hqnr"] >= h and c["ergas"] <= e]
        mode = "PASS_ALL_STEPS" if passed else "BEST_HQNR_FALLBACK"
        kept = [] if bad else passed if passed else sorted(unique,
            key=lambda c: (-c["hqnr"], c["ergas"], c["step"], c["source_result_id"]))[:1]
        last_status = latest[key]["row"][INDEX["Status"]]
        runs.append({"run_key": key, "candidate_count": len(unique), "pass_count": len(passed),
            "kept_count": len(kept), "selection": mode if kept else "NO_VALID_EVALUATION",
            "last_logged_status": last_status, "last_observed_at": latest[key]["created_at"],
            "live_process_verified": False})
        for candidate in kept:
            record = candidate["record"]
            cfg = record["source"]["extra"]["configuration"]
            row = list(record["row"])
            row[INDEX["Result_ID"]] = "VIEW-" + digest({"policy": POLICY, "run_key": key,
                "protocol": candidate["protocol"], "slot": candidate["step"] if passed else "FALLBACK"})
            row[INDEX["Selector"]], row[INDEX["Readback"]] = mode, "VERIFIED"
            row[INDEX["Experiment"]] = cfg["run_display"]
            row[INDEX["Selection@Step"]] = ("충족" if passed else "대표(미달)") + f" · {candidate['step']:,}"
            row[INDEX["Status"]] = last_status
            try:
                artifact = _artifact(candidate, root)
            except (ValueError, KeyError, OSError) as exc:
                conflicts.append({"source_result_id": record["record_id"], "reason": str(exc)})
                continue
            row[INDEX["Checkpoint_SHA"]] = artifact.get("file_sha256") or ""
            publication = {"policy": POLICY, "selection": mode, "hqnr_min": h, "ergas_max": e,
                "candidate_count": len(unique), "passed_count": len(passed),
                "source_result_id": record["record_id"], "evaluation_step": candidate["step"],
                "evaluation_evidence": candidate["evaluation_evidence"], "artifact": artifact,
                "last_logged_status": last_status, "last_observed_at": latest[key]["created_at"],
                "live_process_verified": False, "donor_retention_does_not_change_display_rule": True}
            note = json.loads(row[INDEX["Original description / notes"]])
            note["publication"] = publication
            row[INDEX["Original description / notes"]] = canonical_json(note)
            notes.append(canonical_json({"configuration": cfg, "publication": publication}))
            rows.append(row)
            selected.append({k: candidate[k] for k in ("key", "step", "hqnr", "ergas", "state", "protocol", "source_result_id")})
    return {"policy": POLICY, "destination": destination(server), "rows": rows, "notes": notes,
        "selected": selected, "runs": runs, "conflicts": conflicts, "excluded": excluded,
        "owned_run_keys": sorted(owners), "raw_ids": scan["raw_ids"], "raw_count": scan["raw_count"],
        "summary_count": sum(r["record_type"] == "EVAL_SUMMARY" for r in scan["records"]),
        "source_manifest": scan["source_manifest"], "source_digest": scan["source_digest"],
        "view_hash": digest({"rows": rows, "notes": notes})}


def project(root, server, *, require_cost=True):
    from .cost import attach_cost
    value = project_campaign(root, server)
    attach_cost(value, Path(root), require_cost=require_cost)
    value['view_hash'] = digest({'rows': value['rows'], 'notes': value['notes']})
    return value


def row_requests(snapshot, value, server):
    """Own campaign+server only. appendCells is server-side append, not guessed row.

    Clearing obsolete fallback rows does not delete/shift any worksheet row.
    Foreign and historical cells are never included in a write request.
    """
    from sheet_publish_v2.transport import _typed, _matrix_equal
    from .reporting import CAMPAIGN_ID
    idx = INDEX
    own = {}
    owned_keys = {tuple(k) for k in value.get('owned_run_keys', [])}
    for number, row in enumerate(snapshot['rows'], 5):
        if row[idx['Campaign_ID']] != CAMPAIGN_ID or row[idx['Server']] != server:
            continue
        if row[idx['Dataset']] != 'WV3' or not str(row[idx['Run_ID']]).startswith('P30R_'):
            raise ValueError('Own-row identity mismatch')
        key = tuple(row[idx[k]] for k in ('Campaign_ID','Dataset','Server','Run_ID','Attempt'))
        if key not in owned_keys:
            raise ValueError('Remote own row absent from local evidence; never erase it')
        rid = row[idx['Result_ID']]
        if not isinstance(rid, str) or not rid.startswith('VIEW-') or rid in own:
            raise ValueError('Duplicate/unknown owned display ID; no overwrite')
        own[rid] = (number, row, snapshot['notes'][number-5])
    expected = {row[idx['Result_ID']]: (row, note) for row, note in zip(value['rows'], value['notes'])}
    if len(expected) != len(value['rows']):
        raise ValueError('Duplicate desired IDs')
    def cells(row, note):
        return {'values': [{'userEnteredValue': _typed(v), 'note': note if j == 0 else ''}
                           for j, v in enumerate(row)]}
    requests, additions = [], []
    for rid, (row, note) in expected.items():
        if rid not in own:
            additions.append(cells(row, note)); continue
        number, old, old_note = own[rid]
        if _matrix_equal([old], [row]) and old_note == note:
            continue
        requests.append({'updateCells': {'range': {'sheetId': 261001010,
            'startRowIndex': number, 'endRowIndex': number+1, 'startColumnIndex': 0, 'endColumnIndex': 64},
            'rows': [cells(row, note)], 'fields': 'userEnteredValue,note'}})
    for rid, (number, row, note) in own.items():
        if rid not in expected:
            requests.append({'updateCells': {'range': {'sheetId': 261001010,
                'startRowIndex': number, 'endRowIndex': number+1, 'startColumnIndex': 0, 'endColumnIndex': 64},
                'rows': [cells(['']*64, '')], 'fields': 'userEnteredValue,note'}})
    if additions:
        requests.append({'appendCells': {'sheetId': 261001010, 'rows': additions,
                                        'fields': 'userEnteredValue,note'}})
    return requests


def publish(root, server, *, credentials='gspread/account.json', apply=False,
            watch=False, poll_seconds=120, require_cost=True):
    """Explicit cloud writes only; failures preserve outbox and do not stop training."""
    import fcntl
    import time
    from sheet_publish_v2.transport import GoogleSheets
    from .transport import reconcile_rows
    root = Path(root).resolve(strict=True)
    state = root/'sheet_publish_panda30r'
    state.mkdir(parents=True, exist_ok=True)
    # Distinct campaign ownership: earlier row-scoped writers can finish
    # their own campaigns. Never acquire their lock and silently kill them.
    adapter = GoogleSheets(credentials) if apply else None
    with (state/('writer_'+server+'.lock')).open('a+') as lock:
        if apply:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        terminal_seen = None
        while True:
            try:
                terminal_before = _terminal_state(root)
                value = project(root, server, require_cost=require_cost)
                R.atomic_json(state/'projection.json', value)
                if value['conflicts']:
                    raise ValueError('Evidence/cost conflicts; see projection.json')
                report = (reconcile_rows(adapter, value, state, server, apply=True) if apply
                          else {'action': 'LOCAL_DRY_RUN', 'rows': len(value['rows']),
                                'readback_verified': False})
                R.atomic_json(state/'last_poll.json', report)
                print(canonical_json(report), flush=True)
                if (watch and apply and terminal_before is not None and
                        terminal_upload_complete(root, value, report, terminal_before)):
                    marker = (canonical_json(terminal_before), value['source_digest'])
                    # Two consecutive polls prevent an immediate exit on an
                    # old FAILED state while a relaunched controller imports
                    # Torch and has not yet written its new RUNNING state.
                    if terminal_seen == marker:
                        R.atomic_json(state/'terminal_upload.json', {
                            'campaign_status': terminal_before['status'],
                            'source_digest': value['source_digest'],
                            'raw_count': value['raw_count'], 'row_count': len(value['rows']),
                            'readback_verified': True, 'at': R.utc_now()})
                        return 0
                    terminal_seen = marker
                else:
                    terminal_seen = None
            except Exception as exc:
                terminal_seen = None
                R.atomic_json(state/'last_error.json',
                    {'error': repr(exc), 'at': R.utc_now(), 'training_untouched': True})
                print('[Panda30R publisher] '+repr(exc), flush=True)
                if not watch:
                    return 1
            if not watch:
                return 0
            time.sleep(poll_seconds)


def _terminal_state(root):
    """Only controller terminal state, never an individual completed run."""
    path = Path(root)/'campaign_state.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    return value if value.get('status') in (
        'FINISHED_REGISTERED_QUEUE', 'STOPPED_BUDGET', 'FAILED') else None


def terminal_upload_complete(root, value, report, terminal_before):
    """Exit watcher only after stable terminal evidence and verified final view.

    A failed network request, incomplete cost profile or partial outbox never
    masquerades as upload completion. If the controller is restarted during a
    poll, changed state/ledger keeps the publisher alive.
    """
    if (terminal_before is None or _terminal_state(root) != terminal_before or
            value.get('conflicts') or report.get('readback_verified') is not True):
        return False
    after = scan_campaign(root)
    return not after['conflicts'] and after['source_digest'] == value['source_digest']


if __name__ == '__main__':
    import argparse
    import math
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--server', choices=('s1','s3','s4'), required=True)
    parser.add_argument('--credentials', default='gspread/account.json')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=120)
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error('poll-seconds must be positive')
    raise SystemExit(publish(args.root, args.server, credentials=args.credentials,
                            apply=args.apply, watch=args.watch, poll_seconds=args.poll_seconds))
