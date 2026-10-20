"""Mature30 selected evidence projection; explicit row-scoped cloud writes only.

Stage1: one BEST_VAL_RRE reused-source row, irrespective of HQNR threshold.
Stage2: all jointly passing steps else best-HQNR fallback, plus FINAL50K.
All raw evaluations remain immutable locally; no all-step cloud publication.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from directshift48 import publication as legacy_direct
from . import reporting as R

ROOT = Path(__file__).resolve().parents[1]
INDEX, HEADERS = R.INDEX, R.HEADERS
POLICY = "PANDA_MATURE30_RRE_PASS_OR_BEST_PLUS_FINAL50K_V1"
HQNR_MIN, ERGAS_MAX = .960, 2.045
LANES = {s: ("WV3", "WV3-ablations", 261001010, HQNR_MIN, ERGAS_MAX) for s in ("s1", "s3")}
canonical_json, digest = R.canonical_json, R.digest
destination = R.destination


def _require(test, reason):
    if not test:
        raise ValueError(reason)


def _resolve_evidence_path(raw, root):
    """Resolve Docker /workspace paths on the host without rewriting evidence.

    Only the exact registered repository mount is translated. Absolute paths
    elsewhere must already lie in this repository or the caller's campaign root.
    Traversal and symlink escapes are rejected before any evidence is read.
    """
    _require(isinstance(raw, (str, Path)), "INVALID_EVIDENCE_PATH")
    path = Path(raw)
    _require(path.is_absolute() and ".." not in path.parts, "UNSAFE_EVIDENCE_PATH")
    if path.parts[:2] == ("/", "workspace"):
        path = ROOT.joinpath(*path.parts[2:])
    resolved = path.resolve(strict=True)
    allowed = (ROOT.resolve(), Path(root).resolve(strict=True))
    _require(any(resolved == base or base in resolved.parents for base in allowed),
             "EVIDENCE_PATH_OUTSIDE_REPOSITORY_OR_CAMPAIGN")
    return resolved


def _verify_corrections(evaluation, cfg):
    manifest = R.deployment(evaluation) or {}
    step = evaluation["state_before"]["completed_update"]
    policy = cfg["policy_id"]
    key = {"FULL": "s100", "HALF": "s050", "LOW": "s025", "CAP085": "cap085",
           "DOWN": "s050" if step <= 10000 else "s0375" if step <= 30000 else "s025"}[policy]
    _require(manifest.get("schema") == "PANDA_MATURE30_DEPLOYMENT_V1" and
             manifest.get("policy_id") == policy and manifest.get("completed_update") == step and
             manifest.get("geometry_key") == key, "DEPLOYMENT_PHASE_MISMATCH")
    geometry = manifest.get("geometry", {})
    scale = {"s100": 1., "s050": .5, "s0375": .375, "s025": .25}.get(key)
    cap = .85 if key == "cap085" else None
    for field, expected in (("schema", "PANDA_MATURE30_APPLIED_GEOMETRY_V1"),
            ("geometry_key", key), ("operator", "RADIAL_CAP" if cap else "SCALE"),
            ("scale", scale), ("cap_px", cap), ("upsampler", "interp23"), ("frame", "pan"),
            ("output_coordinates", "MS"), ("ms_shift_factor", 0.), ("output_shift_factor", 0.),
            ("shift_unit", "CURRENT_PAN_PIXEL_NO_X4"),
            ("evaluation_references", "ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20")):
        _require(geometry.get(field) == expected, "APPLIED_GEOMETRY_MISMATCH:" + field)
    _require(evaluation.get("reference_warp") is False and
             evaluation.get("inverse_output_warp") is False and
             evaluation.get("output_coordinates") == "MS" and
             evaluation.get("upsampler") == "interp23" and evaluation.get("frame") == "pan",
             "ORIGINAL_REFERENCE_OR_FRAME_MISMATCH")
    for split in ("rr", "fr"):
        for row in evaluation[split]["per_scene"]:
            _require(row.get("correction_band_order") is None, "UNEXPECTED_BANDWISE_CORRECTION")
            vectors = {}
            for field in ("correction_dy_dx", "pan_applied_dy_dx",
                          "ms_applied_dy_dx", "output_applied_dy_dx"):
                vector = row.get(field)
                _require(isinstance(vector, list) and len(vector) == 2 and
                         all(type(x) in (int, float) and math.isfinite(x) for x in vector),
                         "NONFINITE_OR_MISSING_CORRECTION:" + field)
                vectors[field] = vector
            u = vectors["correction_dy_dx"]
            norm = math.hypot(*u)
            factor = min(1., cap / norm) if cap and norm else (1. if cap else scale)
            expected_pan = [factor * x for x in u]
            _require(all(math.isclose(x,y,rel_tol=1e-6,abs_tol=1e-7) for x,y in
                         zip(expected_pan,vectors["pan_applied_dy_dx"])),
                     "APPLIED_CORRECTION_POLICY_MISMATCH")
            _require(vectors["ms_applied_dy_dx"] == [0.,0.] and
                     vectors["output_applied_dy_dx"] == [0.,0.], "SHIFTED_MS_OR_OUTPUT")

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
    ident, extra = record["identity"], record["source"]["extra"]
    cfg, state = extra["configuration"], extra["identity"]
    _require(ident["server"] == server, "FOREIGN_SERVER")
    _require(ident["eval_scope"] == "EVAL_SUMMARY", "NOT_EVAL_SUMMARY")
    legacy_direct._check_complete(extra)
    _require(R._sha(state.get("model_state_sha256")) and
             R._sha(state.get("evaluator_revision")), "INVALID_STATE_OR_EVALUATOR_SHA")
    _require(state["evaluator_revision"] == ident["eval_protocol"], "EVALUATOR_IDENTITY_MISMATCH")
    metrics = record["source"]["metrics"]
    if cfg["phase"] == "donor":
        receipt = extra["selector_receipt"]
        selected = R.selected_source(receipt)
        full_path = _resolve_evidence_path(receipt["source_receipt_path"], root)
        expected_receipt = (Path(root) / "selection_receipts" /
                            (receipt["source_receipt_sha256"] + ".json")).resolve(strict=True)
        _require(full_path == expected_receipt, "SELECTION_RECEIPT_PATH_MISMATCH")
        full = json.loads(full_path.read_text())
        R.selected_source(full)
        _require(full["receipt_sha256"] == receipt["source_receipt_sha256"],
                 "FULL_SELECTION_RECEIPT_DIGEST_MISMATCH")
        plain_selected = deepcopy(full["selected"])
        plain_selected["validation"] = {k:v for k,v in plain_selected["validation"].items()
                                        if k not in ("per_sample", "protocol")}
        _require(plain_selected == selected, "COMPACT_SELECTION_SOURCE_MISMATCH")
        _require(full.get("selector") == "BEST_VAL_RRE" and full.get("server") == server,
                 "SELECTION_RECEIPT_IDENTITY_MISMATCH")
        candidates = full.get("candidates", [])
        _require(bool(candidates) and full["selected"] in candidates, "MISSING_SELECTION_CANDIDATES")
        geometry = lambda c: (c["validation"]["RRE50"], c["validation"]["RRE90"])
        best_geometry = min(map(geometry, candidates))
        tied = [c for c in candidates if geometry(c) == best_geometry]
        best = min(tied, key=lambda c: (*geometry(c), c["validation"]["Val_ERGAS"],
                                       c["completed_update"]))
        _require(best == full["selected"], "BEST_VAL_RRE_RANK_VIOLATION")
        original = selected["source_config"]
        _require(original["server"] == server and original["dataset"] == "WV3",
                 "CROSS_SERVER_DONOR_REUSE_FORBIDDEN")
        _require(original.get("phase") == "donor" and original.get("upsampler") == "interp23"
                 and original.get("alignment_target") == "SCHARR_OF_MEAN8_GT",
                 "UNREGISTERED_DONOR_SOURCE")
        source_root = selected.get("source_root", receipt.get("source_root"))
        _require(isinstance(source_root, str), "SOURCE_CAMPAIGN_ROOT_REQUIRED")
        source_root = _resolve_evidence_path(source_root, root)
        proof = legacy_direct._verify_full_evaluation(record, original, state, metrics, source_root)
        _require(Path(proof["path"]).resolve() ==
                 _resolve_evidence_path(selected["source_evaluation_path"], root),
                 "SOURCE_EVALUATION_PATH_MISMATCH")
        if selected.get("source_evaluation_sha256"):
            _require(proof["sha256"] == selected["source_evaluation_sha256"],
                     "SOURCE_EVALUATION_DIGEST_MISMATCH")
        evaluation = json.loads(Path(proof["path"]).read_text())
        # Legacy G8 and freshly trained Mature30 Stage1 have the same physical
        # full-u path, but different config geometry manifests. Verify actual
        # corrections directly rather than interpreting new policy as old schema.
        for split in ("rr", "fr"):
            for row in evaluation[split]["per_scene"]:
                u = row.get("correction_dy_dx")
                _require(isinstance(u, list) and len(u) == 2 and all(
                    type(x) in (int,float) and math.isfinite(x) for x in u),
                    "INVALID_STAGE1_CORRECTION")
                _require(row.get("pan_applied_dy_dx") == u and
                         row.get("ms_applied_dy_dx") == [0.,0.] and
                         row.get("output_applied_dy_dx") == [0.,0.],
                         "STAGE1_NOT_FULL_U_ORIGINAL_COORDINATES")
        _require(evaluation.get("reference_warp") is False and evaluation.get("frame") == "pan"
                 and evaluation.get("upsampler") == "interp23", "IMPORTED_REFERENCE_POLICY_MISMATCH")
    else:
        proof = legacy_direct._verify_full_evaluation(record, cfg, state, metrics, root)
        evaluation = json.loads(Path(proof["path"]).read_text())
        _verify_corrections(evaluation, cfg)
        _require(extra.get("deployment_manifest") == R.deployment(evaluation),
                 "DEPLOYMENT_EVIDENCE_MISMATCH")
    _require(0 <= metrics["HQNR"] <= 1 and metrics["ERGAS"] >= 0, "INVALID_METRIC_RANGE")
    _require(extra.get("jqm_variant", "") == evaluation.get("jqm_variant", ""),
             "JQM_VARIANT_EVIDENCE_MISMATCH")
    return {"record": record, "key": run_key(record), "step": ident["selected_step"],
        "hqnr": metrics["HQNR"], "ergas": metrics["ERGAS"], "metrics": metrics,
        "state": state["model_state_sha256"], "protocol": ident["eval_protocol"],
        "config_sha": digest(cfg), "inference_policy": state.get("inference_policy_digest"),
        "source_result_id": record["record_id"], "evaluation_evidence": proof}


def _file_sha(path):
    hashed = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hashed.update(block)
    return hashed.hexdigest()

def _artifact(candidate, root):
    record = candidate["record"]
    extra = record["source"]["extra"]
    if extra["configuration"]["phase"] == "donor":
        selected = R.selected_source(extra["selector_receipt"])
        path = _resolve_evidence_path(selected["checkpoint_path"], root)
        source = selected["source_config"]
        marker = ("runs", source["run_id"], "attempt_%02d" % source["attempt"])
        _require(any(path.parts[i:i+3] == marker for i in range(len(path.parts)-2)),
                 "IMPORTED_CHECKPOINT_LINEAGE_MISMATCH")
        _require(_file_sha(path) == selected["file_sha256"], "IMPORTED_CHECKPOINT_SHA_MISMATCH")
        return {"weights_saved": True, "checkpoint_path": str(path),
                "file_sha256": selected["file_sha256"], "model_state_sha256": candidate["state"],
                "original_source_state": selected["source_state"],
                "verification": "REUSED_SOURCE_FILE_SHA256_VERIFIED"}
    artifact = extra.get("artifact", {})
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
    rows, notes, selected, runs, evidence = [], [], [], [], {}
    for key in sorted(owners):
        candidates = groups[key]
        by_step = defaultdict(list)
        for candidate in candidates:
            by_step[candidate["step"]].append(candidate)
        bad = len({(c["config_sha"], c["protocol"]) for c in candidates}) > 1
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
        best = sorted(unique, key=lambda c: (-c["hqnr"], c["ergas"], c["step"], c["source_result_id"]))[:1]
        donor = latest[key]["source"]["extra"]["configuration"]["phase"] == "donor"
        if donor:
            _require(len(unique) <= 1, "DONOR_SELECTION_IMMUTABLE_CREATE_NEW_ATTEMPT")
            mode, kept = "BEST_VAL_RRE", unique
        else:
            mode = "PASS_ALL_STEPS" if passed else "BEST_HQNR_FALLBACK"
            kept = list(passed if passed else best)
            kept += [c for c in unique if c["step"] == 50000 and c not in kept]
        if bad:
            kept = []
        for candidate in unique:
            evidence[candidate["source_result_id"]] = {
                "candidate": {k:candidate[k] for k in ("key","step","hqnr","ergas","state","protocol","source_result_id")},
                "row": candidate["record"]["row"], "evaluation_evidence": candidate["evaluation_evidence"]}
        last_status = latest[key]["row"][INDEX["Status"]]
        runs.append({"run_key": key, "candidate_count": len(unique), "pass_count": len(passed),
            "kept_count": len(kept), "selection": mode if kept else "NO_VALID_EVALUATION",
            "last_logged_status": last_status, "last_observed_at": latest[key]["created_at"],
            "live_process_verified": False})
        for candidate in kept:
            record = candidate["record"]
            cfg = record["source"]["extra"]["configuration"]
            row = list(record["row"])
            final = candidate["step"] == 50000 and not donor
            display_mode = mode if donor or candidate in (passed if passed else best) else "FINAL50K"
            slot = "RRE" if donor else "FALLBACK" if display_mode == "BEST_HQNR_FALLBACK" else candidate["step"]
            row[INDEX["Result_ID"]] = display_id(POLICY, key, candidate["protocol"], slot)
            row[INDEX["Selector"]], row[INDEX["Readback"]] = display_mode, "VERIFIED"
            row[INDEX["Experiment"]] = cfg["run_display"]
            is_best = bool(best and candidate["source_result_id"] == best[0]["source_result_id"])
            label = ("정합최적" if donor else "충족" if display_mode == "PASS_ALL_STEPS"
                     else "대표(미달)" if display_mode == "BEST_HQNR_FALLBACK" else "FINAL50K")
            row[INDEX["Selection@Step"]] = label + (" / FINAL50K" if final and label != "FINAL50K" else "") + f" · {candidate['step']:,}"
            row[INDEX["Status"]] = last_status
            try:
                artifact = _artifact(candidate, root)
            except (ValueError, KeyError, OSError) as exc:
                conflicts.append({"source_result_id": record["record_id"], "reason": str(exc)})
                continue
            row[INDEX["Checkpoint_SHA"]] = artifact.get("file_sha256") or ""
            publication = {"policy": POLICY, "selection": display_mode, "final50k": final, "hqnr_min": h, "ergas_max": e,
                "candidate_count": len(unique), "passed_count": len(passed),
                "hqnr_primary_rank_best": is_best,
                "checkpoint_selection": "BEST_VAL_RRE" if donor else "HQNR_DESC_ERGAS_ASC_STEP_ASC",
                "target_thresholds_are_annotations_only": donor,
                "source_result_id": record["record_id"], "evaluation_step": candidate["step"],
                "evaluation_evidence": candidate["evaluation_evidence"], "artifact": artifact,
                "last_logged_status": last_status, "last_observed_at": latest[key]["created_at"],
                "live_process_verified": False, "donor_retention_does_not_change_display_rule": True}
            note = json.loads(row[INDEX["Original description / notes"]])
            note["publication"] = publication
            row[INDEX["Original description / notes"]] = canonical_json(note)
            short_note = canonical_json({"configuration": cfg, "publication": publication})
            _require(max(len(row[INDEX["Original description / notes"]]),len(short_note)) < 45000,
                     "PUBLICATION_NOTE_EXCEEDS_SAFE_CELL_SIZE")
            notes.append(short_note)
            rows.append(row)
            selected.append({k: candidate[k] for k in ("key", "step", "hqnr", "ergas", "state", "protocol", "source_result_id")})
    return {"policy": POLICY, "destination": destination(server), "rows": rows, "notes": notes,
        "selected": selected, "runs": runs, "conflicts": conflicts, "excluded": excluded,
        "owned_run_keys": sorted(owners), "raw_ids": scan["raw_ids"], "raw_count": scan["raw_count"],
        "summary_count": sum(r["record_type"] == "EVAL_SUMMARY" for r in scan["records"]),
        "source_manifest": scan["source_manifest"], "source_digest": scan["source_digest"],
        "source_evidence": evidence, "view_hash": digest({"rows": rows, "notes": notes})}


def display_id(policy, key, protocol, slot):
    return "VIEW-" + digest({"policy": policy, "run_key": key, "protocol": protocol, "slot": slot})


def project(root, server, *, require_cost=False):
    value = project_campaign(root, server)
    # Never invent costs or block verified metrics on absent optional profiling.
    # A dedicated deployment profiler can supply this hook later.
    try:
        from .cost import attach_cost
    except ImportError:
        value["cost"] = {"bound_count": 0, "missing": True, "require_cost": require_cost}
        if require_cost and value["rows"]:
            value["conflicts"].append({"reason": "NO_VERIFIED_MATURE30_COST"})
    else:
        attach_cost(value, Path(root), require_cost=require_cost)
    value["view_hash"] = digest({"rows": value["rows"], "notes": value["notes"]})
    return value


def _key(row):
    return tuple(row[INDEX[k]] for k in ('Campaign_ID', 'Dataset', 'Server', 'Run_ID', 'Attempt'))


def _verify_owned(row, value):
    """Never clear/update an unproven row, even with the right campaign label."""
    from sheet_publish_v2.transport import _matrix_equal
    publication = json.loads(row[INDEX['Original description / notes']])['publication']
    source = value['source_evidence'].get(publication.get('source_result_id'))
    if source is None:
        raise ValueError('Remote own row absent from local evidence; never erase it')
    candidate, raw = source['candidate'], source['row']
    if _key(row) != tuple(candidate['key']):
        raise ValueError('Remote source/run mismatch')
    if publication.get('evaluation_step') != candidate['step'] or row[INDEX['Selected_step']] != candidate['step']:
        raise ValueError('Remote source/step mismatch')
    # SHA authenticates the complete same-state RR20+FR20 evaluation, not just H/E.
    if publication.get('evaluation_evidence', {}).get('sha256') != source['evaluation_evidence']['sha256']:
        raise ValueError('Remote evaluation proof mismatch')
    columns = list(range(3, 15)) + [INDEX[k] for k in ('Case_ID', 'Repeat', 'Seed', 'Eval_scope', 'JQM_variant')]
    if not _matrix_equal([[row[i] for i in columns]], [[raw[i] for i in columns]]):
        raise ValueError('Remote metrics/identity differ from immutable source')
    policy, mode = publication.get('policy'), publication.get('selection')
    allowed = ((POLICY, 'BEST_VAL_RRE'), (POLICY, 'FINAL50K'),
               (POLICY, 'PASS_ALL_STEPS'), (POLICY, 'BEST_HQNR_FALLBACK'))
    if (policy, mode) not in allowed:
        raise ValueError('Unknown remote publication policy; never erase it')
    slot = ('RRE' if mode == 'BEST_VAL_RRE' else 'FALLBACK'
            if mode == 'BEST_HQNR_FALLBACK' else candidate['step'])
    if row[INDEX['Result_ID']] != display_id(policy, candidate['key'], candidate['protocol'], slot):
        raise ValueError('Remote display identity mismatch; never erase it')


def row_requests(snapshot, value, server):
    """Reuse proven obsolete own slots; clear surplus cells, never delete rows."""
    from sheet_publish_v2.transport import _typed, _matrix_equal
    if value['conflicts']:
        raise ValueError('Local evidence conflicts; cloud untouched')
    own, owned_keys = {}, {tuple(k) for k in value['owned_run_keys']}
    for number, row in enumerate(snapshot['rows'], 5):
        if row[INDEX['Campaign_ID']] != R.CAMPAIGN_ID or row[INDEX['Server']] != server:
            continue
        if row[INDEX['Dataset']] != 'WV3' or not str(row[INDEX['Run_ID']]).startswith('PM30_'):
            raise ValueError('Own-row identity mismatch')
        if _key(row) not in owned_keys:
            raise ValueError('Remote own run absent from local evidence; never erase it')
        _verify_owned(row, value)
        rid = row[INDEX['Result_ID']]
        if rid in own:
            raise ValueError('Duplicate owned display ID; no overwrite')
        own[rid] = (number, row, snapshot['notes'][number - 5])
    expected = {row[INDEX['Result_ID']]: (row, note) for row, note in zip(value['rows'], value['notes'])}
    if len(expected) != len(value['rows']) or len(value['rows']) != len(value['notes']):
        raise ValueError('Duplicate desired IDs or mismatched notes')
    stale = {rid: item for rid, item in own.items() if rid not in expected}
    def cells(row, note):
        return {'values': [{'userEnteredValue': _typed(v), 'note': note if j == 0 else ''}
                           for j, v in enumerate(row)]}
    def update(number, payload):
        return {'updateCells': {'range': {'sheetId': value['destination']['sheet_id'],
            'startRowIndex': number, 'endRowIndex': number + 1,
            'startColumnIndex': 0, 'endColumnIndex': 64},
            'rows': [payload], 'fields': 'userEnteredValue,note'}}
    requests, additions = [], []
    for rid, (row, note) in expected.items():
        existing = own.get(rid)
        if existing is None:
            reusable = next((key for key, item in stale.items() if _key(item[1]) == _key(row)), None)
            if reusable is not None:
                existing = stale.pop(reusable)
            else:
                additions.append(cells(row, note)); continue
        number, old, old_note = existing
        if not _matrix_equal([old], [row]) or old_note != note:
            requests.append(update(number, cells(row, note)))
    for number, row, note in stale.values():
        requests.append(update(number, {'values': [{} for _ in range(64)]}))
    if additions:
        requests.append({'appendCells': {'sheetId': value['destination']['sheet_id'],
                                        'rows': additions, 'fields': 'userEnteredValue,note'}})
    return requests

def publish(root, server, *, credentials='gspread/account.json', apply=False,
            watch=False, poll_seconds=120, require_cost=False):
    """Explicit cloud writes only; failures preserve outbox and do not stop training."""
    import fcntl
    import time
    from sheet_publish_v2.transport import GoogleSheets
    from .transport import reconcile_rows
    root = Path(root).resolve(strict=True)
    state = root/'sheet_publish_panda_mature30'
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
                print('[Mature30 publisher] '+repr(exc), flush=True)
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
        'FINISHED_REGISTERED_QUEUE', 'FINISHED_CORE_OPTIONALS_BUDGET_SKIPPED',
        'FINISHED_WITH_BUDGET_SKIPS', 'FINISHED_WITH_EXPLICIT_SKIPS',
        'STOPPED_SAFE', 'FAILED') else None


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
    parser.add_argument('--server', choices=('s1','s3'), required=True)
    parser.add_argument('--credentials', default='gspread/account.json')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=120)
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error('poll-seconds must be positive')
    raise SystemExit(publish(args.root, args.server, credentials=args.credentials,
                            apply=args.apply, watch=args.watch, poll_seconds=args.poll_seconds))

