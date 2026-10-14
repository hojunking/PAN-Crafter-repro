"""Fixed64 publication policy. Standard-library only; never imports the trainer.

The original EGR2 records and artifact qualification remain unchanged. Numeric
rows are views of exact, whole-dataset summaries, not additional experiments.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re

POLICY = "PAN_PASSED_STEPS_OR_BEST_HQNR_V1"
CAMPAIGN_ID = "PANDA_S1_EDGERES_TEACHER48H_S123_20261013_v2"
SPREADSHEET_ID = "1_ANKPQjQzMZrw9fUc1OI3B9xF4K3sxNa_fybg1kjBXE"
HEADER_SHA256 = "bfc7f73fd406f447edb9721823e2c1df85540f7068e15647947cd0d2e83a11b1"
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "research_log/gspread_rules/PAN_FIXED64_V1.schema.json").read_text())
HEADERS = tuple(c["header"] for c in SCHEMA["columns"])
INDEX = {h: i for i, h in enumerate(HEADERS)}
LANES = {"s1": ("WV3", "WV3-ablations", 261001010, .960, 2.045),
         "s2": ("QB", "QB-ablations", 261001011, .923, 3.500),
         "s3": ("GF2", "GF2-ablations", 261001012, .965, .522)}
METRICS = {"HQNR": "HQNR↑", "D_s": "D_s↓", "D_lambda": "D_lambda↓", "JQM": "JQM↑",
           "ERGAS": "ERGAS↓", "SCC": "SCC↑", "SAM": "SAM↓", "PSNR": "PSNR↑",
           "SSIM": "SSIM↑", "Q": "Q4/Q8↑", "RMSE": "RMSE↓", "CC": "CC↑"}
METHOD = {"method_revision": "EdgeGuidedResidual-S1-v1",
          "method_sha256": "ac9e2cc7ef58defb03bf94b615faa525c911ead64471e9e7f6348a91d63d30d7",
          "experiment_plan_sha256": "4a606ace09746d957dbf850453b0c2b296ddb4532bcd3472939092dccdd4f1e3"}
FLAGS = ("same_state_verified", "metrics_complete", "metrics_finite", "full_method_eligible")
OBSOLETE = ("q_ref", "tau_R", "alpha", "beta", "lambda_E")


class EvidenceError(ValueError):
    pass


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def scientific_digest(value):
    # Existing evaluator/config digests use json's ensure_ascii=True default.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


if len(HEADERS) != 64 or digest(HEADERS) != HEADER_SHA256 or SCHEMA["spreadsheet_id"] != SPREADSHEET_ID:
    raise EvidenceError("Fixed64 local schema changed")


def _require(condition, reason):
    if not condition:
        raise EvidenceError(reason)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def destination(server):
    _require(server in LANES, "UNSUPPORTED_SERVER_OR_DATASET")
    _, title, gid, _, _ = LANES[server]
    return {"spreadsheet_id": SPREADSHEET_ID, "sheet_title": title, "sheet_id": gid}


def run_key(record):
    i = record["identity"]
    return tuple(i[k] for k in ("campaign", "dataset", "server", "run", "attempt"))


def validate_envelope(record):
    """Authenticate local row/source redundancy without importing watched code."""
    row, i, source = record["row"], record["identity"], record["source"]
    _require(len(row) == 64, "INVALID_ROW_WIDTH")
    _require(record["record_id"] == "EGR2-" + digest(i), "SOURCE_ID_DIGEST_MISMATCH")
    _require(record["payload_sha256"] == digest(row), "SOURCE_PAYLOAD_DIGEST_MISMATCH")
    _require(i["namespace"] == "PAN_FIXED64_V1:2026-10-13", "WRONG_NAMESPACE")
    _require(i["campaign"] == CAMPAIGN_ID and i["dataset"] == LANES.get(i["server"], (None,))[0], "WRONG_CAMPAIGN_OR_LANE")
    _require(record["destination"] == destination(i["server"]), "WRONG_DESTINATION")
    _require(record["record_type"] == i["eval_scope"], "SCOPE_IDENTITY_MISMATCH")
    fields = source["fields"]
    _require(not (set(fields) - set(HEADERS)), "UNKNOWN_SOURCE_FIELDS")
    for col in SCHEMA["columns"]:
        value = row[col["index_1based"] - 1]
        expected = fields.get(col["header"])
        _require(value == ("" if expected is None else expected), "ROW_SOURCE_MISMATCH:" + col["header"])
        if value != "":
            valid = (_number(value) if col["type"] == "number" else
                     type(value) is bool if col["type"] == "boolean" else isinstance(value, str))
            _require(valid, "BAD_COLUMN_TYPE:" + col["header"])
    for col, expected in {"Campaign_ID": i["campaign"], "Dataset": i["dataset"], "Server": i["server"],
                          "Run_ID": i["run"], "Attempt": i["attempt"], "Selected_step": i["selected_step"],
                          "Selector": i["selector"], "Checkpoint_SHA": i["checkpoint"] or "",
                          "Eval_scope": i["eval_scope"], "Result_ID": record["record_id"],
                          "Readback": "PENDING", "Test_aware": True, "Bucket": "ablations", "Date": record["created_at"]}.items():
        _require(row[INDEX[col]] == expected, "ROW_IDENTITY_MISMATCH:" + col)
    observed = dt.datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))
    _require(observed.tzinfo is not None and observed.utcoffset() is not None, "OBSERVATION_TIME_MISSING_TIMEZONE")
    _require(all(row[INDEX[k]] == "" for k in OBSOLETE), "OBSOLETE_COLUMNS_NONEMPTY")
    note = json.loads(row[INDEX["Original description / notes"]])
    _require(note["extra_parameters"] == source["extra"], "NOTE_SOURCE_EXTRA_MISMATCH")
    _require(all(note.get(k) == v for k, v in source["provenance"].items()), "NOTE_PROVENANCE_MISMATCH")
    _require(note.get("model_state_sha256") == i.get("model_state_sha256") and
             note.get("eval_protocol_id") == i["eval_protocol"], "NOTE_IDENTITY_MISMATCH")
    _require(all(note.get(k) == v for k, v in METHOD.items()), "METHOD_IDENTITY_MISMATCH")
    return note


def _check_complete(gate):
    _require(all(gate.get(k) is True for k in FLAGS), "INCOMPLETE_OR_INELIGIBLE_FLAGS")
    for split in ("rr", "fr"):
        c = gate.get("scene_completeness", {}).get(split, {})
        _require(type(c.get("expected")) is int and c["expected"] == 20 and
                 type(c.get("completed")) is int and c["completed"] == 20 and
                 all(c.get(k) == [] for k in ("missing", "duplicates", "unexpected")),
                 "INCOMPLETE_SCENES:" + split)


def _run_directory(root, cfg):
    root = Path(root).resolve()
    path = (root / "runs" / cfg["run_id"] / ("attempt_%02d" % cfg["attempt"])).resolve()
    _require(root in path.parents, "EVIDENCE_PATH_ESCAPE")
    return path


def _verify_full_evaluation(record, cfg, ident, metrics, root):
    directory = _run_directory(root, cfg)
    _require(json.loads((directory / "config.json").read_text()) == cfg, "RESOLVED_CONFIG_FILE_MISMATCH")
    path = directory / ("evaluation_%07d.json" % ident["completed_update"])
    declared = Path(record["source"]["extra"].get("source_evaluation", ""))
    _require(declared.parts[-4:] == path.parts[-4:], "SOURCE_EVALUATION_PATH_MISMATCH")
    raw_evaluation = path.read_bytes()
    evaluation = json.loads(raw_evaluation)
    _require(evaluation.get("record_type") == "EVAL_SUMMARY" and evaluation.get("full_method_eligible") is True,
             "NOT_FORMAL_EVALUATION")
    _require(evaluation.get("measurement_domain") == "OFFICIAL_FULL20_NATIVE" and
             evaluation.get("metric_function_injection") is False and evaluation.get("test_aware") is True,
             "NOT_OFFICIAL_NATIVE_PROTOCOL")
    _require(evaluation.get("eval_protocol_id") == record["identity"]["eval_protocol"], "EVALUATION_PROTOCOL_MISMATCH")
    _require(all(evaluation.get(k) == ident for k in ("state_before", "rr_state", "fr_state", "state_after")),
             "MIXED_RR_FR_MODEL_STATE")
    _check_complete(evaluation.get("qualification", {}))
    q = "Q8" if cfg["dataset"] == "WV3" else "Q4"
    for split, keys in (("rr", ("ERGAS", "SCC", "SAM", "PSNR", "SSIM", q, "RMSE", "CC")),
                        ("fr", ("HQNR", "D_s", "D_lambda", "JQM"))):
        section = evaluation[split]
        scenes = section["per_scene"]
        ids = [s.get("scene_id") for s in scenes]
        _require(len(scenes) == 20 and all(isinstance(x, str) and x for x in ids) and len(set(ids)) == 20,
                 "DUPLICATE_OR_MISSING_SCENES:" + split)
        expected_ids = {f"{cfg['dataset']}:{split}:{index}" for index in range(20)}
        _require(set(ids) == expected_ids, "OFFICIAL_SCENE_IDS_MISMATCH:" + split)
        for scene in scenes:
            _require(all(scene.get(alias) == scene["scene_id"] for alias in ("sample_id", "evaluation_image_id")
                         if alias in scene), "SCENE_LINEAGE_ID_MISMATCH:" + split)
        for k in keys:
            aggregate = section["metrics"].get(k)
            vals = [s.get("metrics", {}).get(k) for s in scenes]
            _require(_number(aggregate) and all(_number(v) for v in vals), "NONFINITE_SCENE_OR_MEAN:" + k)
            _require(aggregate == metrics[k], "SOURCE_EVAL_METRIC_MISMATCH:" + k)
            _require(math.isclose(aggregate, math.fsum(vals) / 20, rel_tol=0, abs_tol=1e-12), "SCENE_MEAN_MISMATCH:" + k)
    return {"path": str(path), "sha256": hashlib.sha256(raw_evaluation).hexdigest()}


def validate_candidate(record, *, server, evidence_root=None):
    note = validate_envelope(record)
    i, source = record["identity"], record["source"]
    _require(i["eval_scope"] == "EVAL_SUMMARY", "NOT_EVAL_SUMMARY")
    _require(i["server"] == server, "FOREIGN_SERVER")
    extra, cfg, ident = source["extra"], source["extra"]["configuration"], source["extra"]["identity"]
    _check_complete(extra)
    _require(all(cfg.get(k) == v and ident.get(k) == v for k, v in METHOD.items()), "METHOD_CONFIG_MISMATCH")
    _require(_integer(i["attempt"], 1) and _integer(i["selected_step"], 1) and _integer(cfg.get("seed")), "INVALID_INTEGER_IDENTITY")
    for outer, inner in (("run", "run_id"), ("attempt", "attempt"), ("selected_step", "completed_update")):
        _require(i[outer] == ident.get(inner), "EVALUATION_IDENTITY_MISMATCH:" + inner)
    for key in ("run_id", "attempt", "seed"):
        _require(cfg.get(key) == ident.get(key), "CONFIG_IDENTITY_MISMATCH:" + key)
    _require(cfg.get("campaign_id") == i["campaign"] and cfg.get("dataset") == i["dataset"] and cfg.get("server") == server,
             "CONFIG_LANE_MISMATCH")
    _require(cfg.get("case") == record["row"][INDEX["Case_ID"]] and
             cfg.get("replica") == "R" + str(record["row"][INDEX["Repeat"]]) and
             cfg["seed"] == record["row"][INDEX["Seed"]], "CASE_SEED_REPEAT_MISMATCH")
    _require(all(_sha(ident.get(k)) for k in ("model_state_sha256", "inference_policy_digest", "evaluator_revision", "edge_protocol_digest")), "INVALID_PROTOCOL_OR_STATE_SHA")
    _require(ident["model_state_sha256"] == i.get("model_state_sha256") and
             ident["edge_protocol_digest"] == cfg.get("edge_protocol_digest") and
             ident["evaluator_revision"] == i["eval_protocol"] == note["evaluator_revision"], "PROTOCOL_OR_MODEL_IDENTITY_MISMATCH")
    _require(note["config_sha256"] == scientific_digest(cfg["baseline_resolved"]), "BASELINE_DIGEST_MISMATCH")
    _require(all(_sha(note.get(k)) for k in ("config_sha256", "dataset_manifest_sha256")), "INVALID_PROVENANCE_SHA")
    metrics = source["metrics"]
    q = "Q8" if i["dataset"] == "WV3" else "Q4"
    for key, col in METRICS.items():
        value = metrics.get(q if key == "Q" else key)
        _require(_number(value), "MISSING_OR_NONFINITE_METRIC:" + key)
        _require(value == record["row"][INDEX[col]], "ROW_METRIC_MISMATCH:" + key)
    _require(0 <= metrics["HQNR"] <= 1 and metrics["ERGAS"] >= 0, "METRIC_RANGE")
    _require(extra.get("HQNR") == metrics["HQNR"] and extra.get("ERGAS") == metrics["ERGAS"], "GATE_METRIC_MISMATCH")
    _require(_integer(cfg.get("planned_horizon"), 1) and i["selected_step"] <= cfg["planned_horizon"], "STEP_OUTSIDE_HORIZON")
    for col, key in (("U_peak_lr", "lr_U"), ("A_peak_lr", "lr_A")):
        _require(_number(cfg.get(key)) and cfg[key] == record["row"][INDEX[col]], "LR_CONFIG_MISMATCH")
    evidence = None
    if evidence_root is not None:
        try:
            evidence = _verify_full_evaluation(record, cfg, ident, metrics, evidence_root)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            # A previously published view must never disappear merely because a
            # source/config file becomes unavailable or malformed during polling.
            raise EvidenceError("PHYSICAL_EVALUATION_PROOF_MISMATCH:" + str(exc)) from exc
    return {"record": record, "key": run_key(record), "step": i["selected_step"], "hqnr": metrics["HQNR"],
            "ergas": metrics["ERGAS"], "state": i["model_state_sha256"], "protocol": i["eval_protocol"],
            "config_sha": scientific_digest(cfg), "inference_policy": ident["inference_policy_digest"],
            "metrics": metrics, "source_result_id": record["record_id"], "evaluation_evidence": evidence}


def _artifact_evidence(candidate, records, root):
    """Never infer saved weights from metric pass or join another step's file."""
    evidence = {"weights_saved": False, "checkpoint_path": None, "file_sha256": None,
                "model_state_sha256": candidate["state"], "qualitative_files": [],
                "verification": "NO_VERIFIED_SAME_STATE_SAVED_FILE"}
    warnings = []
    for record in records:
        i = record["identity"]
        if i.get("model_state_sha256") != candidate["state"] or i.get("selected_step") != candidate["step"]:
            continue
        a = record["source"]["extra"].get("artifact", {})
        if a.get("weights_saved") is not True:
            continue
        try:
            validate_envelope(record)
            _require(run_key(record) == candidate["key"], "ARTIFACT_RUN_MISMATCH")
            _require(a.get("model_state_sha256") == candidate["state"] and _sha(a.get("file_sha256")) and
                     a["file_sha256"] == i["checkpoint"], "ARTIFACT_STATE_OR_FILE_SHA_MISMATCH")
            aid = a.get("identity", {})
            ident = candidate["record"]["source"]["extra"]["identity"]
            _require(aid == ident, "ARTIFACT_EVALUATION_IDENTITY_MISMATCH")
            _require(root is not None, "ARTIFACT_FILE_NOT_LOCALLY_VERIFIED")
            raw = Path(a["checkpoint_path"])
            base = Path(root).resolve()
            # Docker /workspace prefixes may differ. Only remap the known lane suffix.
            marker = list(raw.parts).index(base.name)
            suffix = raw.parts[marker + 1:]
            path = (base.joinpath(*suffix)).resolve()
            _require(base in path.parents and path.is_file(), "ARTIFACT_FILE_MISSING_OR_OUTSIDE_LANE")
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            _require(h.hexdigest() == a["file_sha256"], "ARTIFACT_FILE_DIGEST_MISMATCH")
            _require(not evidence["weights_saved"] or evidence["file_sha256"] == a["file_sha256"], "CONFLICTING_ARTIFACT_FILE")
            qualitative = {}
            declared_files = a.get("qualitative_files", {})
            _require(isinstance(declared_files, dict), "INVALID_QUALITATIVE_MANIFEST")
            for name, expected_sha in declared_files.items():
                target = (path.parent / "qualitative" / name).resolve()
                if (path.parent not in target.parents or not _sha(expected_sha) or not target.is_file()):
                    warnings.append({"source_result_id": record["record_id"], "reason": "QUALITATIVE_FILE_NOT_VERIFIED", "file": name})
                    continue
                file_hash = hashlib.sha256()
                with target.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        file_hash.update(block)
                if file_hash.hexdigest() == expected_sha:
                    qualitative[name] = expected_sha
                else:
                    warnings.append({"source_result_id": record["record_id"], "reason": "QUALITATIVE_FILE_SHA_MISMATCH", "file": name})
            evidence = {**evidence, "weights_saved": True, "checkpoint_path": str(path), "file_sha256": a["file_sha256"],
                        "verification": "SAME_STATE_FILE_SHA256_VERIFIED", "artifact_source_id": record["record_id"],
                        "qualitative_files": qualitative}
        except (ValueError, KeyError, OSError) as exc:
            warnings.append({"source_result_id": record["record_id"], "reason": str(exc)})
    return evidence, warnings


def _display(candidate, mode, candidate_count, pass_count, last, artifacts, root):
    record = candidate["record"]
    row = list(record["row"])
    cfg = record["source"]["extra"]["configuration"]
    b = cfg["baseline_resolved"]
    n = lambda v: format(v, ".6g")
    horizon = cfg["planned_horizon"]
    row[INDEX["Experiment"]] = (f"{cfg['case']} · abs={n(cfg['lambda_abs'])} ε={n(cfg['lambda_epsilon'])} · {cfg['replica']}"
        f" | U/A={n(cfg['lr_U'])}/{n(cfg['lr_A'])} · {horizon//1000}K"
        + (f" · {cfg['optimizer_case']}" if cfg.get("optimizer_case") else ""))
    row[INDEX["Selection@Step"]] = ("충족" if mode == "PASS_ALL_STEPS" else "대표(미달)") + f" · {candidate['step']:,}"
    row[INDEX["Selector"]] = mode
    view_id = "VIEW-" + digest({"policy": POLICY, "run_key": candidate["key"], "protocol": candidate["protocol"],
                               "slot": candidate["step"] if mode == "PASS_ALL_STEPS" else "FALLBACK"})
    row[INDEX["Result_ID"]] = view_id
    row[INDEX["Readback"]] = "VERIFIED"
    artifact, warnings = _artifact_evidence(candidate, artifacts, root)
    row[INDEX["Checkpoint_SHA"]] = artifact["file_sha256"] or ""
    _, _, _, h, e = LANES[candidate["key"][2]]
    note = json.loads(row[INDEX["Original description / notes"]])
    note["publication"] = {"policy": POLICY, "selection": mode, "hqnr_min": h, "ergas_max": e,
        "candidate_count": candidate_count, "passed_count": pass_count, "source_result_id": record["record_id"],
        "source_selector": record["identity"]["selector"], "evaluation_step": candidate["step"],
        "evaluation_evidence": candidate["evaluation_evidence"], "resolved_config_sha256": candidate["config_sha"],
        "artifact": artifact, "artifact_warnings": warnings, "last_observed_at": last["created_at"],
        "last_logged_status": last["row"][INDEX["Status"]], "live_process_verified": False,
        "epsilon_label_means": "lambda_epsilon loss coefficient, not synthetic shift radius",
        "planned_horizon_means": "planned updates, not selected step or completed updates",
        "no_qualitative_files": not bool(artifact["qualitative_files"])}
    row[INDEX["Original description / notes"]] = canonical_json(note)
    _require(len(row[INDEX["Original description / notes"]]) < 45000, "VIEW_NOTE_TOO_LARGE")
    short = {"architecture": b.get("teacher_architecture"), "aligner": b.get("aligner_architecture"),
        "edge_match": b.get("edge_match"), "optimizer": b.get("optimizer"), "betas": b.get("optimizer_betas"),
        "optimizer_eps": b.get("optimizer_eps"), "weight_decay": cfg.get("weight_decay"),
        "optimizer_profile": cfg.get("optimizer_case"), "scheduler": b.get("scheduler_family"),
        "scheduler_indexing": b.get("scheduler_indexing"), "warmup_updates": b.get("warmup_updates0"),
        "min_lr": b.get("min_lr"), "effective_batch_RR": b.get("effective_batch_RR"),
        "microbatch_RR": b.get("microbatch_RR"), "accumulation_RR": b.get("accumulation_RR"),
        "precision": b.get("precision"), "amp": b.get("amp"), "tf32": b.get("tf32"),
        "lambda_abs": cfg["lambda_abs"], "lambda_epsilon": cfg["lambda_epsilon"],
        "epsilon_radius_rr": cfg.get("epsilon_radius_rr"), "planned_horizon": horizon,
        "U_peak_lr": cfg["lr_U"], "A_peak_lr": cfg["lr_A"], "full_resolved_configuration": cfg,
        **note["publication"], "model_state_sha256": candidate["state"], "evaluator_id": candidate["protocol"]}
    return row, canonical_json(short)


def project(records, server, evidence_root=None):
    """Select summaries from complete raw history; reports conflicts fail closed."""
    destination(server)
    groups, excluded, conflicts, all_runs, latest, artifacts = defaultdict(list), [], [], set(), {}, defaultdict(list)
    unique = {}
    for record in records:
        try:
            validate_envelope(record)
            _require(record["identity"]["server"] == server, "FOREIGN_SERVER")
            key = run_key(record)
            all_runs.add(key)
            if key not in latest or (record["created_at"], record["record_id"]) > (latest[key]["created_at"], latest[key]["record_id"]):
                latest[key] = record
            rid = record["record_id"]
            if rid in unique:
                _require(unique[rid] == digest(record), "CONFLICTING_SOURCE_ID")
                continue
            unique[rid] = digest(record)
            if record["record_type"].startswith("QUAL_ARTIFACT"):
                artifacts[key].append(record)
            if record["record_type"] != "EVAL_SUMMARY":
                excluded.append({"source_result_id": rid, "reason": "NOT_EVAL_SUMMARY", "scope": record["record_type"]})
                continue
            groups[key].append(validate_candidate(record, server=server, evidence_root=evidence_root))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            excluded.append({"source_result_id": record.get("record_id"), "reason": str(exc)})
            if any(word in str(exc) for word in ("MISMATCH", "CONFLICT", "MIXED", "ESCAPE")):
                conflicts.append({"source_result_id": record.get("record_id"), "reason": str(exc)})
    rows, notes, selected, reports = [], [], [], []
    for key in sorted(all_runs):
        candidates = groups[key]
        signatures = {(c["config_sha"], c["protocol"], c["inference_policy"]) for c in candidates}
        bad = len(signatures) > 1
        if bad:
            conflicts.append({"run_key": key, "reason": "CONFIG_OR_PROTOCOL_CHANGED_WITHIN_RUN"})
        steps = defaultdict(list)
        for c in candidates:
            steps[c["step"]].append(c)
        valid = []
        state_metrics = defaultdict(set)
        for c in candidates:
            state_metrics[c["state"]].add(digest(c["metrics"]))
        if any(len(values) > 1 for values in state_metrics.values()):
            conflicts.append({"run_key": key, "reason": "SAME_MODEL_STATE_DIFFERENT_METRICS"})
            bad = True
        for step, same_step in sorted(steps.items()):
            if len({(c["state"], digest(c["metrics"])) for c in same_step}) > 1:
                conflicts.append({"run_key": key, "step": step, "reason": "CONFLICTING_STATE_OR_METRICS"})
                bad = True
            valid.append(min(same_step, key=lambda c: c["source_result_id"]))
        _, _, _, h, e = LANES[server]
        passed = [c for c in valid if c["hqnr"] >= h and c["ergas"] <= e]
        mode = "PASS_ALL_STEPS" if passed else "BEST_HQNR_FALLBACK"
        kept = [] if bad else passed if passed else sorted(valid, key=lambda c: (-c["hqnr"], c["ergas"], c["step"], c["source_result_id"]))[:1]
        reports.append({"run_key": key, "candidate_count": len(valid), "raw_summary_count": len(candidates),
                        "pass_count": len(passed), "kept_count": len(kept), "selection": mode if kept else "NO_VALID_EVALUATION",
                        "last_observed_at": latest[key]["created_at"], "last_logged_status": latest[key]["row"][INDEX["Status"]],
                        "live_process_verified": False})
        for c in sorted(kept, key=lambda c: c["step"]):
            row, note = _display(c, mode, len(valid), len(passed), latest[key], artifacts[key], evidence_root)
            rows.append(row); notes.append(note)
            selected.append({k: c[k] for k in ("key", "step", "hqnr", "ergas", "state", "protocol", "source_result_id")})
    return {"policy": POLICY, "destination": destination(server), "rows": rows, "notes": notes, "selected": selected,
            "runs": reports, "excluded": excluded, "conflicts": conflicts, "owned_run_keys": sorted(all_runs),
            "view_hash": digest({"rows": rows, "notes": notes}), "source_digest": digest(sorted(unique.items()))}


def project_campaign(root, server):
    from .source import scan_campaign
    scan = scan_campaign(root)
    result = project(scan["records"], server, evidence_root=root)
    result.update({k: v for k, v in scan.items() if k != "records"})
    result["conflicts"].extend(scan["errors"])
    result["owned_run_keys"] = sorted(set(map(tuple, result["owned_run_keys"])) | set(map(tuple, scan["owned_run_keys"])))
    return result
