"""One ownership-safe publisher for old campaigns and new donor/Stage 2 rows.

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
POLICY, LANES = legacy_direct.POLICY, legacy_direct.LANES
canonical_json, digest = R.canonical_json, R.digest
destination = legacy_direct.destination


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
    ident, extra = record["identity"], record["source"]["extra"]
    cfg, state = extra["configuration"], extra["identity"]
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
            row[INDEX["Experiment"]] = f"{cfg['case']} · {R.phase(cfg)} · {cfg.get('replica', 'R1')}"
            if cfg.get("loss_mask_policy") == R.LOSS_MASK_POLICY:
                row[INDEX["Experiment"]] += " · MASK2"
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


def _history_roots(server):
    from sheet_publish_v2.selection import CAMPAIGN_ID as edge_campaign
    dataset = LANES[server][0]
    candidates = [ROOT / "work_dir/edgeres48" / edge_campaign / server / dataset,
                  ROOT / "work_dir/directshift48" / legacy_direct.CAMPAIGN_ID / server / dataset]
    candidates.extend(ROOT / "work_dir/donor30" / campaign / server / dataset
                      for campaign in R.LEGACY_CAMPAIGNS)
    return [p for p in candidates if p.is_dir()]


def _history_project(root, server):
    from sheet_publish_v2.selection import project_campaign as edge_project
    if list((root / "outbox").glob(R.RECORD_PREFIX + "*.json")):
        return project_campaign(root, server)
    if list((root / "outbox").glob("DS3-*.json")):
        return legacy_direct.project_campaign(root, server)
    # Identify ledger-only sources too, not only their optional outbox replicas.
    ledger = root / "records.jsonl"
    if ledger.exists():
        with ledger.open() as stream:
            first = stream.readline()
        if first:
            prefix = json.loads(first).get("record_id", "")
            if prefix.startswith(R.RECORD_PREFIX):
                return project_campaign(root, server)
            if prefix.startswith("DS3-"):
                return legacy_direct.project_campaign(root, server)
    return edge_project(root, server)


def _deployment_cost_view(measurement, cfg):
    """Relabel a verified same-operator graph without editing its signed cache.

    The architecture key intentionally ignores the semantic T/S role. A cached
    W104 Teacher graph can therefore measure a W104 Student graph exactly, but
    its role label must not claim that Stage 2 deploys the frozen training T.
    Every numeric quantity and the source profile SHA stay unchanged.
    """
    result = deepcopy(measurement)
    source_scopes = {s: measurement["observations"][s]["sheet_fields"]["Cost_scope"]
                     for s in ("rr", "fr")}
    stage2 = cfg.get("phase", cfg.get("stage")) == "stage2"
    if stage2:
        for split in ("rr", "fr"):
            fields = result["observations"][split]["sheet_fields"]
            scope = fields["Cost_scope"].replace("A+warp+Teacher", "A+warp+Student")
            if "frozen training Teacher" not in scope:
                scope += "; frozen training Teacher and training integrity bookkeeping excluded"
            fields["Cost_scope"] = scope
    result["deployment_view_provenance"] = {
        "role": "Student" if stage2 else "Teacher",
        "original_profile_cost_scope": source_scopes,
        "original_profile_deployment_binding": measurement.get("deployment_binding"),
        "role_label_adapted": any(result["observations"][s]["sheet_fields"]["Cost_scope"] != source_scopes[s]
                                  for s in ("rr", "fr")),
        "numeric_measurements_unchanged": True, "signed_profile_not_modified": True,
        "equivalence_basis": "Same verified server-local architecture signature and native inference operators; semantic T/S role is not an extra network"}
    return result


def _attach_cost(projected, root, server, histories, cost_roots, require_cost):
    from experiment_cost.publication import lookup_profile, _validated_fields
    from experiment_cost.timing import timing_for_config
    roots = cost_roots or [ROOT / "work_dir/experiment_cost" / server]
    if isinstance(roots, (str, Path)):
        roots = [roots]
    bindings, missing = [], []
    for i, row in enumerate(projected["rows"]):
        try:
            note = json.loads(row[INDEX["Original description / notes"]])
            cfg = note["extra_parameters"]["configuration"]
            key_cfg = R.cost_config(cfg) if cfg.get("phase", cfg.get("stage")) in ("donor", "stage2") else cfg
            found = [p for p in (lookup_profile(key_cfg, server=server, cost_root=r) for r in roots) if p]
            _require(len({p["profile_sha256"] for p in found}) <= 1, "CONFLICTING_COST_PROFILE_ROOTS")
            if not found:
                missing.append({"run_id": cfg["run_id"], "reason": "NO_VERIFIED_ARCHITECTURE_PROFILE"})
                continue
            envelope = found[0]
            measured_view = _deployment_cost_view(envelope["measurement"], cfg)
            split, fields = _validated_fields(measured_view)
            for column, value in fields.items():
                row[INDEX[column]] = value
            timing = timing_for_config(cfg, row[INDEX["Selected_step"]], root,
                history_roots=histories, cost_root=roots[0])
            for column, value in timing["sheet_fields"].items():
                row[INDEX[column]] = value
            attachment = {"policy": "ARCHITECTURE_COST_SIDECAR_V1",
                "profile_sha256": envelope["profile_sha256"], "architecture_key": envelope["architecture_key"],
                "profile_path": str(envelope["path"]), "default_sheet_split": split,
                "sheet_fields": fields, "observed_timing": timing,
                "deployed_graph": "A+S (Teacher excluded)" if cfg.get("phase") == "stage2" else "A+T",
                "deployment_binding": envelope["measurement"].get("deployment_binding"),
                "deployment_view_provenance": measured_view["deployment_view_provenance"],
                "same_graph_not_same_weights": True,
                "all_domains": {s: measured_view["observations"][s]["sheet_fields"] for s in ("rr", "fr")}}
            note["cost_measurement"] = attachment
            short = json.loads(projected["notes"][i]); short["cost_measurement"] = attachment
            row[INDEX["Original description / notes"]] = canonical_json(note)
            projected["notes"][i] = canonical_json(short)
            _require(max(len(row[INDEX["Original description / notes"]]), len(projected["notes"][i])) < 45000,
                     "COST_NOTE_TOO_LARGE")
            bindings.append({"result_id": row[INDEX["Result_ID"]], **attachment})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            projected["conflicts"].append({"reason": "COST_BINDING_FAILED:" + str(exc),
                                           "result_id": row[INDEX["Result_ID"]]})
    if require_cost:
        projected["conflicts"].extend(missing)
    projected["cost"] = {"bound_count": len(bindings), "bindings": bindings, "missing": missing,
                         "require_cost": require_cost, "cost_roots": list(map(str, roots))}


def project(root, server, legacy_roots=None, cost_roots=None, *, require_cost=True):
    """Complete authenticated tab projection; missing remote-only history blocks at transport."""
    current = Path(root).resolve(strict=True)
    histories = [p for p in _history_roots(server) if p.resolve() != current] if legacy_roots is None else legacy_roots
    if isinstance(histories, (str, Path)):
        histories = [histories]
    histories = [Path(p).resolve(strict=True) for p in histories]
    _require(len(set(histories)) == len(histories) and current not in histories, "DUPLICATE_HISTORY_SOURCE")
    projects = [_history_project(p, server) for p in histories] + [project_campaign(current, server)]
    _require(all(p["destination"] == destination(server) for p in projects), "MIXED_DESTINATIONS")
    keys = [tuple(k) for p in projects for k in p["owned_run_keys"]]
    _require(len(set(keys)) == len(keys), "DUPLICATE_CAMPAIGN_RUN_SOURCE")
    result = {"policy": POLICY, "destination": destination(server), "owned_run_keys": sorted(keys),
        "history_roots": list(map(str, histories)), "current_root": str(current),
        "raw_count": sum(p.get("raw_count", 0) for p in projects),
        "summary_count": sum(p.get("summary_count", 0) for p in projects),
        "raw_ids": sorted({i for p in projects for i in p.get("raw_ids", [])}),
        **{key: [v for p in projects for v in p.get(key, [])] for key in
           ("rows", "notes", "runs", "selected", "excluded", "conflicts", "source_manifest")}}
    _require(len({r[INDEX["Result_ID"]] for r in result["rows"]}) == len(result["rows"]), "DUPLICATE_VIEW_ID")
    _attach_cost(result, current, server, histories, cost_roots, require_cost)
    result["source_digest"] = digest({"sources": [p["source_digest"] for p in projects],
        "costs": sorted({b["profile_sha256"] for b in result["cost"]["bindings"]})})
    result["view_hash"] = digest({"rows": result["rows"], "notes": result["notes"]})
    return result


def competing_publishers(server, *, proc_root="/proc", own_pid=None):
    from experiment_cost.publication import competing_publishers as existing
    import os
    own_pid = os.getpid() if own_pid is None else own_pid
    found = {p["pid"]: p for p in existing(server, proc_root=proc_root, own_pid=own_pid)}
    for path in Path(proc_root).glob("[0-9]*/cmdline"):
        try:
            pid = int(path.parent.name)
            argv = [v.decode("utf-8", "replace") for v in path.read_bytes().split(b"\0") if v]
        except (OSError, ValueError):
            continue
        lane = any(argv[i:i+2] == ["--server", server] for i in range(len(argv)-1)) or "--server=" + server in argv
        module = next((argv[i+1] for i in range(len(argv)-1) if argv[i] == "-m"), None)
        if pid != own_pid and lane and module in ("donor30", "donor30.publication") and "publish" in argv and "--apply" in argv:
            found[pid] = {"pid": pid, "argv": argv, "module": module}
    return sorted(found.values(), key=lambda p: p["pid"])


def _writer_identity(process, proc_root="/proc"):
    """Verify executable intent, repository and process birth before cutover."""
    folder = Path(proc_root) / str(process["pid"])
    argv = [v.decode("utf-8", "replace") for v in (folder / "cmdline").read_bytes().split(b"\0") if v]
    if argv != process["argv"] or (folder / "cwd").resolve() != ROOT.resolve():
        raise ValueError("Publisher repository/argv changed; no signals sent")
    birth = (folder / "stat").read_text().rsplit(")", 1)[1].split()[19]
    return {"pid": process["pid"], "argv": argv, "birth": birth}


def start_publisher(root, server, *, credentials="gspread/account.json", python_executable=None,
                    legacy_roots=None, cost_roots=None):
    """Explicit launch-only writer handoff; never invoke during a review/dry run.

    Authenticated destination inspection precedes stopping a functioning writer.
    Only already-identified same-repository publisher processes are signalled;
    no training process, container or GPU job is stopped by this helper.
    """
    import os
    import signal
    import subprocess
    import sys
    import time
    from sheet_publish_v2.transport import GoogleSheets, _validate_snapshot, validate_ownership
    root = Path(root).resolve(strict=True)
    value = project(root, server, legacy_roots, cost_roots, require_cost=True)
    if value["conflicts"]:
        raise ValueError("Source/cost conflicts; previous publisher untouched")
    credential_path = Path(credentials).resolve(strict=True)
    snapshot = GoogleSheets(str(credential_path)).snapshot(value["destination"])
    _validate_snapshot(snapshot, value)
    validate_ownership(snapshot["rows"], value)
    competitors = competing_publishers(server)
    # Verify all before signalling any: one foreign/ambiguous writer prevents
    # a partial cutover. Recheck birth immediately before each signal.
    identities = [_writer_identity(p) for p in competitors]
    for process, identity in zip(competitors, identities):
        try:
            if _writer_identity(process) != identity:
                raise ValueError("Publisher PID reused before cutover; trainer untouched")
            os.kill(process["pid"], signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError):
            continue
        limit = time.monotonic() + 10
        cmdline = Path("/proc") / str(process["pid"]) / "cmdline"
        while True:
            try:
                running = bool(cmdline.read_bytes())
            except FileNotFoundError:
                running = False
            if not running:
                break
            if time.monotonic() >= limit:
                raise TimeoutError("Known old publisher did not stop; no SIGKILL and trainer untouched")
            time.sleep(.1)
    state = root / "sheet_publish_donor30"
    state.mkdir(parents=True, exist_ok=True)
    command = [python_executable or sys.executable, "-u", "-m", "donor30", "publish",
        "--server", server, "--root", str(root), "--credentials", str(credential_path), "--apply", "--watch"]
    for flag, paths in (("--legacy-root", legacy_roots), ("--cost-root", cost_roots)):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        for path in paths or []:
            command += [flag, str(Path(path).resolve())]
    log_path = state / "publisher.log"
    with log_path.open("a") as log:
        started = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    if started.poll() is not None:
        raise RuntimeError("New publisher exited immediately; inspect " + str(log_path))
    return {"publisher_pid": started.pid, "log": str(log_path),
            "stopped_publisher_pids": [p["pid"] for p in identities], "training_untouched": True,
            "command": command, "cloud_readback_pending": True}


def publish(root, server, *, credentials="gspread/account.json", apply=False,
            watch=False, state_dir=None, legacy_roots=None, cost_roots=None,
            poll_seconds=120, audit_seconds=600, max_polls=0,
            google_read_only=False, require_cost=True):
    """No external writes by default. Competing writer checks never kill a process."""
    import fcntl
    import os
    import sys
    import time
    from sheet_publish_v2.transport import GoogleSheets, TransportError, reconcile
    if (watch and not (apply or google_read_only)) or (apply and google_read_only):
        raise ValueError("Watch requires apply/read-only; apply and read-only are exclusive")
    if not (10 <= poll_seconds <= audit_seconds) or type(max_polls) is not int or max_polls < 0:
        raise ValueError("Require 10 <= poll <= audit and nonnegative max-polls")
    root = Path(root).resolve(strict=True)
    state = Path(state_dir) if state_dir else root / "sheet_publish_donor30"
    state.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / "work_dir/sheet_publication_locks" / (str(destination(server)["sheet_id"]) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        if apply:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another publisher owns this destination") from exc
        revision = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path(__file__).parent.glob("*.py"))}
        R.atomic_json(state / "process.json", {"pid": os.getpid(), "argv": sys.argv,
            "server": server, "root": str(root), "apply": apply, "code_sha256": revision})
        adapter, polls, failures, verified_hash, verified_time = None, 0, 0, None, 0.
        while True:
            try:
                value = project(root, server, legacy_roots, cost_roots, require_cost=require_cost)
                competitors = competing_publishers(server)
                value["competing_publishers"] = competitors
                R.atomic_json(state / "projection.json", value)
                if value["conflicts"]:
                    raise TransportError("Local evidence/cost conflicts; inspect projection.json")
                if apply and competitors:
                    raise TransportError("Existing publisher observed; operator cutover required")
                if apply or google_read_only:
                    if apply and value["view_hash"] == verified_hash and time.monotonic() - verified_time < audit_seconds:
                        report = {"action": "UNCHANGED_LOCAL_HASH", "new_row_count": len(value["rows"]), "fresh_readback": False}
                    else:
                        if adapter is None:
                            adapter = GoogleSheets(credentials)
                        report = reconcile(adapter, value, state, apply=apply)
                        if report.get("readback_verified"):
                            verified_hash, verified_time = value["view_hash"], time.monotonic()
                else:
                    report = {"action": "LOCAL_DRY_RUN", "new_row_count": len(value["rows"]), "cost": value["cost"]}
                R.atomic_json(state / "last_poll.json", report)
                print(canonical_json({"action": report["action"], "rows": len(value["rows"])}), flush=True)
                failures = 0
            except Exception as exc:
                failures += 1
                report = {"error": type(exc).__name__ + ": " + str(exc), "consecutive_failures": failures,
                          "at": R.utc_now(), "training_untouched": True}
                R.atomic_json(state / "last_error.json", report)
                print(canonical_json(report), file=sys.stderr, flush=True)
                # A transient network outage or not-yet-ready cost sidecar must
                # not silently strand the remainder of a 30-hour campaign.
                # Every failure remains local and prevents cloud mutation.
                if not watch:
                    return 1
            polls += 1
            if not watch or max_polls and polls >= max_polls:
                return 1 if failures else 0
            delay = min(60, 5 * 2 ** min(4, max(0, failures - 1))) if failures else poll_seconds
            while delay > 0:
                time.sleep(min(delay, 30)); delay -= min(delay, 30)
