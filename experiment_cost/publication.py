"""Cost-enriched qualified result publication without changing live trainer code.

Architecture measurements are sidecars: they never change scientific selection,
raw result envelopes, model states, metric values, or historical campaign rows.
No Google clients or Torch modules are imported until explicitly needed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from directshift48 import publication as base

ROOT = Path(__file__).resolve().parents[1]
INDEX = base.INDEX
canonical_json = base.canonical_json
digest = base.digest
destination = base.destination
_number = base._number
_integer = base._integer

COST_COLUMNS = ("Params(M)", "FLOPs(G)", "Infer(ms)", "Mem(MB)", "Cost_scope")
COST_POLICY = "ARCHITECTURE_COST_SIDECAR_V1"


def lookup_profile(config, *, server, cost_root):
    """Load a profile only through the integrity/config-validating store."""
    from .store import find_profile
    return find_profile(config, server=server, cost_root=cost_root)


def _validated_fields(measurement):
    if measurement.get("schema_version") != "EXPERIMENT_COST_V1" or measurement.get("status") != "COMPLETE":
        raise ValueError("COST_PROFILE_NOT_COMPLETE")
    split = measurement.get("default_sheet_split")
    if split not in ("rr", "fr"):
        raise ValueError("COST_PROFILE_DEFAULT_DOMAIN_MISSING")
    observations = measurement.get("observations", {})
    for domain in ("rr", "fr"):
        fields = observations.get(domain, {}).get("sheet_fields", {})
        for column in COST_COLUMNS[:-1]:
            if not _number(fields.get(column)) or fields[column] <= 0:
                raise ValueError("COST_PROFILE_NONPOSITIVE_OR_MISSING:" + domain + ":" + column)
        if not isinstance(fields.get("Cost_scope"), str) or not fields["Cost_scope"].strip():
            raise ValueError("COST_PROFILE_SCOPE_MISSING:" + domain)
    return split, {column: observations[split]["sheet_fields"][column] for column in COST_COLUMNS}


def enriched_projection(root, server, *, cost_root=None, history_roots=None, require_cost=True):
    """Preserve complete multi-campaign selection and attach validated costs.

    A missing or malformed sidecar is explicit. By default it blocks this
    cost-aware publisher instead of erasing previously published costs. Local
    audits may set require_cost=False to inspect metric eligibility separately.
    """
    project = deepcopy(base.combined_projection(root, server, history_roots=history_roots))
    cost_root = Path(cost_root).resolve() if cost_root is not None else ROOT / "work_dir/experiment_cost" / server
    records, errors, missing = [], [], []
    for index, row in enumerate(project["rows"]):
        rid = row[INDEX["Result_ID"]]
        try:
            note = json.loads(row[INDEX["Original description / notes"]])
            short_note = json.loads(project["notes"][index])
            config = note.get("extra_parameters", {}).get("configuration")
            if not isinstance(config, dict) or config.get("server") != server:
                raise ValueError("COST_ROW_RESOLVED_CONFIG_OR_SERVER_MISMATCH")
            envelope = lookup_profile(config, server=server, cost_root=cost_root)
            if envelope is None:
                missing.append({"result_id": rid, "run_id": config["run_id"],
                                "reason": "NO_VERIFIED_ARCHITECTURE_PROFILE"})
                continue
            measurement = envelope["measurement"]
            split, fields = _validated_fields(measurement)
            for key in ("profile_sha256", "architecture_key"):
                if not base._sha(envelope.get(key)):
                    raise ValueError("COST_PROFILE_INVALID_DIGEST:" + key)
            for column, value in fields.items():
                row[INDEX[column]] = value
            attachment = {
                "policy": COST_POLICY,
                "profile_sha256": envelope["profile_sha256"],
                "profile_path": str(envelope["path"]),
                "architecture_key": envelope["architecture_key"],
                "server": server,
                "resolved_config_sha256": base.scientific_digest(config),
                "default_sheet_split": split,
                "sheet_fields": fields,
                "all_domains": {domain: measurement["observations"][domain]["sheet_fields"]
                                for domain in ("rr", "fr")},
                "weight_basis": measurement.get("weight_basis"),
                "measurement_scope": "ARCHITECTURE_MATCHED_PROFILE_NOT_SELECTED_CHECKPOINT_BENCHMARK",
                "same_graph_not_same_weights": True,
            }
            # Exact log durations through the displayed step, not ETA or a
            # whole-run duration attached to an earlier selected checkpoint.
            from .timing import timing_for_config
            try:
                timing = timing_for_config(config, row[INDEX['Selected_step']], root,
                    history_roots=history_roots, cost_root=cost_root)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                timing = {'sheet_fields': {}, 'omitted': {'timing': str(exc)}}
            for column, value in timing['sheet_fields'].items():
                if column not in ('Train(h)', 'Eval(h)', 'Wall(h)') or not _number(value) or value < 0:
                    raise ValueError('INVALID_OBSERVED_TIMING:' + column)
                row[INDEX[column]] = value
            attachment['observed_timing'] = timing
            note["cost_measurement"] = attachment
            short_note["cost_measurement"] = attachment
            row[INDEX["Original description / notes"]] = canonical_json(note)
            project["notes"][index] = canonical_json(short_note)
            if max(len(row[INDEX["Original description / notes"]]), len(project["notes"][index])) >= 45000:
                raise ValueError("COST_NOTE_TOO_LARGE")
            records.append({"result_id": rid, **attachment})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append({"result_id": rid, "reason": "COST_BINDING_FAILED:" + str(exc)})
    project["cost"] = {"policy": COST_POLICY, "cost_root": str(cost_root),
                       "bound_count": len(records), "bindings": records,
                       "missing": missing, "errors": errors, "require_cost": require_cost}
    project["conflicts"].extend(errors)
    if require_cost:
        project["conflicts"].extend(missing)
    project["view_hash"] = digest({"rows": project["rows"], "notes": project["notes"]})
    project["source_digest"] = digest({"scientific_source": project["source_digest"],
                                     "cost_policy": COST_POLICY,
                                     "profiles": sorted({r["profile_sha256"] for r in records})})
    return project


def competing_publishers(server, *, proc_root="/proc", own_pid=None):
    """Extend the existing read-only guard to this new writer; never signal."""
    import os
    own_pid = os.getpid() if own_pid is None else own_pid
    found = {p["pid"]: p for p in base.competing_publishers(server, proc_root=proc_root, own_pid=own_pid)}
    for path in Path(proc_root).glob("[0-9]*/cmdline"):
        try:
            pid = int(path.parent.name)
            if pid == own_pid:
                continue
            argv = [v.decode("utf-8", "replace") for v in path.read_bytes().split(b"\0") if v]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        lane = any(argv[i:i+2] == ["--server", server] for i in range(len(argv)-1)) or "--server=" + server in argv
        module = next((argv[i+1] for i in range(len(argv)-1) if argv[i] == "-m"), None)
        if lane and module in ("experiment_cost", "experiment_cost.publication") and "publish" in argv and "--apply" in argv:
            found[pid] = {"pid": pid, "argv": argv, "module": module}
    return sorted(found.values(), key=lambda item: item["pid"])


def publication_revision():
    inherited = base.publication_revision()
    files = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(Path(__file__).parent.glob("*.py"))}
    body = {"files": files, "base_publication": inherited, "cost_policy": COST_POLICY}
    return {**body, "sha256": digest(body)}


def publish(root, server, *, credentials="gspread/account.json", apply=False,
            watch=False, state_dir=None, history_roots=None, poll_seconds=120,
            audit_seconds=600, max_polls=0, google_read_only=False,
            cost_root=None, require_cost=True):
    """One publisher owns the complete server tab; defaults to LOCAL dry-run.

    The operator must stop the old publisher in its owning PID namespace before
    --apply; this function never stops training or another process. A shared
    repository-work_dir lock coordinates new publishers across state directories
    and host/container mounts. Existing legacy publishers are additionally
    detected through visible /proc. No Google client is created for local runs.
    """
    import fcntl
    import os
    import sys
    import time
    from sheet_publish_v2.transport import GoogleSheets, TransportError, atomic_json, reconcile

    destination(server)
    if (watch and not (apply or google_read_only)) or (apply and google_read_only):
        raise ValueError("Watch requires apply or read-only; apply and read-only are exclusive")
    if (not _number(poll_seconds) or not _number(audit_seconds) or
            not 10 <= poll_seconds <= audit_seconds or not _integer(max_polls)):
        raise ValueError("Require finite 10 <= poll <= audit and nonnegative max-polls")
    root = Path(root).resolve(strict=True)
    state = Path(state_dir).resolve() if state_dir else root / "sheet_publish_cost"
    state.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / "work_dir/sheet_publication_locks" / (str(destination(server)["sheet_id"]) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        # Dry-run has no ownership side effects; concurrent dry-run is safe.
        if apply:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another qualified-view publisher owns this server tab") from exc
        revision = publication_revision()
        atomic_json(state / "process.json", {"pid": os.getpid(), "argv": sys.argv, "server": server,
            "root": str(root), "apply": apply, "revision": revision, "lock": str(lock_path)})
        adapter, polls, failures = None, 0, 0
        verified_hash, verified_time = None, 0.
        while True:
            try:
                project = enriched_projection(root, server, history_roots=history_roots,
                    cost_root=cost_root, require_cost=require_cost)
                project["publisher_revision"] = revision
                competitors = competing_publishers(server)
                project["competing_publishers"] = competitors
                atomic_json(state / "projection.json", project)
                if project["conflicts"]:
                    raise TransportError("Local source conflicts; inspect projection.json")
                if apply and competitors:
                    raise TransportError("Existing same-server publisher observed; operator-only cutover required")
                if apply or google_read_only:
                    if (apply and project["view_hash"] == verified_hash
                            and time.monotonic() - verified_time < audit_seconds):
                        report = {"action": "UNCHANGED_LOCAL_HASH", "new_row_count": len(project["rows"]),
                                  "view_hash": project["view_hash"], "fresh_readback": False}
                    else:
                        if adapter is None:
                            adapter = GoogleSheets(credentials)
                        report = reconcile(adapter, project, state, apply=apply)
                        if report["readback_verified"]:
                            verified_hash, verified_time = project["view_hash"], time.monotonic()
                else:
                    report = {"action": "LOCAL_DRY_RUN", "new_row_count": len(project["rows"]),
                              "view_hash": project["view_hash"], "summary_count": project["summary_count"],
                              "history_roots": project["history_roots"], "competing_publishers": competitors, "cost": project["cost"]}
                    atomic_json(state / "last_dry_run.json", report)
                atomic_json(state / "last_poll.json", report)
                print(canonical_json({"action": report["action"], "rows": report["new_row_count"],
                                      "view_hash": project["view_hash"], "state_dir": str(state)}), flush=True)
                failures = 0
            except Exception as exc:
                failures += 1
                error = {"error": type(exc).__name__ + ": " + str(exc), "at_unix": time.time(),
                         "consecutive_failures": failures, "training_untouched": True}
                atomic_json(state / "last_error.json", error)
                print(canonical_json(error), file=sys.stderr, flush=True)
                if not watch or failures >= 5:
                    return 1
            polls += 1
            if not watch or max_polls and polls >= max_polls:
                return 0
            time.sleep(min(60, 5 * 2 ** (failures - 1)) if failures else poll_seconds)
    finally:
        lock.close()

