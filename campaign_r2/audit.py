"""CPU-only R2 evidence collection; never changes the numerical campaign.

Historical receipts are evidence, not permission to declare unfinished science
PASS. This module never infers models, contacts Sheets, credits time, or alters an
original artifact. Optional reports go only into priority_r2/.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import itertools
import subprocess
import tempfile

CAMPAIGN = "WV3_FH20R1_20260919_v1"
REVISION = "FH20R1_PRIORITY_METHOD_PRESERVE_R2_20260919"
GRID = tuple(range(1010, 50000, 1010)) + (50000,)
SELECTIONS = {"target_selection": "TARGET", "raw_max": "RAW_MAX",
              "exact50k": "EXACT50K", "rr_val_selected": "RR_VAL_SELECTED",
              "e_min_diag": "E_MIN_DIAG50"}
S73101 = "FH20R1_S1_S_PLH_W104_D122_WV3_TF1_TS71001_SS73101_BASE_FRESH50_v1"
TARGET_SHA = "3933c471c949962db96e799d7cf0ba8f06f2efa8eac17d795dda722f842d0d50"


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else ({} if default is None else default)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _json(path, doc):
    _atomic(path, json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _csv(path, rows):
    if not rows:
        return
    stream = io.StringIO()
    keys = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(stream, keys)
    writer.writeheader()
    writer.writerows(rows)
    _atomic(path, stream.getvalue())


def _inside(root, path):
    path = Path(path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected repository-relative artifact: " + str(path))
    result = (Path(root) / path).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact escapes repository")
    return result


def verify_source(root, source):
    """Verify original recorded files and HEAD, never bless a changed release."""
    errors = []
    if not source.get("files") or not source.get("git_release"):
        return {"status": "TO_VERIFY", "errors": ["Original execution source identity missing"]}
    for name, expected in source["files"].items():
        try:
            path = _inside(root, name)
            if not path.is_file() or sha(path) != expected:
                errors.append("source hash mismatch: " + name)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if object_sha(source["files"]) != source.get("content_sha256"):
        errors.append("recorded source file-map digest mismatch")
    proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False)
    current = proc.stdout.strip() if proc.returncode == 0 else None
    if current != source["git_release"]:
        errors.append("git HEAD differs from original execution release")
    return dict(status="VERIFIED" if not errors else "MISMATCH", errors=errors,
                recorded_git_release=source["git_release"], current_git_release=current,
                checked_file_count=len(source["files"]),
                runtime_library_parity="Original trainer retains its own full runtime guard; not re-probed here")


def ledger_audit(directory, state):
    """Read interval union without ingesting segments or resetting the clock."""
    budget = read(directory / "campaign_budget.json")
    path = directory / "active_intervals.jsonl"
    errors, intervals, seen, evidence_cache = [], [], {}, {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    def epoch(value):
        stamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Naive ledger timestamp")
        return stamp.timestamp()
    try:
        started = epoch(budget["started_at_utc"])
    except (KeyError, ValueError):
        started = None
        errors.append("Original authorized campaign start is unavailable")
    if budget.get("campaign_id") != CAMPAIGN or not budget.get("actual_start_authorized"):
        errors.append("Original FH20R1 campaign authorization is missing")
    for row in rows:
        try:
            start, end = epoch(row["start_utc"]), epoch(row["end_utc"])
            if row.get("campaign_id") != CAMPAIGN or row.get("server_id") != directory.name:
                raise ValueError("Foreign campaign/server interval")
            if started is None or start < started or end < start or not row.get("evidence"):
                raise ValueError("Invalid interval start/end/evidence")
            if abs(float(row["seconds"]) - (end - start)) > max(.01, (end - start) * 1e-5):
                raise ValueError("Interval seconds disagree with timestamps")
            if row["kind"] != "waiting":
                evidence = _inside(directory.parents[2], row["evidence"])
                if not evidence.is_file():
                    raise ValueError("Missing committed timing evidence: " + row["evidence"])
                if evidence.parent.name == "timing_evidence":
                    # Filename hashes the original JSON bytes. The ledger saves
                    # a sorted-key copy, so its byte digest need not equal that
                    # filename. Validate the committed segment geometry instead.
                    if evidence not in evidence_cache:
                        evidence_cache[evidence] = read(evidence)
                    proof = evidence_cache[evidence]
                    segment = next((s for s in proof.get("segments", [])
                        if row["id"] == str(row.get("run_id")) + ":" + str(s.get("id"))), None)
                    if (not proof.get("valid_fullstate") or proof.get("campaign_id") != CAMPAIGN
                            or proof.get("run_id") != row.get("run_id") or not segment
                            or any(segment.get(k) != row.get(k) for k in
                                   ("kind", "start_utc", "end_utc", "update_from", "update_to"))
                            or int(segment.get("update_to", 0)) > int(proof.get("committed_update", -1))):
                        raise ValueError("Ledger segment disagrees with committed evidence")
            if row["id"] in seen and seen[row["id"]] != row:
                raise ValueError("Mutated duplicate interval ID")
            seen[row["id"]] = row
            if row["kind"] in ("train", "eval", "calibration", "diagnostic"):
                intervals.append((start, end))
            elif row["kind"] != "waiting":
                raise ValueError("Unknown timing category")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    seconds = sum(end - start for start, end in merged)
    active = [name for name, item in state.get("blocks", {}).items() if item.get("status") != "DONE"]
    return dict(status="VERIFIED_EXISTING_LEDGER" if not errors else "MISMATCH", errors=errors,
                effective_seconds=seconds, effective_hours=seconds / 3600,
                remaining_minimum_seconds=max(0., 72000 - seconds), interval_count=len(rows),
                original_started_at_utc=budget.get("started_at_utc"),
                budget_sha256=sha(directory / "campaign_budget.json") if budget else None,
                ledger_sha256=sha(path) if path.exists() else None,
                minimum_satisfied=seconds >= 72000, unfinished_admitted_blocks=active,
                campaign_complete=False, core_completion="Controller must verify all registered CORE blocks",
                time_reset=False, newly_credited_seconds=0, accounting="union, never category sum or waiting")


def reconcile(root, server):
    root = Path(root)
    directory = root / "work_dir/_fh20r1" / server
    state = read(directory / "status.json")
    budget = read(directory / "campaign_budget.json")
    branch = read(directory / "branch_record.json")
    ready = read(directory / "readiness_report.json")
    errors = []
    source = verify_source(root, ready.get("consumer_source_identity", {}))
    if source["status"] != "VERIFIED":
        errors.extend(source["errors"])
    if not state or not budget:
        errors.append("Registered state/budget missing")
    try:
        from campaign_r2.plan import require_branch_record
        require_branch_record(server, branch)
    except ValueError as exc:
        errors.append(str(exc))
    # Older s2-s5 branch records have no lock flag. Their recorded branch is
    # still immutable and verified by the controller's before/after hash.
    if branch.get("locked_before_training") is False:
        errors.append("Branch explicitly records an unlocked pre-training decision")
    for run, expected in budget.get("config_hashes", {}).items():
        path = _inside(root, "config/" + run + ".yaml")
        if not path.exists() or sha(path) != expected:
            errors.append("Registered config file mismatch: " + run)
    runs = set(state.get("runs", {}))
    runs.update(p.name for p in (root / "work_dir").glob("FH20R1_" + server.upper() + "_*") if p.is_dir())
    observations, completed, pending, active = [], [], [], []
    for run in sorted(runs):
        wd = _inside(root, "work_dir/" + run)
        train = read(wd / "meta/training_status.json")
        start = read(wd / "meta/training_start_manifest.json")
        official = read(wd / "official/postrun_status.json")
        receipt = read(wd / "official/upload_receipt.json")
        saved = state.get("runs", {}).get(run, {})
        done = bool(official.get("official_complete") and official.get("actual_updates") == 50000)
        reused = bool(official.get("reused") and official.get("source_official_complete"))
        verified = bool(receipt.get("readback_verified") and receipt.get("run_id") == run
                        and receipt.get("campaign_id") == CAMPAIGN)
        if done or reused:
            completed.append(run)
            if not verified:
                pending.append(run)
        elif saved.get("training_started") or train:
            active.append(run)
        if saved.get("status") in ("DONE", "DONE_REUSED") and not (done or reused):
            errors.append("State DONE without local completed official artifact: " + run)
        if start and start.get("source_identity") != ready.get("consumer_source_identity"):
            errors.append("Run source identity differs from registered source: " + run)
        observations.append(dict(run_id=run, scheduler_status=saved.get("status"),
                                 actual_updates=train.get("actual_updates"),
                                 official_complete=done, reused=reused, upload_verified=verified,
                                 config_sha256=start.get("config_sha256"),
                                 training_status_updated_at=train.get("updated_at_utc")))
    time = ledger_audit(directory, state)
    errors.extend(time["errors"])
    return dict(status="VERIFIED_ARTIFACTS" if not errors else "HOLD_NEW_ADMISSION",
                errors=errors, server_id=server, branch=branch, source=source,
                active_or_unfinished_runs=active, completed_official_run_ids=completed,
                upload_pending_run_ids=pending, blocks=state.get("blocks", {}), runs=observations,
                campaign_time=time, process_observation="Artifact-only; controller/process-lock check still required",
                terminate_active_training=False,
                current_priority_receipt=read(directory / "priority_r2/applied_readback.json"))


def reuse_preflight(root, server, reconciliation):
    directory = Path(root) / "work_dir/_fh20r1" / server
    preflight = read(directory / "preflight_report.json")
    ready = read(directory / "readiness_report.json")
    bound = bool(preflight.get("preflight_pass") and preflight.get("complete")
                 and preflight.get("source_identity") == ready.get("consumer_source_identity")
                 and reconciliation["source"]["status"] == "VERIFIED")
    method_ok = bound and all(preflight.get(key, {}).get("passed")
                              for key in ("frontend", "coupled_init", "loss_routing"))
    frontend = preflight.get("frontend", {})
    sizes = sorted(row.get("size") for row in frontend.get("records", []))
    v01 = dict(status="REUSED_VERIFIED" if method_ok else "TO_VERIFY",
               receipt_path=str(directory / "preflight_report.json"),
               receipt_sha256=sha(directory / "preflight_report.json") if preflight else None,
               source_binding_verified=bound, loss_routing=preflight.get("loss_routing"),
               checks_rerun=False, scope="Original synthetic routing/init/frontend receipt, identical recorded source")
    v08 = dict(status="PARTIAL_REUSED" if bound and sizes == [64, 256, 512] else "TO_VERIFY",
               frontend_receipt=frontend, source_binding_verified=bound,
               canonical_data_note="Preflight reuse does not re-hash raw H5/LP bytes; original trainer/reference guards remain active",
               missing_checks=["Current raw H5 and LP content hash revalidation", "Explicit signed range/native M-frame report"])
    return v01, v08


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Missing or non-finite official metric")
    return value


def selection_audit(root, run, *, verify_checkpoint_bytes=True, expected_campaign=CAMPAIGN):
    """Recompute selectors from stored official50 records, never rerun inference."""
    wd = _inside(root, "work_dir/" + run)
    grid = read(wd / "official/raw_grid.json")
    records = grid.get("records", [])
    if not grid.get("complete") or grid.get("run_id") != run or grid.get("campaign_id") != expected_campaign:
        raise ValueError("Complete original campaign raw grid is required")
    by_step = {r["update"]: r for r in records}
    if sorted(by_step) != list(GRID) or len(records) != 50 or grid.get("expected_steps") != list(GRID):
        raise ValueError("Not exactly the fixed 50-candidate grid")
    for step, row in by_step.items():
        identity = row["checkpoint_identity"]
        folder = wd / "candidates" / str(step)
        if read(wd / "official" / ("candidate_" + str(step) + ".json")) != row:
            raise ValueError("Raw grid differs from retained per-candidate official evidence")
        if read(folder / "identity.json") != identity or identity.get("update") != step:
            raise ValueError("Checkpoint/grid identity mismatch")
        if any(identity.get(k) != grid.get(k) for k in ("source_identity", "config_sha256", "data_sha256")):
            raise ValueError("Mixed source/config/data in checkpoint grid")
        for group, keys in (("rr", ("ergas", "scc", "psnr", "sam", "q8", "ssim")),
                            ("fr", ("hqnr", "d_lambda", "d_s"))):
            for key in keys:
                _number(row[group][key])
            if row[group].get("n_scenes") != 20 or len(row[group].get("per_scene", [])) != 20:
                raise ValueError("Official20 per-scene evidence is missing")
        if not row["rr"].get("official_complete"):
            raise ValueError("RR evaluation is not official-complete")
        _number(row["val_ergas"])
    ordered = [by_step[step] for step in GRID]
    eligible = [r for r in ordered if r["fr"]["hqnr"] >= .9585]
    chosen = {"target_selection": min(eligible, key=lambda r: (r["rr"]["ergas"], -r["rr"]["scc"],
                    -r["rr"]["psnr"], r["update"])) if eligible else None,
              "raw_max": max(ordered, key=lambda r: (r["fr"]["hqnr"], -r["update"])),
              "exact50k": by_step[50000],
              "rr_val_selected": min(ordered, key=lambda r: (r["val_ergas"], r["update"])),
              "e_min_diag": min(ordered, key=lambda r: (r["rr"]["ergas"], r["update"]))}
    selections, hashed = {}, set()
    for filename, row in chosen.items():
        path = wd / "official" / (filename + ".json")
        saved = read(path)
        if (saved.get("selection_id") != SELECTIONS[filename] or saved.get("run_id") != run
                or saved.get("campaign_id") != expected_campaign):
            raise ValueError("Missing or cross-run selection: " + filename)
        if any(saved.get(k) != grid.get(k) for k in ("config_sha256", "data_sha256", "source_identity")):
            raise ValueError("Selection provenance mismatch: " + filename)
        if ((expected_campaign == CAMPAIGN and not saved.get("normal_same_step_A_U"))
                or not saved.get("official_complete")):
            raise ValueError("Not a completed normal same-step A/U selection")
        if row is None:
            if saved.get("target_status") != "no_eligible" or any(saved.get(k) not in (None, {})
                    for k in ("step", "checkpoint_sha256", "rr", "fr")):
                raise ValueError("No-eligible TARGET must be blank, never proxy-filled")
            selections[filename] = dict(selection_id="TARGET", target_status="no_eligible", step=None,
                                        checkpoint_sha256=None, rr=None, fr=None)
            continue
        if any(saved.get(k) != row.get(k) for k in ("rr", "fr", "val_ergas", "checkpoint_identity")):
            raise ValueError("Selection metrics mixed with another checkpoint: " + filename)
        step, checkpoint_sha = row["update"], row["checkpoint_identity"]["model_sha256"]
        if saved.get("step") != step or saved.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Selection order or checkpoint SHA mismatch: " + filename)
        if verify_checkpoint_bytes and step not in hashed:
            if sha(wd / "candidates" / str(step) / "model.safetensors") != checkpoint_sha:
                raise ValueError("Selected checkpoint bytes have changed")
            hashed.add(step)
        selections[filename] = dict(selection_id=SELECTIONS[filename], step=step,
            checkpoint_sha256=checkpoint_sha, rr=row["rr"], fr=row["fr"],
            val_ergas=row["val_ergas"], source_json_sha256=sha(path),
            test_aware=filename in ("target_selection", "raw_max", "e_min_diag"))
    target = read(wd / "official/target_selection.json")
    if target.get("n_eligible") != len(eligible) or target.get("n_evaluated") != 50:
        raise ValueError("TARGET eligible count disagrees with original fixed grid")
    intervals, curve = [], []
    for index, row in enumerate(ordered):
        yes = row["fr"]["hqnr"] >= .9585
        if yes:
            if not intervals or intervals[-1]["end_index"] != index - 1:
                intervals.append(dict(start_step=row["update"], end_step=row["update"], length=0, end_index=index))
            intervals[-1].update(end_step=row["update"], end_index=index, length=intervals[-1]["length"] + 1)
        curve.append(dict(step=row["update"], hqnr=row["fr"]["hqnr"], ergas=row["rr"]["ergas"],
                          d_lambda=row["fr"]["d_lambda"], d_s=row["fr"]["d_s"], eligible=yes,
                          checkpoint_sha256=row["checkpoint_identity"]["model_sha256"]))
    for interval in intervals:
        interval.pop("end_index")
    snapshot = None
    if run == S73101:
        selected = selections["target_selection"]
        snapshot = dict(target_step_matches=selected["step"] == 38380,
                        target_sha_matches=selected["checkpoint_sha256"] == TARGET_SHA,
                        eligible_count_matches=len(eligible) == 19)
        if not all(snapshot.values()):
            raise ValueError("Immutable S73101 target differs from documented step/SHA/eligible snapshot")
    deltas = {}
    for name in ("target_selection", "exact50k"):
        first, second = chosen["raw_max"], chosen[name]
        if second is None:
            continue
        deltas[name + "_minus_raw_max"] = dict(
            from_step=first["update"], to_step=second["update"],
            fr=[{k: b[k] - a[k] for k in ("hqnr", "d_lambda", "d_s")}
                for a, b in zip(first["fr"]["per_scene"], second["fr"]["per_scene"])],
            rr_band_relative_mse=[
                [b - a for a, b in zip(x.get("band_relative_mse", []), y.get("band_relative_mse", []))]
                for x, y in zip(first["rr"]["per_scene"], second["rr"]["per_scene"])])
    return dict(status="VERIFIED_STORED_SELECTIONS", run_id=run,
                raw_grid_sha256=sha(wd / "official/raw_grid.json"), n_eligible=len(eligible),
                eligible_intervals=intervals, longest_eligible_interval=max((x["length"] for x in intervals), default=0),
                first_eligible_step=eligible[0]["update"] if eligible else None,
                last_eligible_step=eligible[-1]["update"] if eligible else None,
                late_eligible=any(r["update"] >= 40000 for r in eligible),
                curve=curve, selections=selections, documented_s73101_snapshot=snapshot,
                selected_checkpoint_bytes_verified=verify_checkpoint_bytes,
                inference_rerun=False, independent_test=False,
                per_scene_selection_deltas=deltas,
                per_scene_candidates=[dict(step=r["update"], fr=r["fr"]["per_scene"],
                    rr=r["rr"]["per_scene"]) for r in ordered])


def branch_review(root, server):
    directory = Path(root) / "work_dir/_fh20r1" / server
    branch = read(directory / "branch_record.json")
    path = directory / "diagnostics/report.json"
    report = read(path)
    if server != "s1":
        return dict(status="NOT_APPLICABLE", branch=branch)
    hash_ok = bool(report and branch.get("report_sha256") == sha(path))
    evidence = bool(report.get("decision_evidence_complete"))
    matrices = []
    for file in sorted((directory / "diagnostics/au_cross").glob("*/matrix.json")):
        item = read(file)
        matrices.append(dict(path=str(file), sha256=sha(file),
                             source_run=item.get("source_run"), complete=item.get("complete"),
                             diagonal_verified=item.get("diagonal_verified"),
                             records=item.get("records")))
    classification = "INCONCLUSIVE"
    if hash_ok and evidence and len(report.get("comparisons", [])) == 4:
        classification = "NO_SUPPORT" if report.get("support_count") == 0 else "MIXED_OR_SUPPORT"
    return dict(status="REVIEWED_STORED_EVIDENCE" if hash_ok else "TO_VERIFY",
                branch=branch, branch_report_hash_verified=hash_ok,
                classification=classification, decision_evidence_complete=evidence,
                comparisons=report.get("comparisons", []), matrices=matrices,
                unavailable=report.get("unavailable", []),
                caveat="NO_SUPPORT for the predeclared contrasts is not proof that A is uninvolved; branch unchanged",
                dependency_bytes_revalidated=False)


def block_summaries(root, server, audits, branch):
    # Registry imports are metadata-only. Historical baselines are not silently
    # manufactured into new independent Student seeds.
    from fh20r1.plan import blocks_for, case_for
    rows, groups = [], []
    for block in blocks_for(server):
        run_ids = block.order_for(branch)
        completed = [run for run in run_ids if run in audits]
        for run in run_ids:
            case = case_for(run)
            report = audits.get(run)
            row = dict(server=server, block_id=block.block_id, tier=block.tier,
                       run_id=run, teacher_alias=case.teacher_alias, teacher_seed=case.teacher_seed,
                       student_seed=case.student_seed, input_layout=case.input_layout,
                       width=case.width, depth="".join(map(str, case.depth)),
                       completed=bool(report), block_completed=len(completed) == len(run_ids),
                       expected_runs=len(run_ids), completed_runs=len(completed),
                       target_test_aware=True, independent_unit="one Student run, not 50 checkpoints")
            if report:
                row.update(n_eligible=report["n_eligible"], late_eligible=report["late_eligible"])
                for name, selected in report["selections"].items():
                    row[name + "_step"] = selected.get("step")
                    row[name + "_sha256"] = selected.get("checkpoint_sha256")
                    for group, keys in (("rr", ("ergas", "scc")), ("fr", ("hqnr", "d_lambda", "d_s"))):
                        for key in keys:
                            row[name + "_" + key] = (selected.get(group) or {}).get(key)
                row["target_eligible"] = bool(report["n_eligible"])
                row["joint_pass"] = bool(report["n_eligible"] and row["target_selection_ergas"] < 2.040)
            rows.append(row)
        same = [row for row in rows if row["block_id"] == block.block_id]
        eligible_e = [row["target_selection_ergas"] for row in same if row.get("target_eligible")]
        differences = []
        for left, right in itertools.combinations(same, 2):
            matching = ("server", "teacher_alias", "teacher_seed", "student_seed")
            if any(left[k] != right[k] for k in matching) or not (left["completed"] and right["completed"]):
                continue
            varied = [key for key in ("input_layout", "width", "depth") if left[key] != right[key]]
            if len(varied) != 1:
                continue
            diff = dict(from_run=left["run_id"], to_run=right["run_id"], changed_axis=varied[0],
                        interpretation="to minus from, same server/Teacher/Student seed")
            for name in SELECTIONS:
                for metric in ("ergas", "hqnr", "d_lambda", "d_s"):
                    key = name + "_" + metric
                    diff[key] = right[key] - left[key] if right.get(key) is not None and left.get(key) is not None else None
            differences.append(diff)
        interactions = []
        for width in sorted({r["width"] for r in same}):
            quartet = {(r["input_layout"], r["depth"]): r for r in same if r["width"] == width and r["completed"]}
            keys = (("PLH", "122"), ("PL", "122"), ("PLH", "121"), ("PL", "121"))
            if all(key in quartet for key in keys):
                four = [quartet[key] for key in keys]
                if len({(r["teacher_alias"], r["teacher_seed"], r["student_seed"]) for r in four}) != 1:
                    continue
                item = dict(width=width, run_ids=[r["run_id"] for r in four])
                for name in SELECTIONS:
                    values = [r.get(name + "_ergas") for r in four]
                    item[name + "_interaction_ergas"] = values[0] - values[1] - values[2] + values[3] if all(v is not None for v in values) else None
                interactions.append(item)
        groups.append(dict(block_id=block.block_id, tier=block.tier, expected_run_count=len(run_ids),
                           completed_run_count=len(completed), eligible_run_count=len(eligible_e),
                           eligible_rate_over_completed=len(eligible_e) / len(completed) if completed else None,
                           target_ergas_conditional_median=statistics.median(eligible_e) if eligible_e else None,
                           target_ergas_conditional_range=[min(eligible_e), max(eligible_e)] if eligible_e else None,
                           target_ergas_individual_values=eligible_e,
                           joint_pass_count=sum(bool(row.get("joint_pass")) for row in same),
                           same_teacher_same_seed_differences=differences, quartet_interactions=interactions,
                           block_complete=len(completed) == len(run_ids),
                           shared_baselines="Not duplicated as independent seeds; bridge comparisons require explicit baseline links"))
    return rows, groups


def collect(root, server, *, write_reports=False, verify_checkpoint_bytes=True):
    if server not in ("s1", "s2", "s3", "s4", "s5"):
        raise ValueError("Unknown FH20R1 server")
    root = Path(root)
    directory = root / "work_dir/_fh20r1" / server
    output = directory / "priority_r2"
    reconciliation = reconcile(root, server)
    method, frontend = reuse_preflight(root, server, reconciliation)
    audits, audit_errors = {}, {}
    for run in reconciliation["completed_official_run_ids"]:
        try:
            from fh20r1.upload import row_values
            # Reuse the production read-only verifier, including resolved recipe,
            # profile, reference bridge, data manifest and selected weight bytes.
            row_values(run, root=root)
            reused = read(root / "work_dir" / run / "meta/reuse_reference.json")
            measured = reused.get("source_run_id", run)
            campaign = "WV3_FH12_TSCRATCH_20260918_v1" if measured.startswith("FH12_") else CAMPAIGN
            audited = selection_audit(root, measured, verify_checkpoint_bytes=verify_checkpoint_bytes,
                                      expected_campaign=campaign)
            if reused:
                audited.update(run_id=run, measured_source_run_id=measured, reused=True,
                               independent_new_measurement=False, credited_new_seconds=0)
            audits[run] = audited
        except (OSError, KeyError, ValueError, TypeError) as exc:
            audit_errors[run] = str(exc)
    recorded_branch = reconciliation["branch"]
    condition = recorded_branch.get("condition", recorded_branch.get("branch", "PRIMARY"))
    rows, groups = block_summaries(root, server, audits, condition)
    v02 = audits.get(S73101, dict(status="TO_VERIFY" if server == "s1" else "NOT_APPLICABLE"))
    branch = branch_review(root, server)
    if audit_errors:
        reconciliation["status"] = "HOLD_NEW_ADMISSION"
        reconciliation["errors"].extend(run + ": " + message for run, message in audit_errors.items())
        reconciliation["completed_official_run_ids"] = list(audits)
    results = {"V00": reconciliation, "V01": method, "V02": v02, "V04": branch, "V08": frontend,
               "V12": dict(status="LOCAL_VERIFIED_LIVE_SHEET_TO_VERIFY" if not audit_errors else "MISMATCH",
                   selection_errors=audit_errors, verified_runs=list(audits),
                   primary_comparison="TARGET", legacy_main_columns="Unchanged RAW_MAX",
                   live_sheet_readback="Not performed by this offline collector; original upload receipts are retained",
                   upload_pending_run_ids=reconciliation["upload_pending_run_ids"]),
               "V13": dict(status="PARTIAL_REPORT", block_aggregates=groups,
                   missing_checks=["Explicit historical baseline pairing and tensor/batch identity verification",
                                   "Historical width-bridge baseline links outside current block"]),
               "V14": reconciliation["campaign_time"]}
    for key, reason in {"V03": "Requires same-checkpoint cross-environment inference; no comparison run here",
                        "V05": "Needs cache distribution/radius/index audit; reference not recalibrated",
                        "V06": "Existing cross diagnostics cataloged in V04; new S73101 cross not run",
                        "V07": "Donor full SHA/step/view must be checked on s2; no donor replaced",
                        "V09": "z-score/GN/GAP C00-C05 diagnostics not implemented by this overlay",
                        "V10": "Loss activity/gradient distributions not computed by this collector",
                        "V11": "Paired common tensors/batches/cost require separate evidence"}.items():
        results[key] = dict(status="NOT_IMPLEMENTED" if key == "V09" else "TO_VERIFY", reason=reason,
                            blocks_all_servers=False)
    result = dict(priority_revision=REVISION, campaign_id=CAMPAIGN, server_id=server,
                  created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), checks=results,
                  admission_integrity_ok=reconciliation["status"] == "VERIFIED_ARTIFACTS"
                    and method["status"] == "REUSED_VERIFIED",
                  no_scientific_pass_implied=True, reports_written=write_reports)
    if write_reports:
        names = {"V00": "runtime_reconciliation.json", "V01": "method_invariant_receipt.json",
                 "V02": "S73101_selection_audit.json", "V08": "frequency_frontend_receipt.json",
                 "V12": "selection_upload_receipt.json", "V14": "campaign_time_and_completion.json"}
        for key, name in names.items():
            _json(output / name, results[key])
        _json(output / "block_summary.json", results["V13"])
        _csv(output / "block_summary.csv", rows)
        if "curve" in v02:
            _csv(output / "eligible_curve.csv", v02["curve"])
        _atomic(output / "s1_branch_evidence_review.md", "# Stored branch evidence\n\n```json\n" +
                json.dumps(branch, indent=2, ensure_ascii=False) + "\n```\n")
        _json(output / "verification_status.json", result)
    return result
