"""Read-only cost sidecars, loaded by the existing publisher extension hook.

No trainer imports on the host. This package does not alter the running
campaign's source pins, optimizer, evaluation, or immutable result envelopes.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from panda_schedule30 import reporting as R

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "PANDA_SCHEDULE30_COST_V1"


def geometry_key(cfg, step):
    R._config(cfg)
    if type(step) is not int or not 0 <= step <= cfg["planned_horizon"]:
        raise ValueError("Invalid selected cost step")
    if cfg["phase"] == "donor":
        return "s100"
    phases = cfg["geometry"]["phases"]
    matching = [p for p in phases if p["first_update"] <= max(1, step) <= p["last_update"]]
    if len(matching) != 1:
        raise ValueError("Unbound selected-step deployment geometry")
    return matching[0]["geometry_key"]


def signature(cfg, step):
    key = geometry_key(cfg, step)
    b = cfg["baseline_resolved"]
    paths = [
        "panda_schedule30/cost/__init__.py", "panda_schedule30/cost/profiling.py",
        "panda_schedule30/method.py", "panda_schedule30/spec.py", "panda_schedule30/contract.py",
        "panda_mature30/stage1.py",
        "panda_band30/aligner.py", "panda_band30/reconstruction.py",
        "panda_warp30/reconstruction.py", "panda_midpoint/resampling.py",
        "panda_midpoint/contract.py", "panda_method/reconstruction.py",
        "panda_method/edge_data.py", "panda_method/stages.py",
        "directshift48/method.py", "directshift48/training.py", "donor30/masking.py",
        "pa/warp.py", "align/__init__.py", "align/resample.py",
        "model/pancrafter_paper.py", "model/pancrafter.py", "fh12/model.py",
        "experiment_cost/measurement.py",
    ]
    return json.loads(json.dumps({
        "schema": SCHEMA, "server": cfg["server"], "dataset": cfg["dataset"],
        "phase": cfg["phase"], "geometry_key": key,
        "pan_shift_scale": 1. if cfg["phase"] == "donor" else cfg["shift_scale"],
        "pan_execution": "BICUBIC_WARP_PAN_PREDICTED_GLOBAL2_TIMES_SCALE",
        "inference_passes": 1, "output_coordinates": "MS",
        "reference_warp": False, "inverse_output_warp": False,
        "aligner": deepcopy(b["aligner_architecture"]),
        "aligner_outputs": cfg["aligner_outputs"], "pan_channels": cfg["pan_channels"],
        "reconstructor": deepcopy(cfg["student_architecture"] if cfg["phase"] == "stage2"
                                else b["teacher_architecture"]),
        "upsampler": cfg["upsampler"], "frame": cfg["frame"],
        "input_shapes": {"rr": [1, 1, 256, 256], "fr": [1, 1, 512, 512]},
        "source_registry_sha256": b["source_registry_sha256"],
        "precision": "FP32", "batch": 1, "amp": False, "tf32": False,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
    }, allow_nan=False))


def validate_measurement(measurement, sig):
    from experiment_cost.publication import _validated_fields
    _validated_fields(measurement)
    isolation = measurement.get("gpu_isolation_assertion", {})
    before = measurement.get("model_state_sha256_before")
    if (measurement.get("weight_basis") != "FRESH_ARCHITECTURE_NOT_CHECKPOINT"
            or measurement.get("optimizer_updates") != 0
            or measurement.get("batch") != 1 or measurement.get("precision") != "FP32"
            or not R._sha(before) or before != measurement.get("model_state_sha256_after")
            or not str(measurement.get("environment", {}).get("device", "")).startswith("cuda")
            or isolation.get("server") != sig["server"]
            or isolation.get("other_compute_pids") != []
            or isolation.get("mode") != "STANDALONE_ISOLATED_NO_TRAINER_GPU"
            or isolation.get("training_state_retained_in_memory") is not False
            or measurement.get("deployment_signature_sha256") != R.digest(sig)):
        raise ValueError("Unverified isolated same-deployment cost")
    expected = {"phase": sig["phase"], "geometry_key": sig["geometry_key"],
        "pan_shift_scale": sig["pan_shift_scale"], "pan_execution": sig["pan_execution"],
        "training_teacher_included": False if sig["phase"] == "stage2" else None,
        "graph": "A+S" if sig["phase"] == "stage2" else "A+T"}
    if measurement.get("deployment") != expected:
        raise ValueError("Cost measurement is not bound to the applied scale/graph")
    for split, shape in sig["input_shapes"].items():
        obs = measurement["observations"][split]
        if obs.get("pan_shape") != shape or len(obs["full_pipeline"]["samples_ms"]) < 10:
            raise ValueError("Cost shapes/repeats differ from production protocol")
        if any(not math.isfinite(v) or v <= 0 for v in obs["full_pipeline"]["samples_ms"]):
            raise ValueError("Invalid latency samples")


def find_profile(cfg, step, root):
    sig = signature(cfg, step)
    key = R.digest(sig)
    path = Path(root) / "cost_profiles" / (key + ".json")
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    proof = value.pop("profile_sha256")
    if (proof != R.digest(value) or value.get("signature") != sig
            or value.get("architecture_key") != key or value.get("schema") != SCHEMA):
        raise ValueError("Cost profile identity/integrity mismatch")
    validate_measurement(value["measurement"], sig)
    return {**value, "profile_sha256": proof, "path": str(path.resolve())}


def save_profile(cfg, step, root, measurement):
    sig = signature(cfg, step)
    validate_measurement(measurement, sig)
    existing = find_profile(cfg, step, root)
    if existing is not None:
        return existing
    body = {"schema": SCHEMA, "signature": sig, "architecture_key": R.digest(sig),
            "measurement": measurement, "created_at": R.utc_now()}
    value = {**body, "profile_sha256": R.digest(body)}
    R.atomic_json(Path(root) / "cost_profiles" / (body["architecture_key"] + ".json"), value)
    return find_profile(cfg, step, root)


def _timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timing timestamp lacks timezone")
    return result


def observed_timing(cfg, step, root, note):
    """Selected-step elapsed costs; reused donors retain original training scope."""
    from .timing import timing_for_config
    from panda_schedule30.publication import _resolve_evidence_path
    receipt = note["extra_parameters"].get("selector_receipt")
    source_cfg, source_root = cfg, Path(root)
    if receipt is not None:
        selected = R.selected_source(receipt)
        source_cfg = selected["source_config"]
        source_root = _resolve_evidence_path(selected.get("source_root", receipt.get("source_root")), root)
    timing = timing_for_config(source_cfg, step, source_root, history_roots=[], cost_root=root)
    directory = source_root / "runs" / source_cfg["run_id"] / ("attempt_%02d" % source_cfg["attempt"])
    try:
        sp, ep = directory / "run_state.json", directory / f"evaluation_{step:07d}.json"
        state, evaluation = json.loads(sp.read_text()), json.loads(ep.read_text())
        identity = evaluation["state_before"]
        if (state.get("run_id") != source_cfg["run_id"] or state.get("attempt") != source_cfg["attempt"]
                or identity["run_id"] != source_cfg["run_id"]
                or identity["attempt"] != source_cfg["attempt"] or identity["completed_update"] != step):
            raise ValueError("Wall-time run/step identity mismatch")
        start, end = _timestamp(state["started_at"]), _timestamp(evaluation["recorded_at"])
        seconds = (end - start).total_seconds()
        measured = sum(timing["sheet_fields"].get(k, 0.) for k in ("Train(h)", "Eval(h)")) * 3600
        if not math.isfinite(seconds) or seconds < 0 or seconds + 1 < measured:
            raise ValueError("Wall interval inconsistent with observed training/evaluation")
        timing["sheet_fields"]["Wall(h)"] = seconds / 3600
        timing["omitted"].pop("Wall(h)", None)
        timing["proof"]["Wall(h)"] = {
            "started_at": state["started_at"], "through_evaluation_at": evaluation["recorded_at"],
            "run_state_path": str(sp), "evaluation_path": str(ep),
            "evaluation_sha256": hashlib.sha256(ep.read_bytes()).hexdigest(),
            "scope": "run start through selected evaluation; includes setup/IO/save overhead and any pauses",
        }
    except (KeyError, TypeError, ValueError, OSError) as exc:
        timing["omitted"]["Wall(h)"] = str(exc)
    timing["source_run_id"] = source_cfg["run_id"]
    timing["reused_source_training_not_new_campaign_charge"] = source_cfg["run_id"] != cfg["run_id"]
    return timing


def attach_cost(projected, root, *, require_cost=False):
    from experiment_cost.publication import _validated_fields
    missing, attached = [], []
    for i, original in enumerate(projected["rows"]):
        row = list(original)
        try:
            note = json.loads(row[R.INDEX["Original description / notes"]])
            cfg = note["extra_parameters"]["configuration"]
            step = row[R.INDEX["Selected_step"]]
            evidence = {"policy": SCHEMA, "same_graph_not_same_weights": True,
                "selected_checkpoint_binding": {
                    "run_id": cfg["run_id"], "step": step,
                    "model_state_sha256": note["extra_parameters"]["identity"]["model_state_sha256"],
                    "checkpoint_file_sha256": row[R.INDEX["Checkpoint_SHA"]] or None,
                    "inference_cost_basis": "FRESH_SAME_DEPLOYMENT_ARCHITECTURE_NOT_CHECKPOINT_WEIGHTS",
                    "timing_basis": "OBSERVED_THROUGH_THIS_SELECTED_EVALUATION",
                }}
            try:
                timing = observed_timing(cfg, step, Path(root), note)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                timing = {"sheet_fields": {}, "omitted": {"timing": str(exc)}}
            for name, value in timing["sheet_fields"].items():
                row[R.INDEX[name]] = value
            # The donor has 35 evaluations. Keep their full hashes locally,
            # not duplicated in a Google cell that is capped at 50k chars.
            timing_sha = R.digest(timing)
            timing_path = Path(root) / "cost_profiles/timing_evidence" / (timing_sha + ".json")
            if not timing_path.exists():
                R.atomic_json(timing_path, timing)
            elif R.digest(json.loads(timing_path.read_text())) != timing_sha:
                raise ValueError("Timing evidence integrity mismatch")
            evidence["observed_timing"] = {
                k: v for k, v in timing.items() if k != "proof"}
            evidence["observed_timing"].update(
                proof_path=str(timing_path.resolve()), proof_sha256=timing_sha)
            profile = find_profile(cfg, step, root)
            if profile is None:
                missing.append({"run_id": cfg["run_id"], "step": step,
                                "reason": "NO_VERIFIED_SCHEDULE30_COST"})
            else:
                split, fields = _validated_fields(profile["measurement"])
                for name, value in fields.items():
                    row[R.INDEX[name]] = value
                evidence.update(profile_sha256=profile["profile_sha256"],
                    architecture_key=profile["architecture_key"], profile_path=profile["path"],
                    sheet_fields=fields, default_sheet_split=split,
                    all_domains={s: profile["measurement"]["observations"][s]["sheet_fields"]
                                 for s in ("rr", "fr")})
                attached.append({"result_id": row[R.INDEX["Result_ID"]],
                                 "profile_sha256": profile["profile_sha256"]})
            short = json.loads(projected["notes"][i])
            note["cost_measurement"] = short["cost_measurement"] = evidence
            row[R.INDEX["Original description / notes"]] = R.canonical_json(note)
            serialized = R.canonical_json(short)
            if max(len(row[R.INDEX["Original description / notes"]]), len(serialized)) >= 45000:
                raise ValueError("Cost provenance exceeds safe Sheet limit")
            projected["rows"][i], projected["notes"][i] = row, serialized
        except (ValueError, KeyError, TypeError, OSError) as exc:
            projected["conflicts"].append({"reason": "SCHEDULE30_COST_FAILED:" + str(exc),
                                          "result_id": original[R.INDEX["Result_ID"]]})
    if require_cost:
        projected["conflicts"].extend(missing)
    projected["cost"] = {"bound_count": len(attached), "bindings": attached,
                         "missing": missing, "require_cost": require_cost}

