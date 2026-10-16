"""Same-state RR20/FR20 for the isolated S4 Direct donor/Stage 2 campaign.

Metric implementations and original, unmasked reference arrays are inherited
unchanged. Stage 2 state identity additionally binds its trainable Student.
Evaluation never saves, steps, clips, shifts references, or exports predictions.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import random
import re
import time

import numpy as np
import torch

from directshift48.evaluation import (expected_ids, metric_functions_for, metric_values,
                                      source_manifest)
from directshift48.method import DirectShiftStage1, DirectPANAligner
from directshift48.plan import FR_METRICS, RR_METRICS
from panda_method.calibration import canonical_sha256, state_dict_sha256
from ta2.evaluation import JQM_VARIANT, digest, evaluator_identity
from donor30.method import DirectStage2
from donor30.masking import MaskedDonorStage1, LOSS_MASK_POLICY
from .reporting import CAMPAIGN_ID, METHOD_REVISION, _config

LANES = {s: {"dataset": "WV3", "hqnr_min": .959, "ergas_max": 2.040} for s in ("s4",)}

def target_gap(h, e, *, server):
    if not _finite(h) or not _finite(e):
        return {"D_inf": None, "target_pair_pass": False}
    return {"D_inf": max(max(0., .959-h)/.959, max(0., e-2.040)/2.040),
            "target_pair_pass": h >= .959 and e <= 2.040}


def model_state_hash(stage):
    roles = {"A": state_dict_sha256(stage.aligner)}
    if stage.teacher is not None:
        roles["T"] = state_dict_sha256(stage.teacher)
    if hasattr(stage, "student"):
        roles["S"] = state_dict_sha256(stage.student)
    return canonical_sha256(roles)


def _tree_hash(value):
    """Exact optimizer tensor/scalar identity, without cloning its GPU state."""
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            x = item.detach().cpu().contiguous()
            visit([str(x.dtype), list(x.shape)])
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b"dict:")
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update((type(item).__name__ + ":" + str(len(item))).encode())
            for sub in item:
                visit(sub)
        else:
            h.update(json.dumps([type(item).__name__, item], sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


@contextmanager
def paused_model(trainer, *, gradients=False):
    if getattr(trainer, "poisoned", False):
        raise RuntimeError("Cannot evaluate a failed optimizer state")
    stage = trainer.stage
    state, updates = model_state_hash(stage), trainer.updates
    policy = digest(stage.pipeline_manifest())
    modes = [(m, m.training) for m in stage.modules()]
    grads = [(p, p.requires_grad, None if p.grad is None else p.grad.detach().clone())
             for p in stage.parameters()]
    extra = copy.deepcopy(stage.get_extra_state())
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    optimizer = getattr(trainer, "optimizer", None)
    optimizer_before = _tree_hash(None if optimizer is None else optimizer.state_dict())
    try:
        stage.eval()
        with torch.set_grad_enabled(gradients):
            yield state
    finally:
        for m, mode in modes:
            m.training = mode
        for p, requires_grad, grad in grads:
            p.requires_grad_(requires_grad)
            p.grad = grad
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.random.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        try:
            stage.set_extra_state(extra)
            changed = (trainer.stage is not stage or getattr(trainer, "optimizer", None) is not optimizer
                       or model_state_hash(stage) != state or trainer.updates != updates
                       or digest(stage.pipeline_manifest()) != policy
                       or _tree_hash(None if optimizer is None else optimizer.state_dict()) != optimizer_before)
        except Exception as exc:
            trainer.poisoned = True
            raise RuntimeError("Inference policy/state changed during paused evaluation") from exc
        if changed:
            trainer.poisoned = True
            raise RuntimeError("A/T/S, optimizer, update or policy changed during paused evaluation")


def _pin(trainer, explicit=None):
    if explicit is not None:
        return explicit
    if getattr(trainer, "evaluator_pin", None) is not None:
        return trainer.evaluator_pin
    if "evaluator_pin" in trainer.config:
        return trainer.config["evaluator_pin"]
    return evaluator_identity()


def alignment_manifest(stage):
    if hasattr(stage, "alignment_manifest"):
        return stage.alignment_manifest()
    return {"aligner": stage.aligner.manifest(), "warp": stage.pipeline_manifest()["warp"],
            "references": "original_PAN/native_LMS-full_frame-no-mask",
            "shift_unit": "CURRENT_PAN_PIXEL"}


def configured_pipeline(trainer, sources):
    from panda_stage2_condq.method import ConditionalQStage2
    c, stage = _config(trainer.config), trainer.stage
    if type(stage.aligner) is not DirectPANAligner:
        raise ValueError("S4 requires the original DirectPANAligner architecture")
    if sources.sensor != c["dataset"]:
        raise ValueError("Source sensor and registered case mismatch")
    pipeline = stage.pipeline_manifest()
    if c["phase"] == "donor":
        if not isinstance(stage, MaskedDonorStage1) or not isinstance(stage, DirectShiftStage1):
            raise ValueError("Stage 1 requires the original Direct MASK2 implementation")
        if c.get("loss_mask_policy") != LOSS_MASK_POLICY:
            raise ValueError("Stage 1 requires explicit MASK2 policy")
        if (stage.sources.sensor != sources.sensor or
                stage.sources.rr_manifest_sha256 != sources.manifest_sha("train")):
            raise ValueError("Stage 1 train-source mismatch")
        expected = c["baseline_resolved"]["teacher_architecture"]
        actual = stage.teacher.spec
        if ((actual.bands, actual.width, list(actual.depth)) !=
                (expected["bands"], expected["width"], list(expected["depth"])) or
                actual.input_layout != "P0" or actual.base_reference != "BICUBIC_MS"):
            raise ValueError("Teacher architecture/base mismatch")
        if (stage.lambda_abs != c["lambda_abs"] or stage.lambda_eps != c["lambda_epsilon"]
                or c.get("lambda_rec", 1.) != 1.):
            raise ValueError("Stage 1 loss coefficients differ from registered case")
    else:
        if not isinstance(stage, ConditionalQStage2):
            raise ValueError("Stage 2 requires new conditional-q implementation")
        trainer.assert_contract()
        if (stage.spec.sensor != sources.sensor or
                stage.donor_binding["source_manifest_sha256"] != sources.manifest_sha("train")):
            raise ValueError("Frozen donor/source/sensor mismatch")
        if c.get("loss_mask_policy") != stage.spec.manifest()["loss_mask_policy"]:
            raise ValueError("Stage 2 loss mask differs from registered case")
        declared = c.get("stage2_spec")
        if declared is not None:
            for key, value in declared.items():
                if getattr(stage.spec, key, object()) != value:
                    raise ValueError("Stage 2 objective mismatch: " + key)
        actual, expected = stage.student.spec, c["student_architecture"]
        if ((actual.bands, actual.width, list(actual.depth)) !=
                (expected["bands"], expected["width"], list(expected["depth"])) or
                actual.input_layout != "P0" or actual.base_reference != "BICUBIC_MS"):
            raise ValueError("Student architecture/base mismatch")
        pinned = c.get("donor_binding")
        if pinned is not None and pinned != stage.donor_binding:
            raise ValueError("Registered Stage 2 donor differs from loaded donor")
    if c.get("alignment_protocol_digest") is not None and digest(alignment_manifest(stage)) != c["alignment_protocol_digest"]:
        raise ValueError("Alignment protocol differs from registered case")
    return pipeline


def model_identity(trainer, sources, *, evaluator_revision=None, evaluator_pin=None):
    c = trainer.config
    pin = _pin(trainer, evaluator_pin)
    pipeline = configured_pipeline(trainer, sources)
    return {"campaign_id": CAMPAIGN_ID, "method_revision": METHOD_REVISION,
            "phase": c["phase"], "server": c["server"], "dataset": c["dataset"],
            "run_id": c["run_id"], "attempt": c["attempt"], "seed": c["seed"],
            "completed_update": trainer.updates, "model_state_sha256": model_state_hash(trainer.stage),
            "inference_policy_digest": digest({"pipeline": pipeline, "eval": pin,
                "references": "original_PAN/native_LMS-full_frame-no-mask",
                "rr_source": sources.manifest_sha("rr"), "fr_source": sources.manifest_sha("fr")}),
            "evaluator_revision": evaluator_revision or digest(pin),
            "alignment_protocol_digest": digest(alignment_manifest(trainer.stage))}


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def qualify_evaluation(record, *, server, expected_rr_ids, expected_fr_ids):
    """Full20 completeness and immutable PandaS4 state; donor gate is separate."""
    lane = LANES[server]
    complete, finite, reasons, counts = True, True, [], {}
    for split, expected in (("rr", expected_rr_ids), ("fr", expected_fr_ids)):
        expected = list(expected)
        rows = record.get(split, {}).get("per_scene", [])
        ids = [r.get("scene_id") for r in rows]
        missing = sorted(set(expected) - set(ids))
        unexpected = sorted(set(ids) - set(expected))
        duplicates = [k for k, n in Counter(ids).items() if n > 1]
        if len(expected) != 20 or len(set(expected)) != 20 or len(rows) != 20 or missing or unexpected or duplicates:
            complete = False
            reasons.append(split.upper() + "_SCENE_SET_INCOMPLETE")
        counts[split] = {"expected": 20, "completed": len(rows), "missing": missing,
                         "unexpected": unexpected, "duplicates": duplicates}
        keys = FR_METRICS if split == "fr" else RR_METRICS + (("Q8" if lane["dataset"] == "WV3" else "Q4"),)
        aggregate = record.get(split, {}).get("metrics", {})
        for values in [aggregate] + [r.get("metrics", {}) for r in rows]:
            if any(k not in values for k in keys):
                complete = False
                reasons.append(split.upper() + "_MISSING_METRIC")
            if any(not _finite(values.get(k)) for k in keys):
                finite = False
                reasons.append(split.upper() + "_NONFINITE_METRIC")
        for key in keys:
            vals = [r.get("metrics", {}).get(key) for r in rows]
            if vals and _finite(aggregate.get(key)) and all(_finite(v) for v in vals):
                if not math.isclose(aggregate[key], math.fsum(vals)/len(vals), rel_tol=0., abs_tol=1e-12):
                    complete = False
                    reasons.append(split.upper() + "_AGGREGATE_MISMATCH:" + key)
    states = [record.get(k) for k in ("state_before", "rr_state", "fr_state", "state_after")]
    first = states[0]
    same = isinstance(first, dict) and all(s == first for s in states[1:])
    required = ("run_id", "attempt", "seed", "completed_update", "model_state_sha256",
                "inference_policy_digest", "evaluator_revision", "alignment_protocol_digest")
    if same:
        same = (all(k in first for k in required) and first.get("campaign_id") == CAMPAIGN_ID
                and first.get("method_revision") == METHOD_REVISION and first.get("phase") in ("donor", "stage2")
                and first.get("server") == server and first.get("dataset") == lane["dataset"]
                and all(type(first.get(k)) is int and first[k] >= minimum
                        for k, minimum in (("attempt", 1), ("seed", 0), ("completed_update", 1)))
                and all(isinstance(first.get(k), str) and first[k] for k in ("run_id", "evaluator_revision"))
                and all(isinstance(first.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", first[k])
                        for k in ("model_state_sha256", "inference_policy_digest", "alignment_protocol_digest")))
    if not same:
        reasons.append("SAME_STATE_IDENTITY_NOT_VERIFIED")
    eligible = (record.get("full_method_eligible") is True and record.get("server") == server
                and record.get("dataset") == lane["dataset"])
    h = record.get("fr", {}).get("metrics", {}).get("HQNR")
    e = record.get("rr", {}).get("metrics", {}).get("ERGAS")
    gap = target_gap(h, e, server=server)
    if gap["D_inf"] is None:
        finite = False
    passed = bool(complete and finite and same and eligible and gap["target_pair_pass"])
    return {**gap, "HQNR": h, "ERGAS": e, "joint_pass": passed,
            "full_method_eligible": eligible, "metrics_complete": complete,
            "metrics_finite": finite, "same_state_verified": bool(same), "scene_completeness": counts,
            "reasons": reasons, "selection": "TARGET_QUALIFIED" if passed else "BEST_OBSERVED_NOT_QUALIFIED"}


def official_evaluate(trainer, sources, *, metric_functions=None, evaluator_revision=None,
                      evaluator_pin=None, device=None, on_scene=None, deadline_unix=None):
    pin = _pin(trainer, evaluator_pin)
    revision = evaluator_revision or digest(pin)
    injected = metric_functions is not None
    if injected and revision != "TEST":
        raise ValueError("Metric injection is allowed only in synthetic TEST evaluations")
    functions = metric_functions or metric_functions_for(sources, pin)
    if set(functions) != {"rr", "fr"}:
        raise ValueError("Both RR and FR metric functions are required")
    if device is None:
        device = next(trainer.stage.parameters()).device
    c = trainer.config
    start = time.monotonic()
    record = {"record_type": "EVAL_SUMMARY", "server": c["server"], "dataset": c["dataset"],
        "phase": c["phase"], "full_method_eligible": not c.get("disposable_preflight", False) and not injected,
        "rr": {}, "fr": {}, "test_aware": True, "jqm_variant": JQM_VARIANT,
        "eval_protocol_id": digest(pin), "metric_function_injection": injected,
        "official_reference": "original_PAN/native_LMS; FR full512 unmasked; RR all20 dim_cut21",
        "measurement_domain": "OFFICIAL_FULL20_NATIVE", "weights_saved": False,
        "checkpoint_path": None, "weights_file_sha256": None,
        "inference_pipeline": copy.deepcopy(trainer.stage.pipeline_manifest())}
    with paused_model(trainer):
        record["state_before"] = model_identity(trainer, sources, evaluator_revision=revision, evaluator_pin=pin)
        for split in ("rr", "fr"):
            rows, tick = [], time.monotonic()
            for i, sample_id in enumerate(expected_ids(sources, split)):
                if deadline_unix is not None and time.time() >= deadline_unix:
                    raise TimeoutError("EVAL_INCOMPLETE: deadline before full RR20/FR20 evaluation")
                batch = sources.batch(split, [i], device=device)
                if tuple(batch.lineage.sample_ids) != (sample_id,):
                    raise ValueError("Official H5 row and evaluation image identity differ")
                side = 256 if split == "rr" else 512
                if not injected and batch.pan.shape[-2:] != (side, side):
                    raise ValueError("Official native evaluation grid dimensions differ")
                out = trainer.stage.native_prediction(batch)
                raw = ((out["y"][0].float() + 1) * (sources.peak/2)).cpu().numpy()
                arrays = {k: v[0] for k, v in sources.raw(split, [i]).items()}
                metrics = functions[split](raw, arrays)
                row = {"scene_id": sample_id, "evaluation_image_id": sample_id, "sample_id": sample_id,
                    "metrics": metric_values(metrics, split, sources.bands),
                    "correction_dy_dx": out["u"][0].cpu().tolist(),
                    "raw_prediction_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                    "raw_min": float(raw.min()), "raw_max": float(raw.max()),
                    "jqm_weights": np.asarray(metrics.get("jqm_weights", [])).tolist(),
                    "jqm_weight_source": metrics.get("jqm_weight_source"),
                    "shift_unit": "CURRENT_PAN_PIXEL", "alignment_kind": type(trainer.stage.aligner).__name__}
                rows.append(row)
                if on_scene is not None:
                    on_scene({"branch": split, "identity": record["state_before"], **row})
                del batch, out, raw, arrays
            record[split] = {"per_scene": rows,
                "metrics": {k: float(np.mean([r["metrics"][k] for r in rows])) for k in rows[0]["metrics"]},
                "std": {k: float(np.std([r["metrics"][k] for r in rows], ddof=1)) for k in rows[0]["metrics"]},
                "elapsed_seconds": time.monotonic()-tick}
            record[split+"_state"] = model_identity(trainer, sources, evaluator_revision=revision, evaluator_pin=pin)
        record["state_after"] = model_identity(trainer, sources, evaluator_revision=revision, evaluator_pin=pin)
    if not injected and evaluator_identity() != pin:
        raise ValueError("Official evaluator changed during evaluation")
    record["elapsed_seconds"] = time.monotonic()-start
    record["qualification"] = qualify_evaluation(record, server=c["server"],
        expected_rr_ids=expected_ids(sources, "rr"), expected_fr_ids=expected_ids(sources, "fr"))
    print(f"[PandaS4:{c['phase']}] step={trainer.updates} HQNR={record['fr']['metrics']['HQNR']:.6f} "
          f"SCC={record['rr']['metrics']['SCC']:.6f} ERGAS={record['rr']['metrics']['ERGAS']:.6f}", flush=True)
    return record


evaluation_context = paused_model
