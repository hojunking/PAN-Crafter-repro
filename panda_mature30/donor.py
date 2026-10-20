"""Strict local same-checkpoint A/T import without rewriting source campaigns."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from donor30.evaluation import model_state_hash
from panda40.stage1 import _check_evaluation, _identity, _optimizer_contract, _scheduler
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_method.data import require_sha256
from panda_method.edge_data import EdgeDataSources
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.sources import RegisteredRRTrain
from ta2.evaluation import file_sha

from .contract import CAMPAIGN_ID, file_identity
from .stage1 import CHECKPOINT_SCHEMA, RESUME_SCHEMA, build_stage1, teacher_binding, validate_config

LEGACY_CAMPAIGN = "PANDA_S1_ALLBAND_WARP_BANDWISE_30H_20261019_v1"


def _read_checkpoint(path, sha=None):
    path = Path(path).resolve(strict=True)
    before = file_identity(path)
    actual = file_sha(path)
    if sha is not None and actual != sha:
        raise ValueError("Donor checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if file_identity(path) != before or file_sha(path) != actual:
        raise ValueError("Checkpoint changed during read")
    return path, actual, payload


def _check_source_config(cfg, server):
    if server not in ("s1", "s3") or cfg.get("server") != server:
        raise ValueError("Cross-server donor import forbidden")
    if (cfg.get("dataset") != "WV3" or cfg.get("upsampler") != "interp23"
            or cfg.get("aligner_outputs") != 2 or cfg.get("pan_channels") != 1
            or cfg.get("alignment_target") != "SCHARR_OF_MEAN8_GT"
            or cfg.get("phase") != "donor" or cfg.get("planned_horizon") != 50000):
        raise ValueError("Local mean8 global2 interp23 H50 Stage1 donor required")
    if cfg.get("campaign_id") == LEGACY_CAMPAIGN:
        if server != "s1" or cfg.get("case") != "G8_S1_U23":
            raise ValueError("Only completed local s1 G8 lineage may be reused")
    elif cfg.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("Unregistered source campaign, no implicit donor fallback")


def checkpoint_evidence(path, *, server):
    path, sha, payload = _read_checkpoint(path)
    cfg = payload.get("config", {})
    _check_source_config(cfg, server)
    step = payload.get("trainer", {}).get("updates")
    if type(step) is not int or not 1 <= step <= 50000:
        raise ValueError("Evaluated donor update required")
    run_dir = path.parent.parent
    if path.name != f"step_{step:07d}.pt" or path.parent.name != "models":
        raise ValueError("Donor must be exact retained step checkpoint")
    evaluation_path = run_dir / f"evaluation_{step:07d}.json"
    evaluation = payload.get("evaluation")
    if (payload.get("resume_only") is not False or not isinstance(evaluation, dict)
            or not evaluation_path.is_file()
            or json.loads(evaluation_path.read_text()) != evaluation):
        raise ValueError("Original checkpoint/external official evaluation mismatch")
    config_path = run_dir / "config.json"
    if not config_path.is_file() or json.loads(config_path.read_text()) != cfg:
        raise ValueError("Original run/checkpoint configuration mismatch")
    identity = _identity(cfg, step, payload.get("state_hash"))
    _check_evaluation(evaluation, identity)
    # Native layout: campaign/server/WV3/runs/run_id/attempt_N/models/step.pt.
    source_root = run_dir.parent.parent.parent
    return {"checkpoint_path": str(path), "file_sha256": sha,
        "source_campaign_id": cfg["campaign_id"], "source_run_id": cfg["run_id"],
        "completed_update": step, "source_config": cfg,
        "source_root": str(source_root), "source_run_dir": str(run_dir),
        "source_evaluation_path": str(evaluation_path),
        "source_evaluation_sha256": file_sha(evaluation_path),
        "source_state": deepcopy(evaluation["state_before"])}


def complete_history(paths, *, server):
    """Prove complete H50 lineage and all scheduled retained candidates."""
    paths = [Path(path).resolve(strict=True) for path in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Nonempty unique retained checkpoints required")
    parents = {path.parent.parent for path in paths}
    if len(parents) != 1:
        raise ValueError("Do not mix independent Stage1 runs in one selection")
    run_dir = next(iter(parents))
    config_path, state_path = run_dir / "config.json", run_dir / "run_state.json"
    cfg, state = json.loads(config_path.read_text()), json.loads(state_path.read_text())
    _check_source_config(cfg, server)
    if (state.get("status") != "COMPLETED" or state.get("step") != 50000
            or state.get("run_id") != cfg["run_id"] or state.get("phase") != "donor"):
        raise ValueError("Re-selection requires a completed full50K local Stage1")
    steps = list(cfg["evaluation_steps"])
    if len(set(steps)) != len(steps) or 50000 not in steps:
        raise ValueError("Complete registered evaluation schedule required")
    expected = {run_dir / "models" / f"step_{step:07d}.pt" for step in steps}
    if set(paths) != expected:
        raise ValueError("RRE selection must include every retained official candidate")
    selection_path = run_dir / "models" / "selection.json"
    selection = json.loads(selection_path.read_text())
    saved = selection.get("saved")
    if (not isinstance(saved, list)
            or len(saved) != len(steps)
            or {r.get("completed_update") for r in saved} != set(steps)):
        raise ValueError("Saved official selection inventory incomplete")
    if state.get("selection") is not None and state["selection"] != selection:
        raise ValueError("Completed source state/selection receipt disagree")
    for row in saved:
        path = run_dir / "models" / f"step_{row['completed_update']:07d}.pt"
        if (any(row.get(k) is not True for k in
                ("checkpoint_verified", "metrics_complete", "metrics_finite", "same_state_verified"))
                or file_sha(path) != row.get("file_sha256")
                or row.get("run_id") != cfg["run_id"]):
            raise ValueError("Source saved candidate hash/identity/evaluation invalid")
    return {"source_run_id": cfg["run_id"], "source_campaign_id": cfg["campaign_id"],
        "server": server, "completed_update": 50000, "candidate_steps": sorted(steps),
        "run_state_sha256": file_sha(state_path),
        "source_selection_sha256": file_sha(selection_path),
        "source_config_sha256": file_sha(config_path),
        "old_records_modified": False}


def load_donor(path, expected_sha256, registered, *, server="s1", device="cpu",
               include_teacher=True, selector_receipt=None):
    """Import exact original A/T tensors, preserving original source identity."""
    require_sha256(expected_sha256, "explicit local donor SHA")
    if type(include_teacher) is not bool:
        raise ValueError("Explicit include_teacher required")
    if isinstance(registered, EdgeDataSources):
        registered = RegisteredRRTrain(registered)
    if not isinstance(registered, RegisteredRRTrain):
        raise TypeError("Pinned RR train inventory required")
    path, sha, payload = _read_checkpoint(path, expected_sha256)
    cfg = payload.get("config", {})
    _check_source_config(cfg, server)
    if cfg["campaign_id"] == LEGACY_CAMPAIGN:
        from panda_band30.donor import load_donor as legacy_load
        donor = legacy_load(path, sha, registered, device=device, include_teacher=include_teacher)
        binding, metadata = deepcopy(donor.binding), deepcopy(donor.checkpoint_metadata)
        aligner, teacher = donor.aligner, donor.teacher
    else:
        inventory = registered.verify()
        validate_config(cfg, registered.sources)
        saved = payload.get("trainer", {})
        update = saved.get("updates")
        if (payload.get("schema") != CHECKPOINT_SCHEMA
                or payload.get("campaign_resume_schema") != RESUME_SCHEMA
                or payload.get("campaign_resume_complete") is not True
                or payload.get("mature30_teacher") != teacher_binding(cfg)
                or type(update) is not int or not 1 <= update <= 50000
                or payload.get("scheduler") != _scheduler(cfg, update)
                or saved.get("schema") != "PANDA_DIRECTSHIFT_STAGE1_TRAINER_V1"):
            raise ValueError("Complete native Mature30 Stage1 checkpoint required")
        _optimizer_contract(saved, cfg, update)
        identity = _identity(cfg, update, payload.get("state_hash"))
        if (payload.get("identity") != identity
                or payload.get("model_state_sha256") != payload.get("state_hash")
                or payload.get("resume_only") is not False):
            raise ValueError("Donor state/update identity mismatch")
        evaluation = payload.get("evaluation")
        if (not isinstance(evaluation, dict)
                or evaluation.get("full_method_eligible") is not True
                or evaluation.get("metric_function_injection") is not False
                or evaluation.get("measurement_domain") != "OFFICIAL_FULL20_NATIVE"
                or evaluation.get("official_reference") !=
                "original_PAN/native_LMS; FR full512 unmasked; RR all20 dim_cut21"):
            raise ValueError("Complete original-reference official donor evaluation required")
        _check_evaluation(evaluation, identity)
        for split in ("rr", "fr"):
            if {r["scene_id"] for r in evaluation[split]["per_scene"]} != {
                    f"WV3:{split}:{i}" for i in range(20)}:
                raise ValueError("Exact official all20 scene identities required")
        with torch.random.fork_rng(devices=[]):
            fresh = build_stage1(cfg, registered.sources)
            fresh.stage.load_state_dict(saved["stage"], strict=True)
            if model_state_hash(fresh.stage) != payload["state_hash"]:
                raise ValueError("Actual donor tensors differ from official checkpoint state")
            aligner, paired_teacher = fresh.stage.aligner, fresh.stage.teacher
        for module in (aligner, paired_teacher):
            if any(p.dtype != torch.float32 or not bool(torch.isfinite(p).all())
                   for p in module.parameters()):
                raise ValueError("Finite FP32 donor parameters required")
            module.requires_grad_(False).eval().to(device)
        teacher = paired_teacher if include_teacher else None
        binding = {"sensor": inventory.sensor, "source_split": "RR_train_only",
            "sample_ids_sha256": inventory.sample_ids_sha256,
            "sample_count": len(inventory.sample_ids),
            "source_manifest_sha256": inventory.source_manifest_sha256,
            "catalog_sha256": inventory.catalog_sha256,
            "source_h5_sha256": inventory.source_h5_sha256,
            "aligner_state_sha256": state_dict_sha256(aligner),
            "teacher_state_sha256": state_dict_sha256(paired_teacher),
            "donor_checkpoint_id": sha, "donor_checkpoint_sha256": sha,
            "donor_run_id": cfg["run_id"], "donor_step": update,
            "donor_source_mask_policy": fresh.stage.pipeline_manifest()["rec_reduction"],
            "reconstruction_spec": {"bands": 8, "width": 112, "depth": [1, 2, 3],
                "input_layout": "P0", "base_reference": "BICUBIC_MS"},
            "scharr": {"kernel": "absolute_xy_/32", "padding": "reflect1",
                "rgb_indices": [4, 2, 1], "roi_margin": 8},
            "warp": "bicubic_border_FP32_align_corners_false_current_pan_px",
            "band30_teacher": teacher_binding(cfg),
            "band30_reconstruction_spec": fresh.stage.teacher.spec.manifest()}
        metadata = {"checkpoint_path": str(path), "checkpoint_sha256": sha,
            "run_id": cfg["run_id"], "completed_update": update,
            "same_checkpoint_A_T": True, "donor_model_state_sha256": payload["state_hash"],
            "source_pipeline": deepcopy(fresh.stage.pipeline_manifest()),
            "checkpoint_identity": identity, "teacher_constructed": True,
            "teacher_retained": include_teacher, "selection": "EXPLICIT_CALLER_PATH_AND_SHA",
            "band30_teacher": teacher_binding(cfg)}
    binding["source_campaign_id"] = cfg["campaign_id"]
    binding["mature30_teacher"] = deepcopy(binding["band30_teacher"])
    metadata["source_campaign_id"] = cfg["campaign_id"]
    metadata["consumer_campaign_id"] = CAMPAIGN_ID
    if selector_receipt is not None:
        from .selection import validate_receipt
        validate_receipt(selector_receipt)
        chosen = selector_receipt["selected"]
        if (selector_receipt["server"] != server or chosen["file_sha256"] != sha
                or Path(chosen["checkpoint_path"]).resolve() != path
                or chosen["source_config"] != cfg
                or chosen["completed_update"] != payload["trainer"]["updates"]):
            raise ValueError("Selected donor differs from validation selection receipt")
        binding["mature30_selector_sha256"] = selector_receipt["receipt_sha256"]
        binding["mature30_validation_inventory_sha256"] = (
            selector_receipt["validation_inventory"]["inventory_sha256"])
        metadata["selector"] = selector_receipt["selector"]
        metadata["selector_receipt_sha256"] = selector_receipt["receipt_sha256"]
        metadata["validation"] = {k: v for k, v in chosen["validation"].items()
                                  if k not in ("per_sample",)}
    result = LoadedDonor(aligner, teacher, binding, metadata)
    result.assert_frozen()
    registered.verify()
    return result
