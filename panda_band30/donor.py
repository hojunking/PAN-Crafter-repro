"""Strict same-checkpoint A/T handoff from newly trained band30 Stage 1.

Only SHA-pinned local experiment artifacts are accepted. A Teacher remains an
ordinary Reconstructor for architecture compatibility; its effective base is
bound separately and must be supplied by the band30 reconstruction adapter.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from donor30.evaluation import model_state_hash
from panda40.stage1 import _check_evaluation, _identity, _optimizer_contract, _scheduler
from panda_method.calibration import state_dict_sha256
from panda_method.data import require_sha256
from panda_method.edge_data import EdgeDataSources
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.sources import RegisteredRRTrain
from directshift48.method import RGB_INDICES
from ta2.evaluation import file_sha

from .reconstruction import upsampler_manifest
from .contract import file_identity
from .stage1 import CHECKPOINT_SCHEMA, RESUME_SCHEMA, build_stage1, teacher_binding, validate_config


def load_donor(path, expected_sha256, registered, *, device="cpu", include_teacher=True):
    """Import one officially evaluated fresh Stage 1 checkpoint, never legacy.

    No selection is performed here. The caller supplies its already selected
    path and SHA; the evaluated Aligner and Teacher are inseparable. Optimizer,
    sampler and epsilon RNG are not transferred into Stage 2.
    """
    require_sha256(expected_sha256, "explicit band30 donor checkpoint SHA256")
    if type(include_teacher) is not bool:
        raise ValueError("Explicit include_teacher boolean required")
    if isinstance(registered, EdgeDataSources):
        registered = RegisteredRRTrain(registered)
    if not isinstance(registered, RegisteredRRTrain):
        raise TypeError("Pinned RegisteredRRTrain required")
    inventory = registered.verify()
    checkpoint = Path(path).resolve(strict=True)
    if not checkpoint.is_file() or file_sha(checkpoint) != expected_sha256:
        raise ValueError("Band30 donor checkpoint SHA mismatch")
    before = file_identity(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if file_identity(checkpoint) != before or file_sha(checkpoint) != expected_sha256:
        raise ValueError("Donor checkpoint changed while loading")
    if (not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("campaign_resume_schema") != RESUME_SCHEMA
            or payload.get("campaign_resume_complete") is not True):
        raise ValueError("Only newly trained complete band30 Stage 1 donors are accepted")
    cfg, saved = payload.get("config", {}), payload.get("trainer", {})
    validate_config(cfg, registered.sources)
    if (saved.get("schema") != "PANDA_DIRECTSHIFT_STAGE1_TRAINER_V1"
            or payload.get("band30_teacher") != teacher_binding(cfg)):
        raise ValueError("Donor trainer/effective upsampler binding mismatch")
    update = saved.get("updates")
    if (type(update) is not int or not 1 <= update <= cfg["planned_horizon"]
            or payload.get("scheduler") != _scheduler(cfg, update)):
        raise ValueError("Completed fresh donor update/schedule required")
    _optimizer_contract(saved, cfg, update)
    identity = _identity(cfg, update, payload.get("state_hash"))
    if (payload.get("identity") != identity
            or payload.get("model_state_sha256") != payload.get("state_hash")):
        raise ValueError("Donor model/checkpoint identity mismatch")
    require_sha256(payload.get("state_hash"), "donor A/T state hash")
    evaluation = payload.get("evaluation")
    if (not isinstance(evaluation, dict) or payload.get("resume_only") is not False
            or evaluation.get("full_method_eligible") is not True
            or evaluation.get("metric_function_injection") is not False
            or evaluation.get("measurement_domain") != "OFFICIAL_FULL20_NATIVE"
            or evaluation.get("official_reference") !=
               "original_PAN/native_LMS; FR full512 unmasked; RR all20 dim_cut21"):
        raise ValueError("Donor must have complete official original-reference evaluation")
    _check_evaluation(evaluation, identity)
    for split in ("rr", "fr"):
        if {row["scene_id"] for row in evaluation[split]["per_scene"]} != {
                f"WV3:{split}:{index}" for index in range(20)}:
            raise ValueError("Exact official WV3 all20 donor scene identities required")
    # Strict stage loading validates the complete pipeline, architecture and
    # source manifest as well as all finite A/T parameters. It also rejects a
    # relabeled legacy or opposite-upsampler state before Stage 2 can consume it.
    with torch.random.fork_rng(devices=[]):
        fresh = build_stage1(cfg, registered.sources)
        fresh.stage.load_state_dict(saved["stage"], strict=True)
        if model_state_hash(fresh.stage) != payload["state_hash"]:
            raise ValueError("Actual A/T tensors differ from the evaluated donor state")
        aligner, paired_teacher = fresh.stage.aligner, fresh.stage.teacher
    for module in (aligner, paired_teacher):
        if any(p.dtype != torch.float32 or not bool(torch.isfinite(p).all())
               for p in module.parameters()):
            raise ValueError("Donor parameters must be finite FP32")
        module.requires_grad_(False).eval().to(device)
    teacher = paired_teacher if include_teacher else None
    bands = registered.sources.bands
    binding = {"sensor": inventory.sensor, "source_split": "RR_train_only",
        "sample_ids_sha256": inventory.sample_ids_sha256, "sample_count": len(inventory.sample_ids),
        "source_manifest_sha256": inventory.source_manifest_sha256,
        "catalog_sha256": inventory.catalog_sha256, "source_h5_sha256": inventory.source_h5_sha256,
        "aligner_state_sha256": state_dict_sha256(aligner),
        "teacher_state_sha256": state_dict_sha256(paired_teacher),
        "donor_checkpoint_id": expected_sha256, "donor_checkpoint_sha256": expected_sha256,
        "donor_run_id": cfg["run_id"], "donor_step": update,
        "donor_source_mask_policy": fresh.stage.pipeline_manifest()["rec_reduction"],
        "reconstruction_spec": {"bands": bands, "width": 112, "depth": [1, 2, 3],
                                "input_layout": "P0", "base_reference": "BICUBIC_MS"},
        "scharr": {"kernel": "absolute_xy_/32", "padding": "reflect1",
                   "rgb_indices": list(RGB_INDICES[inventory.sensor]), "roi_margin": 8},
        "warp": "bicubic_border_FP32_align_corners_false_current_pan_px",
        "band30_teacher": teacher_binding(cfg),
        "band30_teacher_effective_base": upsampler_manifest(cfg["upsampler"]),
        "band30_reconstruction_spec": fresh.stage.teacher.spec.manifest()}
    metadata = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": expected_sha256,
        "run_id": cfg["run_id"], "completed_update": update, "same_checkpoint_A_T": True,
        "donor_model_state_sha256": payload["state_hash"],
        "source_pipeline": deepcopy(fresh.stage.pipeline_manifest()),
        "checkpoint_identity": deepcopy(identity), "teacher_constructed": True,
        "teacher_retained": include_teacher, "selection": "EXPLICIT_CALLER_PATH_AND_SHA",
        "band30_teacher": teacher_binding(cfg)}
    result = LoadedDonor(aligner, teacher, binding, metadata)
    result.assert_frozen()
    registered.verify()
    return result

