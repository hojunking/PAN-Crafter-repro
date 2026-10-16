"""Explicit S2D Stage 1 A/T export and frozen Conditional-q Stage 2 import.

These SHA-pinned torch files are trusted local artifacts, not untrusted input.
The export is a paired-model handoff and trainer audit, NOT a complete campaign
resume: data stream, scheduler, external RNG and evaluation selection are absent.
No exported file is implicitly selected or labelled a qualified/best donor.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import os
import tempfile

import torch

from directshift48.method import RGB_INDICES
from directshift48.training import DirectStage1Trainer
from donor30.plan import LOSS_MASK_POLICY
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_method.data import require_sha256
from panda_method.reconstruction import ReconstructionSpec, build_reconstructor
from panda_stage2_condq.donor import LoadedDonor
from panda_stage2_condq.sources import RegisteredRRTrain
from ta2.evaluation import file_sha

from .aligner import METHOD_VERSION, S2DGlobalPANAligner
from .stage1 import (MASKED_METHOD_REVISION, METHOD_REVISION,
                     S2DMaskedStage1, S2DShiftStage1)


SCHEMA = "PANDA_S2D_STAGE1_PAIRED_DONOR_V1"
TRAINER_SCHEMA = "PANDA_DIRECTSHIFT_STAGE1_TRAINER_V1"
PURPOSES = ("IMPLEMENTATION_DIAGNOSTIC", "EXPERIMENT_CANDIDATE")


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        result = value.copy()
        for key, item in value.items():
            result[key] = _cpu(item)
        if hasattr(value, "_metadata"):
            result._metadata = deepcopy(value._metadata)
        return result
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return deepcopy(value)


def _implementation_pin():
    directory = Path(__file__).resolve().parent
    return {name: file_sha(directory / name)
            for name in ("aligner.py", "stage1.py", "donor.py")}


def _roles(state):
    if (not isinstance(state, dict) or any(not isinstance(k, str) or
            (k != "_extra_state" and not k.startswith(("aligner.", "teacher."))) for k in state)):
        raise ValueError("Only the paired S2D Stage 1 Aligner/Teacher state is a donor")
    roles = []
    for prefix in ("aligner.", "teacher."):
        role = OrderedDict((k[len(prefix):], v) for k, v in state.items() if k.startswith(prefix))
        if not role:
            raise ValueError("Missing paired donor role " + prefix)
        if any(not isinstance(t, torch.Tensor) or t.layout != torch.strided or
               (t.is_floating_point() and (t.dtype != torch.float32 or not bool(torch.isfinite(t).all())))
               for t in role.values()):
            raise ValueError("Paired donor weights must be finite dense FP32 tensors")
        roles.append(role)
    return roles


def _validate_payload(payload, registered, aligner):
    """Verify architecture/source/tensor identities before touching a model."""
    inventory = registered.verify()
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("New S2D Stage 1 paired donor schema required; old donor30/Stage2 forbidden")
    if (payload.get("purpose") not in PURPOSES or payload.get("campaign_resume_complete") is not False
            or payload.get("qualified_donor") is not False
            or payload.get("implementation_pin") != _implementation_pin()):
        raise ValueError("S2D donor implementation/purpose/resume metadata mismatch")
    if payload.get("source_inventory") != inventory.manifest():
        raise ValueError("S2D donor train inventory/catalog/H5 fingerprint mismatch")
    trainer = payload.get("trainer", {})
    if not isinstance(trainer, dict) or trainer.get("schema") != TRAINER_SCHEMA:
        raise ValueError("S2D Stage 1 trainer audit required")
    step = trainer.get("updates")
    run_id = payload.get("run_id")
    if type(step) is not int or step < 1 or not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("At least one completed donor update and explicit run ID required")
    state = trainer.get("stage")
    a_state, t_state = _roles(state)
    extra = state.get("_extra_state", {})
    if not isinstance(extra, dict):
        raise ValueError("S2D donor pipeline metadata required")
    pipeline = extra.get("pipeline", {})
    if not isinstance(pipeline, dict):
        raise ValueError("S2D donor pipeline metadata required")
    spec = ReconstructionSpec(registered.sources.bands, 112, (1, 2, 3), "P0", "BICUBIC_MS")
    policy = pipeline.get("mask_policy")
    expected_mask = {"UNMASKED": (METHOD_REVISION, "all_pixels_bands_mean"),
                     "MASK2": (MASKED_METHOD_REVISION, LOSS_MASK_POLICY)}
    if policy not in expected_mask:
        raise ValueError("Explicit S2D donor reconstruction mask policy required")
    revision, reduction = expected_mask[policy]
    if (extra.get("method") != METHOD_VERSION
            or extra.get("sources") != {"sensor": inventory.sensor,
                "rr_manifest_sha256": inventory.source_manifest_sha256}
            or pipeline.get("method") != METHOD_VERSION or pipeline.get("revision") != revision
            or pipeline.get("aligner") != aligner.manifest()
            or canonical_sha256(pipeline.get("teacher")) != canonical_sha256(spec.manifest())
            or pipeline.get("roi_margin") != 8
            or pipeline.get("boundary_policy") != "require_spatial_support"
            or pipeline.get("reconstruction_warp") != "single_original_PAN_warp_by_u"
            or pipeline.get("warp") != "bicubic_border_align_corners_false_FP32"
            or pipeline.get("inference_units") != "current_PAN_px_no_x4_RR_or_FR"
            or pipeline.get("lms_reference") != "provided_stored_LMS_not_bicubic_base"
            or pipeline.get("rec_reduction") != reduction):
        raise ValueError("S2D donor source/architecture/native geometry/mask contract mismatch")
    hashes = {"A": state_dict_sha256(a_state), "T": state_dict_sha256(t_state)}
    identity = {"run_id": run_id, "sensor": inventory.sensor, "completed_update": step,
                "model_state_sha256": canonical_sha256(hashes),
                "aligner_state_sha256": hashes["A"], "teacher_state_sha256": hashes["T"],
                "method": METHOD_VERSION, "pipeline_sha256": canonical_sha256(pipeline),
                "source_inventory_sha256": canonical_sha256(inventory.manifest())}
    if payload.get("identity") != identity:
        raise ValueError("Actual A/T tensors or source/pipeline disagree with paired checkpoint identity")
    return a_state, t_state, pipeline, identity, spec


def save_donor_checkpoint(path, trainer, registered, *, run_id,
                          purpose="IMPLEMENTATION_DIAGNOSTIC"):
    """Publish one immutable S2D paired donor after an actual completed update.

    The inherited trainer state is retained for audit, but does not include the
    caller's scheduler/stream/global RNG or official checkpoint selection proof.
    This function neither resumes a campaign nor asserts donor qualification.
    """
    if not isinstance(registered, RegisteredRRTrain):
        raise TypeError("Pinned RegisteredRRTrain required")
    if (not isinstance(trainer, DirectStage1Trainer)
            or type(trainer.stage) not in (S2DShiftStage1, S2DMaskedStage1)
            or type(trainer.stage.aligner) is not S2DGlobalPANAligner):
        raise TypeError("Fresh S2D Stage 1 paired trainer required; no legacy donor conversion")
    if (trainer.poisoned or type(trainer.updates) is not int or trainer.updates < 1
            or trainer.stage.forward_calls < 1):
        raise ValueError("At least one completed, unpoisoned S2D donor update required")
    if purpose not in PURPOSES:
        raise ValueError("Explicit diagnostic or unqualified experiment-candidate purpose required")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Explicit nonempty donor run ID required")
    inventory = registered.verify()
    trainer_state = _cpu(trainer.state_dict())
    a_state, t_state = _roles(trainer_state["stage"])
    pipeline = trainer_state["stage"]["_extra_state"]["pipeline"]
    hashes = {"A": state_dict_sha256(a_state), "T": state_dict_sha256(t_state)}
    identity = {"run_id": run_id, "sensor": inventory.sensor, "completed_update": trainer.updates,
                "model_state_sha256": canonical_sha256(hashes),
                "aligner_state_sha256": hashes["A"], "teacher_state_sha256": hashes["T"],
                "method": METHOD_VERSION, "pipeline_sha256": canonical_sha256(pipeline),
                "source_inventory_sha256": canonical_sha256(inventory.manifest())}
    payload = {"schema": SCHEMA, "run_id": run_id, "purpose": purpose,
               "campaign_resume_complete": False, "qualified_donor": False,
               "resume_limitation": "scheduler_stream_global_RNG_and_official_selection_not_exported",
               "implementation_pin": _implementation_pin(), "source_inventory": inventory.manifest(),
               "identity": identity, "trainer": trainer_state}
    _validate_payload(payload, registered, trainer.stage.aligner)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Immutable donor checkpoint already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".s2d-donor-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        serialized = torch.load(temporary, map_location="cpu", weights_only=False)
        _validate_payload(serialized, registered, trainer.stage.aligner)
        if (trainer.updates != identity["completed_update"] or trainer.poisoned
                or state_dict_sha256(trainer.stage.aligner) != hashes["A"]
                or state_dict_sha256(trainer.stage.teacher) != hashes["T"]):
            raise ValueError("Live donor changed during export; no checkpoint published")
        # link fails if a peer created destination after our initial check.
        os.link(temporary, destination)
        directory_fd = os.open(str(destination.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(destination.resolve()), "sha256": file_sha(destination),
            "identity": identity, "purpose": purpose, "qualified_donor": False,
            "campaign_resume_complete": False}


def load_donor(path, expected_sha256, registered, *, include_teacher=True, device="cpu"):
    """Load same-checkpoint frozen S2D A/T; GT-only never constructs Teacher."""
    require_sha256(expected_sha256, "explicit S2D donor checkpoint SHA256")
    if not isinstance(registered, RegisteredRRTrain):
        raise TypeError("Pinned RegisteredRRTrain required")
    if type(include_teacher) is not bool:
        raise ValueError("Explicit include_teacher boolean required")
    inventory = registered.verify()
    checkpoint = Path(path).resolve(strict=True)
    if not checkpoint.is_file() or file_sha(checkpoint) != expected_sha256:
        raise ValueError("S2D donor checkpoint SHA mismatch")
    before = checkpoint.stat()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.stat() != before or file_sha(checkpoint) != expected_sha256:
        raise ValueError("S2D donor changed while loading")
    with torch.random.fork_rng(devices=[]):
        aligner = S2DGlobalPANAligner(registered.sources.bands, view_margin=8)
        a_state, t_state, pipeline, identity, spec = _validate_payload(payload, registered, aligner)
        aligner.load_state_dict(a_state, strict=True)
        teacher = build_reconstructor(spec, seed=0, role="T") if include_teacher else None
        if teacher is not None:
            teacher.load_state_dict(t_state, strict=True)
    for module in (aligner, teacher):
        if module is not None:
            module.requires_grad_(False).eval().to(device)
    binding = {"sensor": inventory.sensor, "source_split": "RR_train_only",
        "sample_ids_sha256": inventory.sample_ids_sha256, "sample_count": len(inventory.sample_ids),
        "source_manifest_sha256": inventory.source_manifest_sha256,
        "catalog_sha256": inventory.catalog_sha256, "source_h5_sha256": inventory.source_h5_sha256,
        "aligner_state_sha256": identity["aligner_state_sha256"],
        "teacher_state_sha256": identity["teacher_state_sha256"],
        "donor_checkpoint_id": expected_sha256, "donor_checkpoint_sha256": expected_sha256,
        "donor_run_id": identity["run_id"], "donor_step": identity["completed_update"],
        "donor_method": METHOD_VERSION, "donor_source_mask_policy": pipeline["rec_reduction"],
        "donor_pipeline_sha256": identity["pipeline_sha256"],
        "aligner_manifest": deepcopy(aligner.manifest()),
        "aligner_manifest_sha256": canonical_sha256(aligner.manifest()),
        "donor_implementation_pin": deepcopy(payload["implementation_pin"]),
        "reconstruction_spec": {"bands": spec.bands, "width": 112, "depth": [1, 2, 3],
                                "input_layout": "P0", "base_reference": "BICUBIC_MS"},
        "scharr": {"kernel": "absolute_xy_/32", "padding": "reflect1",
                   "rgb_indices": list(RGB_INDICES[inventory.sensor]), "roi_margin": 8},
        "warp": "bicubic_border_FP32_align_corners_false_current_pan_px"}
    metadata = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": expected_sha256,
                "run_id": identity["run_id"], "completed_update": identity["completed_update"],
                "same_checkpoint_A_T": True, "donor_model_state_sha256": identity["model_state_sha256"],
                "source_pipeline": deepcopy(pipeline), "checkpoint_identity": identity,
                "teacher_constructed": include_teacher, "selection": "EXPLICIT_CALLER_PATH_AND_SHA",
                "purpose": payload["purpose"], "qualified_donor": False,
                "campaign_resume_complete": False, "schema": SCHEMA}
    result = LoadedDonor(aligner, teacher, binding, metadata)
    result.assert_frozen()
    registered.verify()
    return result
