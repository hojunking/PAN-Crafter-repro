"""Panda30R source import, including minimal-identity legacy resume checkpoints.

Checkpoint files are trusted local experiment artifacts (torch pickle includes
RNG state). Callers must supply the file SHA; untrusted downloads are not inputs.
No legacy loss/calibration, optimizer or RNG state is resumed into Stage 2.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch

from directshift48.method import DirectPANAligner, METHOD_VERSION, RGB_INDICES
from donor30.artifacts import SCHEMA as DONOR_CHECKPOINT_SCHEMA
from panda_method.calibration import canonical_sha256, state_dict_sha256
from panda_method.data import require_sha256
from panda_method.edge_data import EdgeDataSources
from panda_method.reconstruction import ReconstructionSpec, build_reconstructor
from ta2.evaluation import file_sha

from panda_stage2_condq.sources import RegisteredRRTrain
from panda_stage2_condq.donor import LoadedDonor


def _role_state(state, prefix):
    result = OrderedDict((k[len(prefix):], v) for k, v in state.items() if k.startswith(prefix))
    if not result:
        raise ValueError("Checkpoint is missing paired donor role " + prefix)
    for value in result.values():
        if not isinstance(value, torch.Tensor) or (value.is_floating_point() and
                (value.dtype != torch.float32 or not bool(torch.isfinite(value).all()))):
            raise ValueError("Donor weights must be finite FP32 dense tensors")
    return result


def load_donor(checkpoint_path, expected_sha256, sources, *, include_teacher=True, device="cpu"):
    """Strict-load native A and optionally T from one SHA-pinned saved checkpoint.

    ``include_teacher=False`` verifies the saved Teacher tensor hash without
    constructing or executing a Teacher module (the GT-only variant contract).
    """
    require_sha256(expected_sha256, "explicit donor checkpoint SHA256")
    if type(include_teacher) is not bool:
        raise ValueError("Explicit include_teacher boolean required")
    registered = RegisteredRRTrain(sources) if isinstance(sources, EdgeDataSources) else sources
    if not isinstance(registered, RegisteredRRTrain):
        raise TypeError("Pinned RegisteredRRTrain or EdgeDataSources required")
    inventory = registered.verify()
    path = Path(checkpoint_path).resolve(strict=True)
    if not path.is_file() or file_sha(path) != expected_sha256:
        raise ValueError("Donor checkpoint SHA mismatch")
    stat_before = path.stat()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if path.stat() != stat_before or file_sha(path) != expected_sha256:
        raise ValueError("Donor checkpoint changed while loading")
    if not isinstance(payload, dict) or payload.get("schema") != DONOR_CHECKPOINT_SCHEMA:
        raise ValueError("Saved local Stage 1 donor checkpoint schema required")
    cfg, trainer = payload.get("config", {}), payload.get("trainer", {})
    if (cfg.get("phase") != "donor" or cfg.get("dataset") != inventory.sensor
            or cfg.get("stage2") is not False
            or trainer.get("schema") != "PANDA_DIRECTSHIFT_STAGE1_TRAINER_V1"):
        raise ValueError("Only same-sensor Stage 1 donor checkpoints are accepted")
    step = trainer.get("updates")
    if type(step) is not int or step < 1 or not isinstance(cfg.get("run_id"), str) or not cfg["run_id"]:
        raise ValueError("Completed donor step and explicit run ID required")
    bands = registered.sources.bands
    spec = ReconstructionSpec(bands, 112, (1, 2, 3), "P0", "BICUBIC_MS")
    baseline = cfg.get("baseline_resolved", {})
    if (baseline.get("teacher_architecture") != {"bands": bands, "width": 112, "depth": [1, 2, 3]}
            or baseline.get("teacher_input_layout") != "P0"
            or baseline.get("teacher_base") != "BICUBIC_MS"):
        raise ValueError("Stage 2 requires donor W112-D123/P0/BICUBIC_MS")
    state = trainer.get("stage", {})
    if not isinstance(state, dict) or any(k != "_extra_state" and not k.startswith(("aligner.", "teacher.")) for k in state):
        raise ValueError("Mixed Stage 2/unknown model state is not a donor")
    extra = state.get("_extra_state", {})
    pipeline = extra.get("pipeline", {})
    if (extra.get("method") != METHOD_VERSION
            or extra.get("sources") != {"sensor": inventory.sensor,
                "rr_manifest_sha256": inventory.source_manifest_sha256}
            or canonical_sha256(pipeline.get("teacher")) != canonical_sha256(spec.manifest())
            or pipeline.get("roi_margin") != 8
            or pipeline.get("boundary_policy") != "require_spatial_support"
            or pipeline.get("reconstruction_warp") != "single_original_PAN_warp_by_u"
            or pipeline.get("warp") != "bicubic_border_align_corners_false_FP32"
            or pipeline.get("lms_reference") != "provided_stored_LMS_not_bicubic_base"):
        raise ValueError("Donor native source/model/geometry metadata mismatch")
    a_state, t_state = _role_state(state, "aligner."), _role_state(state, "teacher.")
    hashes = {"A": state_dict_sha256(a_state), "T": state_dict_sha256(t_state)}
    joint_hash = canonical_sha256(hashes)
    identity = payload.get("identity", {})
    # donor24's safe-stop payload deliberately omits dataset from its minimal
    # identity. Never relax a conflicting declared dataset. The full cfg and
    # _extra_state.sources sensor/manifest were independently checked above.
    legacy_resume = ("dataset" not in identity and payload.get("resume_only") is True
                     and payload.get("evaluation") is None)
    if (payload.get("state_hash") != joint_hash
            or identity.get("model_state_sha256") != joint_hash
            or identity.get("completed_update") != step
            or identity.get("run_id") != cfg["run_id"]
            or (identity.get("dataset") != inventory.sensor and not legacy_resume)):
        raise ValueError("Actual A/T tensors disagree with paired checkpoint identity")
    # Preserve caller RNG: import must not perturb the paired Student seed/order.
    with torch.random.fork_rng(devices=[]):
        aligner = DirectPANAligner(bands, view_margin=8)
        if pipeline.get("aligner") != aligner.manifest():
            raise ValueError("Donor Aligner architecture/coordinate convention mismatch")
        aligner.load_state_dict(a_state, strict=True)
        teacher = build_reconstructor(spec, seed=0, role="T") if include_teacher else None
        if teacher is not None:
            teacher.load_state_dict(t_state, strict=True)
    for module in (aligner, teacher):
        if module is not None:
            module.requires_grad_(False).eval().to(device)
    binding = {"sensor": inventory.sensor, "source_split": "RR_train_only",
        "sample_ids_sha256": inventory.sample_ids_sha256,
        "sample_count": len(inventory.sample_ids),
        "source_manifest_sha256": inventory.source_manifest_sha256,
        "catalog_sha256": inventory.catalog_sha256, "source_h5_sha256": inventory.source_h5_sha256,
        "aligner_state_sha256": hashes["A"], "teacher_state_sha256": hashes["T"],
        "donor_checkpoint_id": expected_sha256, "donor_checkpoint_sha256": expected_sha256,
        "donor_run_id": cfg["run_id"], "donor_step": step,
        "donor_source_mask_policy": pipeline.get("rec_reduction"),
        "reconstruction_spec": {"bands": bands, "width": 112, "depth": [1, 2, 3],
                                "input_layout": "P0", "base_reference": "BICUBIC_MS"},
        "scharr": {"kernel": "absolute_xy_/32", "padding": "reflect1",
                   "rgb_indices": list(RGB_INDICES[inventory.sensor]), "roi_margin": 8},
        "warp": "bicubic_border_FP32_align_corners_false_current_pan_px"}
    metadata = {"checkpoint_path": str(path), "checkpoint_sha256": expected_sha256,
                "run_id": cfg["run_id"], "completed_update": step,
                "same_checkpoint_A_T": True, "donor_model_state_sha256": joint_hash,
                "source_pipeline": deepcopy(pipeline), "checkpoint_identity": deepcopy(identity),
                "source_identity_schema": "VERIFIED_MINIMAL_RESUME" if legacy_resume else "DATASET_EXPLICIT",
                "teacher_constructed": include_teacher, "selection": "EXPLICIT_CALLER_PATH_AND_SHA"}
    result = LoadedDonor(aligner, teacher, binding, metadata)
    result.assert_frozen()
    registered.verify()
    return result
