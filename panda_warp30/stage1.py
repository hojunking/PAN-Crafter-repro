"""Fresh matched Stage 1 donors with explicit effective interpolation.

Networks, MASK2, absolute Scharr and epsilon objectives reuse existing code.
Only the reconstruction base changes. Checkpoints have a new outer schema and
pin effective interpolation, so neither an old donor nor the other arm can be
silently resumed. No shared model classes or module globals are modified.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from directshift48.training import DirectStage1Trainer, build_direct_stage1
from donor30.artifacts import _cpu, _same, _save_immutable, capture_rng, restore_rng
from donor30.evaluation import model_state_hash
from donor30.masking import MaskedDonorStage1
from panda40.stage1 import (
    Stage1Stream, advance_stage1, _check_evaluation, _identity, _optimizer_contract,
    _pins, _scheduler, _stream_progress, _validate_config as _validate_shared_config,
)
from panda_method.data import require_sha256
from panda_method.reconstruction import ReconstructionSpec
from pa.warp import warp_pan
from ta2.evaluation import file_sha

from .contract import CAMPAIGN_ID, METHOD_REVISION
from .reconstruction import base_from_batch, reconstruct_with_base, upsampler_manifest


CHECKPOINT_SCHEMA = "PANDA_WARP30_STAGE1_CHECKPOINT_V1"
RESUME_SCHEMA = "PANDA_WARP30_STAGE1_COMPLETE_RESUME_V1"


def teacher_binding(upsampler):
    upsampler_manifest(upsampler)
    return {"upsampler": upsampler, "campaign_id": CAMPAIGN_ID}


class WarpStage1(MaskedDonorStage1):
    def __init__(self, aligner, teacher, *, upsampler, **kwargs):
        upsampler_manifest(upsampler)
        super().__init__(aligner, teacher, **kwargs)
        self._upsampler = upsampler

    @property
    def upsampler(self):
        return self._upsampler

    def pipeline_manifest(self):
        result = super().pipeline_manifest()
        result.update(revision=METHOD_REVISION,
                      warp30_teacher=teacher_binding(self.upsampler),
                      effective_reconstruction_base=upsampler_manifest(self.upsampler),
                      architecture_base_reference_is_compatibility_descriptor=True)
        return result

    def alignment_manifest(self):
        return {**super().alignment_manifest(),
                "warp30_teacher": teacher_binding(self.upsampler),
                "effective_reconstruction_base": upsampler_manifest(self.upsampler)}

    def native_prediction(self, batch, *, return_aligner_details=False):
        self._check_observation(batch)
        training_rr = batch.lineage.resolution == "RR" and batch.lineage.split == "train"
        with torch.set_grad_enabled(torch.is_grad_enabled() and training_rr), \
                torch.autocast(device_type=batch.pan.device.type, enabled=False):
            shift = self.predict_shift(batch.pan, batch.native_lms,
                resolution=batch.lineage.resolution, return_aligner_details=return_aligner_details)
            corrected = warp_pan(batch.pan.float(), shift["u"])
            predicted = reconstruct_with_base(self.teacher, corrected,
                                              base_from_batch(batch, self.upsampler), batch.ms)
        return {**predicted, **shift}


def validate_config(cfg, sources):
    _validate_shared_config(cfg, sources)
    if (cfg.get("campaign_id") != CAMPAIGN_ID or cfg.get("method_revision") != METHOD_REVISION
            or cfg.get("server") != "s3" or cfg.get("dataset") != "WV3"
            or cfg.get("aligner_kind") != "direct"
            or cfg.get("lambda_abs") != 2. or cfg.get("lambda_epsilon") != .003
            or cfg.get("lr_A") != 1e-5 or cfg.get("lr_U") != 1e-4
            or cfg.get("initialization") != "FRESH" or cfg.get("start_update", 0) != 0):
        raise ValueError("Fresh s3/WV3 warp30 matched donor configuration required")
    upsampler_manifest(cfg.get("upsampler"))


def build_stage1(cfg, sources, *, device="cpu"):
    validate_config(cfg, sources)
    b, seeds = cfg["baseline_resolved"], cfg["stream_seeds"]
    fresh = build_direct_stage1(sources,
        teacher_spec=ReconstructionSpec(**b["teacher_architecture"],
            input_layout="P0", base_reference="BICUBIC_MS"),
        teacher_seed=seeds["T_init"], aligner_seed=seeds["A_init"], epsilon_seed=seeds["rr_epsilon"],
        aligner_view_margin=8, roi_margin=8, boundary_policy="require_spatial_support",
        lambda_abs=cfg["lambda_abs"], lambda_eps=cfg["lambda_epsilon"])
    stage = WarpStage1(fresh.aligner, fresh.teacher, upsampler=cfg["upsampler"],
        sources=fresh.sources, epsilon_seed=fresh.epsilon_seed, roi_margin=fresh.roi_margin,
        boundary_policy=fresh.boundary_policy, lambda_abs=fresh.lambda_abs,
        lambda_eps=fresh.lambda_eps).to(device)
    optimizer = torch.optim.AdamW([
        {"params": stage.aligner.parameters(), "lr": cfg["lr_A"], "role": "A"},
        {"params": stage.teacher.parameters(), "lr": cfg["lr_U"], "role": "T"}],
        betas=tuple(b["optimizer_betas"]), eps=b["optimizer_eps"], weight_decay=cfg["weight_decay"])
    trainer = DirectStage1Trainer(stage, optimizer,
        gradient_log_interval=b["gradient_log_interval"], effective_batch_size=b["effective_batch_RR"],
        microbatch_size=b["microbatch_RR"])
    trainer.config = deepcopy(cfg)
    trainer._warp30_sources = sources
    return trainer


def save_stage1(path, trainer, stream, *, code_pin, evaluator_pin, evaluation=None):
    """Immutable complete checkpoint; independently strict-reloaded before save."""
    if (not isinstance(stream, Stage1Stream) or not isinstance(trainer, DirectStage1Trainer)
            or not isinstance(trainer.stage, WarpStage1)):
        raise TypeError("Warp30 Stage 1 trainer and complete RR stream required")
    cfg = trainer.config
    validate_config(cfg, stream.sources)
    if not _same(cfg, stream.config) or trainer.stage.upsampler != cfg["upsampler"]:
        raise ValueError("Effective upsampler/trainer/stream configuration mismatch")
    if trainer.poisoned or not 0 <= trainer.updates <= cfg["planned_horizon"]:
        raise ValueError("Cannot checkpoint a failed or out-of-schedule trainer")
    pins = _pins(code_pin, evaluator_pin)
    destination = Path(path).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Immutable Stage 1 checkpoint already exists")
    if destination.parent.is_symlink():
        raise ValueError("Checkpoint directory may not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = capture_rng()
    try:
        state_hash = model_state_hash(trainer.stage)
        identity = _identity(cfg, trainer.updates, state_hash)
        _check_evaluation(evaluation, identity)
        state = _cpu(trainer.state_dict())
        _optimizer_contract(state, cfg, trainer.updates)
        _stream_progress(stream.state_dict(), state, cfg)
        payload = {"schema": CHECKPOINT_SCHEMA, "campaign_resume_schema": RESUME_SCHEMA,
            "campaign_resume_complete": True, "config": deepcopy(cfg), "identity": identity,
            "state_hash": state_hash, "model_state_sha256": state_hash,
            "trainer": state, "stream": _cpu(stream.state_dict()), "stream_protocol": stream.protocol(),
            "rng": _cpu(rng), "scheduler": _scheduler(cfg, trainer.updates), "pins": pins,
            "evaluation": deepcopy(evaluation), "resume_only": evaluation is None,
            "warp30_teacher": teacher_binding(cfg["upsampler"]),
            "selection_rule": "CALLER_MAX_HQNR_THEN_MIN_ERGAS_NO_SCC"}
        reloaded = build_stage1(cfg, stream.sources)
        reloaded.load_state_dict(state)
        restored_stream = Stage1Stream(stream.sources, cfg, device="cpu")
        restored_stream.load_state_dict(payload["stream"])
        if (model_state_hash(reloaded.stage) != state_hash
                or not _same(_cpu(reloaded.state_dict()), state)
                or not _same(restored_stream.state_dict(), payload["stream"])):
            raise ValueError("Fresh model/optimizer/sampler strict reload differs")
        _save_immutable(destination, payload)
        saved = torch.load(destination, map_location="cpu", weights_only=False)
        if (not _same(saved, payload) or model_state_hash(trainer.stage) != state_hash
                or not _same(stream.state_dict(), payload["stream"])
                or not _same(_cpu(trainer.state_dict()), state)):
            raise ValueError("Serialized or paused live trainer changed during checkpoint save")
        return {"checkpoint_path": str(destination), "file_sha256": file_sha(destination),
            "model_state_sha256": state_hash, "state_hash": state_hash, "identity": identity,
            "completed_update": trainer.updates, "weights_saved": True, "checkpoint_verified": True,
            "predictions_reverified": False, "resume_only": evaluation is None,
            "campaign_resume_complete": True,
            "verification": "STRICT_CPU_RELOAD_MODEL_OPTIMIZER_STREAM_AND_FILE_ROUNDTRIP"}
    finally:
        restore_rng(rng)


def load_stage1(path, expected_sha256, cfg, sources, *, device="cpu", code_pin,
                evaluator_pin, restore_global_rng=True):
    """Resume the identical run/upsampler/source/code/evaluator only."""
    require_sha256(expected_sha256, "explicit local Stage 1 checkpoint SHA256")
    validate_config(cfg, sources)
    pins = _pins(code_pin, evaluator_pin)
    checkpoint = Path(path).resolve(strict=True)
    if not checkpoint.is_file() or file_sha(checkpoint) != expected_sha256:
        raise ValueError("Stage 1 checkpoint SHA mismatch")
    before_stat = checkpoint.stat()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.stat() != before_stat or file_sha(checkpoint) != expected_sha256:
        raise ValueError("Checkpoint file changed while loading")
    if (not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("campaign_resume_schema") != RESUME_SCHEMA
            or payload.get("campaign_resume_complete") is not True
            or not _same(payload.get("config"), cfg) or payload.get("pins") != pins
            or payload.get("warp30_teacher") != teacher_binding(cfg["upsampler"])):
        raise ValueError("Warp30 checkpoint/config/code/evaluator/upsampler resume mismatch")
    update = payload.get("trainer", {}).get("updates")
    if (type(update) is not int or not 0 <= update <= cfg["planned_horizon"]
            or payload.get("scheduler") != _scheduler(cfg, update)):
        raise ValueError("Checkpoint completed update/scheduler contract mismatch")
    _optimizer_contract(payload["trainer"], cfg, update)
    identity = _identity(cfg, update, payload.get("state_hash"))
    if (payload.get("identity") != identity
            or payload.get("model_state_sha256") != payload.get("state_hash")
            or payload.get("resume_only") != (payload.get("evaluation") is None)):
        raise ValueError("Checkpoint model/evaluation identity mismatch")
    require_sha256(payload.get("state_hash"), "checkpoint A/T state SHA256")
    _check_evaluation(payload.get("evaluation"), identity)
    rng, success = capture_rng(), False
    try:
        stream = Stage1Stream(sources, cfg, device=device)
        if stream.protocol() != payload.get("stream_protocol"):
            raise ValueError("Training source/sampler protocol changed since checkpoint")
        stream.load_state_dict(payload["stream"])
        _stream_progress(payload["stream"], payload["trainer"], cfg)
        trainer = build_stage1(cfg, sources, device=device)
        trainer.load_state_dict(payload["trainer"])
        if (model_state_hash(trainer.stage) != payload["state_hash"]
                or not _same(_cpu(trainer.state_dict()), payload["trainer"])):
            raise ValueError("Loaded model/optimizer tensor state differs from checkpoint")
        restore_rng(payload["rng"])
        success = True
        return trainer, stream, payload
    finally:
        if not success or not restore_global_rng:
            restore_rng(rng)
