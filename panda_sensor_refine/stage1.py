"""Fresh RGB-reference Stage1 with unified Lanczos4 and complete resume."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from directshift48.method import DirectTrainingSources
import math
from directshift48.training import DirectStage1Trainer
from donor30.artifacts import _cpu, _same, _save_immutable, capture_rng, restore_rng
from donor30.evaluation import model_state_hash
from donor30.masking import MaskedDonorStage1, masked_mean
from panda40.stage1 import (
    Stage1Stream, advance_stage1, _check_evaluation, _identity, _optimizer_contract,
    _pins, _scheduler, _stream_progress, _validate_config as _validate_shared_config,
)
from panda_method.data import require_sha256
from panda_method.reconstruction import ReconstructionSpec
from .warp import warp_pan, warp_support_mask, warp_manifest, boundary_manifest, mask_manifest
from ta2.evaluation import file_sha

from .contract import CAMPAIGN_ID, METHOD_REVISION, file_identity, LANES, validate_loss_pair_config
from panda_sensor_refine.aligner import (BandPANAligner, RGBScharr, GLOBAL_TARGET,
                      aligned_pan_views, support_for_shifts)
from .base_reconstruction import (base_from_batch, reconstruct_with_base,
                             upsampler_manifest, build_reconstructor)

CHECKPOINT_SCHEMA = "PANDA_SENSOR_REFINE_STAGE1_CHECKPOINT_V1"
RESUME_SCHEMA = "PANDA_SENSOR_REFINE_STAGE1_COMPLETE_RESUME_V1"
AUX_POLICY = "HR_LMS_GT_ALIGNER_AUX_C1P5_R0P5"


def sample_epsilon(batch, generator, radius):
    """Area-uniform disk with unchanged dedicated CPU random stream."""
    if (type(batch) is not int or batch <= 0 or radius not in (3., 4.)
            or not isinstance(generator,torch.Generator) or generator.device.type != 'cpu'):
        raise ValueError('Positive batch and registered R3/R4 radius required')
    draws = torch.rand(batch, 2, generator=generator, dtype=torch.float32)
    length, angle = radius * draws[:, 0].sqrt(), 2 * math.pi * draws[:, 1]
    return torch.stack((length * angle.sin(), length * angle.cos()), dim=1)


def teacher_binding(cfg):
    upsampler_manifest(cfg["upsampler"])
    return {key: cfg[key] for key in
            ("campaign_id", "upsampler", "aligner_outputs", "alignment_target", "pan_channels", "lambda_abs", "lambda_epsilon", "epsilon_radius_rr")}


class BandStage1(MaskedDonorStage1):
    def __init__(self, aligner, teacher, *, upsampler, band_augmentation_seed, epsilon_radius_rr, **kwargs):
        upsampler_manifest(upsampler)
        if epsilon_radius_rr not in (3.,4.):
            raise ValueError('Registered R3/R4 epsilon radius required')
        self.epsilon_radius_rr = float(epsilon_radius_rr)
        if not isinstance(aligner, BandPANAligner) or teacher.pan_channels != aligner.outputs // 2:
            raise ValueError("Matched campaign Aligner/output/PAN stem required")
        if type(band_augmentation_seed) is not int:
            raise ValueError("Explicit independent auxiliary RNG seed required")
        super().__init__(aligner, teacher, **kwargs)
        self._upsampler = upsampler
        self.spatial = RGBScharr(aligner.outputs, sensor=self.sources.sensor)
        self.band_augmentation_seed = band_augmentation_seed
        self._band_rng = torch.Generator(device="cpu").manual_seed(band_augmentation_seed)
        self._auxiliary_records = []

    @property
    def upsampler(self):
        return self._upsampler

    @property
    def bandwise(self):
        return False

    def teacher_binding(self):
        return {"campaign_id": CAMPAIGN_ID, "upsampler": self.upsampler,
                "aligner_outputs": self.aligner.outputs, "pan_channels": self.teacher.pan_channels,
                "alignment_target": GLOBAL_TARGET,
                "lambda_abs":self.lambda_abs, "lambda_epsilon":self.lambda_eps,
                "epsilon_radius_rr":self.epsilon_radius_rr}

    def pipeline_manifest(self):
        result = super().pipeline_manifest()
        result.update(method=METHOD_REVISION, revision=METHOD_REVISION, mature30_teacher=self.teacher_binding(),
                      alignment="prior_free_direct_total_global_translation",
                      effective_reconstruction_base=upsampler_manifest(self.upsampler),
                      teacher=self.teacher.spec.manifest(),
                      reconstruction_warp="one_original_PAN_warp_per_declared_PAN_view",
                      loss_mask_intersection="single_PAN",
                      auxiliary=self.auxiliary_manifest())
        result.update(warp=warp_manifest(), boundary=boundary_manifest(self.epsilon_radius_rr),
                      rec_reduction=mask_manifest()['policy'], loss_mask=mask_manifest())
        result['boundary'].update(prediction_dependent_mask=True,
            prediction_dependent_mask_scope='reconstruction_only_spatial_ROI8_unchanged')
        result['aligner_checkpoint_manifest_scope'] = ('UNCHANGED_LEGACY_ARCHITECTURE_IDENTITY_ONLY; '
            'effective_epsilon_and_warp_are_pipeline_eps_and_boundary')
        result['eps'].update(warp=warp_manifest(), radius=self.epsilon_radius_rr,
                             historical_radius_basis='REGISTERED_R3_CONTROL_R4_SENSOR_ABLATION')
        return result

    def auxiliary_manifest(self):
        return {"policy": "NONE"}

    def alignment_manifest(self):
        return {**super().alignment_manifest(), "method": METHOD_REVISION,
                "epsilon": {**super().alignment_manifest()["epsilon"],
                            "radius": self.epsilon_radius_rr, "warp": warp_manifest()},
                "prediction": "prior_free_direct_total_dy_dx",
                "head_dims": [64, 32, self.aligner.outputs],
                "native_warp": "one_original_PAN_warp_per_declared_PAN_view",
                "mature30_teacher": self.teacher_binding(),
                "auxiliary": self.auxiliary_manifest(),
                "effective_reconstruction_base": upsampler_manifest(self.upsampler),
                "warp": warp_manifest(), "boundary": boundary_manifest(self.epsilon_radius_rr)}

    def get_extra_state(self):
        return {**super().get_extra_state(), "method":METHOD_REVISION, "band_augmentation_seed": self.band_augmentation_seed,
                "band_augmentation_rng": self._band_rng.get_state().clone()}

    def _check_checkpoint(self, state):
        super()._check_checkpoint(state)
        if (state.get("epsilon_seed") != self.epsilon_seed
                or state.get("band_augmentation_seed") != self.band_augmentation_seed
                or not isinstance(state.get("band_augmentation_rng"), torch.Tensor)):
            raise ValueError("Complete pinned auxiliary RNG metadata required")
        torch.Generator(device="cpu").set_state(state["band_augmentation_rng"].cpu())

    def set_extra_state(self, state):
        super().set_extra_state(state)
        self._band_rng.set_state(state["band_augmentation_rng"].cpu())

    def native_prediction(self, batch, *, return_aligner_details=False):
        self._check_observation(batch)
        training_rr = batch.lineage.resolution == "RR" and batch.lineage.split == "train"
        with torch.set_grad_enabled(torch.is_grad_enabled() and training_rr), \
                torch.autocast(device_type=batch.pan.device.type, enabled=False):
            shift = self.predict_shift(batch.pan, batch.native_lms,
                resolution=batch.lineage.resolution, return_aligner_details=return_aligner_details)
            corrected = aligned_pan_views(batch.pan.float(), shift["u"])
            predicted = reconstruct_with_base(self.teacher, corrected,
                                              base_from_batch(batch, self.upsampler), batch.ms)
        return {**predicted, **shift}

    def _native_support(self, pan, correction):
        support = support_for_shifts(*pan.shape[-2:], correction)
        good = support[..., 7:-7, 7:-7].flatten(1).all(1)
        if min(pan.shape[-2:]) <= 16 or not bool(good.all()):
            raise ValueError("SUPPORT_VIOLATION: every predicted PAN view must support fixed ROI8/stencil7")
        return {"warp_support_fraction": support.float().mean(), "spatial_support_ok": good}

    def forward(self, rr):
        rr.validate(4, training=True, require_gt=True)
        self.sources.check(rr)
        if not all(p.requires_grad for m in (self.aligner, self.teacher) for p in m.parameters()):
            raise ValueError("Stage1 jointly trains every A/T parameter")
        epsilon_before, aux_before = self._epsilon_rng.get_state(), self._band_rng.get_state()
        try:
            with torch.autocast(device_type=rr.pan.device.type, enabled=False):
                native = self.native_prediction(rr)
                support = self._native_support(rr.pan, native["u"])
                mask = support_for_shifts(*rr.pan.shape[-2:], native["u"])
                rec = masked_mean((native["y"] - rr.gt.detach()).abs(), mask)
                abs_native = self.spatial.absolute_loss(native["pan_aligned"], rr.gt.detach())
                with torch.no_grad():
                    eps = sample_epsilon(len(rr.pan), self._epsilon_rng, self.epsilon_radius_rr).to(rr.pan.device)
                    pan_eps = warp_pan(rr.pan, eps)
                    eps_support = warp_support_mask(*rr.pan.shape[-2:], eps)
                shifted = self.predict_shift(pan_eps, rr.native_lms, resolution="RR",
                                             pan_valid_mask=eps_support)
                broadcast_eps = eps
                eps_pan = (shifted["u"] + broadcast_eps - native["u"].detach()).abs().mean()
                absolute, equivariance = abs_native, eps_pan
                loss = rec + self.lambda_abs * absolute + self.lambda_eps * equivariance
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite band30 Stage1 loss")
                fraction = eps_support[..., 8:-8, 8:-8].float().mean()
        except Exception:
            self._epsilon_rng.set_state(epsilon_before)
            self._band_rng.set_state(aux_before)
            raise
        call = self.forward_calls
        self.forward_calls += 1
        return {"loss": loss, "rec": rec, "abs_rr": absolute, "eps_rr": equivariance,
                "weighted_abs": self.lambda_abs * absolute,
                "weighted_eps_rr": self.lambda_eps * equivariance, "native": native,
                "shifted": shifted, "epsilon_rr": eps, "forward_index": call,
                "epsilon_rr_view_support_fraction": fraction, "loss_mask": mask, **support}


class BandStage1Trainer(DirectStage1Trainer):
    """Restore auxiliary RNG together with parent epsilon on pre-step failures."""
    def step_accumulated(self, microbatches, *, expected_effective_batch=None):
        before = self.stage._band_rng.get_state().clone()
        self.stage._auxiliary_records = []
        try:
            result = super().step_accumulated(microbatches,
                                            expected_effective_batch=expected_effective_batch)
        except Exception:
            if not self.poisoned:
                self.stage._band_rng.set_state(before)
            self.stage._auxiliary_records = []
            raise
        result["band_augmentation"] = self.stage._auxiliary_records
        self.stage._auxiliary_records = []
        return result


def validate_config(cfg, sources):
    _validate_shared_config(cfg, sources)
    validate_loss_pair_config(cfg)
    from .plan import alignment_protocol_manifest, alignment_protocol_digest
    b = cfg['baseline_resolved']
    if (b.get('epsilon_radius_rr') != cfg['epsilon_radius_rr']
            or b.get('alignment_protocol') != alignment_protocol_manifest(b)
            or cfg.get('baseline_alignment_protocol_digest') != alignment_protocol_digest(b)):
        raise ValueError('Stage1 epsilon radius/protocol metadata mismatch')
    if (cfg.get("campaign_id") != CAMPAIGN_ID or cfg.get("method_revision") != METHOD_REVISION
            or cfg.get("server") not in ("s1", "s3") or cfg.get("dataset") not in ("QB", "GF2")
            or cfg.get("aligner_kind") != "direct"
            or cfg.get("lr_A") != 1e-5 or cfg.get("lr_U") != 1e-4
            or cfg.get("initialization") != "FRESH" or cfg.get("start_update", 0) != 0):
        raise ValueError("Fresh local s1/s3 QB/GF2 mature30 RGB-reference donor configuration required")
    if cfg["dataset"] != LANES[cfg["server"]]["dataset"]:
        raise ValueError("Stage1 sensor/server mismatch")
    upsampler_manifest(cfg.get("upsampler"))
    if cfg.get("upsampler") != "interp23":
        raise ValueError("Mature30 fresh Stage1 requires interp23")
    expected = {2: (1, GLOBAL_TARGET, "NONE")}
    if cfg.get("aligner_outputs") not in expected:
        raise ValueError("Explicit global2 configuration required")
    if (cfg.get("pan_channels"), cfg.get("alignment_target"), cfg.get("band_augmentation")) != expected[cfg["aligner_outputs"]]:
        raise ValueError("Aligner head/stem/objective/augmentation mismatch")
    if type(cfg["stream_seeds"].get("rr_band_augmentation")) is not int:
        raise ValueError("Explicit auxiliary RNG seed required")


def build_stage1(cfg, sources, *, device="cpu"):
    validate_config(cfg, sources)
    b, seeds = cfg["baseline_resolved"], cfg["stream_seeds"]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seeds["A_init"])
        aligner = BandPANAligner(cfg["aligner_outputs"])
    teacher = build_reconstructor(ReconstructionSpec(**b["teacher_architecture"],
        input_layout="P0", base_reference="BICUBIC_MS"), seed=seeds["T_init"], role="T",
        pan_channels=cfg["pan_channels"])
    stage = BandStage1(aligner, teacher, upsampler=cfg["upsampler"],
        sources=DirectTrainingSources(sources.sensor, sources.manifest_sha("train")),
        epsilon_seed=seeds["rr_epsilon"], epsilon_radius_rr=cfg["epsilon_radius_rr"], band_augmentation_seed=seeds["rr_band_augmentation"],
        roi_margin=8, boundary_policy="require_spatial_support",
        lambda_abs=cfg["lambda_abs"], lambda_eps=cfg["lambda_epsilon"]).to(device)
    optimizer = torch.optim.AdamW([
        {"params": stage.aligner.parameters(), "lr": cfg["lr_A"], "role": "A"},
        {"params": stage.teacher.parameters(), "lr": cfg["lr_U"], "role": "T"}],
        betas=tuple(b["optimizer_betas"]), eps=b["optimizer_eps"], weight_decay=cfg["weight_decay"])
    trainer = BandStage1Trainer(stage, optimizer,
        gradient_log_interval=b["gradient_log_interval"], effective_batch_size=b["effective_batch_RR"],
        microbatch_size=b["microbatch_RR"])
    trainer.config = deepcopy(cfg)
    trainer._mature30_sources = sources
    return trainer


def save_stage1(path, trainer, stream, *, code_pin, evaluator_pin, evaluation=None):
    """Immutable complete checkpoint; independently strict-reloaded before save."""
    if (not isinstance(stream, Stage1Stream) or not isinstance(trainer, BandStage1Trainer)
            or not isinstance(trainer.stage, BandStage1)):
        raise TypeError("Band30 Stage 1 trainer and complete RR stream required")
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
            "mature30_teacher": teacher_binding(cfg),
            "selection_rule": "VAL_RRE50_THEN_RRE90_THEN_VAL_ERGAS_THEN_EARLIER_UPDATE"}
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
    before_stat = file_identity(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if file_identity(checkpoint) != before_stat or file_sha(checkpoint) != expected_sha256:
        raise ValueError("Checkpoint file changed while loading")
    if (not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("campaign_resume_schema") != RESUME_SCHEMA
            or payload.get("campaign_resume_complete") is not True
            or not _same(payload.get("config"), cfg) or payload.get("pins") != pins
            or payload.get("mature30_teacher") != teacher_binding(cfg)):
        raise ValueError("Band30 checkpoint/config/code/evaluator/upsampler resume mismatch")
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
