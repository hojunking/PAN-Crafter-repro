"""Matched Direct/S2D Stage 1 engines and complete immutable restart files.

Only campaign-owned local torch files are loaded.  A Direct checkpoint retains
the donor30 outer schema so the existing frozen Stage 2 importer can consume
the very same evaluated A/T pair.  The additional campaign contract is required
for *resuming* here: old files are donors, not silently compatible resumes.
S2D files have a separate outer schema and an explicit paired export adapter.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch

from directshift48.training import DirectStage1Trainer
from donor30.artifacts import (SCHEMA as DIRECT_SCHEMA, _cpu, _same,
                               _save_immutable, capture_rng, restore_rng)
from donor30.evaluation import model_state_hash
from donor30.masking import build_masked_stage1
from donor30.plan import LOSS_MASK_POLICY
from edgeres48.stream import RRStream, learning_rates, set_learning_rates
from panda_method.data import require_sha256
from panda_method.reconstruction import ReconstructionSpec
from s2dshift48.stage1 import build_s2d_stage1
from ta2.evaluation import file_sha


RESUME_SCHEMA = "PANDA40_STAGE1_COMPLETE_RESUME_V1"
S2D_SCHEMA = "PANDA40_S2D_STAGE1_CHECKPOINT_V1"


def _validate_config(cfg, sources):
    if (cfg.get("phase") != "donor" or cfg.get("stage2") is not False
            or cfg.get("aligner_kind") not in ("direct", "s2d")
            or cfg.get("dataset") != sources.sensor
            or cfg.get("loss_mask_policy") != LOSS_MASK_POLICY
            or cfg.get("mask_policy", "MASK2") != "MASK2"):
        raise ValueError("Explicit matched Direct/S2D MASK2 Stage 1 config required")
    b = cfg["baseline_resolved"]
    if (b["teacher_architecture"] != {"bands": sources.bands, "width": 112, "depth": [1, 2, 3]}
            or b["teacher_input_layout"] != "P0" or b["teacher_base"] != "BICUBIC_MS"
            or b["aligner_architecture"]["view_margin"] != 8
            or b["absolute_roi_margin"] != 8
            or b["boundary_policy"] != "require_spatial_support"
            or b["precision"] != "fp32" or b["amp"] is not False or b["tf32"] is not False
            or b["optimizer"] != "AdamW" or tuple(b["optimizer_betas"]) != (.9, .999)
            or b["optimizer_eps"] != 1e-8 or cfg["weight_decay"] != .01
            or b["scheduler_family"] != "warmup_cosine"):
        raise ValueError("Unchanged W112/D123/P0, native geometry and FP32 AdamW contract required")
    if (type(cfg["planned_horizon"]) is not int or cfg["planned_horizon"] != 50000
            or b["warmup_updates0"] != 100 or b["min_lr"] != 0
            or cfg.get("lambda_rec", 1.) != 1.):
        raise ValueError("H50/100-update warmup/cosine and unit reconstruction weight required")
    for name in ("lr_A", "lr_U", "lambda_abs", "lambda_epsilon"):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Finite nonnegative " + name + " required")
    if cfg["lr_A"] == 0 or cfg["lr_U"] == 0:
        raise ValueError("Both Stage 1 networks require positive peak learning rates")
    for name in ("A_init", "T_init", "rr_data", "rr_augmentation", "rr_epsilon"):
        if type(cfg["stream_seeds"][name]) is not int:
            raise ValueError("Explicit integer independent stream seeds required")
    if (b["effective_batch_RR"] != b["microbatch_RR"] * b["accumulation_RR"]
            or any(type(b[k]) is not int or b[k] < 1 for k in
                   ("effective_batch_RR", "microbatch_RR", "accumulation_RR"))):
        raise ValueError("Exact effective/micro/accumulation batch contract required")
    if not isinstance(cfg.get("run_id"), str) or not cfg["run_id"]:
        raise ValueError("Explicit nonempty campaign run ID required")


def build_stage1(cfg, sources, *, device="cpu"):
    """Fresh matched initialization; neither architecture is a donor warm-start."""
    _validate_config(cfg, sources)
    b, seeds = cfg["baseline_resolved"], cfg["stream_seeds"]
    kwargs = dict(
        teacher_spec=ReconstructionSpec(**b["teacher_architecture"],
            input_layout="P0", base_reference="BICUBIC_MS"),
        teacher_seed=seeds["T_init"], aligner_seed=seeds["A_init"],
        epsilon_seed=seeds["rr_epsilon"], aligner_view_margin=8, roi_margin=8,
        boundary_policy="require_spatial_support", lambda_abs=cfg["lambda_abs"],
        lambda_eps=cfg["lambda_epsilon"])
    stage = (build_masked_stage1(sources, **kwargs) if cfg["aligner_kind"] == "direct"
             else build_s2d_stage1(sources, mask_policy="MASK2", **kwargs)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": stage.aligner.parameters(), "lr": cfg["lr_A"], "role": "A"},
        {"params": stage.teacher.parameters(), "lr": cfg["lr_U"], "role": "T"}],
        betas=tuple(b["optimizer_betas"]), eps=b["optimizer_eps"], weight_decay=cfg["weight_decay"])
    trainer = DirectStage1Trainer(stage, optimizer,
        gradient_log_interval=b["gradient_log_interval"],
        effective_batch_size=b["effective_batch_RR"], microbatch_size=b["microbatch_RR"])
    trainer.config = deepcopy(cfg)
    trainer._panda40_sources = sources
    return trainer


class Stage1Stream(RRStream):
    """Existing sampling and transforms, with fail-closed complete restoration."""

    def protocol(self):
        return {"sensor": self.sources.sensor,
            "source_train_manifest_sha256": self.sources.manifest_sha("train"),
            "count": self.count, "microbatch": self.micro, "accumulation": self.accumulation,
            "augmentation": deepcopy(self.augmentation),
            "rr_data_seed": self.config["stream_seeds"]["rr_data"],
            "rr_augmentation_seed": self.config["stream_seeds"]["rr_augmentation"],
            "sampling": "SHUFFLED_EPOCH_NO_DROP",
            "augmentation_scope": "ONE_SHARED_MICROBATCH_TRANSFORM"}

    def load_state_dict(self, state):
        reference = self.state_dict()
        if not isinstance(state, dict) or set(state) != set(reference):
            raise ValueError("Complete RR sampler state required")
        if any(state[key] != reference[key] for key in ("count", "sampling", "augmentation_scope")):
            raise ValueError("RR sampler population/protocol changed")
        order = state["order"]
        if (not isinstance(order, list) or len(order) != self.count
                or any(type(i) is not int for i in order) or sorted(order) != list(range(self.count))
                or type(state["cursor"]) is not int or not 0 <= state["cursor"] < self.count
                or type(state["epoch"]) is not int or state["epoch"] < 0):
            raise ValueError("RR permutation/cursor/epoch corrupted")
        # Validate both RNGs before mutating either live RNG or the cursor.
        rng, aug = np.random.default_rng(), np.random.default_rng()
        try:
            rng.bit_generator.state = deepcopy(state["rng"])
            aug.bit_generator.state = deepcopy(state["augmentation_rng"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("Invalid independent sampler/augmentation RNG state") from exc
        self.rng, self.aug = rng, aug
        self.order, self.cursor, self.epoch = deepcopy(order), state["cursor"], state["epoch"]


def advance_stage1(trainer, stream):
    """One complete update with the existing one-based H50 learning-rate law."""
    if not _same(trainer.config, stream.config):
        raise ValueError("Trainer and RR stream configuration differ")
    if trainer.poisoned or trainer.updates >= trainer.config["planned_horizon"]:
        raise ValueError("Cannot advance a failed or completed Stage 1 trainer")
    before_stream = stream.state_dict()
    before_lrs = [group["lr"] for group in trainer.optimizer.param_groups]
    before_update = trainer.updates
    try:
        rates = set_learning_rates(trainer, trainer.config)
        result = trainer.step_accumulated(stream.next_update())
    except Exception:
        # Pre-optimizer errors must not silently consume a different next batch.
        # A partial optimizer failure is poisoned and can only restart from disk.
        if not trainer.poisoned and trainer.updates == before_update:
            stream.load_state_dict(before_stream)
            for group, lr in zip(trainer.optimizer.param_groups, before_lrs):
                group["lr"] = lr
        raise
    result.update(lr=rates, loss_mask_policy=LOSS_MASK_POLICY,
                  loss_valid_fraction=result["support"]["warp_support_fraction"])
    return result


def _scheduler(cfg, update):
    return {"family": "warmup_cosine", "planned_horizon": cfg["planned_horizon"],
            "completed_update": update, "indexing": "ONE_BASED",
            "warmup_updates": cfg["baseline_resolved"]["warmup_updates0"],
            "min_lr": cfg["baseline_resolved"]["min_lr"],
            "last_rates": learning_rates(cfg, update) if update else
                {"A": cfg["lr_A"], "T": cfg["lr_U"]}}


def _finite_tensors(tree):
    if isinstance(tree, torch.Tensor):
        if tree.is_floating_point() and not bool(torch.isfinite(tree).all()):
            raise ValueError("Nonfinite tensor in complete Stage 1 checkpoint")
    elif isinstance(tree, dict):
        for value in tree.values():
            _finite_tensors(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            _finite_tensors(value)


def _optimizer_contract(state, cfg, update):
    groups = state.get("optimizer", {}).get("param_groups", [])
    if len(groups) != 2:
        raise ValueError("Both named Stage 1 optimizer groups required")
    rates = _scheduler(cfg, update)["last_rates"]
    for group, role in zip(groups, ("A", "T")):
        if (group.get("role") != role or group.get("lr") != rates[role]
                or tuple(group.get("betas", ())) != tuple(cfg["baseline_resolved"]["optimizer_betas"])
                or group.get("eps") != cfg["baseline_resolved"]["optimizer_eps"]
                or group.get("weight_decay") != cfg["weight_decay"]
                or group.get("amsgrad") is not False or group.get("maximize") is not False):
            raise ValueError("Optimizer role/LR/AdamW policy differs from the completed scheduler step")
    if update > 0:
        states = state["optimizer"].get("state", {})
        expected_ids = {p for group in groups for p in group["params"]}
        if set(states) != expected_ids:
            raise ValueError("Complete AdamW state for every jointly-trained parameter required")
        for item in states.values():
            if (set(item) != {"step", "exp_avg", "exp_avg_sq"}
                    or not isinstance(item["step"], torch.Tensor)
                    or item["step"].numel() != 1 or float(item["step"]) != update
                    or item["exp_avg"].shape != item["exp_avg_sq"].shape
                    or bool((item["exp_avg_sq"] < 0).any())):
                raise ValueError("AdamW moments/update count corrupted")
    elif state["optimizer"].get("state"):
        raise ValueError("Zero-update checkpoint must have fresh optimizer moments")
    _finite_tensors(state)


def _identity(cfg, update, state_hash):
    result = {key: cfg[key] for key in ("run_id", "attempt", "seed", "dataset")}
    result.update(completed_update=update, model_state_sha256=state_hash)
    return result


def _stream_progress(state, trainer_state, cfg):
    """Prove the optimizer and no-drop sampler describe the same boundary."""
    b, update = cfg["baseline_resolved"], trainer_state["updates"]
    if (state["epoch"] * state["count"] + state["cursor"] != update * b["effective_batch_RR"]
            or trainer_state.get("stage", {}).get("_extra_state", {}).get("forward_calls")
            != update * b["accumulation_RR"]):
        raise ValueError("Optimizer update, data draws and epsilon forward count do not share one boundary")


def _check_evaluation(evaluation, identity):
    if evaluation is None:
        return
    qual = evaluation.get("qualification", {})
    if not all(qual.get(key) is True for key in
               ("same_state_verified", "metrics_complete", "metrics_finite")):
        raise ValueError("Candidate requires complete finite official same-state evaluation")
    states = [evaluation.get(key) for key in ("state_before", "rr_state", "fr_state", "state_after")]
    if (not isinstance(states[0], dict) or any(s != states[0] for s in states[1:])
            or any(states[0].get(k) != v for k, v in identity.items())):
        raise ValueError("Candidate official evaluation is not the exact live A/T state")
    for split, required in (("rr", ("ERGAS", "SCC")), ("fr", ("HQNR",))):
        block = evaluation.get(split, {})
        metrics = block.get("metrics", {})
        if any(k not in metrics or not math.isfinite(float(metrics[k])) for k in required):
            raise ValueError("Finite required official metrics missing")
        rows = block.get("per_scene", [])
        if len(rows) != 20 or len({r.get("scene_id") for r in rows}) != 20:
            raise ValueError("Official candidate requires all20 unique " + split + " scenes")
        for row in rows:
            if not row.get("scene_id"):
                raise ValueError("Official scene identity missing")
            require_sha256(row.get("raw_prediction_sha256"), "official raw prediction SHA256")


def _pins(code_pin, evaluator_pin):
    if not isinstance(code_pin, dict) or not code_pin or not isinstance(evaluator_pin, dict) or not evaluator_pin:
        raise ValueError("Nonempty explicit code and official evaluator pins required")
    for name, sha in code_pin.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Named code pin required")
        require_sha256(sha, "code file SHA256")
    return {"code": deepcopy(code_pin), "evaluator": deepcopy(evaluator_pin)}


def save_stage1_checkpoint(trainer, stream, path, cfg, evaluation=None, *, code_pin, evaluator_pin):
    """Save complete immutable state, strict-load a fresh CPU trainer, return SHA.

    Does not rank, mark HQNR-qualified, advance a best pointer, or write sheets.
    An attached evaluation is proof for the exact paused state, not an assertion
    that its metric target was attained.  ``evaluation=None`` is resume-only.
    """
    if not isinstance(stream, Stage1Stream) or not isinstance(trainer, DirectStage1Trainer):
        raise TypeError("Paired campaign Stage 1 trainer and resumable RR stream required")
    if not _same(cfg, trainer.config) or not _same(cfg, stream.config):
        raise ValueError("Resolved trainer/stream/checkpoint configs differ")
    _validate_config(cfg, stream.sources)
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
        payload = {"schema": DIRECT_SCHEMA if cfg["aligner_kind"] == "direct" else S2D_SCHEMA,
            "campaign_resume_schema": RESUME_SCHEMA, "campaign_resume_complete": True,
            "config": deepcopy(cfg), "identity": identity, "state_hash": state_hash,
            "trainer": state, "stream": _cpu(stream.state_dict()), "stream_protocol": stream.protocol(),
            "rng": _cpu(rng), "scheduler": _scheduler(cfg, trainer.updates), "pins": pins,
            "evaluation": deepcopy(evaluation), "resume_only": evaluation is None,
            "selection_reasons": [], "selection_rule": "CALLER_HQNR_GATE_THEN_ERGAS_NO_SCC"}
        # Validate a real independently constructed model and complete sampler
        # before publishing an immutable destination. This does not re-evaluate
        # forty scenes or pretend serialized reload is prediction verification.
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
            "model_state_sha256": state_hash, "identity": identity,
            "completed_update": trainer.updates, "weights_saved": True,
            "checkpoint_verified": True, "predictions_reverified": False,
            "verification": "STRICT_CPU_RELOAD_MODEL_OPTIMIZER_STREAM_AND_FILE_ROUNDTRIP",
            "resume_only": evaluation is None, "campaign_resume_complete": True,
            "direct_stage2_import_compatible": cfg["aligner_kind"] == "direct"}
    finally:
        restore_rng(rng)


def load_stage1_checkpoint(path, expected_sha256, cfg, sources, *, device="cpu",
                          code_pin, evaluator_pin, restore_global_rng=True):
    """Resume only a fully pinned *same run/config* local complete checkpoint."""
    require_sha256(expected_sha256, "explicit local Stage 1 checkpoint SHA256")
    _validate_config(cfg, sources)
    pins = _pins(code_pin, evaluator_pin)
    checkpoint = Path(path).resolve(strict=True)
    if not checkpoint.is_file() or file_sha(checkpoint) != expected_sha256:
        raise ValueError("Stage 1 checkpoint SHA mismatch")
    before_stat = checkpoint.stat()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.stat() != before_stat or file_sha(checkpoint) != expected_sha256:
        raise ValueError("Checkpoint file changed while loading")
    expected_schema = DIRECT_SCHEMA if cfg["aligner_kind"] == "direct" else S2D_SCHEMA
    if (payload.get("schema") != expected_schema
            or payload.get("campaign_resume_schema") != RESUME_SCHEMA
            or payload.get("campaign_resume_complete") is not True
            or not _same(payload.get("config"), cfg) or payload.get("pins") != pins):
        raise ValueError("Checkpoint architecture/config/code/evaluator resume contract mismatch")
    update = payload.get("trainer", {}).get("updates")
    if (type(update) is not int or not 0 <= update <= cfg["planned_horizon"]
            or payload.get("scheduler") != _scheduler(cfg, update)):
        raise ValueError("Checkpoint completed update/scheduler contract mismatch")
    _optimizer_contract(payload["trainer"], cfg, update)
    identity = _identity(cfg, update, payload.get("state_hash"))
    if payload.get("identity") != identity or payload.get("resume_only") != (payload.get("evaluation") is None):
        raise ValueError("Checkpoint model/evaluation identity mismatch")
    require_sha256(payload.get("state_hash"), "checkpoint A/T state SHA256")
    _check_evaluation(payload.get("evaluation"), identity)
    rng = capture_rng()
    success = False
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
        # Validate the external RNG payload even when callers choose not to
        # change their surrounding RNG. Restoration is last, after all checks.
        restore_rng(payload["rng"])
        success = True
        return trainer, stream, payload
    finally:
        if not success or not restore_global_rng:
            restore_rng(rng)


def export_s2d_donor(path, trainer, registered, *, run_id):
    """Explicit S2D paired handoff in addition to, not instead of, full resume."""
    if trainer.config.get("aligner_kind") != "s2d" or trainer.config.get("run_id") != run_id:
        raise ValueError("S2D export must retain the exact trained campaign run identity")
    from s2dshift48.donor import save_donor_checkpoint
    return save_donor_checkpoint(path, trainer, registered, run_id=run_id,
                                 purpose="EXPERIMENT_CANDIDATE")
