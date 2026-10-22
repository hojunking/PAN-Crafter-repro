"""Existing single-view Hard training with registered scale/coefficients/LR and exact resume."""
from copy import deepcopy

import torch

from panda_stage2_condq.training import StudentTrainer, TrainingSpec
from panda30r.engine import (
    advance as advance_student, save_checkpoint as save_student,
    load_checkpoint as load_student,
)
from .contract import CAMPAIGN_ID, METHOD_REVISION, POLICY, CASES
from .spec import MatureSpec, HORIZON, policy_manifest


class MatureStudentTrainer(StudentTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.training.total_updates != HORIZON or self.stage.completed_update != 0:
            raise ValueError("Fresh mature trainer requires H50 and zero completed updates")
        self.selection_policy = POLICY
        self._initial_selection = POLICY.manifest()

    def assert_contract(self, *, full_audit=True):
        super().assert_contract(full_audit=full_audit)
        if self.stage.completed_update != self.updates:
            raise ValueError("Mature phase counter must match completed optimizer updates")

    def step_accumulated(self, batches):
        self.assert_contract(full_audit=False)
        if self.updates >= HORIZON:
            raise ValueError("Mature training horizon is already complete")
        geometry_key = self.stage.spec.geometry_at_next(self.updates)
        previous = self.updates
        try:
            result = super().step_accumulated(batches)
        except Exception:
            # The inherited trainer may already have incremented its update
            # before a trailing diagnostic allocation fails. Never rewind the
            # data stream or publish a mismatched phase after such a failure.
            if self.updates != previous:
                self.poisoned = True
            raise
        try:
            self.stage.set_completed_update(self.updates)
            result["mature30_geometry_key"] = geometry_key
            result["mature30_deployment"] = self.stage.deployment_manifest()
            result["student_views"] = 1
            result["consistency_lambda"] = 0.
        except Exception:
            self.poisoned = True
            raise
        return result

    def load_state_dict(self, state):
        saved = state.get("stage", {}).get("_completed_update_state")
        if (not isinstance(saved, torch.Tensor) or saved.shape != ()
                or saved.dtype != torch.int64 or int(saved) != state.get("updates")):
            raise ValueError("Checkpoint schedule state and completed updates disagree")
        return super().load_state_dict(state)


def build_fresh_stage2(cfg, donor, stats_bank, *, evaluator_pin=None, device="cpu"):
    from .method import MatureStage2
    from .calibration import bind_donor
    if (cfg.get("campaign_id") != CAMPAIGN_ID or cfg.get("method_revision") != METHOD_REVISION
            or cfg.get("phase") != "stage2" or cfg.get("engine_kind") != "stage2_mature"
            or cfg.get("server") not in ("s1", "s3") or cfg.get("dataset") != "WV3"):
        raise ValueError("Explicit s1/s3 WV3 SCALE30 fresh Stage2 configuration required")
    spec = MatureSpec(**cfg["stage2_spec"])
    registered = CASES.get(cfg.get("case"))
    if (registered is None or cfg.get("tuning_parameters") != registered
            or cfg.get("policy_id") != registered["policy_id"]
            or cfg.get("shift_scale") != registered["scale"]
            or (spec.alpha, spec.gamma) != (registered["alpha"], registered["gamma"])
            or cfg.get("student_views") != 1 or cfg.get("rotation_consistency_weight") != 0.
            or cfg.get("pair_transform") != "NONE" or cfg.get("inference_tta") is not False):
        raise ValueError("Exact registered positive scale, single-view Hard coefficients and no consistency required")
    if cfg.get("policy_id") != spec.policy_id:
        raise ValueError("Configuration and immutable mature policy disagree")
    if cfg.get("geometry") != policy_manifest(spec):
        raise ValueError("Configuration and immutable phase schedule disagree")
    if cfg.get("stream_seeds", {}).get("S_init") != spec.seed:
        raise ValueError("Student initialization must match the declared paired seed")
    for name in ("upsampler", "frame", "aligner_outputs", "alignment_target", "pan_channels"):
        if cfg.get(name, getattr(spec, name)) != getattr(spec, name):
            raise ValueError("Configuration and mature geometry/dimensions disagree: " + name)
    donor = bind_donor(donor, spec)
    if cfg.get("donor_binding") != donor.binding:
        raise ValueError("Resolved mature donor binding must match actual same-checkpoint A/T")
    training = TrainingSpec(**cfg["training_spec"])
    if not (training.lr == cfg.get("lr_U") == cfg.get("lr_S") == registered["lr_S"]):
        raise ValueError("Student LR differs from the registered optimizer and reporting configuration")
    if training.total_updates != HORIZON or cfg.get("planned_horizon") != HORIZON:
        raise ValueError("Mature fixed/phasewise policies require the registered 50K horizon")
    stage = MatureStage2(spec, donor.aligner, donor.teacher,
                        donor_binding=donor.binding, stats_bank=stats_bank).to(device)
    trainer = MatureStudentTrainer(stage, training, evaluation_pin=evaluator_pin, hqnr_target=.959)
    trainer.config, trainer.engine_kind = deepcopy(cfg), "stage2_mature"
    return trainer


def advance(trainer, stream):
    if trainer.config["phase"] != "stage2":
        raise ValueError("SCALE30 imports a mature donor; no Stage1 training is registered")
    return advance_student(trainer, stream)


def save_checkpoint(path, trainer, stream, *, code_pin, evaluator_pin, evaluation=None):
    if trainer.config["phase"] != "stage2":
        raise ValueError("SCALE30 cannot save a newly trained donor")
    if evaluation is not None:
        if evaluation.get("mature30_deployment") != trainer.stage.deployment_manifest():
            raise ValueError("Evaluation deployment geometry differs from saved update/phase")
    return save_student(path, trainer, stream, code_pin=code_pin,
                        evaluator_pin=evaluator_pin, evaluation=evaluation)


def load_checkpoint(path, sha, trainer, stream, *, code_pin, evaluator_pin):
    if trainer.config["phase"] != "stage2":
        raise ValueError("SCALE30 restores only complete fresh-Student states")
    loaded = load_student(path, sha, trainer, stream,
                          code_pin=code_pin, evaluator_pin=evaluator_pin)
    evidence = loaded.get("evaluation")
    if evidence is not None and evidence.get("mature30_deployment") != trainer.stage.deployment_manifest():
        trainer.poisoned = True
        raise ValueError("Restored checkpoint evaluation has a different deployed phase")
    return trainer, stream
