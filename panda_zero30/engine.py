"""ZERO30 Student with complete resume and paired-loss diagnostics."""
from copy import deepcopy

import torch

from panda_stage2_condq.training import StudentTrainer, TrainingSpec, _norm
from panda30r.engine import (
    advance as advance_student, save_checkpoint as save_student,
    load_checkpoint as load_student,
)
from .contract import CAMPAIGN_ID, METHOD_REVISION, POLICY
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
        self.stage._microbatch_diagnostics = []
        diagnostic = self.updates == 0 or (self.updates + 1) % self.training.gradient_log_interval == 0
        self.stage._consistency_gradient_accumulator = (
            [None] * len(list(self.stage.student.parameters())) if diagnostic else None)
        self.stage._diagnostic_effective_batch = self.training.effective_batch_size
        try:
            result = super().step_accumulated(batches)
        except Exception:
            # The inherited trainer may already have incremented its update
            # before a trailing diagnostic allocation fails. Never rewind the
            # data stream or publish a mismatched phase after such a failure.
            if self.updates != previous:
                self.poisoned = True
            raise
        finally:
            diagnostics = self.stage._microbatch_diagnostics
            consistency_gradients = self.stage._consistency_gradient_accumulator
            self.stage._microbatch_diagnostics = None
            self.stage._consistency_gradient_accumulator = None
            self.stage._diagnostic_effective_batch = None
        try:
            self.stage.set_completed_update(self.updates)
            result["mature30_geometry_key"] = geometry_key
            result["mature30_deployment"] = self.stage.deployment_manifest()
            total = sum(count for count, _ in diagnostics)
            if total != self.training.effective_batch_size:
                raise ValueError("Paired diagnostics lost an effective-batch microbatch")
            result.update({name: sum(count * item[name] for count, item in diagnostics) / total
                           for name in ("hard_canonical", "hard_rotated", "consistency_raw",
                                        "consistency_weighted")})
            result["student_views"] = 2 if self.stage.spec.paired_views else 1
            result["consistency_lambda"] = self.stage.spec.consistency_lambda
            result["gradient"]["hard_norm_excludes_consistency"] = True
            if diagnostic:
                total_gradients = [p.grad for p in self.stage.student.parameters()]
                cn = _norm(consistency_gradients)
                hard_gradients = [(total if cons is None else total - cons)
                                  if total is not None else None
                                  for total, cons in zip(total_gradients, consistency_gradients)]
                hn = _norm(hard_gradients)
                dot = sum(float(hard.double().mul(cons.double()).sum())
                          for hard, cons in zip(hard_gradients, consistency_gradients)
                          if hard is not None and cons is not None)
                result["gradient"].update(lambda_consistency_norm=cn,
                    hard_consistency_cosine=max(-1., min(1., dot / (hn * cn))) if hn and cn else None)
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
        raise ValueError("Explicit s1/s3 WV3 zero30 fresh Stage2 configuration required")
    spec = MatureSpec(**cfg["stage2_spec"])
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
    if training.total_updates != HORIZON or cfg.get("planned_horizon") != HORIZON:
        raise ValueError("Mature fixed/phasewise policies require the registered 50K horizon")
    stage = MatureStage2(spec, donor.aligner, donor.teacher,
                        donor_binding=donor.binding, stats_bank=stats_bank).to(device)
    trainer = MatureStudentTrainer(stage, training, evaluation_pin=evaluator_pin, hqnr_target=.959)
    trainer.config, trainer.engine_kind = deepcopy(cfg), "stage2_mature"
    return trainer


def advance(trainer, stream):
    if trainer.config["phase"] != "stage2":
        raise ValueError("ZERO30 imports a mature donor; no Stage1 training is registered")
    return advance_student(trainer, stream)


def save_checkpoint(path, trainer, stream, *, code_pin, evaluator_pin, evaluation=None):
    if trainer.config["phase"] != "stage2":
        raise ValueError("ZERO30 cannot save a newly trained donor")
    if evaluation is not None:
        if evaluation.get("mature30_deployment") != trainer.stage.deployment_manifest():
            raise ValueError("Evaluation deployment geometry differs from saved update/phase")
    return save_student(path, trainer, stream, code_pin=code_pin,
                        evaluator_pin=evaluator_pin, evaluation=evaluation)


def load_checkpoint(path, sha, trainer, stream, *, code_pin, evaluator_pin):
    if trainer.config["phase"] != "stage2":
        raise ValueError("ZERO30 restores only complete fresh-Student states")
    loaded = load_student(path, sha, trainer, stream,
                          code_pin=code_pin, evaluator_pin=evaluator_pin)
    evidence = loaded.get("evaluation")
    if evidence is not None and evidence.get("mature30_deployment") != trainer.stage.deployment_manifest():
        trainer.poisoned = True
        raise ValueError("Restored checkpoint evaluation has a different deployed phase")
    return trainer, stream

