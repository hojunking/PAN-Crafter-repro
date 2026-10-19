"""Fresh midpoint Student construction; unchanged AdamW and complete-state resume.

The proven complete-checkpoint adapter is reused as an implementation utility,
not as a campaign controller. Its payload binds engine_kind, complete config,
stage geometry and code hashes, preventing original-coordinate resume here.
"""
from copy import deepcopy

from panda30r.engine import advance, save_checkpoint, load_checkpoint, load_mature_donor
from panda_stage2_condq.training import StudentTrainer, TrainingSpec
from .calibration import bind_donor
from .method import MidpointStage2
from .spec import MidpointSpec


def build_fresh_stage2(cfg, donor, stats, *, evaluator_pin=None, device="cpu"):
    if cfg.get("engine_kind") != "stage2_midpoint" or cfg.get("phase") != "stage2":
        raise ValueError("Explicit midpoint Stage 2 campaign config required")
    donor = bind_donor(donor)
    if cfg.get("donor_binding") != donor.binding:
        raise ValueError("Resolved midpoint donor binding is required before trainer build")
    spec = MidpointSpec(**cfg["stage2_spec"])
    stage = MidpointStage2(spec, donor.aligner, donor.teacher,
        donor_binding=donor.binding, stats=stats).to(device)
    trainer = StudentTrainer(stage, TrainingSpec(**cfg["training_spec"]),
        evaluation_pin=evaluator_pin, hqnr_target=.959)
    trainer.config, trainer.engine_kind = deepcopy(cfg), "stage2_midpoint"
    return trainer
