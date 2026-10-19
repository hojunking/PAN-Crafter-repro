"""Fresh Student, immutable complete restart, HQNR-first campaign selection."""
from copy import deepcopy

from panda30r.engine import (advance as advance_student, save_checkpoint as save_student,
                             load_checkpoint as load_student)
from panda_stage2_condq.training import StudentTrainer, TrainingSpec
from .contract import POLICY


class WarpStudentTrainer(StudentTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the new policy once during construction, before any update/state
        # snapshot. All inherited contract checks continue to enforce it.
        self.selection_policy = POLICY
        self._initial_selection = POLICY.manifest()


def build_fresh_stage2(cfg, donor, stats, *, evaluator_pin=None, device='cpu'):
    from .spec import WarpSpec
    from .method import WarpStage2
    from .calibration import bind_donor
    if cfg.get('engine_kind') != 'stage2_warp' or cfg.get('phase') != 'stage2':
        raise ValueError('Explicit fresh warp Stage2 configuration required')
    spec = WarpSpec(**cfg['stage2_spec'])
    donor = bind_donor(donor, spec)
    if cfg.get('donor_binding') != donor.binding:
        raise ValueError('Resolved matching warp donor/calibration binding required')
    stage = WarpStage2(spec, donor.aligner, donor.teacher,
                      donor_binding=donor.binding, stats=stats).to(device)
    trainer = WarpStudentTrainer(stage, TrainingSpec(**cfg['training_spec']),
                                evaluation_pin=evaluator_pin, hqnr_target=.959)
    trainer.config, trainer.engine_kind = deepcopy(cfg), 'stage2_warp'
    return trainer


def advance(trainer, stream):
    if trainer.config['phase'] == 'donor':
        from .stage1 import advance_stage1
        return advance_stage1(trainer, stream)
    return advance_student(trainer, stream)


def save_checkpoint(path, trainer, stream, *, code_pin, evaluator_pin, evaluation=None):
    if trainer.config['phase'] == 'donor':
        from .stage1 import save_stage1
        return save_stage1(path, trainer, stream, code_pin=code_pin,
                           evaluator_pin=evaluator_pin, evaluation=evaluation)
    return save_student(path, trainer, stream, code_pin=code_pin,
                        evaluator_pin=evaluator_pin, evaluation=evaluation)


def load_checkpoint(path, sha, trainer, stream, *, code_pin, evaluator_pin):
    if trainer.config['phase'] == 'donor':
        from .stage1 import load_stage1
        loaded, restored, _ = load_stage1(path, sha, trainer.config, stream.sources,
            device=stream.device, code_pin=code_pin, evaluator_pin=evaluator_pin)
        return loaded, restored
    load_student(path, sha, trainer, stream, code_pin=code_pin, evaluator_pin=evaluator_pin)
    return trainer, stream
