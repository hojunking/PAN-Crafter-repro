"""Fresh Student, immutable complete restart, HQNR-first campaign selection."""
from copy import deepcopy

from panda30r.engine import (advance as advance_student, save_checkpoint as save_student,
                             load_checkpoint as load_student)
from panda_stage2_condq.training import StudentTrainer, TrainingSpec
from .contract import POLICY, CAMPAIGN_ID, METHOD_REVISION


class BandStudentTrainer(StudentTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set the new policy once during construction, before any update/state
        # snapshot. All inherited contract checks continue to enforce it.
        self.selection_policy = POLICY
        self._initial_selection = POLICY.manifest()


def build_fresh_stage2(cfg, donor, stats, *, evaluator_pin=None, device='cpu'):
    from .spec import BandSpec
    from .method import BandStage2
    from .calibration import bind_donor
    if (cfg.get('engine_kind') != 'stage2_band' or cfg.get('phase') != 'stage2'
            or cfg.get('campaign_id') != CAMPAIGN_ID or cfg.get('method_revision') != METHOD_REVISION
            or cfg.get('server') != 's1' or cfg.get('dataset') != 'WV3'):
        raise ValueError('Explicit fresh s1/WV3 Band30 Stage2 configuration required')
    spec = BandSpec(**cfg['stage2_spec'])
    if any(cfg.get(key) != getattr(spec,key) for key in
           ('upsampler','frame','shift_scale','aligner_outputs','alignment_target','pan_channels')):
        raise ValueError('Band30 configuration and Stage2 geometry/dimensions disagree')
    donor = bind_donor(donor, spec)
    if cfg.get('donor_binding') != donor.binding:
        raise ValueError('Resolved matching Band30 donor/calibration binding required')
    stage = BandStage2(spec, donor.aligner, donor.teacher,
                      donor_binding=donor.binding, stats=stats).to(device)
    trainer = BandStudentTrainer(stage, TrainingSpec(**cfg['training_spec']),
                                evaluation_pin=evaluator_pin, hqnr_target=.959)
    trainer.config, trainer.engine_kind = deepcopy(cfg), 'stage2_band'
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
