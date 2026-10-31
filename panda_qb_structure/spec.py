"""Expose the existing two PAN inputs and two upsamplers in this campaign only."""
from panda_sensor_focus.spec import MatureSpec as OriginalSpec
from panda_stage2_condq.spec import Stage2Spec, MASK2_POLICY
from panda_sensor_focus.reconstruction import StudentReconstructionSpec
from . import registry as G


class MatureSpec(OriginalSpec):
    def __post_init__(self):
        Stage2Spec.__post_init__(self)
        actual = (self.sensor, self.variant, self.alpha, self.beta, self.gamma, self.delta,
                  self.mask_policy, self.policy_id, self.upsampler, self.frame,
                  self.aligner_outputs, self.alignment_target, self.pan_channels,
                  self.total_updates, self.sampling_variant, self.input_variant, self.student_width)
        allowed = {('QB', 'panda_condq', p['alpha'], 0., p['gamma'], 1e-6, MASK2_POLICY,
                    'S100', 'interp23', 'pan', 2, 'SCHARR_OF_RGB_MEAN_GT', 1,
                    p['total_updates'], p['sampling_variant'], p['input_variant'], p['student_width'])
                   for p in G.cases().values()}
        if actual not in allowed:
            raise ValueError('Unregistered QB input/sampling/objective specification')

    @property
    def reconstruction_spec(self):
        # The Teacher is always W112/PAN1/BASE, independently of Student input.
        if self.input_variant == 'CORRECTED':
            return self.teacher_reconstruction_spec
        return StudentReconstructionSpec(4, self.student_width, (1, 2, 3), 'P0', 'BICUBIC_MS',
                                         2)
