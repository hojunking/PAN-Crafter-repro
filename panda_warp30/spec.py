"""Explicit upsampler, shift strength and Stage 2 coordinate contracts."""
from copy import deepcopy
from dataclasses import dataclass
import math

from panda_stage2_condq.spec import Stage2Spec, MASK2_POLICY
from panda_midpoint.contract import UPSAMPLING

METHOD_VERSION = 'PANDA_WARP30_STAGE2_V1'
GEOMETRY_VERSION = 'PANDA_WARP30_GEOMETRY_V1'
FRAMES = ('pan', 'mid', 'mid_return', 'ms_return')
UPSAMPLERS = ('bicubic', 'interp23')


def upsampling_manifest(upsampler):
    if upsampler == 'interp23':
        value = deepcopy(UPSAMPLING)
        value.pop('frozen_teacher_base_reference', None)
        return value
    if upsampler == 'bicubic':
        return {'schema': 'TORCH_BICUBIC_X4_ALIGN_CORNERS_FALSE',
                'base_reference': 'BICUBIC_MS', 'ratio': 4,
                'augmentation_order': 'LRMS_AUGMENT_THEN_UPSAMPLE',
                'translation_warp': 'EXISTING_BICUBIC_UNCHANGED'}
    raise ValueError('Explicit bicubic/interp23 upsampler required')


def geometry_manifest(spec):
    rho = spec.pan_fraction
    return {'schema': GEOMETRY_VERSION, 'upsampler': spec.upsampler,
            'frame': spec.frame, 'shift_scale': spec.shift_scale,
            'pan_shift_factor': rho * spec.shift_scale,
            'ms_shift_factor': (rho - 1) * spec.shift_scale,
            'output_shift_factor': (1 - rho) * spec.shift_scale if spec.returns_output else 0.,
            'output_coordinates': spec.output_coordinates,
            'loss_coordinates': spec.output_coordinates,
            'q_pan_shift_factor': (rho if spec.frame == 'mid' else 1.) * spec.shift_scale,
            'shift_unit': 'CURRENT_PAN_PIXEL_NO_X4',
            'teacher_policy': 'FIXED_STAGE1_FORWARD_PAN_PLUS_U',
            'training_target_policy': 'TRANSPORT_MINUS_HALF_D_ONLY_MID_ELSE_ORIGINAL',
            'zero_strength_policy': 'BYPASS_ALL_STUDENT_AND_TARGET_WARPS_KEEP_FIXED_TEACHER',
            'inverse_location': 'INSIDE_TRAIN_AUTOGRAD_AND_DEPLOYMENT' if spec.returns_output else 'NONE',
            'loss_support': 'INTERSECTION_EACH_ACTUAL_BICUBIC_STAGE_NO_NET_SHIFT_CANCELLATION',
            'support_scope': 'SAMPLER_ONLY_NOT_UPSAMPLER_CIRCULAR_OR_CNN_RECEPTIVE_FIELD',
            'evaluation_references': 'ORIGINAL_UNSHIFTED_UNMASKED_RR20_FR20'}


@dataclass(frozen=True)
class WarpSpec(Stage2Spec):
    mask_policy: str = MASK2_POLICY
    upsampler: str = 'interp23'
    frame: str = 'pan'
    shift_scale: float = 1.

    def __post_init__(self):
        super().__post_init__()
        if self.sensor != 'WV3' or self.mask_policy != MASK2_POLICY:
            raise ValueError('Warp30 preserves WV3/MASK2')
        if self.upsampler not in UPSAMPLERS or self.frame not in FRAMES:
            raise ValueError('Explicit Warp30 upsampler/frame required')
        if (type(self.shift_scale) not in (int, float) or not math.isfinite(self.shift_scale)
                or self.shift_scale not in (0., .25, .5, 1.)):
            raise ValueError('Registered shift_scale is 0/.25/.5/1')
        object.__setattr__(self, 'shift_scale', float(self.shift_scale))

    @property
    def method_version(self):
        return METHOD_VERSION

    @property
    def pan_fraction(self):
        return {'pan': 1., 'mid': .5, 'mid_return': .5, 'ms_return': 0.}[self.frame]

    @property
    def returns_output(self):
        return self.frame in ('mid_return', 'ms_return')

    @property
    def output_coordinates(self):
        return 'MIDPOINT' if self.frame == 'mid' else 'MS'

    def manifest(self):
        value = super().manifest()
        constructor = deepcopy(value['student'])
        value['student'] = {**constructor,
            'base_reference': upsampling_manifest(self.upsampler)['base_reference'],
            'base_execution': 'EXTERNAL_BASE_X_IN_AND_RESIDUAL_ADDITION'}
        return {**value, 'student_constructor_spec': constructor,
                'student_upsampling': upsampling_manifest(self.upsampler),
                'warp30_geometry': geometry_manifest(self)}
