"""Coefficient endpoints on the unchanged RGB Stage1 and anchored Stage2 operators."""
import math
from panda_wv3_ablation_rgb.model import Stage1
from panda_wv3_ablation.model import Stage2 as AnchoredStage2


class Stage2(AnchoredStage2):
    def __init__(self, cfg, donor, stats):
        if (any(not math.isfinite(float(cfg[key])) or cfg[key] < 0 for key in ('alpha','gamma'))
                or cfg['beta'] != 0 or cfg['student_input'] != 'CORRECTED_DIFF'
                or cfg['scale'] != 1. or cfg['architecture'] != 'W112_D123_UBIL'):
            raise ValueError('Hyper ablation requires nonnegative coefficients and unchanged W112 full input')
        donor.assert_frozen()
        if donor.binding.get('alignment_target') != 'SCHARR_OF_RGB_MEAN_GT':
            raise ValueError('Exact same-repeat RGB Stage1 reference required')
        # Keep the exact calibration identity for zero endpoints as well. The
        # inherited forward always keeps Teacher/Student support intersection,
        # while avoiding unused Teacher reconstruction when alpha=gamma=0.
        stats.validate_binding(donor.binding)
        stats.assert_integrity(full_hash=True)
        super().__init__(cfg, donor, stats)
        self.selection_receipt = donor.selection_receipt
