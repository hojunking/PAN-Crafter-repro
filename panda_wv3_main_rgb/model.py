"""Full Ours with a freshly trained RGB Stage1 reference on the same server."""
from panda_wv3_ablation_rgb.model import Stage1
from panda_wv3_ablation.model import Stage2 as AnchoredStage2


class Stage2(AnchoredStage2):
    def __init__(self, cfg, donor, stats):
        if (cfg['alpha'] <= 0 or cfg['gamma'] <= 0 or cfg['beta'] != 0
                or cfg['student_input'] != 'CORRECTED_DIFF' or cfg['scale'] != 1.):
            raise ValueError('RGB main requires every Ours component and full-scale alignment')
        donor.assert_frozen()
        if donor.binding.get('alignment_target') != 'SCHARR_OF_RGB_MEAN_GT':
            raise ValueError('Fresh RGB Stage1 reference required for RGB main')
        stats.validate_binding(donor.binding)
        stats.assert_integrity(full_hash=True)
        super().__init__(cfg, donor, stats)
        self.selection_receipt = donor.selection_receipt
