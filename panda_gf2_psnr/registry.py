"""Registered fresh Students: four controlled comparisons, two matched seeds each."""
from copy import deepcopy

CAMPAIGN_ID = 'PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1'
METHOD_REVISION = 'PANDA-GF2-PSNR4508-S100-DIFFUBIL-RGB-L4-HARDONLY-v1'
OLD_CAMPAIGN = 'PANDA_S13_SENSOR_FOCUS_FIXEDDONOR_20261029_v1'
TARGET = {'HQNR': .9645, 'ERGAS': .522, 'PSNR': 45.08}
OUTPUT = 'work_dir/panda_gf2_psnr'
OLD_RUNS = (('GF2_C60', 'R2'), ('GF2_C60LR150', 'R1'), ('GF2_C60LR150', 'R2'),
            ('GF2_C60W128', 'R1'), ('GF2_C60W128', 'R2'))
RUNS = tuple((case, rep) for case in (
    'GF2P_C60LR200', 'GF2P_C50LR150', 'GF2P_C60LR150A0125', 'GF2P_C60LR150W128')
    for rep in ('R1', 'R2'))


def cases():
    from panda_sensor_focus.contract import _fresh
    return {
        'GF2P_C60LR200': _fresh('GF2', updates=60000, lr=2e-4),
        'GF2P_C50LR150': _fresh('GF2', updates=50000, lr=1.5e-4),
        'GF2P_C60LR150A0125': _fresh('GF2', updates=60000, lr=1.5e-4, alpha=.125),
        'GF2P_C60LR150W128': _fresh('GF2', updates=60000, lr=1.5e-4, width=128),
    }


def manifest():
    return {'campaign_id': CAMPAIGN_ID, 'server': 's3', 'dataset': 'GF2',
            'joint_target': deepcopy(TARGET), 'cases': cases(), 'runs': RUNS,
            'stage1': 'REUSE_EXACT_AUTHENTICATED_RGB_MATURE_U23_42000',
            'initialization': 'FRESH_STUDENT', 'after': 'EXISTING_FOCUS_CONTROLLER_COMPLETED_INCLUDING_WV3',
            'selection': 'KEEP_HE_ALL_PASSES_AND_HE_REPRESENTATIVE; ANNOTATE_PSNR_AND_THREE_METRIC_PASS',
            'prohibited': ['SWA', 'EMA', 'TTA', 'rotation_loss', 'soft_loss', 'band_weighted_loss']}
