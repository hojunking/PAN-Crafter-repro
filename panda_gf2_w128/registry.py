from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CAMPAIGN_ID = 'PANDA_S3_GF2_W128_FIXEDDONOR_20261106_v1'
METHOD_REVISION = 'PANDA-GF2-W128-S100-DIFFUBIL-RGB-L4-HARDONLY-v1'
OUTPUT = 'work_dir/panda_gf2_w128'
TARGET = {'HQNR': .9645, 'ERGAS': .522, 'PSNR': 45.08}
RECIPES = (
    ('G1_LR200', 2e-4, .25, .5),
    ('G2_LR150', 1.5e-4, .25, .5),
    ('G3_LR250', 2.5e-4, .25, .5),
    ('G4_LR200_H050', 2e-4, .125, .25),
    ('G5_LR150_H050', 1.5e-4, .125, .25),
    ('G6_LR250_H050', 2.5e-4, .125, .25),
)
RUNS = tuple((case, rep) for rep in ('R1', 'R2') for case, *_ in RECIPES)
BASELINE_CAMPAIGN = 'PANDA_S3_GF2_PSNR4508_FIXEDDONOR_20261030_v1'
BASELINE_RUN = 'PSFC_GF2_S3_GF2P_C60LR200_NOSOFT_R2_S261018102'
BASELINE_DIRECTORY = f'work_dir/panda_gf2_psnr/{BASELINE_CAMPAIGN}/s3/GF2/runs/{BASELINE_RUN}/attempt_01'
BASELINE_SHA = '9027206526d343b5a82aa735057f13de7ed0778ec2ad8fadaec73d27760182d6'
BASELINE_EVALUATION_SHA = '4760c2745ad9ee071a6b72bacd06f62cb9fcba74998f7207abf01c961b30a951'


def cases():
    from panda_sensor_focus.contract import _fresh
    return {case: _fresh('GF2', updates=60000, lr=lr, width=128, alpha=alpha, gamma=gamma)
            for case, lr, alpha, gamma in RECIPES}


def manifest():
    from panda_sensor_focus.contract import SOURCE_DONORS
    return {'campaign_id': CAMPAIGN_ID, 'server': 's3', 'dataset': 'GF2',
            'target': deepcopy(TARGET), 'recipes': cases(), 'runs': RUNS,
            'stage1_runs': 0, 'student_runs': 12, 'round_major': True,
            'frozen_donor': deepcopy(SOURCE_DONORS['s3']),
            'student_width': 128, 'teacher_width': 112, 'bands': 4,
            'horizon': 60000, 'batch': 48, 'microbatch': 12,
            'beta': 0., 'aligner_scale': 1., 'time_limit': None,
            'source_stage1_server': 's3', 'cross_server_weight_transfer': False,
            'baseline': {'campaign_id': BASELINE_CAMPAIGN, 'run_id': BASELINE_RUN,
                'selected_step': 47000, 'width': 112, 'lr': 2e-4,
                'checkpoint_sha256': BASELINE_SHA, 'evaluation_sha256': BASELINE_EVALUATION_SHA},
            'comparison': 'G1 changes only Student width versus GF2P_C60LR200; all runs fresh',
            'historical_W128_LR100': 'Already completed R1/R2 in SENSOR_FOCUS; do not duplicate',
            'excluded': ['new_Stage1', 'rotation_loss', 'SWA', 'EMA', 'TTA', 'soft_loss', 'band_weighted_loss']}
