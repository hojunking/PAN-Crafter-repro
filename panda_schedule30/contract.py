"""Isolated schedule extension; historical campaigns and source files stay immutable."""
from panda_mature30.contract import file_identity, POLICY, STAGE1_SELECTION

CAMPAIGN_ID = 'PANDA_S13_SCHEDULE_HARDONLY_20261023_v1'
METHOD_REVISION = 'PANDA-SCHEDULE30-POSITIVE-HARDONLY-v1'
HQNR_MIN, ERGAS_MAX = .9585, 2.040
BUDGET = {
    'total_seconds': None, 'closeout_seconds': 0,
    'safety_factor': 1.12, 'calibration_reserve_seconds': 600,
    'scope': 'FINITE_REGISTERED_QUEUE_NO_WALL_TIME_LIMIT',
    'restart': 'PRESERVE_ELAPSED_AND_COMPLETE_STATE',
}
SCALE_POLICIES = {'S010': .1, 'S020': .2}


def _case(scale, updates=30000, *, lr=1e-4, alpha=.25, gamma=.5, source=None):
    return dict(policy_id=f'S{round(scale*100):03d}', scale=scale,
                alpha=alpha, gamma=gamma, lr_S=lr, total_updates=updates,
                warmup_updates=100 if source is None else 0,
                min_lr=0. if source is None else 1e-6,
                source_case=source, inherited_updates=0 if source is None else 25000,
                schedule_kind='WARMUP_COSINE' if source is None else 'SOURCE25K_THEN_COSINE')


CASES = {
    'S010C30': _case(.1),
    'S020C30': _case(.2),
    'S020C35': _case(.2, 35000),
    'S020C30LR150': _case(.2, lr=1.5e-4),
    'S020C30H050': _case(.2, alpha=.125, gamma=.25),
    'S020C35H050': _case(.2, 35000, alpha=.125, gamma=.25),
    'S010BASE': _case(.1, 50000),
    'S020BASE': _case(.2, 50000),
    'S020H050': _case(.2, 50000, alpha=.125, gamma=.25),
    'S010F25C35': _case(.1, 35000, lr=2e-5, source='S010BASE'),
    'S020F25C35': _case(.2, 35000, lr=2e-5, source='S020BASE'),
}
