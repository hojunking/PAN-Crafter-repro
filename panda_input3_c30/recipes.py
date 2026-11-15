"""Only the horizon and campaign identity change; preserve paired initialization."""
from copy import deepcopy
import json
from panda_input3 import plan as original

ROOT = original.ROOT
CAMPAIGN_ID = 'PANDA_S123_INPUT3_C30_20261115_v1'
ORIGINAL_CAMPAIGN = original.CAMPAIGN_ID
CASES = tuple(original.CASES)
ORIGINAL_CONFIGS = {s: {c: original.resolve(s, c) for c in CASES}
                    for s in ('s1', 's2', 's3')}


def ORIGINAL_ROOT(server):
    # Do not retain the old function: its globals include the plan.CAMPAIGN_ID
    # that the isolated C30 worker replaces.
    return ROOT / 'work_dir/panda_input3' / ORIGINAL_CAMPAIGN / server / original.sensor_for(server)


def campaign_root(server):
    return ROOT / 'work_dir/panda_input3_c30' / CAMPAIGN_ID / server / original.sensor_for(server)


def resolve(server, case, repeat=1):
    if type(repeat) is not int or repeat != 1:
        raise ValueError('One fresh paired-seed C30 run per input case')
    cfg = deepcopy(ORIGINAL_CONFIGS[server][case])
    parent = cfg['config_sha256']
    cfg.update(campaign_id=CAMPAIGN_ID,
        method_revision='INPUT3_UNCHANGED_METHOD_FRESH_COSINE30_V1',
        run_id=cfg['run_id'].replace('INPUT3_', 'INPUT3C30_', 1),
        experiment=cfg['experiment'].replace('-INPUT3-', '-INPUT3-C30-', 1),
        horizon=30000, primary_step=30000, common_comparison_step=30000,
        evaluation_steps=list(range(1000, 30001, 1000)),
        schedule={'name': 'WARMUP_COSINE_HOLD', 'decay_end': 30000, 'min_lr': 0.},
        schedule_comparison={'parent_config_sha256': parent,
            'parent_campaign': ORIGINAL_CAMPAIGN,
            'initialization': 'FRESH_SAME_SEED_NOT_CHECKPOINT_FINETUNING',
            'changed_training_fields': ['horizon', 'schedule.decay_end']})
    cfg.pop('config_sha256')
    cfg['config_sha256'] = original.digest(cfg)
    return cfg


def canonical_queue(server):
    return [resolve(server, case) for case in CASES]


def queue(server):
    path = campaign_root(server) / 'cutover.json'
    if not path.exists():
        return canonical_queue(server)
    from .transition import read_cutover
    return [resolve(server, case) for case in read_cutover(server)['c30_order']]


def validate(cfg):
    if cfg != resolve(cfg.get('server'), cfg.get('case_id'), cfg.get('repeat')):
        raise ValueError('Registered fresh INPUT3-C30 recipe required')
