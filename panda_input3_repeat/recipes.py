"""R2 changes Student randomness only; reuse each server's frozen main donor."""
from copy import deepcopy
from panda_input3_c30 import recipes as previous
from panda_unified_campaign.plan import seeds

ROOT = previous.ROOT
CAMPAIGN_ID = 'PANDA_S123_INPUT3_C30_R2_20261116_v1'
CASES = previous.CASES
PARENTS = {s: {c: previous.resolve(s, c) for c in CASES} for s in ('s1', 's2', 's3')}


def campaign_root(server):
    return ROOT / 'work_dir/panda_input3_repeat' / CAMPAIGN_ID / server / previous.original.sensor_for(server)


def resolve(server, case, repeat=2):
    if type(repeat) is not int or repeat != 2:
        raise ValueError('This queue adds exactly one R2 cycle')
    cfg = deepcopy(PARENTS[server][case]); parent = cfg.pop('config_sha256')
    seed = 261018102
    cfg.update(campaign_id=CAMPAIGN_ID, repeat=2, replica='R2', seed=seed,
        stream_seeds=seeds(seed), reference_repeats=2,
        run_id=f'INPUT3C30_{cfg["sensor"]}_{server.upper()}_{case}_R2_S{seed}',
        experiment=cfg['experiment'].replace(' R1 ', ' R2 '),
        official_selection_reference='ORIGINAL_PAN',
        checkpoint_selection='ORIGINAL_H_QUALIFIED_MIN_E_NO_FALLBACK',
        repetition_comparison={'parent_campaign': previous.CAMPAIGN_ID,
            'parent_config_sha256': parent, 'parent_seed': PARENTS[server][case]['seed'],
            'changed_training_fields': ['seed', 'stream_seeds'],
            'initialization': 'FRESH_STUDENT_NEW_SEED_SHARED_ACROSS_A_B_C',
            'new_stage1_training': False})
    cfg['schedule_comparison']['initialization'] = 'FRESH_R2_PAIRED_SEED_NOT_CHECKPOINT_FINETUNING'
    cfg['config_sha256'] = previous.original.digest(cfg)
    return cfg


def queue(server):
    return [resolve(server, c) for c in CASES]


def validate(cfg):
    if cfg != resolve(cfg.get('server'), cfg.get('case_id'), cfg.get('repeat')):
        raise ValueError('Registered INPUT3-C30 R2 recipe required')
