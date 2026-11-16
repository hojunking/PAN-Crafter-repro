"""C30 input C, fixed local s3 donor34K, three Student seeds."""
from copy import deepcopy
import csv
import io
import json
from panda_input3_c30.recipes import resolve as c30_resolve
from panda_qb_s3_parallel import plan as donor_plan

ROOT = donor_plan.ROOT
CONFIG = ROOT / 'config/panda_qb_s3_input_c3'
CAMPAIGN_ID = 'PANDA_QB_S3_INPUT_C3_20261116_v1'
CASE = 'C_CORRECTED_DIFF'
digest, seeds, local_path = donor_plan.digest, donor_plan.seeds, donor_plan.local_path
FIXED = json.loads((CONFIG / 'donor.json').read_text())
TEMPLATE = c30_resolve('s2', CASE)
DONOR_CONFIG = donor_plan.donor_config('s3')


def sensor_for(server):
    if server != 's3': raise ValueError('This QB C3 queue belongs to physical s3')
    return 'QB'


def targets(sensor):
    if sensor != 'QB': raise ValueError('QB only')
    return {'HQNR_min': .9315, 'ERGAS_max': 3.5}


def campaign_root(server):
    return ROOT / 'work_dir/panda_qb_s3_input_c3' / CAMPAIGN_ID / server / sensor_for(server)


def donor_root(server):
    sensor_for(server)
    return ROOT / 'work_dir/panda_qb_s3_parallel' / donor_plan.CAMPAIGN_ID / 's3/QB'


def donor_config(server):
    sensor_for(server)
    return deepcopy(DONOR_CONFIG)


def resolve(server, case, repeat=1):
    sensor_for(server)
    if case != CASE or type(repeat) is not int or repeat not in (1, 2, 3):
        raise ValueError('Only input C with repeats R1/R2/R3 is registered')
    cfg = deepcopy(TEMPLATE); parent = cfg.pop('config_sha256'); seed = 261018100 + repeat
    cfg.update(server='s3', campaign_id=CAMPAIGN_ID, case_id=CASE,
        run_id=f'QBC3_QB_S3_{CASE}_R{repeat}_S{seed}',
        experiment=f'QB-INPUT3-C30-C-CORRECTED-DIFF s3 R{repeat} (RGBN)',
        repeat=repeat, replica=f'R{repeat}', seed=seed, stream_seeds=seeds(seed),
        donor_run_id=FIXED['run_id'], donor_step=FIXED['step'], fixed_donor=deepcopy(FIXED),
        donor_campaign_id=donor_plan.CAMPAIGN_ID,
        donor_selection={'policy':'EXISTING_S3_QB_MAIN_FIXED_34K_ALL_THREE_SEEDS',
                         'uses_test_for_selection':True, 'new_stage1_runs':0},
        official_selection_reference='ORIGINAL_PAN',
        checkpoint_selection='ORIGINAL_H_QUALIFIED_MIN_E_NO_FALLBACK', reference_repeats=3,
        evaluation_reference='SAME_FIXED_LOCAL_S3_QB_34K_DONOR_ALL_THREE_SEEDS',
        repetition_comparison={'parent_s2_c30_config_sha256':parent,
            'new_stage1_runs':0,'fresh_student':True,
            'changed_training_fields':['local_donor','calibration_binding','seed','stream_seeds']})
    cfg['schedule_comparison']['initialization'] = 'FRESH_THREE_SEEDS_NOT_CHECKPOINT_FINETUNING'
    cfg['config_sha256'] = digest(cfg)
    return cfg


def queue(server):
    return [resolve(server, CASE, r) for r in (1, 2, 3)]


def validate(cfg):
    if cfg != resolve(cfg.get('server'), cfg.get('case_id'), cfg.get('repeat')):
        raise ValueError('Registered s3 QB C3 recipe required')


def catalog(server):
    out = io.StringIO()
    keys = ['run_id','server','sensor','repeat','seed','student_input','horizon','width','lr',
            'alpha','beta','gamma','donor_run_id','donor_step']
    w = csv.DictWriter(out, fieldnames=keys, lineterminator='\n'); w.writeheader()
    w.writerows({k:c[k] for k in keys} for c in queue(server))
    return out.getvalue()
