"""Existing same-server 40/48/50K donors crossed with three Stage2 schedules."""
import csv
import io
from pathlib import Path
from panda_qb_lateval import plan as source

ROOT = source.ROOT
CONFIG = ROOT / 'config/panda_qb_donorgrid'
CAMPAIGN_ID = 'PANDA_QB_S23_DONORGRID_20261115_v1'
REVISION = 'QB_RGBN_L4_COMMON_W128_UBIL_FIXED_DONOR3_SCHEDULE3_V1'
digest, local_path, seeds = source.digest, source.local_path, source.seeds
sensor_for, targets, assert_method = source.sensor_for, source.targets, source.assert_method
donor_config, donor_root = source.donor_config, source.donor_root
CANDIDATE_STEPS = (40000, 48000, 50000)
SELECTOR = 'PREDECLARED_FIXED_STAGE1_40K_48K_50K_NO_RANKING'
SCHEDULES = [('C50', 50000, 0.), ('C15_HOLD', 15000, 1e-6), ('C25_HOLD', 25000, 1e-6)]


def campaign_root(server):
    return ROOT / 'work_dir/panda_qb_donorgrid' / CAMPAIGN_ID / server / sensor_for(server)


def reference_root(root, step):
    if type(step) is not int or step not in CANDIDATE_STEPS:
        raise ValueError('Only existing 40K, 48K and 50K donors are registered')
    return Path(root) / 'donors' / f'step_{step:07d}'


def case_id(step, schedule):
    return f'D{step//1000:02d}_{schedule}'


def resolve(server, case, repeat=1):
    sensor_for(server)
    choices={case_id(step,key):(step,key,end,floor) for key,end,floor in SCHEDULES for step in CANDIDATE_STEPS}
    if case not in choices or type(repeat) is not int or repeat != 1:
        raise ValueError('Registered donor/schedule combination and one paired Student seed required')
    step,key,end,floor=choices[case]
    cfg=source.resolve(server)
    cfg.update(campaign_id=CAMPAIGN_ID, method_revision=REVISION, case_id=case,
        run_id=f'QBDG_QB_{server.upper()}_{case}_R1_S261018101',
        experiment=f'QB-DONORGRID-{case.replace("_", "-")} {server} R1 (RGBN)',
        donor_step=step,
        donor_selection={'policy':SELECTOR, 'fixed_step':step,
                         'uses_test_for_selection':False, 'performance_admission_gate':False},
        schedule={'name':'WARMUP_COSINE_HOLD','decay_end':end,'min_lr':floor},
        reuse_existing=False,experiment_kind='MAIN_FIXED_DONOR_SCHEDULE_COMPARISON',
        baseline_variant='EXISTING_LOCAL_FIXED_STAGE1_STEP',
        evaluation_reference='FIXED_LOCAL_STAGE1_CHECKPOINT_CORRECTED_PAN')
    cfg.pop('config_sha256')
    assert_method(cfg)
    cfg['config_sha256']=digest(cfg)
    return cfg


def queue(server):
    # First compare all three donors under the same baseline, then each schedule.
    return [resolve(server,case_id(step,key)) for key,_,_ in SCHEDULES for step in CANDIDATE_STEPS]


def validate(cfg):
    if cfg!=resolve(cfg.get('server'),cfg.get('case_id'),cfg.get('repeat')):
        raise ValueError('Registered QB donor-grid configuration required')


def catalog(server):
    out=io.StringIO()
    keys=['server','case_id','donor_step','horizon','width','lr','gamma','decay_end','min_lr','seed','donor_run_id']
    writer=csv.DictWriter(out,fieldnames=keys,lineterminator='\n');writer.writeheader()
    for cfg in queue(server):
        value={**cfg,**cfg['schedule']};writer.writerow({key:value[key] for key in keys})
    return out.getvalue()
