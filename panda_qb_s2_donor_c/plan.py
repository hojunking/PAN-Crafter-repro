"""Five predeclared fresh30K Students; no architecture or loss sweep."""
from copy import deepcopy
import csv
import io
from panda_input3_c30 import recipes
from panda_qb_lateval import plan as donor_source

ROOT=recipes.ROOT
CAMPAIGN_ID='PANDA_QB_S2_DONOR_C_20261116_v1'
REVISION='QB_S2_RGBN_LATE_DONOR_C_ALPHA0_FIXED30_V1'
CONFIG=ROOT/'config/panda_qb_s2_donor_c'
TEMPLATE=deepcopy(recipes.resolve('s2','C_CORRECTED_DIFF'))
DONOR_CONFIG=deepcopy(donor_source.donor_config('s2'))
DONOR_ROOT=donor_source.donor_root('s2')
STEPS=(40000,48000,50000)
JOBS=((48000,1),(40000,1),(50000,1),(48000,2),(48000,3))
digest,local_path,seeds=donor_source.digest,donor_source.local_path,donor_source.seeds
SELECTOR='PREDECLARED_S2_EXISTING_RGBN_40K_48K_50K_NOT_TEST_RANKED'


def sensor_for(server):
    if server!='s2':raise ValueError('This campaign runs only on physical s2')
    return 'QB'


def targets(sensor):
    if sensor!='QB':raise ValueError('QB only')
    return dict(HQNR_min=.9315,ERGAS_max=3.5)


def campaign_root(server):
    return ROOT/'work_dir/panda_qb_s2_donor_c'/CAMPAIGN_ID/server/sensor_for(server)


def donor_config(server):
    sensor_for(server);return deepcopy(DONOR_CONFIG)


def donor_root(server):
    sensor_for(server);return DONOR_ROOT


def reference_root(root,step):
    if step not in STEPS:raise ValueError('Unregistered late donor')
    return root/'donors'/f'step_{step:07d}'


def resolve(server,case,repeat=1):
    sensor_for(server)
    available={f'D{step//1000}_C30':step for step in STEPS}
    if case not in available or (available[case],repeat) not in JOBS or type(repeat) is not int:
        raise ValueError('Exactly the five predeclared donor/seed cases are registered')
    step=available[case];seed=261018100+repeat
    cfg=deepcopy(TEMPLATE)
    cfg.update(campaign_id=CAMPAIGN_ID,method_revision=REVISION,case_id=case,
        run_id=f'QBDC_QB_S2_{case}_R{repeat}_S{seed}',
        experiment=f'QB-DONOR-C-D{step//1000}-C30 R{repeat} (RGBN)',
        repeat=repeat,replica=f'R{repeat}',seed=seed,stream_seeds=seeds(seed),
        donor_step=step,donor_run_id=DONOR_CONFIG['run_id'],donor_campaign_id=DONOR_CONFIG['campaign_id'],
        donor_selection=dict(policy=SELECTOR,fixed_step=step,uses_test_for_selection=False,
                             thresholds_are_admission_gates=False),
        official_selection_reference='ORIGINAL_PAN',
        checkpoint_selection='ORIGINAL_H_QUALIFIED_MIN_E_ELSE_MAX_H',
        evaluation_reference='SAME_LOCAL_STAGE1_DIFFERENT_PREDECLARED_CHECKPOINT',
        reference_repeats=3 if step==48000 else 1,
        baseline_variant='S2_C30_32K_DONOR_EXISTING_R1_R2_COMPARATORS',
        experiment_kind='FIXED_ARCH_ALPHA0_DONOR_MATURITY_COMPARISON',
        donor_comparison=dict(baseline_donor_step=32000,changed_training_factor='FIXED_PAIRED_ALIGNER_TEACHER_CHECKPOINT',
            repeat_policy='48K_SELECTED_BEFORE_RESULTS_NOT_BEST_OF_40K_48K_50K'))
    for key in ('fixed_donor','config_sha256','schedule_comparison'):
        cfg.pop(key,None)
    cfg['config_sha256']=digest(cfg)
    return cfg


def queue(server):
    return [resolve(server,f'D{step//1000}_C30',repeat) for step,repeat in JOBS]


def validate(cfg):
    if cfg!=resolve(cfg.get('server'),cfg.get('case_id'),cfg.get('repeat')):
        raise ValueError('Immutable five-case donor/architecture/alpha contract changed')


def catalog(server):
    stream=io.StringIO();keys=['run_id','donor_step','repeat','seed','horizon','student_input','width','alpha','gamma','lr']
    writer=csv.DictWriter(stream,fieldnames=keys,lineterminator='\n');writer.writeheader()
    for cfg in queue(server):writer.writerow({k:cfg[k] for k in keys})
    return stream.getvalue()
