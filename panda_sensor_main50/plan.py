"""Five fresh 50K Stage2 main recipes per sensor; no performance admission gate."""
import csv
import io
import json
from panda_eq40_zeroq import plan as main
from panda_unified_campaign.plan import seeds

ROOT=main.ROOT
CONFIG=ROOT/'config/panda_sensor_main50'
PLAN=json.loads((CONFIG/'plan.json').read_text())
CAMPAIGN_ID=PLAN['campaign_id']
REVISION='MAIN50_CORRECTED_ONLY_RGBN_ZEROQ_NO_PERFORMANCE_STOP_V1'
digest,local_path,targets=main.digest,main.local_path,main.targets

def sensor_for(server):
    if server not in PLAN['servers']:raise ValueError('s2 QB or s3 GF2 required')
    return PLAN['servers'][server]

def campaign_root(server):
    return ROOT/'work_dir/panda_sensor_main50'/CAMPAIGN_ID/server/sensor_for(server)

def donor_config(server):
    sensor=sensor_for(server)
    return main.donor_config(server,PLAN['donors'][sensor]['repeat'])

def donor_root(server):return main.donor_root(server)

def resolve(server,case,repeat=1):
    sensor=sensor_for(server)
    matches=[r for r in PLAN['cases'][sensor] if r[0]==case]
    if len(matches)!=1 or type(repeat)is not int or repeat!=1:
        raise ValueError('Five recipes, one paired Student seed each')
    _,lr,gamma=matches[0]
    cfg=main.queue(server)[0]
    donor=PLAN['donors'][sensor]
    seed=PLAN['seed']
    cfg.update(campaign_id=CAMPAIGN_ID,method_revision=REVISION,case_id=case,
        run_id=f'MAIN50_{sensor}_{server.upper()}_{case}_R1_S{seed}',
        experiment=f'{sensor}-MAIN50-{case.replace("_","-")} R1 (RGBN)',
        stage='stage2',repeat=1,replica='R1',seed=seed,stream_seeds=seeds(seed),
        horizon=50000,primary_step=50000,common_comparison_step=50000,
        evaluation_steps=list(range(1000,50001,1000)),validation_steps=[],
        lr=lr,alpha=0.,gamma=gamma,beta=0.,
        early_stop={'enabled':False,'reason':'USER_REQUEST_FULL_50K_NO_METRIC_STOP'},
        fixed_donor=donor,donor_run_id=donor['run_id'],
        donor_campaign_id=donor_config(server)['campaign_id'],
        donor_repeat=donor['repeat'],donor_selection={'policy':'PINNED_SAME_SERVER_NO_PERFORMANCE_GATE'},
        student_input='CORRECTED',input_alignment=True,weight_variant='PRODUCT',
        weight=f'1+{gamma:g}*d_e*r_q; r_q=0 if q=0',
        baseline_variant='MAIN_TUNING_FULL50_NO_METRIC_STOP',
        evaluation_reference='FIXED_LOCAL_DONOR_CORRECTED_PAN',
        training_support='COMMON_ERODED_LANCZOS4_SUPPORT',
        official_selection_reference='CORRECTED_PAN',
        checkpoint_selection='CORRECTED_HQNR_QUALIFIED_MIN_ERGAS_ELSE_MAX_CORRECTED_HQNR',
        reference_repeats=1,reuse_existing=False,experiment_kind='MAIN_TUNING')
    cfg.pop('config_sha256');cfg['config_sha256']=digest(cfg)
    return cfg

def queue(server):return [resolve(server,c[0]) for c in PLAN['cases'][sensor_for(server)]]

def validate(cfg):
    if cfg!=resolve(cfg.get('server'),cfg.get('case_id'),cfg.get('repeat')):
        raise ValueError('Unregistered main50 config')

def catalog(server):
    out=io.StringIO();keys=['case_id','sensor','width','lr','gamma','horizon','seed','donor_run_id']
    writer=csv.DictWriter(out,fieldnames=keys,lineterminator='\n');writer.writeheader()
    for c in queue(server):writer.writerow({k:c[k] for k in keys})
    return out.getvalue()
