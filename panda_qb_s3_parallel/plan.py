"""Independent same-method s3 QB donor; five complementary full50K Students."""
import csv,io,json
from pathlib import Path
from panda_unified_campaign import plan as base
ROOT=base.ROOT
CONFIG=ROOT/'config/panda_qb_s3_parallel'
CAMPAIGN_ID='PANDA_QB_S3_PARALLEL_20261114_v1'
REVISION='S3_LOCAL_RGBN_A4_E003_CORRECTED_ONLY_ZEROQ_50K_V1'
digest,seeds,local_path=base.digest,base.seeds,base.local_path
STAGE1_CASE='S1_RGBN_FULL'
CASES=[('P1_LR100_G025',1e-4,.25),('P2_LR075_G025',7.5e-5,.25),
       ('P3_LR100_G000',1e-4,0.),('P4_LR100_G0125',1e-4,.125),('P5_LR075_G0125',7.5e-5,.125)]
PLAN={'donor_selection':{'QB':{'ergas_max':3.5,'psnr_min':None,'order':'PSNR_HQNR'}}}
def sensor_for(server):
    if server!='s3':raise ValueError('This independent QB campaign is s3 only')
    return 'QB'
def targets(sensor):return base.targets(sensor)
def campaign_root(server):return ROOT/'work_dir/panda_qb_s3_parallel'/CAMPAIGN_ID/server/sensor_for(server)
def donor_root(server):return campaign_root(server)
def donor_config(server,repeat=1):return resolve(server,STAGE1_CASE,repeat)
def resolve(server,case,repeat=1):
    sensor_for(server)
    if type(repeat)!=int or repeat!=1:raise ValueError('One fresh donor and one paired Student seed per recipe')
    s1=case==STAGE1_CASE
    if not s1 and case not in [c[0] for c in CASES]:raise ValueError('Unregistered recipe')
    c=base.resolve('s2',STAGE1_CASE if s1 else 'S2_RGBN_G025',1)
    seed=261016303 if s1 else 261018101
    lr,gamma=(1e-4,0.) if s1 else next((l,g) for k,l,g in CASES if k==case)
    c.update(server=server,campaign_id=CAMPAIGN_ID,method_revision=REVISION,case_id=case,
        run_id=f'QBS3_QB_S3_{case}_R1_S{seed}',experiment=f'QB-S3-PARALLEL-{case.replace("_","-")} R1 (RGBN)',
        seed=seed,stream_seeds=seeds(seed),lr=lr,horizon=50000,primary_step=50000,common_comparison_step=50000,
        alpha=0.,gamma=gamma,beta=0.,student_input='CORRECTED',weight='1' if s1 else '1+gamma*d_e*r_q',
        q_zero_policy='EXACT_ZERO_RANK_ZERO_POSITIVE_RIGHT_ECDF_V1',
        early_stop={'enabled':False,'reason':'FULL50K_NO_METRIC_STOP_OR_ADMISSION_GATE'},
        validation_steps=[30000,40000,50000] if s1 else [],
        fr_references=['ORIGINAL_PAN','CORRECTED_PAN'],official_selection_reference='CORRECTED_PAN',
        evaluation_reference='SAME_LOCAL_S3_DONOR_CORRECTED_PAN',input_alignment=True,training_support=c['mask_policy'],
        donor_selection=PLAN['donor_selection']['QB'],
        donor_run_id=None if s1 else donor_config(server)['run_id'],
        donor_campaign_id=None if s1 else CAMPAIGN_ID,
        checkpoint_selection='ERGAS_QUALIFIED_MAX_PSNR_CORRECTED_H_ELSE_MIN_ERGAS' if s1 else 'CORRECTED_H_QUALIFIED_MIN_E_ELSE_MAX_H')
    c.pop('config_sha256');c['config_sha256']=digest(c);return c
def queue(server):return [donor_config(server)]+[resolve(server,k) for k,_,_ in CASES]
def validate(cfg):
    if cfg!=resolve(cfg.get('server'),cfg.get('case_id'),cfg.get('repeat')):raise ValueError('Registered s3 QB config required')
def catalog(server):
    s=io.StringIO();keys=['run_id','stage','horizon','width','lr','gamma','lambda_abs','lambda_epsilon','donor_run_id']
    w=csv.DictWriter(s,fieldnames=keys,lineterminator='\n');w.writeheader()
    for c in queue(server):w.writerow({k:c[k] for k in keys})
    return s.getvalue()
def pick_donor(rows):
    if not rows:raise ValueError('No donor evaluations')
    good=[r for r in rows if r['rr']['ERGAS']<=3.5]
    return min(good,key=lambda r:(-r['rr']['PSNR'],-r['fr_corrected_pan']['HQNR'],r['rr']['ERGAS'],r['step'])) if good else min(rows,key=lambda r:(r['rr']['ERGAS'],-r['rr']['PSNR'],-r['fr_corrected_pan']['HQNR'],r['step']))
