"""s2 QB local RGBN continuation and s3 fresh GF2 paired runs."""
import csv,io,json
from panda_unified_campaign import plan as old
ROOT=old.ROOT
CONFIG=ROOT/'config/panda_sensor_eq40'
PLAN=json.loads((CONFIG/'plan.json').read_text())
CAMPAIGN_ID,REVISION=PLAN['campaign_id'],PLAN['method_revision']
STAGE1_CASE='S1_RGBN_FULL'
STAGE2_CASE='S2_RGBN_CORRECTED_A000_G025_C40'
digest,seeds,local_path=old.digest,old.seeds,old.local_path

def sensor_for(server):
    if server not in PLAN['servers']:raise ValueError('Explicit s2 QB or s3 GF2 required')
    return PLAN['servers'][server]
def targets(sensor):return dict(PLAN['targets'][sensor])
def reference_bands(sensor,mode):
    if sensor not in ('QB','GF2') or mode!='RGBN':raise ValueError('Four-band RGBN only')
    return [2,1,0,3]
def campaign_root(server):return ROOT/'work_dir/panda_sensor_eq40'/CAMPAIGN_ID/server/sensor_for(server)
def donor_root(server):return old.campaign_root('s2') if server=='s2' else campaign_root(server)
def donor_config(server,repeat):
    return old.resolve('s2',STAGE1_CASE,1) if server=='s2' else resolve(server,STAGE1_CASE,repeat)
def stage1_cases(server):return [STAGE1_CASE]
def donor_case(cfg):return STAGE1_CASE

def resolve(server,case,repeat):
    sensor=sensor_for(server)
    if repeat not in (1,2) or type(repeat) is not int:raise ValueError('R1/R2 required')
    if case not in (STAGE1_CASE,STAGE2_CASE) or (server=='s2' and case==STAGE1_CASE):
        raise ValueError('QB Stage1 is the existing external local run; no new training')
    s1=case==STAGE1_CASE
    base=old.resolve('s2',STAGE1_CASE if s1 else 'S2_RGBN_G025',1)
    seed=261016300+repeat if s1 else 261018100+repeat
    base.update(campaign_id=CAMPAIGN_ID,method_revision=REVISION,server=server,sensor=sensor,dataset=sensor,
        case_id=case,repeat=repeat,replica=f'R{repeat}',seed=seed,stream_seeds=seeds(seed),
        run_id=f'SEQ40_{sensor}_{server.upper()}_{case}_R{repeat}_S{seed}',
        experiment=f'{sensor}-EQ40-{case.replace("_","-")} R{repeat} (RGBN)',
        lambda_abs=2. if s1 else 0.,lambda_epsilon=.003 if s1 else 0.,
        alpha=0.,gamma=0. if s1 else .25,beta=0.,student_input='CORRECTED',
        horizon=50000 if s1 else 40000,primary_step=50000 if s1 else 40000,
        common_comparison_step=50000 if s1 else 40000,
        lr=1e-4 if s1 else PLAN['student_lr'][sensor],
        weight='1' if s1 else '1+0.25*d_e*r_q',
        baseline_variant='FRESH_UNIFIED_RGBN_STAGE1' if s1 else 'CORRECTED_ONLY_NO_ALPHA',
        donor_selection=PLAN['donor_selection'][sensor],
        fr_references=['ORIGINAL_PAN','CORRECTED_PAN'],official_selection_reference='ORIGINAL_PAN')
    base['evaluation_steps']=list(range(1000,base['horizon']+1,1000))
    base['validation_steps']=[30000,40000,50000] if s1 else []
    base['donor_run_id']=None if s1 else donor_config(server,repeat)['run_id']
    base['donor_campaign_id']=None if s1 else donor_config(server,repeat)['campaign_id']
    base['checkpoint_selection']='EXPLICIT_TEST_AWARE_ERGAS_PSNR_HQNR' if s1 else 'HQNR_QUALIFIED_MIN_ERGAS_ELSE_MAX_HQNR'
    base['early_stop']['enabled']=sensor=='QB' and not s1
    base.pop('config_sha256');base['config_sha256']=digest(base)
    return base

def queue(server):
    return ([resolve(server,STAGE2_CASE,r) for r in (1,2)] if server=='s2' else
            [resolve(server,c,r) for c in (STAGE1_CASE,STAGE2_CASE) for r in (1,2)])
def validate(cfg):
    if cfg!=resolve(cfg.get('server'),cfg.get('case_id'),cfg.get('repeat')):raise ValueError('Unregistered sensor EQ40 configuration')
def catalog(server):
    s=io.StringIO();keys=['run_id','stage','repeat','horizon','width','lr','alpha','gamma','lambda_abs','lambda_epsilon','donor_run_id']
    w=csv.DictWriter(s,fieldnames=keys,lineterminator='\n');w.writeheader()
    for c in queue(server):w.writerow({k:c[k] for k in keys})
    return s.getvalue()
