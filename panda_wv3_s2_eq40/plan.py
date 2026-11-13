"""s1-only corrected-PAN Student, preserving the already trained local Stage1."""
import csv, io, json
from panda_unified_campaign import plan as old
ROOT=old.ROOT
CONFIG=ROOT/'config/panda_wv3_s2_eq40'
PLAN=json.loads((CONFIG/'plan.json').read_text())
CAMPAIGN_ID=PLAN['campaign_id']
REVISION=PLAN['method_revision']
STAGE2_CASE='S2_RGBN_CORRECTED_A000_G025_C40_LR110'
digest=old.digest
local_path=old.local_path

def sensor_for(server):
    if server!='s1':raise ValueError('This follow-up is s1 WV3 only')
    return 'WV3'
def targets(sensor):return old.targets(sensor)
def campaign_root(server):return ROOT/'work_dir/panda_wv3_s2_eq40'/CAMPAIGN_ID/server/sensor_for(server)
def donor_root():return old.campaign_root('s1')
def donor_config(repeat):return old.resolve('s1',old.STAGE1_CASE,repeat)
def resolve(server,case,repeat):
    sensor_for(server)
    if case!=STAGE2_CASE:raise ValueError('Unknown EQ40 case')
    cfg=old.resolve(server,old.STAGE2_CASE,repeat)
    cfg.update(campaign_id=CAMPAIGN_ID,method_revision=REVISION,case_id=case,
        run_id=f'EQ40_WV3_S1_{case}_R{repeat}_S{cfg["seed"]}',
        experiment=f'WV3-EQ40-CORRECTED-A000-G025-LR110 R{repeat} (RGBN)',
        alpha=0.,gamma=.25,horizon=40000,primary_step=40000,common_comparison_step=40000,
        student_input='CORRECTED',weight='1+0.25*d_e*r_q',
        baseline_variant='CORRECTED_ONLY_NO_ALPHA_FROZEN_UNIFIED_STAGE1',
        evaluation_steps=list(range(1000,40001,1000)),
        donor_selection=PLAN['donor_selection'],donor_campaign_id=old.CAMPAIGN_ID,
        fr_references=['ORIGINAL_PAN','CORRECTED_PAN'],official_selection_reference='ORIGINAL_PAN')
    cfg.pop('config_sha256');cfg['config_sha256']=digest(cfg)
    return cfg
def queue(server):return [resolve(server,STAGE2_CASE,r) for r in (1,2)]
def validate(cfg):
    if cfg!=resolve(cfg.get('server'),cfg.get('case_id'),cfg.get('repeat')):raise ValueError('Unregistered EQ40 config')
def catalog(server):
    s=io.StringIO();keys=['run_id','repeat','horizon','lr','alpha','gamma','student_input','donor_run_id']
    w=csv.DictWriter(s,fieldnames=keys,lineterminator='\n');w.writeheader()
    for c in queue(server):w.writerow({k:c[k] for k in keys})
    return s.getvalue()
