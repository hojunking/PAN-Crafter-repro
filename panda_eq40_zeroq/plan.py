"""One shared Stage2 method, original per-sensor recipes and local Stage1 donors."""
import csv,io,json
from panda_wv3_s2_eq40 import plan as wv3
from panda_sensor_eq40 import plan as sensors
ROOT=wv3.ROOT
CONFIG=ROOT/'config/panda_eq40_zeroq'
CAMPAIGN_ID='PANDA_S123_EQ40_ZEROQ_20261113_v1'
REVISION='CORRECTED_ONLY_A0_G025_C40_RGBN_DUAL_FR_ZEROQ_V1'
STAGE2_CASE='S2_RGBN_CORRECTED_A000_G025_C40_QZERO'
Q_ZERO_POLICY='EXACT_ZERO_RANK_ZERO_POSITIVE_RIGHT_ECDF_V1'
digest,local_path=wv3.digest,wv3.local_path

def sensor_for(server):
    if server not in ('s1','s2','s3'):raise ValueError('Explicit server required')
    return {'s1':'WV3','s2':'QB','s3':'GF2'}[server]
def prior_plan(server):
    sensor_for(server);return wv3 if server=='s1' else sensors
def prior_config(server,repeat):
    p=prior_plan(server);return p.resolve(server,p.STAGE2_CASE,repeat)
def prior_root(server):return prior_plan(server).campaign_root(server)
def donor_config(server,repeat):
    return wv3.donor_config(repeat) if server=='s1' else sensors.donor_config(server,repeat)
def donor_root(server):return wv3.donor_root() if server=='s1' else sensors.donor_root(server)
def targets(sensor):return wv3.targets(sensor) if sensor=='WV3' else sensors.targets(sensor)
def campaign_root(server):return ROOT/'work_dir/panda_eq40_zeroq'/CAMPAIGN_ID/server/sensor_for(server)
def resolve(server,case,repeat):
    if case!=STAGE2_CASE or type(repeat)is not int or repeat not in (1,2):raise ValueError('Two registered Stage2 repeats only')
    base=prior_config(server,repeat)
    base.update(campaign_id=CAMPAIGN_ID,method_revision=REVISION,case_id=case,
        previous_stage2_config_sha256=base['config_sha256'],q_zero_policy=Q_ZERO_POLICY,
        run_id=f'ZQ40_{sensor_for(server)}_{server.upper()}_R{repeat}_S{base["seed"]}',
        experiment=f'{sensor_for(server)}-EQ40-QZERO-A000-G025-LR{round(base["lr"]*1e6):03d} R{repeat} (RGBN)',
        weight='1+0.25*d_e*r_q; r_q=0 if q=0, right-ECDF otherwise',
        baseline_variant='CORRECTED_ONLY_ZERO_DISCREPANCY_GUARD')
    base.pop('config_sha256');base['config_sha256']=digest(base)
    return base
def queue(server):return [resolve(server,STAGE2_CASE,r) for r in (1,2)]
def validate(cfg):
    if cfg!=resolve(cfg.get('server'),cfg.get('case_id'),cfg.get('repeat')):raise ValueError('Unregistered ZEROQ config')
def catalog(server):
    out=io.StringIO();keys=['run_id','repeat','width','lr','horizon','alpha','gamma','q_zero_policy','donor_run_id']
    w=csv.DictWriter(out,fieldnames=keys,lineterminator='\n');w.writeheader()
    for c in queue(server):w.writerow({k:c[k] for k in keys})
    return out.getvalue()
