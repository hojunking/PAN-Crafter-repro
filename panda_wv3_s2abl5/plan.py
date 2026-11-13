import csv
import io
import json
from panda_eq40_zeroq import plan as main
from panda_unified_campaign.plan import seeds

ROOT = main.ROOT
CONFIG = ROOT/'config/panda_wv3_s2abl5'
PLAN = json.loads((CONFIG/'plan.json').read_text())
CAMPAIGN_ID = PLAN['campaign_id']
REVISION = 'WV3_S2ABL5_FIXED_DONOR_RGBN_L4_UBIL_C40_V1'
digest, local_path, targets = main.digest, main.local_path, main.targets
CASES = {
    'A_RAW_L1': ('RAW', 'UNIFORM', '1'),
    'B_ALIGN_L1': ('CORRECTED', 'UNIFORM', '1'),
    'C_DE_ONLY': ('CORRECTED', 'DE_ONLY', '1+0.25*d_e'),
    'D_RQ_ONLY': ('CORRECTED', 'RQ_ONLY', '1+0.25*r_q'),
    'E_OURS': ('CORRECTED', 'PRODUCT', '1+0.25*d_e*r_q'),
}

def sensor_for(server):
    if server != 's1': raise ValueError('Fixed local s1 donor; no cross-server weights')
    return 'WV3'

def campaign_root(server):
    return ROOT/'work_dir/panda_wv3_s2abl5'/CAMPAIGN_ID/server/sensor_for(server)

def source_config(): return main.queue('s1')[0]
def source_directory(): return main.campaign_root('s1')/'runs'/source_config()['run_id']

def resolve(server, case, repeat):
    sensor_for(server)
    if case not in CASES or type(repeat) is not int or repeat not in (1,2,3):
        raise ValueError('Exactly five cases and three repeats are registered')
    cfg = source_config()
    view, variant, formula = CASES[case]
    seed = 261018100 + repeat
    cfg.update(campaign_id=CAMPAIGN_ID, method_revision=REVISION, case_id=case,
        run_id=f'S2ABL5_WV3_S1_{case}_R{repeat}_S{seed}',
        experiment=f'WV3-S2ABL5-{case.replace("_","-")} R{repeat} (RGBN)',
        repeat=repeat, replica=f'R{repeat}', seed=seed, stream_seeds=seeds(seed),
        donor_repeat=1, donor_run_id=PLAN['fixed_donor']['run_id'],
        fixed_donor=PLAN['fixed_donor'], student_input=view, input_alignment=view=='CORRECTED',
        baseline_variant='FIXED_DONOR_COMPONENT_'+case,
        alignment='REFERENCE_ONLY_FROZEN' if view=='RAW' else 'LEARNED_FROZEN',
        weight_variant=variant, weight=formula, alpha=0., gamma=0. if variant=='UNIFORM' else .25,
        beta=0., reference_repeats=3, evaluation_reference=PLAN['reference'],
        training_support='FIXED_DONOR_COMMON_ERODED_LANCZOS_SUPPORT_ALL_CASES',
        official_selection_reference='CORRECTED_PAN',
        checkpoint_selection='CORRECTED_HQNR_QUALIFIED_MIN_ERGAS_ELSE_MAX_CORRECTED_HQNR',
        reuse_existing=bool(case=='E_OURS' and repeat==1),
        source_main_config_sha256=source_config()['config_sha256'])
    cfg.pop('config_sha256'); cfg['config_sha256']=digest(cfg)
    return cfg

def queue(server): return [resolve(server,c,r) for r in (1,2,3) for c in PLAN['case_order']]
def validate(cfg):
    if cfg != resolve(cfg.get('server'),cfg.get('case_id'),cfg.get('repeat')):
        raise ValueError('Unregistered S2ABL5 configuration')

def catalog(server):
    out=io.StringIO();keys=['order','case_id','repeat','seed','student_input','weight','horizon','lr','reuse_existing','donor_run_id']
    writer=csv.DictWriter(out,fieldnames=keys,lineterminator='\n');writer.writeheader()
    for i,c in enumerate(queue(server),1):writer.writerow(dict(order=i,**{k:c[k] for k in keys if k!='order'}))
    return out.getvalue()
