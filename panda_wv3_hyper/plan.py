"""Registered coefficient-only WV3 ablations; paired, round-major on each server."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
from panda_wv3_main_rgb import plan as main
from panda_wv3_ablation_rgb import plan as stage1

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/panda_wv3_hyper'
PLAN = json.loads((CONFIG / 'plan.json').read_text())
CAMPAIGN_ID = PLAN['campaign_id']
REVISION = PLAN['method_revision']
CASES = {c['case_id']: c for c in PLAN['cases']}
HQNR_MIN, ERGAS_MAX = .9585, 2.040
digest, local_path = main.digest, main.local_path


def campaign_root(server):
    if server not in ('s1', 's3'):
        raise ValueError('WV3 hyperparameter campaign belongs to s1 and s3')
    return ROOT / 'work_dir/panda_wv3_hyper' / CAMPAIGN_ID / server / 'WV3'


def donor_pin(repeat):
    if type(repeat) is not int or repeat not in (1, 2, 3):
        raise ValueError('Unregistered donor repetition')
    return deepcopy(PLAN['donors'][f'R{repeat}'])


def resolve(server, case, repeat):
    campaign_root(server)
    if type(repeat) is not int or repeat not in (1, 2, 3):
        raise ValueError('Unregistered repetition')
    recipe = CASES[case]
    if recipe['server'] != server:
        raise ValueError('Case belongs to another server')
    if recipe['stage'] == 'stage1':
        cfg = stage1.resolve(server, 'S1_A5_ALL', repeat)
        cfg.update(lambda_abs=recipe['lambda_spa'], lambda_epsilon=recipe['lambda_epsilon'],
                   label=case, alpha=0., gamma=0., donor_run_id=None,
                   student_width=None, teacher_architecture='W112_D123_BASE',
                   stage1_retrained=True, stage1_warp='BICUBIC_BORDER')
    else:
        cfg = main.resolve(server, 'M0_OURS', repeat)
        pin = donor_pin(repeat)
        cfg.update(alpha=recipe['alpha'], gamma=recipe['gamma'], student_width=112,
                   teacher_architecture='W112_D123_BASE', donor_run_id=pin['run_id'],
                   donor_pin=pin, donor_origin_server='s1', stage1_retrained=False)
    cfg.update(campaign_id=CAMPAIGN_ID, method_revision=REVISION, case_id=case,
               run_id=f'WHYP_WV3_{server.upper()}_{case}_R{repeat}_S{cfg["seed"]}',
               experiment=f'WV3-HYP-{case.replace("_", "-")} R{repeat} (RGB)',
               existing_control_case=recipe['existing_control_case'],
               primary_step=recipe['primary_step'], horizon=recipe['horizon'],
               teacher_width=112, scale=1., beta=0., initialization='FRESH',
               weight_averaging=False, conditional_groups=5,
               baseline_variant=recipe['baseline_variant'],
               historical_control_policy='EXACT_EVIDENCE_OR_FRESH_SAME_SEED',
               historical_primary_source_campaign=(PLAN['source_campaign'] if server=='s1'
                                                   else PLAN['source_stage1_campaign']))
    cfg['config_sha256'] = digest({k:v for k,v in cfg.items() if k!='config_sha256'})
    return cfg


def queue(server):
    campaign_root(server)
    return [resolve(server, case, repeat) for repeat in (1, 2, 3)
            for case, recipe in CASES.items() if recipe['server']==server]


def validate(cfg):
    if cfg != resolve(cfg.get('server'), cfg.get('case_id'), cfg.get('repeat')):
        raise ValueError('Configuration differs from registered coefficient-only recipe')


def catalog(server):
    out = io.StringIO()
    fields = ['order','server','stage','case_id','repeat','seed','horizon','primary_step',
              'architecture','lr','lr_aligner','lambda_abs','lambda_epsilon','alpha','gamma',
              'beta','warp','existing_control_case','experiment']
    writer = csv.DictWriter(out,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for i,cfg in enumerate(queue(server),1):
        writer.writerow(dict(order=i,**{k:cfg.get(k,'') for k in fields if k!='order'}))
    return out.getvalue()
