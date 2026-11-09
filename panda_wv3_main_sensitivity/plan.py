"""Paired Stage2 coefficient sensitivity at the actual main LR, without reselection."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
from panda_wv3_main_rgb import plan as main

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/panda_wv3_main_sensitivity'
PLAN = json.loads((CONFIG / 'plan.json').read_text())
CAMPAIGN_ID, REVISION = PLAN['campaign_id'], PLAN['method_revision']
CASES = {c['case_id']: c for c in PLAN['cases']}
HQNR_MIN, ERGAS_MAX = .9585, 2.040
digest, local_path = main.digest, main.local_path


def campaign_root(server):
    if server != 's1':
        raise ValueError('This additional main-matched campaign belongs to s1 only')
    return ROOT / 'work_dir/panda_wv3_main_sensitivity' / CAMPAIGN_ID / server / 'WV3'


def donor_pin(repeat):
    if type(repeat) is not int or repeat not in (1, 2, 3):
        raise ValueError('Unregistered donor repetition')
    return deepcopy(PLAN['donors'][f'R{repeat}'])


def resolve(server, case, repeat):
    campaign_root(server)
    pin = donor_pin(repeat)
    # The shared CUDA preflight checks the literal unit-weight endpoint. This
    # diagnostic is never queued, trained as a production case, or published.
    diagnostic = case == 'H2_GT'
    recipe = (dict(CASES['H2_REF'], alpha=0., gamma=0., existing_control_case=None)
              if diagnostic else CASES[case])
    cfg = main.resolve(server, 'M5_LR110', repeat)
    cfg.update(campaign_id=CAMPAIGN_ID, method_revision=REVISION, case_id=case,
        run_id=f'WMSENS_WV3_S1_{case}_R{repeat}_S{cfg["seed"]}',
        experiment=f'WV3-MAIN-SENS-LR110-{case.replace("H2_", "")} R{repeat} (RGB)',
        alpha=recipe['alpha'], gamma=recipe['gamma'], student_width=112,
        teacher_width=112, teacher_architecture='W112_D123_BASE',
        donor_run_id=pin['run_id'], donor_pin=pin, donor_origin_server='s1',
        stage1_retrained=False, conditional_groups=5,
        existing_control_case=recipe['existing_control_case'],
        baseline_variant=recipe['baseline_variant'],
        historical_control_policy='EXACT_EVIDENCE_OR_FRESH_SAME_SEED',
        historical_primary_source_campaign=PLAN['source_campaign'],
        reporting_policy=deepcopy(PLAN['reporting']), diagnostic_only=diagnostic)
    cfg['config_sha256'] = digest({k:v for k,v in cfg.items() if k!='config_sha256'})
    return cfg


def queue(server):
    campaign_root(server)
    return [resolve(server, case, repeat) for repeat in (1, 2, 3) for case in CASES]


def validate(cfg):
    if cfg != resolve(cfg.get('server'), cfg.get('case_id'), cfg.get('repeat')):
        raise ValueError('Configuration differs from registered main-matched sensitivity')


def catalog(server):
    out = io.StringIO()
    fields = ['order','case_id','repeat','seed','lr','alpha','gamma','horizon',
              'primary_step','donor_run_id','existing_control_case','experiment']
    writer = csv.DictWriter(out,fieldnames=fields,lineterminator='\n')
    writer.writeheader()
    for i,cfg in enumerate(queue(server),1):
        writer.writerow(dict(order=i,**{k:cfg.get(k,'') for k in fields if k!='order'}))
    return out.getvalue()
