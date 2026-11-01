"""The accepted 40-hour plan is the single ordered job catalog."""
import json
from copy import deepcopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / 'config/panda_qb40/plan.json'
PLAN = json.loads(PLAN_PATH.read_text())
CAMPAIGN_ID = 'PANDA_S13_QB40_LOCALSTAGE1_20261101_v1'
METHOD_REVISION = 'PANDA-QB40-LOCALSTAGE1-RGB-L4-DIFFUBIL-HARDONLY-v1'
OUTPUT = 'work_dir/panda_qb40'
TARGET = deepcopy(PLAN['joint_target'])
RUNS = {s: tuple((r['case'], r['replica']) for r in rows) for s, rows in PLAN['queues'].items()}
SOURCES = deepcopy(PLAN['existing_stage1_sources'])
SOURCES['s1']['package'] = 'panda_s1loss9'
SOURCES['s3']['package'] = 'panda_qb_deepen'
SEEDS = {'R1': 261018101, 'R2': 261018102, 'R3': 261018103}

def job(server, case, replica='R1'):
    return deepcopy(next(r for r in PLAN['queues'][server] if (r['case'],r['replica']) == (case,replica)))

def donor_case(row):
    return ('S1_A200_E030' if row['donor'] == 'EXISTING_OWN_A200_E030' else
            row['case'] if row['phase']=='stage1' else 'QB5_S1_'+row['donor'].removeprefix('NEW_OWN_'))

def manifest():
    return dict(campaign_id=CAMPAIGN_ID,method_revision=METHOD_REVISION,plan=deepcopy(PLAN),
                execution='REGISTERED_40H_LOCAL_QUEUE',cross_server_donor_transfer=False)
