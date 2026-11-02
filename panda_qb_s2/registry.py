"""Six fresh local Stage1 donors, each followed by one matched Student."""
import json
from copy import deepcopy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / 'config/panda_qb_s2/plan.json'
PLAN = json.loads(PLAN_PATH.read_text())
CAMPAIGN_ID = 'PANDA_S2_QB_STAGE1_BALANCE_20261102_v1'
METHOD_REVISION = 'PANDA-QB-S2-BALANCE-RGB-L4-DIFFUBIL-HARDONLY-v1'
OUTPUT = 'work_dir/panda_qb_s2'
TARGET = deepcopy(PLAN['joint_target'])
RUNS = {s: tuple((r['case'], r['replica']) for r in rows) for s, rows in PLAN['queues'].items()}
SEEDS = {'R1': 261018101}

def job(server, case, replica='R1'):
    return deepcopy(next(r for r in PLAN['queues'][server] if (r['case'],r['replica']) == (case,replica)))

def donor_case(row):
    return row['donor_case']

def manifest():
    return dict(campaign_id=CAMPAIGN_ID,method_revision=METHOD_REVISION,plan=deepcopy(PLAN),
                execution='REGISTERED_LOCAL_QUEUE_NO_DEADLINE',cross_server_donor_transfer=False)
