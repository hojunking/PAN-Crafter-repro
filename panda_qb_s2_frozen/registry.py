"""Approved round-major Student queue; source donors never enter the queue."""
import json
from copy import deepcopy
from pathlib import Path
from panda_qb_s2.registry import METHOD_REVISION

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / 'config/panda_qb_s2_frozen/plan.json'
PLAN = json.loads(PLAN_PATH.read_text())
CAMPAIGN_ID = 'PANDA_S2_QB_FROZEN_S2_20261104_v1'
OUTPUT = 'work_dir/panda_qb_s2_frozen'
TARGET = deepcopy(PLAN['joint_target'])
SEEDS = {'R1':261018101, 'R2':261018102, 'R3':261018103}
RUNS = {'s2': tuple((r['case'], r['replica']) for r in PLAN['queues']['s2'])}

def job(server, case, replica='R1'):
    return deepcopy(next(r for r in PLAN['queues'][server]
                         if (r['case'], r['replica']) == (case, replica)))

def donor_case(row):
    return row['donor_case']

def donor(case_or_alias):
    return deepcopy(next(d for alias,d in PLAN['donors'].items()
                         if case_or_alias in (alias,d['source_case'])))

def manifest():
    return deepcopy(PLAN)
