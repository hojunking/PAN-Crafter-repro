from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from panda_band30.tests.test_reporting import evaluation as original_evaluation
from panda_mature30.tests.test_reporting import reused_receipt
from panda_qb_s2_frozen import best_publication as B, plan, reporting as R
from panda_qb_s2_frozen.spec import geometry_manifest

def persist(root, case=None, *, server='s2', step=1000, h=None, e=None):
    sensor = R.DATASETS[server]
    case = case or 'Q3_W128H050'
    hmin, emax = R.thresholds(server)
    h, e = hmin if h is None else h, emax if e is None else e
    cfg = plan.resolve_case(server, case, 'R1')
    ev = original_evaluation(cfg, step, h, e)
    key = cfg['policy_id'].lower()
    ev['mature30_deployment'] = {'schema':'PANDA_NEXT30_DEPLOYMENT_V1',
        'policy_id':cfg['policy_id'], 'completed_update':step, 'geometry_key':key,
        'geometry':geometry_manifest(key), 'student_sampling':deepcopy(cfg['student_sampling']),
        'student_input':deepcopy(cfg['student_input']), 'student_warp':deepcopy(cfg['student_warp'])}
    for split in ('rr', 'fr'):
        for scene in ev[split]['per_scene']:
            scene['correction_dy_dx'] = [.3,-.4]
            scene['pan_applied_dy_dx'] = [.3,-.4]
    directory = Path(root)/'runs'/cfg['run_id']/'attempt_01'
    directory.mkdir(parents=True, exist_ok=True)
    R.atomic_json(directory/'config.json', cfg)
    R.atomic_json(directory/f'evaluation_{step:07d}.json', ev)
    return R.emit_evaluation(root, cfg, ev, timestamp='2026-10-26T00:00:00+00:00')
