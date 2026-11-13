"""Reuse only the same server's unchanged Stage1 selection and raw calibration."""
import importlib
from pathlib import Path
from directshift48.reporting import atomic_json
from . import plan

def old_assets(server):
    return importlib.import_module(('panda_wv3_s2_eq40' if server=='s1' else 'panda_sensor_eq40')+'.assets')
def sources(server):return old_assets(server).sources(server)
def prepare_reference(cfg,registered,root,pins,device='cpu'):
    plan.validate(cfg)
    donor,stats=old_assets(cfg['server']).prepare_reference(
        plan.prior_config(cfg['server'],cfg['repeat']),registered,plan.prior_root(cfg['server']),pins,device=device)
    record={'q_zero_policy':cfg['q_zero_policy'],'raw_calibration_unchanged':True,
        'calibration_sha256':stats.artifact_sha256,'donor_binding':donor.binding,
        'donor_selection':donor.selection_receipt,'previous_stage2_config_sha256':cfg['previous_stage2_config_sha256'],
        'checkpoint_imported_from_previous_stage2':False,'same_server_only':True}
    path=Path(root)/'donors'/f'R{cfg["repeat"]}'/'reference.json'
    if path.exists():
        import json
        if json.loads(path.read_text())!=record:raise ValueError('Frozen local reference changed')
    else:atomic_json(path,record)
    return donor,stats
