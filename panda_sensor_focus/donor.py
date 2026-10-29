"""Authenticate exact original QB/GF2 selected donor; never train or reselect it."""
import importlib,json
from pathlib import Path
from ta2.evaluation import file_sha
from .contract import SOURCE_DONORS,LANES
from .plan import ROOT

def receipt_path(server,root=ROOT):
    s=SOURCE_DONORS[server]
    suffix=(f"donors/{s['case']}/rre_selection.json" if server=='s1' else 'donors/rre_selection.json')
    return Path(root)/'work_dir'/s['package']/s['campaign_id']/server/LANES[server]['dataset']/suffix

def local(path,root=ROOT):
    p=Path(path)
    if p.parts[:2]==('/','workspace'): return Path(root)/p.relative_to('/workspace')
    return p if p.is_absolute() else Path(root)/p

def validate_source(server,root=ROOT,*,verify_checkpoint=True):
    s=SOURCE_DONORS[server];path=receipt_path(server,root)
    receipt=importlib.import_module(s['package']+'.selection').validate_receipt(json.loads(path.read_text()))
    selected=receipt['selected'];cfg=selected['source_config']
    if (receipt.get('server')!=server or receipt.get('campaign_id')!=s['campaign_id']
            or cfg.get('server')!=server or cfg.get('dataset')!=LANES[server]['dataset']
            or cfg.get('case')!=s['case'] or cfg.get('campaign_id')!=s['campaign_id']
            or selected.get('completed_update')!=s['selected_step']
            or selected.get('file_sha256')!=s['checkpoint_sha256']
            or (s['receipt_sha256'] is not None and receipt.get('receipt_sha256')!=s['receipt_sha256'])
            or cfg.get('planned_horizon')!=50000 or cfg.get('phase')!='donor'
            or cfg.get('lambda_abs')!=s['lambda_abs'] or cfg.get('lambda_epsilon')!=s['lambda_epsilon']
            or cfg.get('epsilon_radius_rr',3.)!=3.):
        raise ValueError('Exact authenticated original fixed donor differs from registration')
    checkpoint=local(selected['checkpoint_path'],root)
    if verify_checkpoint and (not checkpoint.is_file() or file_sha(checkpoint)!=s['checkpoint_sha256']):
        raise ValueError('Fixed donor checkpoint missing or hash mismatch')
    run_state=json.loads((checkpoint.parent.parent/'run_state.json').read_text())
    if (run_state.get('status')!='COMPLETED' or run_state.get('step')!=50000
            or run_state.get('run_id')!=cfg['run_id']):
        raise ValueError('Original donor must have completed registered 50K selection horizon')
    return receipt

def load_donor(path,expected_sha256,registered,*,server='s1',device='cpu',selector_receipt=None,include_teacher=True):
    source=SOURCE_DONORS[server]
    receipt=validate_source(server)
    if selector_receipt!=receipt or expected_sha256!=source['checkpoint_sha256']:
        raise ValueError('Caller donor evidence differs from exact original source')
    return importlib.import_module(source['package']+'.donor').load_donor(path,expected_sha256,
        registered,server=server,device=device,selector_receipt=receipt,include_teacher=include_teacher)
