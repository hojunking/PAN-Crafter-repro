"""Validate and read the two original s2 artifacts; never train or reselect."""
from copy import deepcopy
import json
from pathlib import Path
from panda_qb_s2.donor import load_donor
from panda_qb_s2.selection import validate_receipt
from ta2.evaluation import file_sha
from . import registry as G


def local(path,root):
    path=Path(path)
    expected=Path('/workspace/work_dir/panda_qb_s2')/G.PLAN['source_campaign_id']/'s2/QB'
    if not path.is_relative_to(expected):
        raise ValueError('Donor artifact must be in the original s2 campaign')
    return Path(root)/path.relative_to('/workspace')


def checked_receipt(pin,root):
    if pin['server']!='s2' or pin['source_campaign_id']!=G.PLAN['source_campaign_id']:
        raise ValueError('Cross-server/campaign donor rejected')
    receipt_path=local(pin['selector_receipt_path'],root)
    receipt=validate_receipt(json.loads(receipt_path.read_text()))
    selected=receipt['selected'];cfg=selected['source_config']
    if (receipt['receipt_sha256']!=pin['selector_receipt_sha256'] or receipt['server']!='s2'
        or selected['checkpoint_path']!=pin['checkpoint_path']
        or selected['file_sha256']!=pin['checkpoint_sha256'] or selected['completed_update']!=48000
        or cfg['run_id']!=pin['source_run_id'] or cfg['case']!=pin['source_case']
        or cfg['seed']!=pin['source_stage1_seed'] or cfg['server']!='s2'
        or cfg['campaign_id']!=pin['source_campaign_id']):
        raise ValueError('Pinned s2 selector/checkpoint identity differs')
    checkpoint=local(pin['checkpoint_path'],root)
    if file_sha(checkpoint)!=pin['checkpoint_sha256']:
        raise ValueError('Pinned s2 checkpoint SHA differs')
    directory=checkpoint.parent.parent
    state=json.loads((directory/'run_state.json').read_text())
    if state.get('status')!='COMPLETED' or state.get('step')!=50000 or state.get('run_id')!=pin['source_run_id']:
        raise ValueError('Original source Stage1 must have completed 50K')
    history=receipt['source_history']
    for filename,key in [('config.json','source_config_sha256'),('models/selection.json','source_selection_sha256')]:
        if file_sha(directory/filename)!=history[key]:
            raise ValueError('Original source config/retention inventory changed')
    for entry in pin['calibration_artifacts'].values():
        if file_sha(local(entry['path'],root))!=entry['file_sha256']:
            raise ValueError('Original calibration SHA differs; no automatic recalibration')
    return receipt_path,receipt,checkpoint


def verify_baselines(root):
    """Link real completed R1 evidence; never create a new result row for it."""
    from panda_qb_s2 import plan as source_plan
    verified=[]
    for pin in G.PLAN['donors'].values():
        ref=pin['baseline_R1'];checkpoint=local(ref['artifact']['checkpoint_path'],root)
        directory=checkpoint.parent.parent
        state=json.loads((directory/'run_state.json').read_text())
        cfg=json.loads((directory/'config.json').read_text())
        expected=json.loads(json.dumps(source_plan.resolve_case('s2',pin['source_case'].replace('_S1_','_S2_'))))
        if (state.get('status')!='COMPLETED' or state.get('step')!=30000
            or state.get('run_id')!=ref['run_id']
            or any(cfg.get(k)!=v for k,v in expected.items())
            or cfg.get('donor_checkpoint_sha256')!=pin['checkpoint_sha256']
            or file_sha(checkpoint)!=ref['artifact']['file_sha256']):
            raise ValueError('Archived baseline is not the completed identical s2 R1 recipe')
        evidence=local(ref['evaluation_evidence']['path'],root)
        if file_sha(evidence)!=ref['evaluation_evidence']['sha256']:
            raise ValueError('Archived baseline evaluation SHA differs')
        ev=json.loads(evidence.read_text())
        if (ev['state_before']['run_id']!=ref['run_id']
            or ev['state_before']['completed_update']!=ref['selected_step']
            or ev['fr']['metrics']['HQNR']!=ref['HQNR']
            or ev['rr']['metrics']['ERGAS']!=ref['ERGAS']):
            raise ValueError('Archived baseline metrics/identity differ')
        verified.append(dict(run_id=ref['run_id'],source_checkpoint=str(checkpoint),
            source_evaluation=str(evidence),new_training=False,publish_duplicate=False))
    return verified


def verify_sources(root,registered=None):
    rows={}
    for pin in G.PLAN['donors'].values():
        receipt_path,receipt,checkpoint=checked_receipt(pin,root)
        selected=receipt['selected']
        binding=None
        if registered is not None:
            loaded=load_donor(checkpoint,pin['checkpoint_sha256'],registered,server='s2',
                              selector_receipt=receipt,device='cpu')
            loaded.assert_frozen();binding=deepcopy(loaded.binding)
            del loaded
        rows[pin['source_case']]=dict(label=pin['source_case'],source_config=selected['source_config'],
            checkpoint_path=str(checkpoint),checkpoint_sha256=pin['checkpoint_sha256'],
            evaluation_path=selected['source_evaluation_path'],selected_step=48000,binding=binding,
            source_server='s2',selector_receipt_path=str(receipt_path),
            selector_sha256=pin['selector_receipt_sha256'],complete_source_horizon=50000,
            exploratory=True,stage1_retrained=False)
    return dict(status='S2_FROZEN_SOURCES_VERIFIED',donors=rows,stage1_training_runs=0,
                archived_baselines=verify_baselines(root),cross_server_transfer=False)
