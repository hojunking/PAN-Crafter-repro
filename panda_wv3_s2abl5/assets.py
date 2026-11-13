"""One immutable local donor/calibration for every case and Student seed."""
import json
from pathlib import Path
from directshift48.reporting import atomic_json
from ta2.evaluation import file_sha
from panda_unified_campaign.assets import sources, _load_selected, calibration
from . import plan

def prepare_reference(cfg,registered,root,pins,device='cpu'):
    plan.validate(cfg)
    fixed=plan.PLAN['fixed_donor']
    selection=plan.ROOT/fixed['selection_path']
    if file_sha(selection)!=fixed['selection_file_sha256']:
        raise ValueError('Pinned Stage1 selection receipt changed')
    receipt=json.loads(selection.read_text())
    if receipt['receipt_sha256']!=fixed['selection_receipt_sha256']:
        raise ValueError('Wrong fixed donor receipt')
    donor_root=plan.main.donor_root('s1');dc=plan.main.donor_config('s1',1)
    oldpins=json.loads((donor_root/'admission.json').read_text())
    for key in ('evaluator','train_inventory','split_manifests'):
        if pins[key]!=oldpins[key]:raise ValueError('Donor data/evaluator mismatch: '+key)
    if receipt['selected']['step']!=27000 or receipt['selected']['file_sha256']!=fixed['checkpoint_file_sha256']:
        raise ValueError('Only the exact selected R1 27K donor is allowed')
    folder=plan.ROOT/fixed['calibration_path']
    if not (folder/'full_s100.npz').is_file() or not (folder/'full_s100.json').is_file():
        raise FileNotFoundError('Original shared calibration must be retained; no recalibration')
    donor=_load_selected(dc,receipt,registered,donor_root,oldpins,device=device)
    stats=calibration(donor,registered,folder,device=device)
    if stats.artifact_sha256!=fixed['calibration_artifact_sha256']:
        raise ValueError('Different calibration population')
    proof={'fixed_donor':fixed,'same_for_all_cases_and_seeds':True,'original_selection':receipt,
           'calibration_sha256':stats.artifact_sha256,'binding':donor.binding}
    path=Path(root)/'donors/fixed_reference.json'
    if path.exists() and json.loads(path.read_text())!=proof:raise ValueError('Frozen reference changed')
    if not path.exists():atomic_json(path,proof)
    return donor,stats
