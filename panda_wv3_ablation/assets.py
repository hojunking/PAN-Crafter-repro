"""Verified original sources and the existing local s1 A/T/calibration only."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from ta2.evaluation import file_sha
from . import plan


def sources():
    from panda_method.edge_data import EdgeDataSources
    from panda_stage2_condq.sources import RegisteredRRTrain
    p = plan.local_path(plan.REFERENCES['baseline']['source_registry'])
    if file_sha(p) != plan.REFERENCES['baseline']['source_registry_sha256']:
        raise ValueError('WV3 data registry differs from the anchored experiment')
    source = EdgeDataSources(p,root=plan.ROOT,sensor='WV3')
    for split in ('train','val','rr','fr'):
        source.raw(split,[0])
    return source,RegisteredRRTrain(source)


def anchor_spec():
    from panda_tune30.spec import MatureSpec
    return MatureSpec('WV3','panda_condq',plan.REFERENCES['stage2']['stream_seeds']['S_init'],
        .25,0.,.25,1e-6,policy_id='S100',total_updates=30000,
        sampling_variant='UBIL',input_variant='CORRECTED_DIFF')


def load_reference(registered,*,server,device='cpu'):
    if server != 's1':
        raise ValueError('s3 Stage1 has no dependency on any existing donor weights')
    from panda_mature30.donor import load_donor
    from panda_mature30.selection import validate_receipt,validate_inventory
    from panda_tune30.calibration import bind_donor
    receipt = validate_receipt(json.loads(plan.local_path(plan.REFERENCES['source_receipt']).read_text()))
    chosen = receipt['selected']; ref = plan.REFERENCES['stage1']
    if (receipt['server']!='s1' or chosen['file_sha256']!=ref['file_sha256']
            or chosen['source_config']!=plan.REFERENCES['source_stage1_config']
            or chosen['completed_update']!=50000
            or plan.local_path(chosen['checkpoint_path'])!=plan.local_path(ref['checkpoint'])):
        raise ValueError('Reference is not the user-selected s1 50K checkpoint')
    validate_inventory(receipt['validation_inventory'],registered.sources)
    # Validate original receipt above, preserving its /workspace path and digest.
    # The loader receives the actual host/container path, never a rewritten receipt.
    donor = load_donor(plan.local_path(ref['checkpoint']),ref['file_sha256'],registered,
                       server='s1',device=device,include_teacher=True)
    binding = deepcopy(donor.binding)
    binding['mature30_selector_sha256'] = receipt['receipt_sha256']
    binding['mature30_validation_inventory_sha256'] = receipt['validation_inventory']['inventory_sha256']
    return bind_donor(replace(donor,binding=binding),anchor_spec())


def calibration(donor,registered,root,*,device='cpu'):
    from panda_tune30.calibration import calibrate_geometry,binding_for_geometry
    from panda_stage2_condq.calibration import CalibrationStats
    from directshift48.reporting import atomic_json
    spec = anchor_spec()
    folder = Path(root)/'calibration'; folder.mkdir(parents=True,exist_ok=True)
    # Original statistics can be shared locally; copying or changing donor tensors is not needed.
    original = plan.local_path(plan.REFERENCES['source_receipt']).parent.parent/'calibration'
    candidates = sorted(original.glob('*.json'))+[folder/'s100.json']
    for meta in candidates:
        if not meta.exists():
            continue
        record = json.loads(meta.read_text())
        if record.get('donor_binding')!=binding_for_geometry(donor.binding,'s100'):
            continue
        path = plan.local_path(record['path'])
        if not path.is_file():
            raise ValueError('Pinned calibration receipt exists but its artifact is missing')
        stats = CalibrationStats.load(path,expected_binding=record['donor_binding'],
            expected_file_sha256=record['file_sha256'],memory_budget_bytes=4*1024**3,
            expected_mask_policy=spec.mask_policy)
        atomic_json(folder/'binding.json',{'source_receipt':str(meta),'file_sha256':record['file_sha256'],
            'artifact_sha256':stats.artifact_sha256,'tau_e':stats.tau_e,'donor_sha256':plan.REFERENCES['stage1']['file_sha256']})
        return stats.to(device,memory_budget_bytes=1024**3)
    record = calibrate_geometry(donor,registered,spec=spec,geometry_key='s100',output=folder/'s100.npz',
        k=5,batch_size=48,memory_budget_bytes=4*1024**3,device=device,
        on_progress=lambda r: print('Calibration',r['seen'],r['expected'],flush=True))
    atomic_json(folder/'s100.json',record)
    return calibration(donor,registered,root,device=device)
