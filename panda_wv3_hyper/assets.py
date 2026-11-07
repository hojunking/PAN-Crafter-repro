"""Read-only local RGB donor/calibration import for the coefficient ablation.

The Student experiments reuse the original s1 repeat's frozen A/T pair.  s3
trains fresh Stage1 models and never needs these s1 weights.
"""
from copy import deepcopy
import json
from pathlib import Path

from panda_wv3_main_rgb import assets as original, plan as source_plan
from panda_stage2_condq.calibration import CalibrationStats
from panda_stage2_condq.spec import MASK2_POLICY
from ta2.evaluation import evaluator_identity, file_sha
from . import plan

sources = original.sources


def _read(path):
    return json.loads(Path(path).read_text())


def source_evidence(cfg):
    """Check the exact retained source, without reselection or recalibration."""
    plan.validate(cfg)
    if cfg['server'] != 's1' or cfg['stage'] != 'stage2':
        raise ValueError('Only s1 Stage2 imports the local s1 RGB reference')
    pin = cfg['donor_pin']
    root = source_plan.campaign_root('s1')
    receipt_path = plan.ROOT / pin['selection_receipt_path']
    receipt = _read(receipt_path)
    if (pin['server'] != 's1' or pin['run_id'] != cfg['donor_run_id']
            or receipt.get('receipt_sha256') != pin['selection_receipt_sha256']
            or receipt['receipt_sha256'] != plan.digest(
                {k: v for k, v in receipt.items() if k != 'receipt_sha256'})
            or receipt.get('server') != 's1' or receipt.get('repeat') != cfg['repeat']
            or receipt.get('donor_run_id') != cfg['donor_run_id']
            or receipt.get('uses_test_for_selection') is not False):
        raise ValueError('Original same-repeat s1 donor receipt differs')
    chosen = receipt['selected']
    if (chosen['step'] != pin['selected_step']
            or chosen['file_sha256'] != pin['checkpoint_sha256']
            or chosen != min(receipt['candidates'], key=lambda r: (
                r['validation']['RRE50'], r['validation']['RRE90'],
                r['validation']['Val_ERGAS'], r['step']))):
        raise ValueError('Pinned original validation selection differs')
    state = _read(root / 'campaign_state.json')
    jobs = [r for r in state['runs'] if r['run_id'] == cfg['donor_run_id']]
    if (state.get('server') != 's1' or state.get('campaign_id') != source_plan.CAMPAIGN_ID
            or len(jobs) != 1 or jobs[0]['status'] != 'COMPLETED'
            or jobs[0]['step'] != 50000):
        raise ValueError('Original Stage1 is not completed locally at 50K')
    source_cfg = source_plan.resolve('s1', 'S1_RGB_FULL', cfg['repeat'])
    directory = root / 'runs' / cfg['donor_run_id']
    if _read(directory / 'config.json') != source_cfg:
        raise ValueError('Original Stage1 configuration differs')
    checkpoint = plan.ROOT / pin['checkpoint_path']
    if checkpoint.resolve() != (root / chosen['checkpoint_path']).resolve():
        raise ValueError('Donor path does not belong to original local campaign')
    if file_sha(checkpoint) != pin['checkpoint_sha256']:
        raise ValueError('Donor checkpoint SHA mismatch')
    for name, key in (('evaluation', 'evaluation_sha256'), ('validation', 'validation_sha256')):
        if file_sha(directory / f"{name}_{chosen['step']:07d}.json") != chosen[key]:
            raise ValueError('Donor evaluation/validation evidence differs')
    calibration = plan.ROOT / pin['calibration_path']
    meta = _read(plan.ROOT / pin['calibration_receipt_path'])
    if (file_sha(calibration) != pin['calibration_file_sha256']
            or meta['file_sha256'] != pin['calibration_file_sha256']
            or meta['artifact_sha256'] != pin['calibration_artifact_sha256']
            or meta['donor_binding']['donor_checkpoint_sha256'] != pin['checkpoint_sha256']
            or meta['donor_binding']['donor_run_id'] != cfg['donor_run_id']
            or meta['donor_binding']['selector_receipt_sha256'] != receipt['receipt_sha256']):
        raise ValueError('Exact original calibration SHA/binding mismatch')
    pins = _read(root / 'admission.json')
    if (receipt['source_release'] != pins['source_release']
            or receipt['source_config_sha256'] != source_cfg['config_sha256']
            or meta['donor_binding']['source_release'] != pins['source_release']):
        raise ValueError('Original donor source release/config binding differs')
    return source_cfg, receipt, pins, meta


def prepare_reference(cfg, registered, root, pins, device='cpu'):
    """Load the retained donor and exact old full-train calibration, read-only."""
    source_cfg, receipt, old_pins, meta = source_evidence(cfg)
    if any(pins[k] != old_pins[k] for k in ('evaluator', 'train_inventory', 'split_manifests')):
        raise ValueError('Original donor and new Student data/evaluator differ')
    donor = original._load_selected(source_cfg, receipt, registered,
        source_plan.campaign_root('s1'), old_pins, device=device)
    if meta['donor_binding'] != donor.binding:
        raise ValueError('Calibration binding does not match frozen A/T pair')
    stats = CalibrationStats.load(plan.ROOT / cfg['donor_pin']['calibration_path'],
        expected_binding=donor.binding,
        expected_file_sha256=cfg['donor_pin']['calibration_file_sha256'],
        expected_mask_policy=MASK2_POLICY, memory_budget_bytes=4 * 1024**3)
    if stats.artifact_sha256 != cfg['donor_pin']['calibration_artifact_sha256']:
        raise ValueError('Embedded calibration identity differs')
    stats.validate_binding(donor.binding)
    stats.assert_integrity(full_hash=True)
    donor.assert_frozen()
    registered.verify()
    folder = Path(root) / 'donors' / f"R{cfg['repeat']}"
    original._immutable_json(folder / 'selection.json', deepcopy(receipt))
    original._immutable_json(folder / 'import.json', dict(source=cfg['donor_pin'],
        source_pins=old_pins, recomputed_calibration=False, retrained_stage1=False))
    return donor, stats.to(device, memory_budget_bytes=1024**3)


def verify_references(server, device='cpu'):
    """Preflight evidence audit; s3 deliberately has no s1 reference dependency."""
    plan.campaign_root(server)
    native, registered = sources()
    current = {'evaluator': evaluator_identity(),
        'train_inventory': registered.verify().manifest(),
        'split_manifests': {s: native.manifest_sha(s) for s in ('train', 'val', 'rr', 'fr')}}
    references = []
    if server == 's1':
        for repeat in (1, 2, 3):
            cfg = plan.resolve(server, 'H2_REF', repeat)
            _, receipt, old_pins, meta = source_evidence(cfg)
            if any(current[k] != old_pins[k] for k in current):
                raise ValueError('Current data/evaluator differs from original RGB donor')
            references.append({'repeat': repeat, 'step': receipt['selected']['step'],
                'checkpoint_sha256': receipt['selected']['file_sha256'],
                'calibration_artifact_sha256': meta['artifact_sha256'],
                'same_server': True, 'retrained': False})
    return {'server': server, 'references': references, 'pins': current,
        'cross_server_weight_transfer': False}
