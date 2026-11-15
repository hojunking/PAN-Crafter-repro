"""Read-only existing local 40/48/50K donors; separate calibration per donor."""
from copy import deepcopy
from pathlib import Path
from directshift48.reporting import atomic_json
from panda_qb_lateval.assets import (read, sources, check_source, load_candidate,
                                     calibration, package)
from . import plan


def select(cfg, registered, root, pins):
    plan.validate(cfg)
    dc, source_root, source_pins, rows = check_source(cfg['server'], registered, pins)
    if dc['run_id'] != cfg['donor_run_id']:
        raise ValueError('Wrong same-server donor source')
    chosen = next(row for row in rows if row['step'] == cfg['donor_step'])
    receipt = dict(schema='PANDA_QB_DONORGRID_DONOR_V1', server=cfg['server'],
        sensor='QB', reference_mode='RGBN', repeat=1, donor_run_id=dc['run_id'],
        source_config_sha256=dc['config_sha256'], source_release=source_pins['source_release'],
        selector=plan.SELECTOR, uses_test_for_selection=False,
        candidate_steps=list(plan.CANDIDATE_STEPS), candidates=rows,
        selected=deepcopy(chosen), thresholds_are_admission_gates=False,
        selection_reference='PREDECLARED_FIXED_STAGE1_STEP')
    receipt['receipt_sha256'] = plan.digest(receipt)
    path = plan.reference_root(root, cfg['donor_step']) / 'selection.json'
    if path.exists() and read(path) != receipt:
        raise ValueError('Frozen donor evidence changed')
    atomic_json(path, receipt)
    return dc, receipt, source_root, source_pins


def prepare_reference(cfg, registered, root, pins, device='cpu'):
    dc, receipt, source_root, source_pins = select(cfg, registered, root, pins)
    donor = load_candidate(dc, receipt['selected'], registered, source_root,
                           source_pins, device, receipt)
    stats = calibration(donor, registered,
        plan.reference_root(root, cfg['donor_step']) / 'calibration', device=device)
    return donor, stats
