"""Verify all retained source files, pin receipts, calibrate each paired donor."""
from copy import deepcopy
from pathlib import Path
from directshift48.reporting import atomic_json
from panda_qb_lateval.assets import check_source,load_candidate,sources,calibration,read
from . import plan


def select(cfg,registered,root,pins):
    plan.validate(cfg)
    dc,source_root,sp,rows=check_source('s2',registered,pins)
    if dc!=plan.donor_config('s2'):raise ValueError('Different Stage1 architecture or source recipe')
    chosen=next(r for r in rows if r['step']==cfg['donor_step'])
    receipt=dict(schema='PANDA_INPUT3_FIXED_DONOR_V1',server='s2',sensor='QB',reference_mode='RGBN',repeat=1,
        donor_run_id=dc['run_id'],source_config_sha256=dc['config_sha256'],source_release=sp['source_release'],
        selector=plan.SELECTOR,uses_test_for_selection=False,thresholds_are_admission_gates=False,
        candidate_steps=list(plan.STEPS),candidates=rows,selected=deepcopy(chosen),
        rationale='Same Stage1 trajectory and architecture; predeclared later checkpoint only')
    receipt['receipt_sha256']=plan.digest(receipt)
    path=plan.reference_root(Path(root),cfg['donor_step'])/'selection.json'
    if path.exists() and read(path)!=receipt:raise ValueError('Fixed donor evidence changed')
    atomic_json(path,receipt)
    return dc,receipt,source_root,sp


def selected_reference(server,registered,pins):
    plan.sensor_for(server)
    return select(plan.queue(server)[0],registered,plan.campaign_root(server),pins)


def prepare_reference(cfg,registered,root,pins,device='cpu'):
    dc,receipt,source_root,sp=select(cfg,registered,root,pins)
    donor=load_candidate(dc,receipt['selected'],registered,source_root,sp,device,receipt)
    stats=calibration(donor,registered,plan.reference_root(Path(root),cfg['donor_step'])/'calibration',device=device)
    return donor,stats
