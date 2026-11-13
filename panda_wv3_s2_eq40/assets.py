"""Read-only original Stage1 weights; explicit test-aware donor criterion."""
import json, math
from pathlib import Path
from ta2.evaluation import file_sha
from directshift48.reporting import atomic_json
from panda_unified_campaign.assets import sources, _load_selected, calibration, _immutable_json
from panda_unified_campaign.evaluation import verify_record
from . import plan

def _read(path):return json.loads(Path(path).read_text())
def rank_candidates(rows):
    t=plan.PLAN['donor_selection']
    eligible=[r for r in rows if r['rr']['ERGAS']<=t['ERGAS_max'] and r['rr']['PSNR']>=t['PSNR_min']]
    if not eligible:raise ValueError('NO_QUALIFIED_STAGE1_DONOR: ERGAS<=2.04 and PSNR>=38.0; no silent fallback')
    return min(eligible,key=lambda r:(-r['fr']['HQNR'],r['rr']['ERGAS'],-r['rr']['PSNR'],r['step']))

def select(cfg,registered,pins):
    plan.validate(cfg);root=plan.donor_root();dc=plan.donor_config(cfg['repeat'])
    oldpins=_read(root/'admission.json')
    for key in ('evaluator','train_inventory','split_manifests'):
        if pins[key]!=oldpins[key]:raise ValueError('Original donor/current data or evaluator mismatch: '+key)
    if oldpins['train_inventory']!=registered.verify().manifest():raise ValueError('Training inventory changed')
    state=_read(root/'campaign_state.json')
    jobs=[r for r in state['runs'] if r['run_id']==dc['run_id']]
    if len(jobs)!=1 or jobs[0]['status']!='COMPLETED' or jobs[0]['step']!=50000:
        raise ValueError('Local same-repeat Stage1 must complete 50K first')
    directory=root/'runs'/dc['run_id']
    if _read(directory/'config.json')!=dc:raise ValueError('Original Stage1 configuration changed')
    rows=[]
    for step in dc['evaluation_steps']:
        path=directory/f'evaluation_{step:07d}.json';e=_read(path);verify_record(e,dc)
        if e['step']!=step or e['eval_protocol_id']!=plan.digest(oldpins['evaluator']):raise ValueError('Donor evaluation mismatch')
        ck=directory/'models'/f'step_{step:07d}.pt';meta=_read(ck.with_suffix('.json'))
        if (meta!=e['checkpoint'] or meta['model_sha256']!=e['model_sha256']
            or meta['config_sha256']!=dc['config_sha256'] or meta['step']!=step
            or plan.local_path(meta['path']).resolve()!=ck.resolve() or file_sha(ck)!=meta['file_sha256']):
            raise ValueError('Donor checkpoint evidence mismatch')
        rows.append(dict(step=step,rr=e['rr']['metrics'],fr=e['fr']['metrics'],
            checkpoint_path=str(ck.relative_to(root)),file_sha256=meta['file_sha256'],
            model_sha256=meta['model_sha256'],evaluation_sha256=file_sha(path)))
    chosen=rank_candidates(rows)
    receipt=dict(schema='PANDA_EQ40_DONOR_SELECTION_V1',server='s1',sensor='WV3',reference_mode='RGBN',
        repeat=cfg['repeat'],donor_run_id=dc['run_id'],source_config_sha256=dc['config_sha256'],
        source_release=oldpins['source_release'],selector='RR_ERGAS_LE_2P04_PSNR_GE_38_THEN_ORIGINAL_FR_HQNR_MAX',
        uses_test_for_selection=True,thresholds=plan.PLAN['donor_selection'],candidates=rows,
        candidate_steps=[r['step'] for r in rows],selected=chosen)
    receipt['receipt_sha256']=plan.digest(receipt)
    return dc,receipt,oldpins

def prepare_reference(cfg,registered,root,pins,device='cpu'):
    dc,receipt,oldpins=select(cfg,registered,pins)
    donor=_load_selected(dc,receipt,registered,plan.donor_root(),oldpins,device=device)
    folder=Path(root)/'donors'/f"RGBN_R{cfg['repeat']}"
    _immutable_json(folder/'selection.json',receipt)
    stats=calibration(donor,registered,folder/'calibration',device=device)
    return donor,stats
