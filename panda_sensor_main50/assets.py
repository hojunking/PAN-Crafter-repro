"""Pinned existing local A/T weights; Stage1 target misses never block Stage2."""
from copy import deepcopy
import json
from pathlib import Path
from directshift48.reporting import atomic_json
from ta2.evaluation import file_sha
from panda_sensor_eq40.assets import sources,calibration
from . import plan

def read(path):return json.loads(Path(path).read_text())

def selected_reference(server,registered,pins):
    sensor=plan.sensor_for(server);fixed=plan.PLAN['donors'][sensor]
    cfg=plan.donor_config(server);root=plan.donor_root(server)
    source_pins=read(root/'admission.json')
    for key in ('evaluator','train_inventory','split_manifests'):
        if source_pins[key]!=pins[key]:raise ValueError('Donor data/evaluator differs: '+key)
    if registered.verify().manifest()!=source_pins['train_inventory']:
        raise ValueError('Donor/current training inventory differs')
    state=read(root/'campaign_state.json')
    rows=[r for r in state['runs'] if r['run_id']==fixed['run_id']]
    if (state['campaign_id']!=cfg['campaign_id'] or state['server']!=server
            or len(rows)!=1 or rows[0]['status']!='COMPLETED' or rows[0]['step']!=50000):
        raise ValueError('Completed local 50K Stage1 required; no remote donor substitution')
    directory=root/'runs'/fixed['run_id']
    if read(directory/'config.json')!=cfg:raise ValueError('Original local Stage1 config changed')
    step=fixed['step'];path=directory/f'evaluation_{step:07d}.json'
    record=read(path)
    if server=='s2':
        from panda_unified_campaign.evaluation import verify_record
    else:
        from panda_sensor_eq40.evaluation import verify_record
    verify_record(record,cfg)
    ck=directory/'models'/f'step_{step:07d}.pt';meta=read(ck.with_suffix('.json'))
    if (file_sha(path)!=fixed['evaluation_sha256'] or record['step']!=step
        or record['eval_protocol_id']!=plan.digest(source_pins['evaluator'])
        or meta!=record['checkpoint'] or meta['step']!=step
        or plan.local_path(meta['path']).resolve()!=ck.resolve()
        or meta['config_sha256']!=cfg['config_sha256']
        or meta['file_sha256']!=fixed['checkpoint_file_sha256']
        or file_sha(ck)!=fixed['checkpoint_file_sha256']
        or record['model_sha256']!=fixed['model_sha256']
        or meta['model_sha256']!=fixed['model_sha256']):
        raise ValueError('Pinned local donor/evaluation SHA identity differs')
    chosen={'step':step,'checkpoint_path':str(ck.relative_to(root)),
        'file_sha256':meta['file_sha256'],'model_sha256':meta['model_sha256'],
        'evaluation_sha256':file_sha(path),'rr':record['rr']['metrics'],'fr':record['fr']['metrics']}
    if 'fr_corrected_pan' in record:chosen['fr_corrected_pan']=record['fr_corrected_pan']['metrics']
    receipt={'schema':'PANDA_MAIN50_PINNED_DONOR_V1','server':server,'sensor':sensor,
        'reference_mode':'RGBN','repeat':cfg['repeat'],'donor_run_id':cfg['run_id'],
        'source_config_sha256':cfg['config_sha256'],'source_release':source_pins['source_release'],
        'uses_test_for_selection':True,'selector':'PREDECLARED_LOCAL_CHECKPOINT_NO_METRIC_GATE',
        'thresholds_are_admission_gates':False,'selected':chosen,
        'rationale':('Same QB donor as previous stopped Stage2; isolate schedule/LR/gamma'
                     if sensor=='QB' else 'Existing unified GF2 R2@42K: ERGAS .516366, PSNR45.059975; final Student goals do not gate Teacher')}
    receipt['receipt_sha256']=plan.digest(receipt)
    return cfg,receipt,root,source_pins

def prepare_reference(cfg,registered,root,pins,device='cpu'):
    plan.validate(cfg)
    dc,receipt,source_root,source_pins=selected_reference(cfg['server'],registered,pins)
    if cfg['server']=='s2':
        from panda_unified_campaign.assets import _load_selected
    else:
        from panda_sensor_eq40.assets import _load_selected
    donor=_load_selected(dc,receipt,registered,source_root,source_pins,device=device)
    folder=Path(root)/'donor';path=folder/'selection.json'
    if path.exists() and read(path)!=receipt:raise ValueError('Frozen main donor receipt changed')
    atomic_json(path,receipt)
    stats=calibration(donor,registered,folder/'calibration',device=device)
    return donor,stats
