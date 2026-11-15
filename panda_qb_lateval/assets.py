"""Read-only local Stage1 evidence; validation-only selection, no new Aligner."""
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import torch
from directshift48.reporting import atomic_json
from panda_donor_read_guard.guard import protect
from panda_sensor_eq40.assets import calibration, sources as qb_sources
from panda_unified_campaign.selection import validation_inventory, aggregate_rows, collect_validation
from ta2.evaluation import file_sha
from . import plan


def read(path):
    return json.loads(Path(path).read_text())


def sources(server):
    plan.sensor_for(server)
    # s2 here selects QB dataset paths only. Donor, server marker and output
    # provenance below always use the actual requested server.
    return qb_sources('s2')


def package(server):
    plan.sensor_for(server)
    return 'panda_unified_campaign' if server == 's2' else 'panda_qb_s3_parallel'


def check_source(server, registered, pins):
    cfg=plan.donor_config(server);plan.assert_method(cfg)
    root=plan.donor_root(server);sp=read(root/'admission.json')
    for key in ('evaluator','train_inventory','split_manifests'):
        if sp[key]!=pins[key]:raise ValueError('Source/current data or evaluator differs: '+key)
    if registered.verify().manifest()!=sp['train_inventory']:
        raise ValueError('Registered RR train changed')
    state=read(root/'campaign_state.json')
    jobs=[r for r in state['runs'] if r['run_id']==cfg['run_id']]
    if (state['campaign_id']!=cfg['campaign_id'] or state['server']!=server
        or len(jobs)!=1 or jobs[0]['status']!='COMPLETED' or jobs[0]['step']!=50000):
        raise ValueError('Complete this server RGBN Stage1 50K first; no training fallback')
    directory=root/'runs'/cfg['run_id']
    if read(directory/'config.json')!=cfg:raise ValueError('Source Stage1 recipe differs')
    verifier=importlib.import_module(package(server)+'.evaluation').verify_record
    candidates=[]
    for step in plan.CANDIDATE_STEPS:
        ev=directory/f'evaluation_{step:07d}.json';e=read(ev);verifier(e,cfg)
        ck=directory/'models'/f'step_{step:07d}.pt';meta=read(ck.with_suffix('.json'))
        if (e['checkpoint']!=meta or meta['step']!=step or e['step']!=step
            or meta['config_sha256']!=cfg['config_sha256']
            or meta['model_sha256']!=e['model_sha256']
            or plan.local_path(meta['path']).resolve()!=ck.resolve()
            or file_sha(ck)!=meta['file_sha256']
            or e['eval_protocol_id']!=plan.digest(sp['evaluator'])):
            raise ValueError('Local candidate checkpoint/evaluation identity differs')
        candidates.append(dict(step=step,checkpoint_path=str(ck.relative_to(root)),
            file_sha256=meta['file_sha256'],model_sha256=meta['model_sha256'],
            evaluation_sha256=file_sha(ev),rr=e['rr']['metrics'],fr=e['fr']['metrics']))
    return cfg,root,sp,candidates


def load_candidate(cfg, row, registered, root, pins, device, receipt=None):
    if receipt is None:
        receipt={'selected':row,'donor_run_id':cfg['run_id'],'selector':'VALIDATION_CANDIDATE_ONLY'}
        receipt['receipt_sha256']=plan.digest(receipt)
    loader=importlib.import_module(package(cfg['server'])+'.assets')._load_selected
    return protect(loader)(cfg,receipt,registered,root,pins,device=device)


def validate_score(value, inventory):
    if value.get('protocol')!=inventory['protocol']:
        raise ValueError('Candidate validation must use the same RGBN protocol')
    aggregate=aggregate_rows(value['per_sample'],inventory)
    for key,expected in aggregate.items():
        if value.get(key)!=expected:raise ValueError('Validation rows/aggregate mismatch: '+key)
    for key in ('RRE50','RRE90','Val_ERGAS'):
        if not isinstance(value.get(key),(int,float)) or not math.isfinite(value[key]) or value[key]<0:
            raise ValueError('Invalid validation score: '+key)
    return {k:v for k,v in value.items() if k not in ('per_sample','protocol')}


def pick(rows):
    if sorted(r['step'] for r in rows)!=list(plan.CANDIDATE_STEPS):
        raise ValueError('Exactly 40K, 48K, 50K candidates required; no missing-candidate fallback')
    return min(rows,key=lambda r:tuple(r['validation'][k] for k in ('RRE50','RRE90','Val_ERGAS'))+(r['step'],))


def select(server, registered, root, pins, device='cpu'):
    cfg,source_root,sp,rows=check_source(server,registered,pins)
    folder=Path(root)/'donor';folder.mkdir(parents=True,exist_ok=True)
    inventory=validation_inventory(registered.sources,'RGBN')
    atomic_json(folder/'validation_inventory.json',inventory)
    directory=source_root/'runs'/cfg['run_id']
    for row in rows:
        step=row['step'];cache=folder/f'validation_{step:07d}.json'
        original=directory/f'validation_{step:07d}.json'
        if cache.exists():
            stored=read(cache)
            if (stored['checkpoint_sha256']!=row['file_sha256']
                or stored['inventory_sha256']!=inventory['inventory_sha256']):
                raise ValueError('Cached validation identity changed')
        elif original.exists():
            value=read(original);validate_score(value,inventory)
            # The source evaluation was verified against the checkpoint above.
            recorded=read(directory/f'evaluation_{step:07d}.json').get('validation',{})
            for k in ('RRE50','RRE90','Val_ERGAS','validation_inventory_sha256'):
                if recorded.get(k)!=value[k]:raise ValueError('Validation is not bound to source checkpoint')
            stored={'checkpoint_sha256':row['file_sha256'],'inventory_sha256':inventory['inventory_sha256'],
                'origin':'VERIFIED_EXISTING_VALIDATION','source_path':str(original),'source_sha256':file_sha(original),
                'result':value}
        else:
            print(f'{server}: score existing Stage1 {step} using RGBN validation only',flush=True)
            donor=load_candidate(cfg,row,registered,source_root,sp,device)
            value=collect_validation(donor,registered.sources,inventory=inventory,include_ergas=True,
                epsilon=True,device=device,batch_size=48,
                on_progress=lambda r:print(f"validation {step}: {r['seen']}/{r['expected']}",flush=True))
            stored={'checkpoint_sha256':row['file_sha256'],'inventory_sha256':inventory['inventory_sha256'],
                'origin':'READ_ONLY_CHECKPOINT_REEVALUATION','result':value}
            del donor
        row['validation']=validate_score(stored['result'],inventory)
        atomic_json(cache,stored)
        row.update(validation_path=str(cache),validation_sha256=file_sha(cache))
    receipt=dict(schema='PANDA_QB_LATEVAL_DONOR_V1',server=server,sensor='QB',reference_mode='RGBN',
        repeat=1,donor_run_id=cfg['run_id'],source_config_sha256=cfg['config_sha256'],
        source_release=sp['source_release'],selector=plan.SELECTOR,uses_test_for_selection=False,
        candidate_steps=list(plan.CANDIDATE_STEPS),candidates=rows,selected=deepcopy(pick(rows)),
        thresholds_are_admission_gates=False,selection_reference='RR_VALIDATION_RGBN_GT')
    receipt['receipt_sha256']=plan.digest(receipt)
    path=folder/'selection.json'
    if path.exists() and read(path)!=receipt:raise ValueError('Frozen selected donor changed')
    atomic_json(path,receipt)
    return cfg,receipt,source_root,sp


def prepare_reference(cfg, registered, root, pins, device='cpu'):
    plan.validate(cfg);plan.assert_method(cfg)
    dc,receipt,source_root,sp=select(cfg['server'],registered,root,pins,device)
    donor=load_candidate(dc,receipt['selected'],registered,source_root,sp,device,receipt)
    stats=calibration(donor,registered,Path(root)/'donor/calibration',device=device)
    return donor,stats
