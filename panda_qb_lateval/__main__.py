"""Read-only local donor selection, then one unchanged 50K Student per server."""
import argparse
import gc
import json
import shutil
from directshift48.reporting import atomic_json
from . import plan
from .release import verify


def pins_for(source,registered,proof):
    from ta2.evaluation import evaluator_identity
    return {'source_release':proof['source_sha256'],'evaluator':evaluator_identity(),
        'train_inventory':registered.inventory.manifest(),
        'split_manifests':{s:source.manifest_sha(s) for s in ('train','val','rr','fr')}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('catalog','check','check-sheet','publish','select','run'))
    p.add_argument('--server',choices=('s2','s3'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    if a.command=='catalog':print(plan.catalog(a.server),end='');return
    from . import assets
    proof=verify(environment=a.command in ('check','select','run'))
    if a.command in ('select','run'):
        from panda_stage2_condq.training import configure_precision
        configure_precision()  # Validation and Teacher reporting precede Trainer construction.
    from .reporting import install
    install()
    from .publication import publish,destination
    root=plan.campaign_root(a.server)
    if a.command in ('select','run') or a.apply:
        if (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:
            raise ValueError('Execute on matching physical server; no donor transfer')
    if a.command=='check':
        source,registered=assets.sources(a.server);pins=pins_for(source,registered,proof)
        dc,_,_,rows=assets.check_source(a.server,registered,pins)
        proof.update(donor=dc['run_id'],candidate_steps=[r['step'] for r in rows],new_stage1_runs=0,stage2_runs=1)
    elif a.command in ('publish','check-sheet'):
        if a.apply:
            from panda_fr_columns.adapter import GoogleSheets
            GoogleSheets(plan.ROOT/'gspread/account.json').ensure_layout(destination(a.server),root/'publication/fr_columns',apply=True)
        proof=publish(root,a.server,apply=a.command=='publish' and a.apply)
        if a.apply and (root/'stage1_reference/evaluation.json').exists():
            from .reference_report import publish_source
            publish_source(a.server,apply=True)
    else:
        from panda_sensor_main50.__main__ import wait_for_gpu
        if a.device.startswith('cuda'):wait_for_gpu(root)
        if a.command=='run':
            from panda_fr_columns.adapter import GoogleSheets
            GoogleSheets(plan.ROOT/'gspread/account.json').ensure_layout(destination(a.server),root/'publication/fr_columns',apply=True)
        source,registered=assets.sources(a.server);pins=pins_for(source,registered,proof)
        admission=root/'admission.json'
        if admission.exists() and assets.read(admission)!=pins:raise ValueError('Campaign code/data changed')
        atomic_json(admission,pins)
        if a.command=='select':
            _,proof,_,_=assets.select(a.server,registered,root,pins,a.device)
        else:
            from .runtime import read_state,run,publish_safely
            state=read_state(root,a.server);atomic_json(root/'campaign_state.json',state)
            publish_safely(root,a.server)
            donor,stats=assets.prepare_reference(plan.resolve(a.server),registered,root,pins,a.device)
            from .reference_report import prepare
            prepare(a.server,a.device)
            from .reuse import apply_if_identical
            # A restored or already completed local run keeps its own trajectory.
            if state['runs'][0]['step']==0:
                apply_if_identical(a.server,root,donor,stats,pins)
            del donor,stats;gc.collect()
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
            state=read_state(root,a.server)
            if state['status']=='COMPLETED':
                publish_safely(root,a.server);proof=state
            else:
                from .preflight import run as preflight
                check=preflight(a.server,a.device);sizes=check['checkpoint_bytes']
                needed=50*sizes['evaluated']+2*sizes['complete_resume']+3*2**30
                if shutil.disk_usage(root).free<needed:raise RuntimeError('Insufficient disk for 50K evidence')
                proof=run(a.server,device=a.device)
    print(json.dumps(proof,ensure_ascii=False,default=str))

if __name__=='__main__':main()
