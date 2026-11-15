"""Run each server's existing-donor input comparison; no Stage1 training."""
import argparse,json,shutil
from directshift48.reporting import atomic_json
from . import plan
from .release import verify

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('catalog','check','check-sheet','publish','run'))
    p.add_argument('--server',choices=('s1','s2','s3'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    if a.command=='catalog':print(plan.catalog(a.server),end='');return
    proof=verify(environment=a.command in ('check','run'))
    from .assets import sources,selected_reference
    from .reporting import install
    install()
    from .publication import publish,destination
    root=plan.campaign_root(a.server)
    if a.command=='run' or a.apply:
        if (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:
            raise ValueError('Execute on matching physical server; no donor transfer')
    if a.command=='check':
        from ta2.evaluation import evaluator_identity
        source,registered=sources(a.server)
        pins={'evaluator':evaluator_identity(),'train_inventory':registered.inventory.manifest(),
              'split_manifests':{s:source.manifest_sha(s) for s in ('train','val','rr','fr')}}
        dc,receipt,_,_=selected_reference(a.server,registered,pins)
        proof.update(sensor=source.sensor,donor=dc['run_id'],donor_step=receipt['selected']['step'],
                     new_stage1_runs=0,stage2_runs=3)
    elif a.command in ('publish','check-sheet'):
        proof=publish(root,a.server,apply=a.command=='publish' and a.apply)
    else:
        from panda_sensor_main50.__main__ import wait_for_gpu
        wait_for_gpu(root)
        from panda_fr_columns.adapter import GoogleSheets
        GoogleSheets(plan.ROOT/'gspread/account.json').ensure_layout(destination(a.server),root/'publication/fr_columns',apply=True)
        from .runtime import read_state,run,publish_safely
        state=read_state(root,a.server);atomic_json(root/'campaign_state.json',state)
        publish_safely(root,a.server)
        if all(r['status'] in ('COMPLETED','FAILED','BLOCKED_DONOR') for r in state['runs']):proof=state
        else:
            from .preflight import run as preflight
            check=preflight(a.server,a.device);sizes=check['checkpoint_bytes']
            needed=int(1.1*sum(len(c['evaluation_steps'])*sizes['evaluated']+2*sizes['complete_resume'] for c in plan.queue(a.server)))+3*2**30
            if shutil.disk_usage(root).free<needed:raise RuntimeError('Insufficient disk for all evaluated weights')
            proof=run(a.server,device=a.device)
    print(json.dumps(proof,ensure_ascii=False,default=str))

if __name__=='__main__':main()
