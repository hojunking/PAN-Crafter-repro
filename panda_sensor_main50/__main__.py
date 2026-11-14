"""Remote main experiment entrypoint. Five cases per server, full 50K."""
import argparse
import json
import shutil
import subprocess
import time
from directshift48.reporting import atomic_json,utc_now
from . import plan
from .release import verify

def check_assets(server):
    from ta2.evaluation import evaluator_identity
    from .assets import sources,selected_reference
    source,registered=sources(server)
    pins={'evaluator':evaluator_identity(),'train_inventory':registered.inventory.manifest(),
          'split_manifests':{s:source.manifest_sha(s) for s in ('train','val','rr','fr')}}
    _,receipt,_,_=selected_reference(server,registered,pins)
    return {'sensor':source.sensor,'donor_run_id':receipt['donor_run_id'],
        'donor_step':receipt['selected']['step'],'donor_sha256':receipt['selected']['file_sha256'],
        'performance_gate':False}

def wait_for_gpu(root):
    while True:
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if not active:
            atomic_json(root/'queue_wait.json',{'status':'READY','at':utc_now()});return
        atomic_json(root/'queue_wait.json',{'status':'WAITING_GPU','active_pids':active.splitlines(),'at':utc_now()})
        print('Waiting for current GPU work to finish; no forced stop',flush=True)
        time.sleep(30)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('catalog','check','check-sheet','publish','run'))
    p.add_argument('--server',choices=('s2','s3'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    if a.command=='catalog':print(plan.catalog(a.server),end='');return
    proof=verify(environment=a.command in ('check','run'))
    from .reporting import install
    install()
    if a.command=='check':
        proof.update(check_assets(a.server),runs=5,horizon=50000,early_stop=False)
    else:
        from .publication import publish,destination
        root=plan.campaign_root(a.server)
        if a.command=='run' or a.apply:
            if (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:
                raise ValueError('Run on matching server; do not transfer donors')
            from panda_fr_columns.adapter import GoogleSheets
            GoogleSheets(plan.ROOT/'gspread/account.json').ensure_layout(destination(a.server),root/'publication/fr_columns',apply=True)
        if a.command!='run':proof=publish(root,a.server,apply=a.command=='publish' and a.apply)
        else:
            from .runtime import read_state,run,publish_safely
            state=read_state(root,a.server);atomic_json(root/'campaign_state.json',state)
            publish_safely(root,a.server)
            wait_for_gpu(root)
            from .preflight import run as preflight
            check=preflight(a.server,a.device)
            sizes=check['checkpoint_bytes']
            needed=5*(50*sizes['evaluated']+2*sizes['complete_resume'])+3*2**30
            if shutil.disk_usage(root).free<needed:
                raise RuntimeError('Insufficient disk for full50K evidence and checkpoints')
            proof=run(a.server,device=a.device)
    print(json.dumps(proof,ensure_ascii=False,default=str))

if __name__=='__main__':main()
