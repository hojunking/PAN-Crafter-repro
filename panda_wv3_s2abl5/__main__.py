import argparse
import json
import shutil
import subprocess
import time
from directshift48.reporting import atomic_json,utc_now
from . import plan
from .release import verify

def wait_for_main(root):
    source=plan.main.campaign_root('s1')/'campaign_state.json'
    while True:
        state=json.loads(source.read_text())
        done=all(r['status']=='COMPLETED' and r['step']==40000 for r in state['runs'])
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if done and not active:
            atomic_json(root/'queue_wait.json',{'status':'READY','at':utc_now(),'main_completed':True})
            return
        status='WAITING_MAIN' if not done else 'WAITING_GPU'
        atomic_json(root/'queue_wait.json',{'status':status,'at':utc_now(),
            'main_runs':[{k:r[k] for k in ('run_id','status','step')} for r in state['runs']],
            'active_compute_pids':active.splitlines()})
        print(status,'main steps',[r['step'] for r in state['runs']],flush=True)
        time.sleep(30)

def main():
    p=argparse.ArgumentParser(description='WV3 fixed-donor Stage2 ablation: five cases, three Student seeds')
    p.add_argument('command',choices=('catalog','check','check-sheet','publish','run','summary'))
    p.add_argument('--server',choices=('s1',),default='s1')
    p.add_argument('--device',default='cuda:0');p.add_argument('--apply',action='store_true')
    a=p.parse_args();root=plan.campaign_root(a.server)
    if a.command=='catalog':print(plan.catalog(a.server),end='');return
    result=verify(environment=a.command in ('check','run'))
    from .reporting import install
    install()
    if a.command=='check':
        from .assets import sources
        from .reuse import validate_recipe
        src,_=sources(a.server);validate_recipe(plan.resolve('s1','E_OURS',1))
        result.update(sensor=src.sensor,runs=15,new_training_runs=14,fixed_donor=plan.PLAN['fixed_donor'])
    elif a.command=='summary':
        from .summary import write
        result=write(root,a.server)
    else:
        from .publication import publish,destination
        if a.command=='run' or a.apply:
            if (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:raise ValueError('Wrong physical server')
            from panda_fr_columns.adapter import GoogleSheets
            GoogleSheets(plan.ROOT/'gspread/account.json').ensure_layout(destination(a.server),root/'publication/fr_columns',apply=True)
            from .reuse import initialize
            initialize(root,a.server)
        if a.command!='run':
            result=publish(root,a.server,apply=a.command=='publish' and a.apply)
        else:
            # Queue visibility is published now; no GPU training/preflight until main completes.
            from .runtime import publish_safely
            publish_safely(root,a.server)
            wait_for_main(root)
            from .preflight import run as check
            proof=check(a.server,a.device)
            sizes=proof['checkpoint_bytes']
            needed=sum(40*sizes['evaluated']+2*sizes['complete_resume'] for c in plan.queue(a.server) if not c['reuse_existing'])+3*2**30
            if shutil.disk_usage(root).free<needed:raise RuntimeError('Insufficient storage for ablation exceptions/retained weights')
            from .runtime import run
            result=run(a.server,device=a.device)
    print(json.dumps(result,ensure_ascii=False,default=str))

if __name__=='__main__':main()
