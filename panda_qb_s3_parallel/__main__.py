import argparse,json,shutil
from directshift48.reporting import atomic_json
from . import plan

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('catalog','check','check-sheet','run','publish'))
    p.add_argument('--server',choices=('s3',),default='s3');p.add_argument('--apply',action='store_true');p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.command=='catalog':print(plan.catalog(a.server));return
    from .release import verify
    result=verify(environment=a.command in ('check','run'))
    from .reporting import install
    install()
    from .publication import publish,destination
    root=plan.campaign_root(a.server)
    if a.command=='check':
        from .assets import sources
        data,registered=sources(a.server);result.update(sensor=data.sensor,runs=6,stage1=1,stage2=5,original_qb_lms=True)
    elif a.command in ('check-sheet','publish'):result=publish(root,a.server,apply=a.command=='publish' and a.apply)
    else:
        if (plan.ROOT/'gspread/server.txt').read_text().strip()!='s3':raise ValueError('s3 only')
        from panda_fr_columns.adapter import GoogleSheets
        GoogleSheets(plan.ROOT/'gspread/account.json').ensure_layout(destination(a.server),root/'publication/fr_columns',apply=True)
        from .runtime import read_state,run,publish_safely
        atomic_json(root/'campaign_state.json',read_state(root,a.server));publish_safely(root,a.server)
        from .preflight import run as preflight
        proof=preflight(a.server,a.device);size=proof['checkpoint_bytes']
        needed=11*(50*size['evaluated']+2*size['complete_resume'])+3*2**30
        if shutil.disk_usage(root).free<needed:raise RuntimeError('Insufficient storage for donor + five full50K runs')
        result=run(a.server,device=a.device)
    print(json.dumps(result,ensure_ascii=False,default=str))
if __name__=='__main__':main()
