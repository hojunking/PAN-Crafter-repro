"""Shared three-server zero-safe Stage2 entrypoint, including FR67 publication."""
import argparse,json
from . import plan
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('catalog','check','check-sheet','publish','run'))
    p.add_argument('--server',choices=('s1','s2','s3'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    if a.command=='catalog':print(plan.catalog(a.server),end='');return
    from .release import verify
    result=verify(environment=a.command in ('check','run'))
    from panda_fr_columns.publication import install
    install()
    if a.command=='check':
        from .assets import sources
        src,registered=sources(a.server)
        result.update(server=a.server,sensor=src.sensor,runs=len(plan.queue(a.server)),
            q_zero_policy=plan.Q_ZERO_POLICY,source_manifests={s:src.manifest_sha(s) for s in ('train','val','rr','fr')})
    elif a.command in ('publish','check-sheet'):
        from .publication import publish,destination
        if a.apply:
            from panda_fr_columns.adapter import GoogleSheets
            if (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:raise ValueError('Wrong server')
            GoogleSheets(plan.ROOT/'gspread/account.json').ensure_layout(destination(a.server),
                plan.campaign_root(a.server)/'publication/fr_columns',apply=True)
        result=publish(plan.campaign_root(a.server),a.server,apply=a.command=='publish' and a.apply)
    else:
        from .publication import destination
        from panda_fr_columns.adapter import GoogleSheets
        if (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:raise ValueError('Wrong server')
        GoogleSheets(plan.ROOT/'gspread/account.json').ensure_layout(destination(a.server),
            plan.campaign_root(a.server)/'publication/fr_columns',apply=True)
        from .coordinator import run
        result=run(a.server,device=a.device)
    print(json.dumps(result,ensure_ascii=False,default=str))
if __name__=='__main__':main()
