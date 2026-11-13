"""Portable QB/GF2 catalog, admission checks, execution and scoped Sheet rows."""
import argparse,json
from . import plan

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('catalog','check','check-sheet','run','publish'))
    p.add_argument('--server',choices=('s2','s3'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--apply',action='store_true')
    p.add_argument('--no-publish',action='store_true');a=p.parse_args()
    if a.command=='catalog':print(plan.catalog(a.server),end='');return
    from .release import verify
    value=verify(environment=a.command in ('check','run'))
    if a.command=='check':
        from .assets import sources
        source,registered=sources(a.server)
        value.update(server=a.server,sensor=source.sensor,runs=len(plan.queue(a.server)),
            source_manifests={s:source.manifest_sha(s) for s in ('train','val','rr','fr')},
            donor_policy=plan.PLAN['donor_selection'][source.sensor],
            original_qb_state_exists=(plan.donor_root('s2')/'campaign_state.json').is_file() if a.server=='s2' else None)
    elif a.command in ('publish','check-sheet'):
        from .publication import publish
        value=publish(plan.campaign_root(a.server),a.server,apply=a.apply and a.command=='publish')
    else:
        from .coordinator import run
        value=run(a.server,device=a.device,publish=not a.no_publish)
    print(json.dumps(value,ensure_ascii=False,default=str))

if __name__=='__main__':main()
