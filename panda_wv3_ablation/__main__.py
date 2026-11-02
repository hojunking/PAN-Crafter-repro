"""WV3 component catalog, read-only checks, training, and own-row publication."""
import argparse
import json
import time
from . import plan


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('catalog','check','check-sheet','run','publish'))
    p.add_argument('--server',required=True,choices=('s1','s3'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--apply',action='store_true',help='Apply Sheet writes for publish')
    p.add_argument('--watch',action='store_true',help='Repeat publication every 45 seconds')
    p.add_argument('--no-publish',action='store_true',help='Explicit local-only training; normal run publishes automatically')
    args=p.parse_args(argv)
    if args.command=='catalog':print(plan.catalog(args.server),end='');return
    from .release import verify
    value=verify(environment=args.command in ('check','run'))
    root=plan.campaign_root(args.server)
    if args.command=='check':
        from .assets import sources,load_reference
        source,registered=sources()
        if args.server=='s1':
            donor=load_reference(registered,server='s1')
            value['donor_file_sha256']=donor.binding['donor_checkpoint_sha256']
        value.update(server=args.server,run_count=len(plan.queue(args.server)),
                     source_manifests={s:source.manifest_sha(s) for s in ('train','val','rr','fr')},
                     stage1_weights_required=args.server=='s1')
    elif args.command in ('check-sheet','publish'):
        from .publication import publish
        while True:
            try:
                value=publish(root,args.server,apply=args.apply and args.command=='publish')
            except Exception as exc:
                if not args.watch or args.command=='check-sheet':raise
                from directshift48.reporting import atomic_json,utc_now
                value={'action':'PUBLICATION_PENDING','at':utc_now(),'error':repr(exc)}
                atomic_json(root/'publication/last_error.json',value)
            print(json.dumps(value),flush=True)
            if not args.watch or args.command=='check-sheet':return
            time.sleep(45)
    elif args.command=='run':
        from .runtime import run
        value=run(args.server,device=args.device,publish=not args.no_publish)
    print(json.dumps(value,ensure_ascii=False,default=str))


if __name__=='__main__':main()
