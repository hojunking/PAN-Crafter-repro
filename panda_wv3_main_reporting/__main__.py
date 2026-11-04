"""Resume the unchanged main trainer with the cost/rounding reporting extension."""
import argparse
import json
from . import publication, cost
from panda_wv3_main_rgb import plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('run','publish','profile'))
    p.add_argument('--server',choices=('s1',),default='s1')
    p.add_argument('--case',default='M0_OURS')
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--step',type=int,default=24000)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--apply',action='store_true')
    args=p.parse_args();root=plan.campaign_root(args.server)
    if args.command=='profile':
        result=cost.measure(root,plan.resolve(args.server,args.case,args.repeat),args.step,device=args.device)
        print(json.dumps({'profile_sha256':result['profile_sha256'],
            'cost':{s:v['sheet_fields'] for s,v in result['measurement']['observations'].items()}}));return
    if args.command=='publish':
        result=publication.publish(root,args.server,apply=args.apply)
    else:
        from panda_wv3_main_rgb import publication as legacy, runtime
        # Only replace the reporting entry point; source/resume/optimizer/RNG checks remain original.
        cost.load(root,plan.resolve(args.server,'M0_OURS',1))
        legacy.publish=publication.publish
        result=runtime.run(args.server,device=args.device,publish=True)
    print(json.dumps(result,ensure_ascii=False,default=str))


if __name__=='__main__':main()
