"""One explicit launch after pull; all admission/calibration steps automated."""
import argparse
import json
from pathlib import Path

def main(argv=None):
    parser=argparse.ArgumentParser(description='40h per server: s1 S2D / s3 fixed Direct donor Stage2')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('mounts')
    sub.add_parser('smoke')
    for name in ('catalog','check','run'):
        p=sub.add_parser(name)
        p.add_argument('--server',choices=('s1','s3'),required=True)
        if name=='catalog':
            p.add_argument('--csv',action='store_true')
        if name in ('check','run'):
            p.add_argument('--sources',default='ablr2/sensor_sources.json')
        if name=='run':
            p.add_argument('--output',default='work_dir/panda40')
            p.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    if args.command=='mounts':
        from edgeres48.docker_mounts import targets
        for path in targets(Path(__file__).resolve().parents[1],'s1'): # WV3 on BOTH servers
            print(path)
        return 0
    from . import plan
    if args.command=='smoke':
        import unittest
        tests=unittest.defaultTestLoader.discover(str(Path(__file__).parent),
                                                 pattern='test*.py',top_level_dir=str(plan.ROOT))
        return not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful()
    if args.command=='catalog':
        print(plan.catalog_csv() if args.csv else json.dumps(plan.default_config(args.server),indent=2))
        return 0
    if args.command=='check':
        # Read-only admission before replacing an existing job; no training,
        # calibration, GPU allocation, checkpoint mutation or sheet write.
        from panda_stage2_condq.sources import RegisteredRRTrain
        from ta2.evaluation import file_sha,evaluator_identity
        from .runtime import code_pin
        cfg=plan.default_config(args.server)
        if file_sha(args.sources)!=cfg['baseline_resolved']['source_registry_sha256']:
            raise ValueError('Data registry SHA mismatch')
        registered=RegisteredRRTrain.from_catalog(args.sources,root=plan.ROOT,sensor='WV3')
        for split in ('val','rr','fr'):
            registered.sources.raw(split,[0])
        code_pin()
        evaluator_identity()
        print(json.dumps({'status':'READ_ONLY_CHECK_OK','server':args.server,'dataset':'WV3',
                          'train_count':len(registered.sample_ids),'hours':40}))
        return 0
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu(args.device)
    from .runtime import Campaign
    campaign=Campaign(args.server,output=args.output,device=args.device)
    state=campaign.run(catalog=Path(args.sources))
    print(json.dumps({'root':str(campaign.root),'status':state['status']}))
    return 1 if state['status']=='FAILED' else 0


if __name__=='__main__':
    raise SystemExit(main())
