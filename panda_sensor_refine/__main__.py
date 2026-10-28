"""Explicit read-only checks and local sensor training."""
import argparse
import json
from pathlib import Path

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('smoke')
    for name in ('mounts','catalog','check','run'):
        p=sub.add_parser(name); p.add_argument('--server',choices=('s1','s3'),required=True)
        if name=='catalog': p.add_argument('--csv',action='store_true')
        if name in ('check','run'): p.add_argument('--sources',default='ablr2/sensor_sources.json')
        if name=='run':
            p.add_argument('--output',default='work_dir/panda_sensor_refine')
            p.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    from . import plan
    if args.command=='smoke':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent/'tests'),pattern='test*.py')
        return int(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    cfg=plan.default_config(args.server)
    if args.command=='catalog':
        print(plan.catalog_csv(args.server) if args.csv else json.dumps(cfg,indent=2)); return 0
    if args.command=='mounts':
        from edgeres48.docker_mounts import targets
        for p in targets(plan.ROOT,'s2' if args.server=='s1' else 's3'): print(p)
        return 0
    if args.command=='check':
        from panda_stage2_condq.sources import RegisteredRRTrain
        from ta2.evaluation import file_sha,evaluator_identity
        from .runtime import code_pin
        if file_sha(args.sources)!=cfg['baseline_resolved']['source_registry_sha256']:
            raise ValueError('Registered source catalog changed')
        reg=RegisteredRRTrain.from_catalog(args.sources,root=plan.ROOT,sensor=cfg['dataset'],
                                           lms_overrides=plan.lms_overrides(cfg['dataset']))
        for split in ('val','rr','fr'): reg.sources.raw(split,[0])
        import h5py
        with h5py.File(reg.sources._verify('val'),'r') as f: val_count=len(f['pan'])
        lane=plan.LANES[args.server]
        if len(reg.sample_ids)!=lane['train_count'] or val_count!=lane['val_count']:
            raise ValueError('Wrong sensor train/validation population')
        code_pin(); evaluator_identity()
        print(json.dumps({'status':'READ_ONLY_CHECK_OK','server':args.server,
            'dataset':cfg['dataset'],'train_count':len(reg.sample_ids),'val_count':val_count,
            'source_sha_verified':True,'donor_policy':cfg['donor_policy'],
            'stage1_runs':3,'stage2_runs':7,'joint_target':cfg['joint_target']})); return 0
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!=args.server:
        raise ValueError('Local server marker mismatch')
    from directshift48.__main__ import require_idle_gpu
    from .runtime import Campaign
    require_idle_gpu(args.device)
    campaign=Campaign(args.server,output=args.output,device=args.device)
    state=campaign.run(catalog=Path(args.sources))
    print(json.dumps({'root':str(campaign.root),'status':state['status']}))
    return int(state['status'] not in ('FINISHED_REGISTERED_QUEUE','FINISHED_WITH_EXPLICIT_FAILURES'))

if __name__=='__main__':
    raise SystemExit(main())
