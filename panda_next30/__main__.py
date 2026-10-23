"""Explicit local s1/s3 execution; catalog/check/smoke do not launch experiments."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description='s1/s3 WV3 positive PAN scale, fresh Lanczos4, sampling and full-shift PAN-difference experiments without a wall deadline')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('mounts')
    sub.add_parser('smoke')
    for name in ('catalog','check','run'):
        p=sub.add_parser(name)
        p.add_argument('--server',choices=('s1','s3'),required=True)
        if name=='catalog': p.add_argument('--csv',action='store_true')
        if name in ('check','run'): p.add_argument('--sources',default='ablr2/sensor_sources.json')
        if name=='run':
            p.add_argument('--output',default='work_dir/panda_next30')
            p.add_argument('--device',default='cuda')
    args=parser.parse_args(argv)
    if args.command=='mounts':
        from edgeres48.docker_mounts import targets
        # Historical helper maps s1 to the WV3 sensor; both physical servers
        # in this campaign use WV3. Passing physical s3 would select GF2.
        for path in targets(Path(__file__).resolve().parents[1],'s1'): print(path)
        return 0
    from . import plan
    if args.command=='smoke':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent/'tests'),
            pattern='test*.py')
        return int(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    cfg=plan.default_config(args.server)
    if args.command=='catalog':
        print(plan.catalog_csv(args.server) if args.csv else json.dumps(cfg,indent=2))
        return 0
    if args.command=='check':
        from panda_stage2_condq.sources import RegisteredRRTrain
        from ta2.evaluation import evaluator_identity,file_sha
        from .runtime import code_pin
        if file_sha(args.sources)!=cfg['baseline_resolved']['source_registry_sha256']:
            raise ValueError('Registered data catalog SHA changed')
        registered=RegisteredRRTrain.from_catalog(args.sources,root=plan.ROOT,sensor='WV3')
        for split in ('val','rr','fr'): registered.sources.raw(split,[0])
        from .donor import source_receipt, load_donor
        _, receipt = source_receipt(args.server, plan.ROOT)
        selected = receipt['selected']
        donor = load_donor(selected['checkpoint_path'], selected['file_sha256'], registered,
                          server=args.server, selector_receipt=receipt)
        donor.assert_frozen()
        code_pin(); evaluator_identity()
        print(json.dumps({'status':'READ_ONLY_CHECK_OK','server':args.server,'dataset':'WV3',
                          'train_count':len(registered.sample_ids),'stage1':cfg['donor_policy'],
                          'old_checkpoint_required':True,'budget_hours':None,
                          'donor_selected_step':selected['completed_update']}))
        return 0
    marker=plan.ROOT/'gspread/server.txt'
    if not marker.is_file() or marker.read_text().strip() != args.server:
        raise ValueError('Local server marker must match requested s1/s3; nothing started')
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu(args.device)
    from .runtime import Campaign
    campaign=Campaign(args.server,output=args.output,device=args.device)
    state=campaign.run(catalog=Path(args.sources))
    print(json.dumps({'root':str(campaign.root),'status':state['status']}))
    return int(state['status']=='FAILED')


if __name__=='__main__':
    raise SystemExit(main())
