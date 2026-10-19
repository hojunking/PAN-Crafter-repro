"""Explicit s3-only execution; help/catalog/check do not start training."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='s3/WV3 midpoint Stage 2, three cases, no inherited deadline')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('mounts', help='Read only registered WV3 data mount targets')
    sub.add_parser('smoke', help='CPU synthetic contracts; no training or Sheets writes')
    for name in ('catalog', 'check', 'run'):
        command = sub.add_parser(name)
        command.add_argument('--server', choices=('s3',), required=True)
        command.add_argument('--replicas', choices=('R1', 'R2'), nargs='+', default=['R1'])
        if name == 'catalog':
            command.add_argument('--csv', action='store_true')
        if name in ('check', 'run'):
            command.add_argument('--sources', default='ablr2/sensor_sources.json')
        if name == 'run':
            command.add_argument('--output', default='work_dir/panda_midpoint')
            command.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.command == 'mounts':
        from edgeres48.docker_mounts import targets
        # The legacy helper maps s1->WV3 and s3->GF2; dataset is WV3 here.
        for path in targets(Path(__file__).resolve().parents[1], 's1'):
            print(path)
        return 0
    from . import plan
    if args.command == 'smoke':
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent/'tests'),
            pattern='test*.py', top_level_dir=str(plan.ROOT))
        return int(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    plan.default_config(args.server, args.replicas)
    if args.command == 'catalog':
        print(plan.catalog_csv(args.server, args.replicas) if args.csv else
              json.dumps(plan.default_config(args.server, args.replicas), indent=2))
        return 0
    if args.command == 'check':
        from panda_stage2_condq.sources import RegisteredRRTrain
        from ta2.evaluation import evaluator_identity, file_sha
        from .runtime import code_pin, discover_mature_donor
        cfg = plan.default_config(args.server, args.replicas)
        if file_sha(args.sources) != cfg['baseline_resolved']['source_registry_sha256']:
            raise ValueError('Data registry SHA mismatch')
        source = RegisteredRRTrain.from_catalog(args.sources, root=plan.ROOT, sensor='WV3')
        for split in ('val', 'rr', 'fr'):
            source.sources.raw(split, [0])
        code_pin()
        evaluator_identity()
        discover_mature_donor(args.server, root=plan.ROOT)
        print(json.dumps({'status': 'READ_ONLY_CHECK_OK', 'server': args.server,
                          'dataset': 'WV3', 'train_count': len(source.sample_ids),
                          'runs': 3*len(args.replicas), 'donor_import': 'LOCAL_M_RECEIPT_PRESENT'}))
        return 0
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu(args.device)
    from .runtime import Campaign
    campaign = Campaign(args.server, output=args.output, device=args.device, replicas=args.replicas)
    state = campaign.run(catalog=Path(args.sources))
    print(json.dumps({'root': str(campaign.root), 'status': state['status']}))
    return int(state['status'] == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
