"""Explicit 30h WV3 launch; catalog/help/mounts never start an experiment."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='s1/s3/s4: local-donor schedule ablations, WV3, durable 30h budget')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('mounts', help='Print only registered WV3 data mount targets')
    sub.add_parser('smoke', help='Offline CPU contract tests; no training or Sheets writes')
    for name in ('catalog', 'check', 'run'):
        command = sub.add_parser(name)
        command.add_argument('--server', choices=('s1', 's3', 's4'), required=True)
        if name == 'catalog':
            command.add_argument('--csv', action='store_true')
        if name in ('check', 'run'):
            command.add_argument('--sources', default='ablr2/sensor_sources.json')
        if name == 'run':
            command.add_argument('--output', default='work_dir/panda30r')
            command.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.command == 'mounts':
        from edgeres48.docker_mounts import targets
        # The legacy mount helper's s1 means WV3. Its s3 would select GF2.
        for path in targets(Path(__file__).resolve().parents[1], 's1'):
            print(path)
        return 0
    from . import plan
    if args.command == 'smoke':
        import unittest
        tests = unittest.defaultTestLoader.discover(
            str(Path(__file__).parent / 'tests'), pattern='test*.py',
            top_level_dir=str(plan.ROOT))
        return int(not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful())
    if args.command == 'catalog':
        print(plan.catalog_csv(args.server) if args.csv else
              json.dumps(plan.default_config(args.server), indent=2))
        return 0
    if args.command == 'check':
        from panda_stage2_condq.sources import RegisteredRRTrain
        from ta2.evaluation import evaluator_identity, file_sha
        from .runtime import code_pin
        cfg = plan.default_config(args.server)
        if file_sha(args.sources) != cfg['baseline_resolved']['source_registry_sha256']:
            raise ValueError('Data registry SHA mismatch')
        registered = RegisteredRRTrain.from_catalog(args.sources, root=plan.ROOT, sensor='WV3')
        for split in ('val', 'rr', 'fr'):
            registered.sources.raw(split, [0])
        code_pin()
        evaluator_identity()
        print(json.dumps({'status': 'READ_ONLY_CHECK_OK', 'server': args.server,
                          'dataset': 'WV3', 'train_count': len(registered.sample_ids),
                          'hours': 30}))
        return 0
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu(args.device)
    from .runtime import Campaign
    campaign = Campaign(args.server, output=args.output, device=args.device)
    state = campaign.run(catalog=Path(args.sources))
    print(json.dumps({'root': str(campaign.root), 'status': state['status']}))
    return int(state['status'] == 'FAILED')


if __name__ == '__main__':
    raise SystemExit(main())
