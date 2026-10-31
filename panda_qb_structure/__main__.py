"""Fresh QB Students using each server's independently trained local Stage1."""
import argparse
import json
import sys
from . import registry as G


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('catalog', 'check', 'cost', 'run', 'publish', 'verify-donor'))
    parser.add_argument('--server', choices=('s1', 's3'), required=True)
    parser.add_argument('--csv', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    from .adapter import install
    plan = install(args.server)
    root = plan.ROOT / G.OUTPUT / G.CAMPAIGN_ID / args.server / 'QB'
    if args.command == 'catalog':
        print(plan.catalog_csv(args.server) if args.csv else json.dumps(plan.default_config(args.server), indent=2))
        return 0
    if args.command == 'verify-donor':
        from .donor import verify
        print(json.dumps(verify(plan.ROOT, args.server)))
        return 0
    if args.command == 'publish':
        from panda_sensor_focus.best_publication.__main__ import publish
        return publish(root, args.server, apply=args.apply, watch=args.watch)
    if args.command == 'cost':
        from .donor import validate_source
        validate_source(args.server)
        from panda_sensor_focus.cost.__main__ import main as cost_main
        sys.argv = ['cost', '--server', args.server, '--root', str(root), '--seconds', '1200']
        return cost_main()
    from panda_sensor_focus.__main__ import main as training_main
    return training_main([args.command, '--server', args.server] +
                         (['--output', str(plan.ROOT / G.OUTPUT)] if args.command == 'run' else []))


if __name__ == '__main__':
    raise SystemExit(main())
