"""Dedicated QB process using the existing focus engine and H/E Sheet policy."""
import argparse
import json
import sys
from . import registry as G


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('catalog', 'check', 'cost', 'run', 'publish'))
    p.add_argument('--server', choices=('s1',), default='s1')
    p.add_argument('--csv', action='store_true')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    from .adapter import install
    plan = install()
    root = plan.ROOT / G.OUTPUT / G.CAMPAIGN_ID / 's1/QB'
    if args.command == 'catalog':
        print(plan.catalog_csv('s1') if args.csv else json.dumps(plan.default_config('s1'), indent=2))
        return 0
    if args.command == 'cost':
        from .prerequisites import evidence
        evidence(plan.ROOT)
        from panda_sensor_focus.cost.__main__ import main as cost_main
        sys.argv = ['cost', '--server', 's1', '--root', str(root), '--seconds', '1200']
        return cost_main()
    if args.command == 'publish':
        from panda_sensor_focus.best_publication.__main__ import publish
        return publish(root, 's1', apply=args.apply, watch=args.watch)
    from panda_sensor_focus.__main__ import main as training_main
    return training_main([args.command, '--server', 's1'] +
                         (['--output', str(plan.ROOT / G.OUTPUT)] if args.command == 'run' else []))


if __name__ == '__main__':
    raise SystemExit(main())
