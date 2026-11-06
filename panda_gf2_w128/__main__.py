"""GF2-only entrypoints built on the unchanged validated sensor-focus engine."""
import argparse
import json
import sys
from . import registry as G


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['catalog', 'check', 'check-sheet', 'cost', 'run', 'publish'])
    parser.add_argument('--server', choices=['s3'], default='s3')
    parser.add_argument('--csv', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    from .adapter import install
    plan = install()
    root = G.ROOT / G.OUTPUT / G.CAMPAIGN_ID / 's3/GF2'
    if args.command == 'catalog':
        print(plan.catalog_csv('s3') if args.csv else json.dumps(plan.default_config('s3'), indent=2)); return 0
    from .release import verify
    verify(check_environment=args.command in ('check', 'cost', 'run'))
    if args.command in ('check-sheet', 'publish'):
        from .publication import install as install_publication
        install_publication()
        if args.command == 'publish':
            from panda_sensor_focus.best_publication.__main__ import publish
            return publish(root, 's3', apply=args.apply, watch=args.watch)
        from sheet_publish_v2.transport import GoogleSheets
        from panda_sensor_focus.best_publication import project
        from panda_sensor_focus.best_publication.transport import prepare
        from panda_sensor_focus.best_publication.header import reconcile_header
        root.mkdir(parents=True, exist_ok=True)
        value = project(root, 's3')
        adapter = GoogleSheets('gspread/account.json')
        prepare(adapter.snapshot(value['destination']), value, 's3')
        reconcile_header(adapter, value['destination'], root / 'sheet_publish_panda_sensor_focus', 's3', apply=False)
        print(json.dumps({'status': 'SHEET_READ_ONLY_CHECK_OK', 'destination': value['destination']})); return 0
    from .prerequisites import evidence
    proof = evidence(G.ROOT)
    if args.command == 'cost':
        from panda_sensor_focus.cost.__main__ import main as cost_main
        sys.argv = ['cost', '--server', 's3', '--root', str(root), '--seconds', '1200']
        return cost_main()
    from panda_sensor_focus.__main__ import main as engine_main
    result = engine_main([args.command, '--server', 's3'] +
                        (['--output', str(G.ROOT / G.OUTPUT)] if args.command == 'run' else []))
    if args.command == 'check':
        print(json.dumps({'local_reference': proof}))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
