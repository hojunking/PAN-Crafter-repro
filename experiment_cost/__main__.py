import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent RR/FR cost measurement and complete qualified-result publication')
    sub = parser.add_subparsers(dest='command', required=True)
    guard = sub.add_parser('_resume-guard')
    for name in ('pid', 'starttime', 'cancel', 'parent', 'seconds'):
        guard.add_argument(name)
    measure = sub.add_parser('_measure')
    measure.add_argument('--request', required=True)
    for name in ('start', 'measure', 'publish', 'audit'):
        command = sub.add_parser(name)
        command.add_argument('--server', choices=('s1', 's3'), required=True)
        command.add_argument('--root')
        command.add_argument('--cost-root')
        if name in ('start', 'measure'):
            command.add_argument('--container', help='Verified running DirectShift container (defaults panda-directshift48-SERVER)')
        if name in ('start', 'publish'):
            command.add_argument('--credentials', default='gspread/account.json')
        if name == 'publish':
            command.add_argument('--apply', action='store_true')
            command.add_argument('--watch', action='store_true')
            command.add_argument('--google-read-only', action='store_true')
            command.add_argument('--poll-seconds', type=float, default=120)
    args = parser.parse_args(argv)
    from . import operations
    if args.command == '_resume-guard':
        operations.resume_guard(int(args.pid), args.starttime, args.cancel, int(args.parent), float(args.seconds))
        return 0
    if args.command == '_measure':
        operations.measure_request(args.request)
        return 0
    root = Path(args.root).resolve() if args.root else operations.campaign_root(args.server)
    if args.command == 'publish':
        from .publication import publish
        return publish(root, args.server, credentials=args.credentials, apply=args.apply,
            watch=args.watch, google_read_only=args.google_read_only, cost_root=args.cost_root,
            poll_seconds=args.poll_seconds)
    if args.command == 'audit':
        from .audit import audit_campaign
        result = audit_campaign(root, args.server)
    elif args.command == 'measure':
        result = operations.measure_needed(root, args.server, args.container or f'panda-directshift48-{args.server}',
                                           cost_root=args.cost_root)
    else:
        result = operations.start(root, args.server, args.container or f'panda-directshift48-{args.server}',
                                  credentials=args.credentials, cost_root=args.cost_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get('status') == 'INCOMPLETE' else 0


if __name__ == '__main__':
    raise SystemExit(main())
