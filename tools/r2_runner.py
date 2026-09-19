#!/usr/bin/env python
"""HEAD-preserving FH20R1 R2 priority overlay; explicit start, no automatic activation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from tools.fh12_runner import detect_server
    from campaign_r2 import controller as C
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'audit', 'start', 'run', 'ensure', 'status'))
    parser.add_argument('--server', choices=('s1', 's2', 's3', 's4', 's5'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    server = detect_server(ROOT, args.server)
    if args.action in ('check', 'audit'):
        result = C.check(ROOT, server, write_reports=args.action == 'audit' and not args.dry_run)
    elif args.action == 'start':
        result = C.start(ROOT, server, dry_run=args.dry_run)
    elif args.action == 'run':
        return C.run(ROOT, server)
    elif args.action == 'ensure':
        result = C.ensure(ROOT, server)
    else:
        from fh20r1.ledger import read, report
        folder = C.directory(ROOT, server)
        state = read(ROOT / 'work_dir/_fh20r1' / server / 'status.json')
        result = dict(registration=read(folder / 'registration.json'),
                      applied=read(folder / 'applied_readback.json'),
                      controller=read(folder / 'controller_status.json'), timing=report(ROOT, server),
                      live_state=state, live_updates={run: read(ROOT / 'work_dir' / run / 'meta/training_status.json')
                          for run, row in state.get('runs', {}).items() if row.get('status') not in ('DONE', 'DONE_REUSED')})
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.action in ('check', 'audit') and not result['audit'].get('admission_integrity_ok'):
        return 2
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, BlockingIOError) as error:
        print(f'FH20R1 R2: {error}', file=sys.stderr)
        sys.exit(2)
