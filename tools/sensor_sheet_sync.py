#!/usr/bin/env python
"""Metadata-only GF2 Sheet supplement; no learning, inference or campaign restart."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reporting_extra.sensor_backfill import TABS, connect, ready_runs, sync_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--server', choices=tuple(TABS), required=True)
    parser.add_argument('--apply', action='store_true', help='default is read-only Sheet planning')
    parser.add_argument('--repair-layout', action='store_true',
                        help='also align this server GF2 tab with the common WV3-like column order')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--watch', nargs='?', const=60., type=float, metavar='SECONDS')
    mode.add_argument('--once', action='store_true', help='one pass (default)')
    parser.add_argument('--run', action='append', help='restrict to an existing local run; repeatable')
    parser.add_argument('--spreadsheet')
    parser.add_argument('--credentials', type=Path)
    args = parser.parse_args(argv)
    if args.watch is not None and args.watch < 60:
        parser.error('--watch interval must be at least 60 seconds')
    root = args.root.resolve()
    book = worksheet = None
    layout_done = False
    while True:
        failures = 0
        try:
            ready = ready_runs(root, args.server)
        except Exception as exc:
            print(json.dumps(dict(status='REPORTING_PENDING', error=f'{type(exc).__name__}: {exc}')), flush=True)
            if args.watch is None:
                return 2
            time.sleep(args.watch)
            continue
        if args.run:
            ready = [wd for wd in ready if wd.name in set(args.run)]
            missing = set(args.run) - {wd.name for wd in ready}
            for run in sorted(missing):
                print(json.dumps(dict(run_id=run, status='NOT_READY', reason='Complete verified local GF2 upload required')))
            failures += len(missing)
        if (ready or args.repair_layout) and worksheet is None:
            try:
                book = connect(root, spreadsheet=args.spreadsheet, credentials=args.credentials)
                worksheet = book.worksheet(TABS[args.server])  # Existing tab only.
            except Exception as exc:
                print(json.dumps(dict(status='REPORTING_PENDING', error=f'{type(exc).__name__}: {exc}')), flush=True)
                if args.watch is None:
                    return 2
                time.sleep(args.watch)
                continue
        if args.repair_layout and not layout_done:
            try:
                from reporting_extra.sensor_repair_layout import repair_layout
                result = repair_layout(worksheet, root, args.server, apply=args.apply)
                print(json.dumps({k: v for k, v in result.items()
                                  if k not in ('batch_requests', 'final_headers')},
                                 ensure_ascii=False), flush=True)
                # The native column operations can invalidate cached geometry.
                worksheet = book.worksheet(TABS[args.server])
                layout_done = True
            except Exception as exc:
                failures += 1
                print(json.dumps(dict(status='LAYOUT_PENDING',
                                      error=f'{type(exc).__name__}: {exc}'),
                                 ensure_ascii=False), flush=True)
                # Fail closed on structure; no metadata update from a stale view.
                if args.watch is None:
                    return 2
                time.sleep(args.watch)
                continue
        for wd in ready:
            try:
                result = sync_run(wd, args.server, worksheet, apply=args.apply)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as exc:
                failures += 1
                print(json.dumps(dict(run_id=wd.name, status='REPORTING_PENDING',
                                      error=f'{type(exc).__name__}: {exc}'), ensure_ascii=False), flush=True)
        if args.watch is None:
            return 2 if failures else 0
        # A reporting-only sidecar; never installs cron or changes controller state.
        time.sleep(args.watch)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
