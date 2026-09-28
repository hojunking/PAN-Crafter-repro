#!/usr/bin/env python3
"""Explicit B01 run/status/report. Does not install cron or upload Sheets."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panda_rb.common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan', 'status', 'run', 'report', 'package'])
    p.add_argument('--server', choices=['s1', 's3', 's5'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--binding', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--retry-technical', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.command == 'plan':
        from panda_rb.plan import validate_plan, schedule
        result = validate_plan(args.root)
        if args.server: result['schedule'] = schedule(args.server, args.root)
    elif args.command in ['status', 'run']:
        if not args.server: p.error('--server is required')
        from panda_rb.controller import status, run
        result = status(args.server, args.root) if args.command == 'status' else run(
            args.server, args.root, args.binding, args.device, args.retry_technical)
    else:
        from panda_rb.reporting import summarize, package
        servers = [args.server] if args.server else ['s1', 's3', 's5']
        result = summarize(args.root, servers=servers, verify=True)
        if args.command == 'package':
            if not args.output: p.error('--output ZIP path is required')
            result = package(args.root, servers, args.output)
        elif args.output:
            from panda_rb.common import atomic_json
            atomic_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__': main()
