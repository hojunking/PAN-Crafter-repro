#!/usr/bin/env python
"""One-command supplemental evaluation and backfill, outside campaign source hashes."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('start', 'watch', 'once', 'process', 'status'))
    parser.add_argument('--server', choices=('s1', 's2', 's3', 's4', 's5'))
    parser.add_argument('--run')
    parser.add_argument('--if-enabled', action='store_true')
    parser.add_argument('--no-cron', action='store_true')
    args = parser.parse_args()
    # Also enforce CPU constraints for a manually launched `process`.
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    from tools.fh12_runner import detect_server
    from reporting_extra.worker import start, watch, process_run, running, state_dir
    from fh12.common import read_json
    server = detect_server(ROOT, args.server)
    if args.command == 'start':
        result = start(ROOT, server, if_enabled=args.if_enabled, cron=not args.no_cron)
    elif args.command in ('watch', 'once'):
        result = watch(ROOT, server, once=args.command == 'once')
    elif args.command == 'process':
        if not args.run:
            parser.error('process requires --run')
        result = process_run(ROOT, args.run)
    else:
        path = state_dir(ROOT, server) / 'status.json'
        result = read_json(path) if path.is_file() else {}
        result['worker_running'] = running(ROOT, server)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
