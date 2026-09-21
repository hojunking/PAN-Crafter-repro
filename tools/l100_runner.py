#!/usr/bin/env python3
"""LOCAL-T explicit CLI. Import/build/status never launches an experiment."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'window', 'start', 'run', 'status',
        'preflight', 'train', 'calibrate', 'endpoint', 'postrun', 'partial', 'upload', 'bundle'])
    parser.add_argument('--server', choices=['s3', 's4', 's5'])
    parser.add_argument('--t0', help='Same timezone-aware actual preparation start for ALL servers')
    parser.add_argument('--window', help='Previously created common window JSON; no clock restart')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--in-place', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--run')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--dataset-manifest')
    parser.add_argument('--output', help='Bundle output directory')
    args = parser.parse_args(argv)
    from l100 import controller as C
    from l100.common import apply_runtime_policy, utcnow
    result, code = {}, 0
    if args.action == 'build':
        result = C.build_artifacts(ROOT)
    elif args.action == 'window':
        if not args.t0:
            parser.error('window requires --t0; use now ONCE centrally, then share that exact timestamp')
        result = C.write_window(ROOT, utcnow() if args.t0 == 'now' else args.t0)
    elif args.action == 'bundle':
        from l100.deployment import build_bundle
        result = build_bundle(ROOT, args.output)
    else:
        if args.server is None:
            parser.error('--server s3|s4|s5 is required')
        if args.action == 'status':
            result = C.status(ROOT, args.server)
        elif args.action == 'start':
            if args.t0 == 'now':
                parser.error('start --t0 now would reset clocks separately; share the window command timestamp')
            if not args.dry_run and not args.in_place:
                # Fix the shared clock before transition/setup time is spent.
                if args.t0:
                    C.write_window(ROOT, args.t0)
                window = C.shared_window(ROOT, args.window)
                C.write_window(ROOT, window.to_dict()['t0_utc'])
                from l100.deployment import frozen_checkout
                frozen = frozen_checkout(ROOT, args.server)
                forward = ['start', '--server', args.server, '--in-place']
                if args.foreground:
                    forward.append('--foreground')
                return subprocess.call([sys.executable, str(frozen / 'tools/l100_runner.py'), *forward], cwd=frozen)
            result = C.start(ROOT, args.server, t0=args.t0, window_path=args.window,
                foreground=args.foreground, dry_run=args.dry_run)
            code = result.get('exit_code', 0)
        else:
            apply_runtime_policy(ROOT)
            C.verify_registration(ROOT, args.server)
            window = C.shared_window(ROOT)
            deadline = window.deadline_utc.isoformat()
            if args.action == 'run':
                code = C.run(ROOT, args.server)
                result = dict(exit_code=code)
            elif args.action == 'preflight':
                from l100.preflight import execute
                result = execute(ROOT, args.server, window.train_finish_utc.isoformat(),
                                 manifest_path=args.dataset_manifest)
            elif args.action == 'train':
                if not args.config:
                    parser.error('train requires --config')
                C.authorize_train(ROOT, args.server, args.config)
                from l100.training import train_run
                code = int(train_run(args.config, device='cuda', resume=args.resume,
                                     deadline_arg=deadline, root=ROOT))
                result = dict(exit_code=code)
            elif args.action == 'calibrate':
                if not args.run:
                    parser.error('calibrate requires --run')
                from l100.calibration import calibrate
                result = dict(reference_manifest=str(calibrate(args.run, root=ROOT, server=args.server,
                                                              device='cuda', deadline=deadline)))
            else:
                if not args.run:
                    parser.error('evaluation action requires --run')
                from l100.plan import case_for
                if case_for(args.run).server_id != args.server:
                    parser.error('Run does not belong to this local server')
                from l100.postrun import process, process_endpoint, process_partial
                if args.action == 'endpoint':
                    result = process_endpoint(args.run, device='cuda', deadline=deadline, root=ROOT)
                    code = 0 if result.get('endpoint_complete') else 1
                elif args.action == 'partial':
                    result = process_partial(args.run, device='cuda', deadline=deadline, root=ROOT)
                else:
                    code = process(args.run, device='cuda', deadline=deadline,
                        upload=args.upload or args.action == 'upload',
                        upload_only=args.action == 'upload', root=ROOT)
                    result = dict(exit_code=code)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
