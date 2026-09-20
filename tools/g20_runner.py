#!/usr/bin/env python
"""G20 explicit operations. build/check/status never activate learning or a clock."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'bundle', 'inventory', 'window', 'check', 'status',
        'start', 'run', 'ensure', 'preflight', 'train', 'calibrate', 'adopt', 'export', 'import',
        'parity', 'postrun', 'upload', 'transfer', 'transfer-select'))
    parser.add_argument('--server', choices=('s1', 's2', 's3', 's4', 's5'))
    parser.add_argument('--t0')
    parser.add_argument('--window')
    parser.add_argument('--spec')
    parser.add_argument('--config')
    parser.add_argument('--run')
    parser.add_argument('--reference', choices=('R0', 'R1', 'R2', 'R3', 'R4'))
    parser.add_argument('--manifest')
    parser.add_argument('--archive')
    parser.add_argument('--evidence')
    parser.add_argument('--reference-screen')
    parser.add_argument('--student-screen')
    parser.add_argument('--global-receipt')
    parser.add_argument('--before-confirmation', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from g20 import controller as C
    if args.action == 'build':
        result = C.build_artifacts(ROOT)
    elif args.action == 'bundle':
        from g20.deployment import bundle
        result = bundle(ROOT)
    elif args.action == 'inventory':
        from g20.data import inventory_sources
        from g20.references import inventory_references
        result = dict(data=inventory_sources(ROOT, 'GF2'), references=inventory_references(ROOT))
    elif args.action == 'window':
        if not args.t0:
            parser.error('--t0 timezone-aware common preparation start is required')
        result = C.write_window(ROOT, args.t0)
    else:
        from tools.fh12_runner import detect_server
        server = detect_server(ROOT, args.server)
        if args.action in ('check', 'status'):
            result = C.status(ROOT, server, window_path=args.window)
        elif args.action == 'start':
            result = C.start(ROOT, server, spec_path=args.spec, window_path=args.window,
                dry_run=args.dry_run, device=args.device, foreground=args.foreground)
        elif args.action == 'run':
            return C.run(ROOT, server)
        elif args.action == 'ensure':
            result = C.ensure(ROOT, server)
        else:
            deadline = C.shared_window(ROOT).deadline_utc.isoformat()
            if args.action == 'preflight':
                from g20.preflight import execute
                result = execute(ROOT, server, args.spec or C.camp(ROOT, server) / 'sensor_spec.json', deadline, args.device)
            elif args.action == 'train':
                if not args.config or args.device != 'cuda':
                    parser.error('Production train requires --config and CUDA; CPU smoke uses isolated tests')
                case = C.authorize_train(ROOT, server, args.config)
                from g20.training import train_run
                from g20.common import locked
                with locked(ROOT / 'work_dir' / case.run_id / 'meta/.training.lock'):
                    return train_run(args.config, args.device, args.resume, deadline, ROOT)
            elif args.action == 'calibrate':
                if not args.run:
                    parser.error('--run required')
                from g20.calibration import calibrate
                result = dict(manifest=str(calibrate(args.run, ROOT, server, args.device, deadline)))
            elif args.action in ('adopt', 'export', 'import'):
                if not args.reference:
                    parser.error('--reference required')
                from g20.references import adopt_reference, export_reference, import_reference
                if args.action == 'adopt':
                    if not args.manifest:
                        parser.error('--manifest original reference manifest required')
                    result = dict(manifest=str(adopt_reference(args.manifest, args.reference, server, ROOT, args.device, deadline=deadline)))
                elif args.action == 'export':
                    result = dict(archive=str(export_reference(args.reference, server, ROOT, args.device, deadline=deadline)))
                else:
                    if not args.archive:
                        parser.error('--archive required')
                    result = dict(manifest=str(import_reference(args.archive, args.reference, server, ROOT, args.device, deadline=deadline)))
            elif args.action == 'parity':
                from g20.parity import verify_r0_parity
                result = verify_r0_parity(ROOT, server, args.device, deadline, args.evidence)
            elif args.action == 'transfer':
                if not args.global_receipt:
                    parser.error('--global-receipt s1-issued immutable transfer_selection.json required')
                result = C.register_transfer(ROOT, server, args.global_receipt)
            elif args.action == 'transfer-select':
                if server != 's1' or not args.reference_screen or not args.student_screen or not args.reference:
                    parser.error('s1 only: --reference-screen, --student-screen, and --reference required')
                result = C.publish_transfer_selection(ROOT, args.reference_screen, args.student_screen, args.reference,
                    before_confirmation=args.before_confirmation)
            else:
                if not args.run:
                    parser.error('--run required')
                from g20.postrun import process
                return process(args.run, root=ROOT, device=args.device, deadline=deadline,
                    upload=args.action == 'upload', upload_only=args.action == 'upload')
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, TimeoutError) as error:
        print(f'G20: {type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(2)
