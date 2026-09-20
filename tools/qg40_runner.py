#!/usr/bin/env python
"""Explicit QG40 entrypoints. Import/check/build never starts learning or a clock."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'bundle', 'inventory', 'binding-template', 'window',
        'check', 'status', 'start', 'run', 'ensure', 'preflight', 'train', 'calibrate', 'export',
        'import', 'postrun', 'upload', 'branch-evidence'))
    parser.add_argument('--server', choices=('s1', 's2', 's3', 's4', 's5'))
    parser.add_argument('--sensor', choices=('QB', 'GF2'))
    parser.add_argument('--t0')
    parser.add_argument('--window')
    parser.add_argument('--spec')
    parser.add_argument('--config')
    parser.add_argument('--run')
    parser.add_argument('--reference')
    parser.add_argument('--archive')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from qg40 import controller as C
    if args.action == 'build':
        result = C.build_artifacts(ROOT)
    elif args.action == 'bundle':
        from qg40.deployment import bundle
        result = bundle(ROOT)
    elif args.action in ('inventory', 'binding-template'):
        if not args.sensor:
            parser.error('--sensor is required')
        from qg40.data import inventory_sources
        result = inventory_sources(ROOT, args.sensor)
        if args.action == 'binding-template':
            from qg40.plan import sensor_spec
            from qg40.common import sha256
            template = sensor_spec(args.sensor).to_dict()
            template['splits'] = {key: dict(path=value['resolved_path'],
                sha256=sha256(value['resolved_path']) if value['exists'] else None,
                source_identity=None, lp_path=None, lp_sha256=None)
                for key, value in result['splits'].items()}
            from qg40.data import PROVENANCE_FIELDS
            template['source_provenance'] = {key: None for key in PROVENANCE_FIELDS}
            if args.sensor == 'QB':
                for split, name in (('train', 'train_qb.h5'), ('val', 'valid_qb.h5')):
                    raw = (ROOT / 'data/PanCollection/QB' / name).resolve()
                    template['source_provenance'][f'raw_{split}_path'] = str(raw)
                    template['source_provenance'][f'raw_{split}_sha256'] = sha256(raw) if raw.is_file() else None
                template['source_provenance']['msfix_recipe'] = None
            result = template  # stdout only; no manufactured provenance or activation.
    elif args.action == 'window':
        if not args.t0:
            parser.error('--t0 is required; all servers share the same preparation start')
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
            window = C.shared_window(ROOT)
            deadline = window.deadline_utc.isoformat()
            if args.action == 'preflight':
                from qg40.preflight import execute
                result = execute(ROOT, server, args.spec or C.camp(ROOT, server) / 'sensor_spec.json', deadline, args.device)
            elif args.action == 'train':
                if not args.config:
                    parser.error('--config required')
                if args.device != 'cuda':
                    raise ValueError('Production training CLI requires registered CUDA; use isolated CPU unit tests for smoke')
                case = C.authorize_train(ROOT, server, args.config)
                from qg40.training import train_run
                from qg40.common import locked
                with locked(ROOT / 'work_dir' / case.run_id / 'meta/.training.lock'):
                    return train_run(args.config, device=args.device, resume=args.resume, deadline_arg=deadline, root=ROOT)
            elif args.action == 'calibrate':
                if not args.run:
                    parser.error('--run required')
                from qg40.calibration import calibrate
                result = dict(reference_manifest=str(calibrate(args.run, ROOT, server, args.device, deadline_utc=deadline)))
            elif args.action in ('export', 'import'):
                if not args.reference:
                    parser.error('--reference required')
                from qg40.references import export_reference, import_reference
                if args.action == 'export':
                    result = dict(archive=str(export_reference(args.reference, server, ROOT, device=args.device, deadline=deadline)))
                else:
                    if not args.archive:
                        parser.error('--archive required')
                    result = dict(manifest=str(import_reference(args.archive, args.reference, server, ROOT, args.device, deadline=deadline)))
            elif args.action in ('postrun', 'upload'):
                if not args.run:
                    parser.error('--run required')
                from qg40.postrun import process
                return process(args.run, root=ROOT, device=args.device, deadline=deadline,
                               upload=args.action == 'upload', upload_only=args.action == 'upload')
            elif args.action == 'branch-evidence':
                from qg40.branch_evidence import student_evidence, c3_evidence
                result = (student_evidence(ROOT, server, args.device, deadline) if server in ('s2', 's5')
                          else c3_evidence(ROOT, server))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        print(f'QG40: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(2)
