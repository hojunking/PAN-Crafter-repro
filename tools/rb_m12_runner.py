#!/usr/bin/env python3
"""M12 explicit finite run; read-only report/package cannot launch inference."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panda_rb_m12.plan import ROOT, PLAN_DIR, binding_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate-plan', 'status', 'preflight', 'run',
                                           'retry-diagnostics', 'report', 'package', 'import-package', 'sheet-setup', 'sheet-sync'])
    parser.add_argument('--server', choices=['s1', 's3', 's5'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--retry-technical', action='store_true')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    allowed_plans = {(args.root / PLAN_DIR).resolve(),
                     (args.root / PLAN_DIR / 'planning/experiment_registry.json').resolve()}
    if args.plan and args.plan.resolve() not in allowed_plans:
        parser.error('Only the registered M12 plan bundle is executable')
    binding = args.binding or binding_path(args.root)
    if args.command == 'validate-plan':
        from panda_rb_m12.plan import validate_plan
        result = validate_plan(args.root)
    elif args.command in ('status', 'run', 'preflight', 'retry-diagnostics'):
        if not args.server:
            parser.error('--server required')
        if args.command == 'preflight':
            from panda_rb_m12.preflight import prepare
            result = prepare(args.server, binding, args.root, args.device)
        else:
            from panda_rb_m12.controller import status, run, retry_diagnostics
            if args.command == 'status':
                result = status(args.server, args.root)
            elif args.command == 'run':
                result = run(args.server, args.root, binding, args.device, args.retry_technical, args.activate)
            else:
                result = retry_diagnostics(args.server, args.root, binding, args.device, args.activate)
    elif args.command == 'import-package':
        if args.server != 's1' or not args.input or not args.sha256:
            parser.error('--server s1 --input ZIP --sha256 sender_receipt_sha are required')
        from panda_rb_m12.transfer import import_package
        result = import_package(args.input, args.sha256, args.root, activate=args.activate)
    elif args.command in ('sheet-setup', 'sheet-sync'):
        if args.server != 's1':
            parser.error('s1 is the sole explicitly activated Sheet writer')
        from reporting_bridge.rb_m12_sheets import Uploader, GoogleSheetsAdapter
        from panda_rb_m12.plan import campaign_dir
        writer = Uploader(GoogleSheetsAdapter(args.root), campaign_dir(args.root) / 'reporting/sheets', writer_server='s1')
        if args.command == 'sheet-setup':
            result = writer.setup(activate=args.activate)
        else:
            from panda_rb_m12.reporting import summarize
            report = summarize(args.root, verify=True)
            result = writer.sync(report['rows'], report['point_records'], report['status_records'], activate=args.activate)
    else:
        from panda_rb_m12.reporting import summarize, package
        servers = [args.server] if args.server else ['s1', 's3', 's5']
        if args.command == 'package':
            if not args.output:
                parser.error('--output ZIP path required')
            result = package(args.root, servers, args.output)
        else:
            result = summarize(args.root, servers, verify=True)
            if args.output:
                from panda_rb_m12.reporting import export_tables
                result = dict(result, exported=export_tables(result, args.output))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
