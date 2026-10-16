"""Offline implementation verification only; never starts/replaces training."""
import argparse
import json
import os
from pathlib import Path
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    smoke = commands.add_parser('smoke', help='Synthetic CPU unit/integration tests')
    smoke.add_argument('--report-dir')
    profile = commands.add_parser('profile', help='Untrained CPU cost/latency; no GPU or real data')
    profile.add_argument('--bands', type=int, choices=(4, 8), default=8)
    profile.add_argument('--sides', nargs='+', type=int, default=[64, 256, 512])
    profile.add_argument('--repeats', type=int, default=10)
    profile.add_argument('--warmup', type=int, default=3)
    profile.add_argument('--threads', type=int, default=2)
    profile.add_argument('--output', help='Optional NEW local JSON receipt')
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from .runtime import code_pin
    before = code_pin()
    if args.command == 'smoke':
        import unittest
        tests = unittest.defaultTestLoader.discover(str(Path(__file__).parent / 'tests'),
                                                    top_level_dir=str(Path(__file__).resolve().parents[1]))
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        success = result.wasSuccessful() and before == code_pin()
        if args.report_dir:
            from directshift48.reporting import atomic_json, utc_now
            destination = Path(args.report_dir) / ('unit_tests_' + str(time.time_ns()) + '.json')
            atomic_json(destination, {'at': utc_now(), 'successful': success, 'tests_run': result.testsRun,
                'failures': [(t.id(), error) for t, error in result.failures],
                'errors': [(t.id(), error) for t, error in result.errors],
                'skipped': [(t.id(), reason) for t, reason in result.skipped],
                'code_pin': before, 'code_unchanged': before == code_pin(),
                'scope': 'OFFLINE_CPU_SYNTHETIC_NO_CLOUD_NO_FORMAL_EXPERIMENT'})
            print('Validation receipt:', destination)
        return 0 if success else 1
    if args.output and Path(args.output).exists():
        parser.error('Profile output exists; choose a new receipt path')
    from .profile import compare
    report = compare(bands=args.bands, sides=args.sides, repeats=args.repeats,
                     warmup=args.warmup, threads=args.threads)
    if before != code_pin():
        raise RuntimeError('Implementation changed during profiling')
    report['code_pin'] = before
    if args.output:
        from directshift48.reporting import atomic_json
        atomic_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'code_pin')}, indent=2))
    for row in report['rows']:
        print(json.dumps({k: v for k, v in row.items() if k not in ('module_costs', 'manifest', 'latency_ms_samples')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
