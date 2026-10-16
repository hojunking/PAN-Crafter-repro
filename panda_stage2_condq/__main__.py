"""Implementation validation only; no automatic donor selection or campaign."""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    smoke = commands.add_parser('smoke', help='Offline CPU unit/pipeline tests; no real training/data/cloud')
    smoke.add_argument('--report-dir', help='Optional local validation evidence directory')
    for name in ('inspect-donor', 'calibrate', 'diagnostic'):
        sub = commands.add_parser(name)
        sub.add_argument('--checkpoint', required=True, help='Explicit trusted local Stage 1 donor path')
        sub.add_argument('--checkpoint-sha256', required=True)
        sub.add_argument('--sensor', required=True, choices=('WV3', 'GF2', 'QB'))
        sub.add_argument('--sources', default=str(ROOT/'ablr2/sensor_sources.json'))
        sub.add_argument('--lms-overrides', help='Required explicit original train/val LMS JSON for QB')
        sub.add_argument('--device', default='cpu')
        sub.add_argument('--cpu-threads', type=int, default=2)
        if name == 'inspect-donor':
            sub.add_argument('--without-teacher', action='store_true')
        if name in ('calibrate', 'diagnostic'):
            sub.add_argument('--output', required=True, help='New artifact path / new diagnostic directory')
            sub.add_argument('--memory-gib', type=float, required=True, help='Explicit exact-table memory budget')
        if name == 'calibrate':
            sub.add_argument('--kind', choices=('e', 'condq'), required=True)
            sub.add_argument('--k', type=int, required=True, help='Requested bins; 5 is provisional, not implicit')
            sub.add_argument('--batch-size', type=int, default=12)
        if name == 'diagnostic':
            sub.add_argument('--variant', choices=('gt_only', 'panda_e', 'panda_condq'), required=True)
            for field in ('alpha', 'beta', 'gamma', 'delta'):
                sub.add_argument('--' + field, type=float, required=True)
            for field in ('seed', 'data-seed', 'augmentation-seed'):
                sub.add_argument('--' + field, type=int, required=True)
            sub.add_argument('--calibration')
            sub.add_argument('--calibration-file-sha256')
            sub.add_argument('--updates', type=int, choices=(1, 2), default=2)
            sub.add_argument('--microbatch', type=int, default=12)
            sub.add_argument('--effective-batch', type=int, default=48)
            sub.add_argument('--hqnr-target', type=float,
                             help='WV3 defaults to 0.959; other sensors require an explicit target')
    args = parser.parse_args(argv)
    if args.command == 'smoke':
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import unittest
        tests = unittest.defaultTestLoader.discover(str(Path(__file__).parent/'tests'),
                                                    top_level_dir=str(ROOT))
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        if args.report_dir:
            import time
            from directshift48.reporting import atomic_json, utc_now
            from .runtime import code_pin
            report = Path(args.report_dir)/('unit_tests_' + str(time.time_ns()) + '.json')
            atomic_json(report, {'at': utc_now(), 'tests_run': result.testsRun,
                'successful': result.wasSuccessful(), 'failures': [(t.id(), message) for t, message in result.failures],
                'errors': [(t.id(), message) for t, message in result.errors],
                'skipped': [(t.id(), reason) for t, reason in result.skipped],
                'code_pin': code_pin(), 'scope': 'OFFLINE_CPU_SYNTHETIC_NO_CLOUD_NO_OFFICIAL_METRICS'})
            print('Validation receipt: ' + str(report))
        return 0 if result.wasSuccessful() else 1
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    if args.command in ('calibrate', 'diagnostic'):
        import math
        if not math.isfinite(args.memory_gib) or args.memory_gib <= 0:
            parser.error('--memory-gib must be finite and positive')
        if Path(args.output).exists():
            parser.error('Output already exists; never overwrite calibration/diagnostic artifacts')
        budget = int(args.memory_gib * 1024**3)
    if str(args.device).startswith('cuda'):
        # Implementation validation must never share/stop an occupied GPU.
        from directshift48.__main__ import require_idle_gpu
        require_idle_gpu(args.device)
    import torch
    from .training import TrainingSpec, configure_precision
    configure_precision()
    torch.set_num_threads(args.cpu_threads)
    from .sources import RegisteredRRTrain
    from .donor import load_donor
    overrides = json.loads(Path(args.lms_overrides).read_text()) if args.lms_overrides else None
    registered = RegisteredRRTrain.from_catalog(args.sources, root=ROOT, sensor=args.sensor,
                                                lms_overrides=overrides)
    include_teacher = not (args.command == 'inspect-donor' and args.without_teacher or
                           args.command == 'diagnostic' and args.variant == 'gt_only')
    donor = load_donor(args.checkpoint, args.checkpoint_sha256, registered,
                       include_teacher=include_teacher, device=args.device)
    if args.command == 'inspect-donor':
        print(json.dumps({'binding': donor.binding, 'checkpoint': donor.checkpoint_metadata,
                          'sample_count': len(registered.sample_ids), 'donor_selected': False,
                          'training_started': False}, indent=2))
        return 0
    from . import runtime
    if args.command == 'calibrate':
        def progress(value):
            if value['seen'] % (args.batch_size * 50) == 0 or value['seen'] == value['expected']:
                print(json.dumps({'calibration_progress': value}), flush=True)
        report = runtime.calibrate(donor, registered, output=args.output, k=args.k,
            batch_size=args.batch_size, memory_budget_bytes=budget,
            include_condq=args.kind == 'condq', device=args.device, on_progress=progress)
    else:
        from .calibration import CalibrationStats
        from .spec import Stage2Spec
        from .stream import StreamSpec
        spec = Stage2Spec(args.sensor, args.variant, args.seed,
                          args.alpha, args.beta, args.gamma, args.delta)
        stats = None
        if args.variant == 'gt_only':
            if args.calibration or args.calibration_file_sha256:
                parser.error('gt_only must not load calibration')
        else:
            if not args.calibration or not args.calibration_file_sha256:
                parser.error('Explicit new calibration file AND its file SHA256 required')
            stats = CalibrationStats.load(args.calibration, expected_binding=donor.binding,
                expected_file_sha256=args.calibration_file_sha256, memory_budget_bytes=budget)
        training = TrainingSpec(microbatch_size=args.microbatch, effective_batch_size=args.effective_batch)
        stream = StreamSpec(args.data_seed, args.augmentation_seed)
        report = runtime.diagnostic(donor, registered, spec, training, stream, output=args.output,
            stats=stats, device=args.device, updates=args.updates, calibration_device_budget_bytes=budget,
            hqnr_target=args.hqnr_target)
    if args.command == 'diagnostic':
        report = {key: report[key] for key in ('status', 'official_experiment', 'completed_updates',
            'elapsed_seconds', 'checkpoint', 'calibration_table_bytes', 'cuda_peak_bytes')}
        report['manifest_path'] = str((Path(args.output)/'manifest.json').resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
