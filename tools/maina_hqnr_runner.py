#!/usr/bin/env python3
"""Independent s4/s5 MAIN-A service and explicit recovery commands."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preview', 'start', 'status', 'stop', 'preflight',
        'assets', 'train', 'postrun', 'upload', 'report', 'retry'))
    parser.add_argument('--server', choices=('s4', 's5'), required=True)
    parser.add_argument('--cycle', type=int, default=0)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--asset-map', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--in-place', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--host-cutover-done', action='store_true')
    parser.add_argument('--no-upload', action='store_true')
    parser.add_argument('--safe-now', action='store_true')
    parser.add_argument('--phase', choices=('train', 'postrun'))
    args = parser.parse_args(argv)
    from maina_hqnr.common import camp, read, read_json, atomic_json, utcnow, RuntimePaused, locked, selected_gpu_uuid
    from maina_hqnr.plan import cycle_cases
    folder = camp(ROOT, args.server)
    if args.command == 'preview':
        print(json.dumps(cycle_cases(args.server, args.cycle), ensure_ascii=False, indent=2)); return 0
    if args.command == 'status':
        state = read(folder / 'state.json')
        if state.get('active_run_id'):
            from maina_hqnr.common import run_dir
            item = state['runs'][state['active_run_id']]
            training = read(run_dir(state['active_run_id'], ROOT, item['attempt']) / 'meta/training_status.json')
            state = dict(state, training=training)
            if state.get('status') == 'ADMITTED' and training.get('actual_updates', 0) > 0 and training.get('status') == 'RUNNING':
                state['effective_status'] = 'RUNNING'
        print(json.dumps(state, ensure_ascii=False, indent=2)); return 0
    if args.command == 'stop':
        atomic_json(folder / 'STOP_AFTER_RUN', dict(operator_request=True, at_utc=utcnow()))
        if args.safe_now:
            atomic_json(folder / 'STOP_NOW_SAFE', dict(operator_request=True, at_utc=utcnow()))
        return 0
    if args.command == 'retry':
        if not args.phase:
            parser.error('retry requires --phase train|postrun')
        from maina_hqnr.controller import retry
        print(json.dumps(retry(ROOT, args.server, args.phase), ensure_ascii=False)); return 0
    if args.command == 'assets':
        from maina_hqnr.assets import verify_assets
        # Inspection is read-only: only the final frozen runtime may own bindings.
        result = verify_assets(args.server, root=ROOT, overrides=read_json(args.asset_map) if args.asset_map else None, persist=False)
        print(json.dumps(result, ensure_ascii=False, default=str)); return 0
    if args.command == 'preflight':
        from maina_hqnr.preflight import run_preflight
        from maina_hqnr.handoff import ensure_gpu_idle
        release = read(folder / 'runtime_release.json')
        if not release or Path(release['path']).resolve() != ROOT.resolve():
            raise PermissionError('Production preflight must run in the committed campaign runtime; use the launcher')
        ensure_gpu_idle(selected_gpu_uuid())
        result = run_preflight(ROOT, args.server, overrides=read_json(args.asset_map) if args.asset_map else None)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True); return 0
    if args.command in ('train', 'postrun'):
        if args.config is None:
            parser.error('--config required')
        from maina_hqnr.common import read_config
        cfg = read_config(args.config)
        if cfg.get('maina_hqnr', {}).get('case', {}).get('server') != args.server:
            raise PermissionError('--server and the configured server must match')
        with locked(args.config.parent / (args.command + '.lock')):
            if args.command == 'train':
                from maina_hqnr.training import train_run
                return train_run(args.config, root=ROOT, resume=args.resume)
            from maina_hqnr.postrun import run_postrun
            run_postrun(args.config, root=ROOT, stopcheck=lambda: (folder / 'STOP_NOW_SAFE').exists())
            return 0
    if args.command == 'upload':
        from maina_hqnr.upload import flush_outbox
        print(json.dumps(flush_outbox(ROOT, args.server, activated=True), ensure_ascii=False, default=str)); return 0
    if args.command == 'report':
        from maina_hqnr.reporting import build_report
        print(json.dumps(build_report(ROOT, args.server), ensure_ascii=False, default=str)); return 0
    if (folder / 'STOP_NOW_SAFE').exists() or (folder / 'STOP_AFTER_RUN').exists():
        raise RuntimePaused('Operator stop markers retained. Inspect and explicitly clear only the intended local markers to resume.')
    gpu_uuid = selected_gpu_uuid()
    os.environ['PANCRAFTER_MAINA_GPU_UUID'] = gpu_uuid
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    if not args.in_place:
        from maina_hqnr.deployment import frozen_checkout
        from maina_hqnr.handoff import wait_boundary
        frozen = frozen_checkout(ROOT, args.server)
        wait_boundary(ROOT, args.server, gpu_uuid, activated=True)
        command = [sys.executable, str(frozen / 'tools/maina_hqnr_runner.py'), 'start',
                   '--server', args.server, '--in-place', '--foreground', '--host-cutover-done']
        if args.asset_map:
            command += ['--asset-map', str(args.asset_map.resolve())]
        if args.no_upload:
            command.append('--no-upload')
        if args.foreground:
            return subprocess.call(command, cwd=frozen)
        with (folder / 'runner.log').open('a') as stream:
            child = subprocess.Popen(command, cwd=frozen, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(status='START_REQUESTED', pid=child.pid, log=str(folder / 'runner.log')))); return 0
    if not args.host_cutover_done:
        raise RuntimePaused('Host-side verified safe cutover required; use the documented launcher')
    from maina_hqnr.handoff import verify_local_receipt
    with locked(folder / 'service.lock'), locked(ROOT / 'work_dir/maina_hqnr' / ('gpu-' + gpu_uuid + '.lock')):
        verify_local_receipt(ROOT, args.server, gpu_uuid)
        command = [sys.executable, str(ROOT / 'tools/maina_hqnr_runner.py'), 'preflight', '--server', args.server]
        if args.asset_map:
            command += ['--asset-map', str(args.asset_map.resolve())]
        code = subprocess.call(command, cwd=ROOT)
        if code:
            return code
        from maina_hqnr.controller import run
        return run(ROOT, args.server, activated=True, no_upload=args.no_upload)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, InterruptedError):
        raise SystemExit(75)
    except Exception as error:
        from maina_hqnr.common import RuntimePaused
        import traceback
        traceback.print_exc()
        code = (75 if isinstance(error, RuntimePaused) else 3 if isinstance(error, FloatingPointError)
                else 74 if isinstance(error, OSError) or 'out of memory' in str(error).lower() else 65)
        raise SystemExit(code)
