#!/usr/bin/env python3
"""G23 SENS: preview/start/status/stop and private preflighted phase workers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preview','start','status','stop','preflight','train','postrun','upload','report'))
    parser.add_argument('--server',choices=('s4','s5'),required=True)
    parser.add_argument('--cycle',type=int,default=0)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--foreground',action='store_true')
    parser.add_argument('--in-place',action='store_true')
    parser.add_argument('--no-upload',action='store_true')
    parser.add_argument('--safe-now',action='store_true')
    args=parser.parse_args(argv)
    from g23sens.common import camp,read,atomic_json,utcnow,RuntimePaused,locked
    from g23sens.plan import cycle_cases
    folder=camp(ROOT,args.server)
    if args.command=='preview':
        print(json.dumps(cycle_cases(args.server,args.cycle),ensure_ascii=False,indent=2));return 0
    if args.command=='status':
        print(json.dumps(read(folder/'state.json'),ensure_ascii=False,indent=2));return 0
    if args.command=='stop':
        atomic_json(folder/('STOP_NOW_SAFE' if args.safe_now else 'STOP_AFTER_RUN'),
                    dict(at_utc=utcnow(),operator_request=True));return 0
    if args.command=='preflight':
        from g23sens.preflight import run_preflight
        print(json.dumps(run_preflight(ROOT,args.server),ensure_ascii=False,default=str),flush=True);return 0
    if args.command=='train':
        if args.config is None:parser.error('--config required')
        from g23sens.training import train_run
        with locked(args.config.parent/'training.lock'):
            return train_run(args.config,root=ROOT,resume=args.resume)
    if args.command=='postrun':
        if args.config is None:parser.error('--config required')
        from g23sens.postrun import run_postrun
        with locked(args.config.parent/'postrun.lock'):
            run_postrun(args.config,root=ROOT,stopcheck=lambda:(folder/'STOP_NOW_SAFE').exists())
        return 0
    if args.command=='upload':
        from g23sens.upload import flush_outbox
        print(json.dumps(flush_outbox(ROOT,args.server,activated=True),ensure_ascii=False,default=str));return 0
    if args.command=='report':
        from g23sens.reporting import build_report
        print(json.dumps(build_report(ROOT,args.server),ensure_ascii=False,default=str));return 0
    if not args.in_place:
        from g23sens.deployment import frozen_checkout
        frozen=frozen_checkout(ROOT,args.server)
        command=[sys.executable,str(frozen/'tools/g23sens_runner.py'),'start','--server',args.server,'--in-place','--foreground']
        if args.no_upload:command.append('--no-upload')
        if args.foreground:return subprocess.call(command,cwd=frozen)
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'runner.log').open('a') as stream:
            child=subprocess.Popen(command,cwd=frozen,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(status='START_REQUESTED',pid=child.pid,log=str(folder/'runner.log'),
            note='Current-run handoff, actual CUDA smoke and asset checks precede training.')));return 0
    # A stale stop is never silently cleared by start; operator can inspect/remove the exact marker.
    if (folder/'STOP_NOW_SAFE').exists() or (folder/'STOP_AFTER_RUN').exists():
        raise RuntimePaused('Stop marker retained; remove the intended local marker explicitly to resume')
    from g23sens.common import selected_gpu_uuid
    gpu_uuid=selected_gpu_uuid()
    os.environ['PANCRAFTER_G23SENS_GPU_UUID']=gpu_uuid
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_uuid
    with locked(folder/'service.lock'), locked(ROOT/'work_dir/g23sens'/('gpu-'+gpu_uuid+'.lock')):
        from g23sens.handoff import wait_boundary
        wait_boundary(ROOT,args.server,gpu_uuid,activated=True)
        # Exit the smoke worker before training: do not retain its CUDA context in the supervisor.
        code=subprocess.call([sys.executable,str(ROOT/'tools/g23sens_runner.py'),
            'preflight','--server',args.server],cwd=ROOT)
        if code:return code
        from g23sens.controller import run
        return run(ROOT,args.server,activated=True,no_upload=args.no_upload)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (KeyboardInterrupt,InterruptedError) as exc:
        print(str(exc),file=sys.stderr);raise SystemExit(75)
    except Exception as exc:
        from g23sens.common import RuntimePaused
        import traceback
        traceback.print_exc()
        if isinstance(exc,RuntimePaused):code=75
        elif isinstance(exc,FloatingPointError):code=3
        elif isinstance(exc,OSError):code=74
        elif 'out of memory' in str(exc).lower():code=74
        else:code=65
        raise SystemExit(code)
