#!/usr/bin/env python3
"""Explicit GFB20 actions. Build/inspect do not create a clock or launch work."""
import argparse
import errno
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('build','inspect-assets','clock','start','run','status','stop',
        'preflight','mixed','train','postrun','report','upload','retry-uploads'))
    p.add_argument('--server',choices=('s3','s4','s5'))
    p.add_argument('--t0',help='Same timezone-aware actual start timestamp on all three servers')
    p.add_argument('--clock-file',help='Copy of the single shared GFB20 campaign_window.json')
    p.add_argument('--bindings',help='Explicit JSON parent asset paths when exact local discovery is insufficient')
    p.add_argument('--foreground',action='store_true')
    p.add_argument('--in-place',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--no-upload',action='store_true')
    p.add_argument('--config')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--deadline')
    p.add_argument('--run')
    p.add_argument('--command',choices=('STOP_NOW_SAFE','STOP_AFTER_BLOCK','CONTINUE'))
    args=p.parse_args(argv)
    from gfb20 import controller as C
    from gfb20.common import apply_runtime_policy,read,read_json,camp
    result={};code=0
    if args.action=='build':result=C.build(ROOT)
    elif args.action=='clock':
        if not args.t0:p.error('clock requires explicit shared --t0')
        result=C.write_window(ROOT,args.t0)
    else:
        if not args.server:p.error('--server s3|s4|s5 required')
        if args.action=='inspect-assets':
            from gfb20.assets import discover_bindings
            result=dict(discovered=discover_bindings(args.server,ROOT),bytes_verified=False,clock_created=False)
        elif args.action=='status':result=C.status(ROOT,args.server)
        elif args.action=='stop':
            if not args.command:p.error('stop requires --command')
            result=C.control(ROOT,args.server,args.command)
        elif args.action=='report':
            from gfb20.reporting import rebuild
            result=rebuild(ROOT,args.server)
        elif args.action=='start' and not args.in_place:
            # Freeze before actual setup: merely copying/reading the bundle is not t0.
            from gfb20.deployment import frozen_checkout
            frozen=frozen_checkout(ROOT,args.server)
            forwarded=['start','--in-place','--server',args.server]
            for flag,value in (('--t0',args.t0),('--clock-file',args.clock_file),('--bindings',args.bindings)):
                if value:
                    forwarded += [flag,str(Path(value).resolve()) if flag!='--t0' else value]
            if args.foreground:forwarded+=['--foreground']
            if args.no_upload:forwarded+=['--no-upload']
            return subprocess.call([sys.executable,str(frozen/'tools/gfb20_runner.py'),*forwarded],cwd=frozen)
        elif args.action=='start':
            if args.t0:C.write_window(ROOT,args.t0)
            if args.clock_file:C.shared_window(ROOT,args.clock_file)
            result=C.start(ROOT,args.server,args.foreground,args.bindings,not args.no_upload)
            code=result.get('exit_code',0)
        else:
            if not args.in_place and args.action in ('upload','retry-uploads'):
                release=read(camp(ROOT,args.server)/'runtime_release.json')
                if release and Path(release['path']).resolve()!=ROOT:
                    forwarded=list(argv if argv is not None else sys.argv[1:])+['--in-place']
                    return subprocess.call([sys.executable,str(Path(release['path'])/'tools/gfb20_runner.py'),*forwarded],cwd=release['path'])
            apply_runtime_policy(ROOT);C.verify_registration(ROOT,args.server)
            window=C.shared_window(ROOT)
            final=window.deadline_utc.isoformat()
            prep=window.train_finish_utc.isoformat()
            if args.deadline not in (None,final,prep):p.error('Deadline must match registered shared clock')
            if args.action=='run':code=C.run(ROOT,args.server,args.bindings,not args.no_upload);result=dict(exit_code=code)
            elif args.action=='retry-uploads':result=C.retry_uploads(ROOT,args.server)
            elif args.action=='upload':
                if not args.run:p.error('upload requires --run')
                from gfb20.plan import case_for
                if case_for(args.run).server!=args.server:p.error('Wrong server run')
                from gfb20.upload import upload_run
                result=upload_run(args.run,ROOT,activated=True)
            else:
                def cooperative_stop(*_):raise TimeoutError('Own preparation/evaluation received safe-stop signal')
                if args.action in ('preflight','mixed','postrun'):
                    signal.signal(signal.SIGTERM,cooperative_stop);signal.signal(signal.SIGINT,cooperative_stop)
                if args.action=='preflight':
                    from gfb20.preflight import execute
                    result=execute(ROOT,args.server,prep,args.bindings)
                elif args.action=='mixed':result=C.prepare_mixed(ROOT,args.server,prep)
                else:
                    if not args.config:p.error('train/postrun requires --config')
                    from gfb20.common import read_config
                    from gfb20.plan import validate_config
                    if validate_config(read_config(args.config)).server!=args.server:p.error('Wrong server config')
                    if args.action=='train':
                        from gfb20.training import train_run
                        code=train_run(args.config,device='cuda',resume=args.resume,deadline_arg=final,root=ROOT)
                        result=dict(exit_code=code)
                    elif args.action=='postrun':
                        from gfb20.postrun import process
                        result=process(args.config,root=ROOT,device='cuda',deadline=final)
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    return code


if __name__=='__main__':
    try:raise SystemExit(main())
    except TimeoutError as exc:
        print(str(exc),file=sys.stderr);raise SystemExit(75)
    except OSError as exc:
        transient={errno.EAGAIN,errno.EINTR,errno.ETIMEDOUT,errno.ECONNRESET,errno.ECONNREFUSED,errno.ENETUNREACH}
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(74 if exc.errno in transient else 1)
