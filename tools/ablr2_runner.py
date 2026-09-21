#!/usr/bin/env python3
"""ABLR2 s1/WV3 and s2/QB: explicit renewable leases, no implicit training."""
import argparse
import datetime as dt
import errno
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','lease','start','run','status','stop','verify',
                                        'preflight','train','calibrate','postrun','upload','retry-uploads'])
    parser.add_argument('--server',choices=['s1','s2'])
    parser.add_argument('--hours',type=float,default=72.)
    parser.add_argument('--lease-hours',type=float,help='Explicitly grant/renew lease with start, maximum72h')
    parser.add_argument('--foreground',action='store_true')
    parser.add_argument('--in-place',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--command',choices=['STOP_AFTER_RUN','STOP_AFTER_SWEEP','PAUSE_AFTER_BLOCK','STOP_NOW_SAFE','CONTINUE'])
    parser.add_argument('--config')
    parser.add_argument('--run')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--upload',action='store_true')
    parser.add_argument('--no-upload',action='store_true')
    parser.add_argument('--deadline')
    parser.add_argument('--dataset-manifest')
    args=parser.parse_args(argv)
    from ablr2 import controller as C
    from ablr2.common import apply_runtime_policy
    if args.action in ('verify','retry-uploads','upload') and not args.in_place:
        from ablr2.common import camp,read
        release=read(camp(ROOT,args.server) / 'runtime_release.json') if args.server else {}
        if release and Path(release['path']).resolve()!=ROOT.resolve():
            forwarded=list(argv if argv is not None else sys.argv[1:])+['--in-place']
            return subprocess.call([sys.executable,str(Path(release['path'])/'tools/ablr2_runner.py'),*forwarded],cwd=release['path'])
    result={};code=0
    if args.action=='build': result=C.build(ROOT,args.server)
    else:
        if args.server is None: parser.error('--server s1|s2 is required')
        if args.action=='status': result=C.status(ROOT,args.server)
        elif args.action=='lease': result=C.grant_lease(ROOT,args.server,args.hours)
        elif args.action=='stop':
            if not args.command: parser.error('stop requires --command')
            result=C.control(ROOT,args.server,args.command)
        elif args.action=='verify': result=C.request_verify(ROOT,args.server)
        elif args.action=='start':
            if not args.in_place:
                from ablr2.deployment import frozen_checkout
                frozen=frozen_checkout(ROOT,args.server)
                forward=['start','--server',args.server,'--in-place']
                if args.foreground: forward+=['--foreground']
                if args.no_upload: forward+=['--no-upload']
                if args.lease_hours is not None: forward+=['--lease-hours',str(args.lease_hours)]
                return subprocess.call([sys.executable,str(frozen/'tools/ablr2_runner.py'),*forward],cwd=frozen)
            if args.lease_hours is not None: C.grant_lease(ROOT,args.server,args.lease_hours)
            result=C.start(ROOT,args.server,args.foreground,not args.no_upload)
            code=result.get('exit_code',0)
        else:
            apply_runtime_policy(ROOT)
            C.verify_registration(ROOT,args.server)
            if args.action=='run':
                code=C.run(ROOT,args.server,not args.no_upload);result=dict(exit_code=code)
            elif args.action=='retry-uploads':
                result=C.retry_uploads(ROOT,args.server)
            else:
                lease=C._lease(ROOT,args.server)
                deadline=lease['expires_utc']
                if args.deadline:
                    provided=dt.datetime.fromisoformat(args.deadline.replace('Z','+00:00'))
                    if provided.tzinfo is None: parser.error('--deadline requires an explicit timezone')
                    deadline=min(provided,dt.datetime.fromisoformat(deadline)).isoformat()
                if args.action=='preflight':
                    from ablr2.preflight import run
                    result=run(ROOT,args.server,deadline_utc=deadline,manifest_path=args.dataset_manifest)
                elif args.action=='train':
                    if not args.config: parser.error('train requires --config')
                    C.authorize_train(ROOT,args.server,args.config)
                    from ablr2.training import train_run
                    code=int(train_run(args.config,device='cuda',resume=args.resume,deadline_arg=deadline,root=ROOT))
                    result=dict(exit_code=code)
                else:
                    if not args.run: parser.error('action requires --run')
                    from ablr2.plan import case_for
                    if case_for(args.run,ROOT).server_id!=args.server: parser.error('Run belongs to another lane')
                    if args.action=='calibrate':
                        from ablr2.calibration import calibrate
                        result=dict(reference_manifest=str(calibrate(args.run,root=ROOT,server=args.server,
                                                                      device='cuda',deadline=deadline)))
                    else:
                        from ablr2.postrun import process
                        code=process(args.run,root=ROOT,deadline=deadline,upload=args.upload or args.action=='upload',
                                     upload_only=args.action=='upload')
                        result=dict(exit_code=code)
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    return code


if __name__=='__main__':
    from ablr2.common import RuntimePaused
    try:
        raise SystemExit(main())
    except (RuntimePaused,TimeoutError) as exc:
        print(str(exc),file=sys.stderr);raise SystemExit(75)
    except OSError as exc:
        transient={errno.EAGAIN,errno.EINTR,errno.ETIMEDOUT,errno.ECONNRESET,errno.ECONNREFUSED,errno.ENETUNREACH}
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(74 if exc.errno in transient else 1)
