#!/usr/bin/env python3
"""GFP40 explicit actions; build/inspection never start training or a clock."""
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
    p.add_argument('action',choices=('build','inspect-assets','clock','start','run','status','stop','preflight',
        'family','train','postrun','replay','report','upload','retry-uploads','aggregate'))
    p.add_argument('--server',choices=('s3','s4','s5'))
    p.add_argument('--t0',help='Optional local start time; first start defaults to now');p.add_argument('--bindings')
    p.add_argument('--family',choices=('G025','G0125','G050','P025','P075','LOW','HIGH'))
    p.add_argument('--foreground',action='store_true');p.add_argument('--in-place',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--no-upload',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--config');p.add_argument('--deadline');p.add_argument('--run');p.add_argument('--ctrl');p.add_argument('--mix')
    p.add_argument('--reports',nargs=3);p.add_argument('--output')
    p.add_argument('--command',choices=('STOP_NOW_SAFE','STOP_AFTER_BLOCK','CONTINUE'))
    a=p.parse_args(argv)
    from gfp40 import controller as C
    from gfp40.common import apply_runtime_policy,read,camp
    result={};code=0
    if a.action=='report' and a.server and not a.in_place:
        release=read(camp(ROOT,a.server)/'runtime_release.json')
        if release and Path(release['path']).resolve()!=ROOT:
            return subprocess.call([sys.executable,str(Path(release['path'])/'tools/gfp40_runner.py'),
                *(argv if argv is not None else sys.argv[1:]),'--in-place'],cwd=release['path'])
    if a.action=='build':result=C.build(ROOT)
    elif a.action=='clock':
        if not a.t0 or not a.server:p.error('clock requires local --server and --t0')
        result=C.write_window(ROOT,a.t0,a.server)
    elif a.action=='aggregate':
        if not a.reports or not a.output:p.error('aggregate requires three --reports and --output')
        from gfp40.reporting import aggregate
        result=aggregate(a.reports,a.output)
    else:
        if not a.server:p.error('--server s3|s4|s5 required')
        if a.action=='inspect-assets':
            from gfp40.assets import discover_bindings
            from gfp40.preflight import inspect_b20_handoff
            result=dict(discovered=discover_bindings(a.server,ROOT),b20_handoff=inspect_b20_handoff(ROOT,a.server),bytes_verified=False)
        elif a.action=='status':result=C.status(ROOT,a.server)
        elif a.action=='stop':
            if not a.command:p.error('stop requires --command')
            result=C.control(ROOT,a.server,a.command)
        elif a.action=='report':
            from gfp40.reporting import rebuild
            result=rebuild(ROOT,a.server)
        elif a.action=='start' and not a.in_place:
            from gfp40.deployment import frozen_checkout
            frozen=frozen_checkout(ROOT,a.server);forward=['start','--in-place','--server',a.server]
            for flag,value in (('--t0',a.t0),('--bindings',a.bindings)):
                if value:forward += [flag,value if flag=='--t0' else str(Path(value).resolve())]
            if a.foreground:forward+=['--foreground']
            if a.no_upload:forward+=['--no-upload']
            return subprocess.call([sys.executable,str(frozen/'tools/gfp40_runner.py'),*forward],cwd=frozen)
        elif a.action=='start':
            if a.t0:C.write_window(ROOT,a.t0,a.server)
            result=C.start(ROOT,a.server,a.foreground,a.bindings,not a.no_upload);code=result.get('exit_code',0)
        else:
            if not a.in_place and a.action in ('upload','retry-uploads'):
                release=read(camp(ROOT,a.server)/'runtime_release.json')
                if release and Path(release['path']).resolve()!=ROOT:
                    return subprocess.call([sys.executable,str(Path(release['path'])/'tools/gfp40_runner.py'),
                        *(argv if argv is not None else sys.argv[1:]),'--in-place'],cwd=release['path'])
            apply_runtime_policy(ROOT);C.verify_registration(ROOT,a.server);w=C.local_window(ROOT,a.server)
            final,prep=w.deadline_utc.isoformat(),w.train_finish_utc.isoformat()
            if a.deadline not in (None,final,prep):p.error('Deadline must equal registered clock')
            if a.action=='run':code=C.run(ROOT,a.server,a.bindings,not a.no_upload);result=dict(exit_code=code)
            elif a.action=='retry-uploads':result=C.retry_uploads(ROOT,a.server)
            elif a.action=='upload':
                from gfp40.upload import upload_run
                from gfp40.plan import case_for
                if not a.run or case_for(a.run).server!=a.server:p.error('upload requires local --run')
                result=upload_run(a.run,ROOT,activated=True)
            else:
                def safe_signal(*_):raise TimeoutError('Own preparation/evaluation safely interrupted')
                if a.action!='train':
                    signal.signal(signal.SIGTERM,safe_signal);signal.signal(signal.SIGINT,safe_signal)
                if a.action=='preflight':
                    from gfp40.preflight import execute
                    result=execute(ROOT,a.server,prep,a.bindings)
                elif a.action=='family':
                    if not a.family:p.error('family requires --family')
                    result=C.prepare_family(ROOT,a.server,a.family,prep,a.bindings)
                elif a.action=='replay':
                    if not a.ctrl or not a.mix:p.error('replay requires --ctrl and --mix')
                    from gfp40.replay import replay_pair
                    from gfp40.plan import validate_config
                    from gfp40.common import read_config
                    if any(validate_config(read_config(path)).server!=a.server for path in (a.ctrl,a.mix)):p.error('Cross-server replay')
                    result=replay_pair(a.ctrl,a.mix,root=ROOT,device='cuda',deadline=prep)
                else:
                    if not a.config:p.error('train/postrun requires --config')
                    from gfp40.plan import validate_config
                    from gfp40.common import read_config
                    if validate_config(read_config(a.config)).server!=a.server:p.error('Cross-server config')
                    if a.action=='train':
                        from gfp40.training import train_run
                        code=train_run(a.config,device='cuda',resume=a.resume,deadline_arg=final,root=ROOT);result=dict(exit_code=code)
                    else:
                        from gfp40.postrun import process
                        result=process(a.config,root=ROOT,device='cuda',deadline=final)
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str));return code

if __name__=='__main__':
    try:raise SystemExit(main())
    except TimeoutError as exc:print(str(exc),file=sys.stderr);raise SystemExit(75)
    except OSError as exc:
        retry={errno.EAGAIN,errno.EINTR,errno.ETIMEDOUT,errno.ECONNRESET,errno.ECONNREFUSED,errno.ENETUNREACH}
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);raise SystemExit(74 if exc.errno in retry else 1)
