#!/usr/bin/env python3
"""Explicit local PCREPRO entrypoint; importing it has no execution side effects."""
import argparse
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pcrepro.common import ROOT,atomic_json,camp,read,read_config,locked,machine_lock
from pcrepro.plan import SERVERS,DATASETS,case_for,verify_server

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','inspect-data','preflight','start','launch','run','train','postrun',
        'status','control','report','upload','retry-uploads','retry-evaluations'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--server',choices=SERVERS)
    p.add_argument('--dataset',choices=DATASETS)
    p.add_argument('--bindings',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--run-id')
    p.add_argument('--cycle',type=int,default=0)
    p.add_argument('--command',choices=['STOP_NOW_SAFE','STOP_AFTER_CURRENT_RUN','STOP_AFTER_CYCLE','CONTINUE'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--foreground',action='store_true')
    p.add_argument('--no-upload',action='store_true')
    return p

def main(argv=None):
    args=parser().parse_args(argv);root=args.root.resolve();s=args.server
    if args.action not in ('build','inspect-data') and s is None:raise ValueError('--server s3/s4/s5 required')
    # Operational commands remain bound to the frozen release after future pulls.
    if s and args.action not in ('start','build','inspect-data'):
        release=read(camp(root,s)/'runtime_release.json')
        if release and Path(release['path']).resolve()!=ROOT.resolve():
            target=Path(release['path']).resolve()
            if target.parent!=root.parent or not target.name.startswith(root.name+'-runtime-pcrepro-'+s+'-'):
                raise ValueError('Unexpected release routing target')
            import subprocess
            args_list=list(sys.argv[1:] if argv is None else argv)
            if '--root' in args_list:
                i=args_list.index('--root');del args_list[i:i+2]
            return dict(exit_code=subprocess.call([sys.executable,str(target/'tools/pcrepro_runner.py'),*args_list],cwd=target))
    interrupted=[False]
    def stop(*_):
        interrupted[0]=True
        if args.action in ('start','launch','run'):
            from pcrepro.controller import control
            control(root,s,'STOP_NOW_SAFE')
    if args.action not in ('train',):
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    if args.action=='inspect-data':
        from pcrepro.data import inventory_candidates
        return inventory_candidates(root)
    if args.action=='build':
        from pcrepro.controller import build
        return build(root,s,args.cycle)
    if args.action=='start':
        from pcrepro.deployment import frozen_checkout
        target=frozen_checkout(root,s)
        # Re-exec so imported modules and ROOT agree with the immutable release.
        import subprocess
        binding=args.bindings or read(camp(target,s)/'registration.json').get('bindings_path')
        if not binding:raise ValueError('First start requires --bindings; see pcrepro/DATA_BINDINGS.md')
        binding=Path(binding).resolve(strict=True)
        command=[sys.executable,str(target/'tools/pcrepro_runner.py'),'launch','--server',s,'--bindings',str(binding)]
        if args.foreground:command.append('--foreground')
        if args.no_upload:command.append('--no-upload')
        returncode=subprocess.call(command,cwd=target)
        return dict(exit_code=returncode,release=str(target))
    if args.action=='launch':
        receipt=read(camp(root,s)/'runtime_release.json')
        if Path(receipt.get('path','')).resolve()!=ROOT.resolve() or root!=ROOT.resolve():
            raise ValueError('launch is reserved for the committed immutable release')
        from pcrepro.controller import start
        return start(root,s,args.bindings,args.foreground,not args.no_upload)
    if args.action=='run':
        from pcrepro.controller import run
        return dict(exit_code=run(root,s))
    if args.action=='control':
        if not args.command:raise ValueError('--command required')
        from pcrepro.controller import control
        return control(root,s,args.command)
    if args.action=='status':
        from pcrepro.controller import status
        return status(root,s)
    if args.action=='preflight':
        if not args.dataset or not args.bindings:raise ValueError('--dataset and --bindings required')
        from pcrepro.preflight import execute
        try:
            from pcrepro.handoff import gpu_processes
            with locked(machine_lock('worker')):
                if gpu_processes()!=[]:raise ValueError('GPU busy/unavailable; preflight smoke will not overlap')
                return execute(root,s,args.dataset,args.bindings,stopcheck=lambda:interrupted[0])
        except Exception as exc:
            atomic_json(camp(root,s)/'preflight'/f'{args.dataset}.failure.json',dict(reason=f'{type(exc).__name__}: {exc}'))
            raise
    if args.action in ('train','postrun'):
        if not args.config:raise ValueError('--config required')
        from pcrepro.plan import validate_config
        case=validate_config(read_config(args.config),require_bound=True)
        if case.server!=s:raise ValueError('Config/server differs')
        if args.action=='train':
            from pcrepro.training import train_run
            from pcrepro.handoff import gpu_processes
            with locked(machine_lock('worker')):
                if gpu_processes()!=[]:raise ValueError('GPU busy/unavailable; trainer will not overlap')
                return dict(exit_code=train_run(args.config,root,resume=args.resume))
        from pcrepro.postrun import execute
        from pcrepro.handoff import gpu_processes
        with locked(machine_lock('worker')):
            if gpu_processes()!=[]:raise ValueError('GPU busy/unavailable; postrun will not overlap')
            return execute(args.config,root,stopcheck=lambda:interrupted[0])
    if args.action=='report':
        from pcrepro.reporting import rebuild
        return rebuild(root,s)
    if args.action=='upload':
        if not args.run_id or case_for(args.run_id).server!=s:raise ValueError('--run-id for same server required')
        from pcrepro.upload import upload_run
        return upload_run(args.run_id,root,activated=True)
    if args.action=='retry-uploads':
        from pcrepro.upload import retry_pending
        return retry_pending(root,s,activated=True)
    if args.action=='retry-evaluations':
        from pcrepro.controller import _state,_save,retry_evaluations,verify_registration
        from pcrepro.handoff import gpu_processes
        verify_registration(root,s)
        with locked(machine_lock()),locked(camp(root,s)/'runner.lock'):
            if gpu_processes()!=[]:raise ValueError('GPU busy or unavailable')
            state=_state(root,s)
            pending=state.setdefault('pending_evaluations',{})
            pending.update(state.pop('blocked_evaluations',{}));_save(root,s,state)
            return dict(exit_code=retry_evaluations(root,s,state,limit=len(pending)))

if __name__=='__main__':
    try:
        result=main();print(json.dumps(result,ensure_ascii=False,default=str,allow_nan=False,indent=2))
        sys.exit(result.get('exit_code',0) if isinstance(result,dict) else 0)
    except (InterruptedError,KeyboardInterrupt) as exc:print(str(exc),file=sys.stderr);sys.exit(75)
    except OSError as exc:print(f'IO_ERROR: {exc}',file=sys.stderr);sys.exit(74)
    except (FloatingPointError,) as exc:print(f'NUMERICAL_ERROR: {exc}',file=sys.stderr);sys.exit(2)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(2 if 'out of memory' in str(exc).lower() else 1)
