"""Explicit operator API for the isolated s2 campaign."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import time

from .common import CAMPAIGN, atomic_json, read_json, timestamp
from .preflight import PROJECT, check_identity, prepare
from .registry import build_registry, register
from .safety import SafetyStop, admit, control, exclusive_lock, isolated_paths, inventory

DEFAULT_WORK = PROJECT/'work_dir'/CAMPAIGN/'s2'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('start','preflight','verify','register','run','status','pause','resume','stop','sync-sheet'):
        q=sub.add_parser(command)
        q.add_argument('--work-root',type=Path,default=DEFAULT_WORK)
        if command in ('start','preflight','resume'):
            q.add_argument('--legacy-root',type=Path,action='append')
            q.add_argument('--data-root',type=Path)
            q.add_argument('--data-catalog',type=Path,default=PROJECT/'vendor_reference/model_source/ablr2/sensor_sources.json')
            q.add_argument('--server-identity',type=Path)
            q.add_argument('--microbatch',type=int,choices=(48,24,12),default=None if command=='resume' else 48)
        if command in ('start','resume','run'):
            q.add_argument('--server',choices=('s2',),required=True)
            q.add_argument('--until-operator-stop',action='store_true',required=True)
            q.add_argument('--device',default='cuda:0')
        if command in ('start','resume','run','verify','sync-sheet'):
            q.add_argument('--credentials',type=Path)
        if command=='verify':
            q.add_argument('--all-gates',action='store_true',required=True)
            q.add_argument('--device',default='cuda:0')
        if command=='stop':
            choice=q.add_mutually_exclusive_group(required=True)
            choice.add_argument('--after-current-update',action='store_true')
            choice.add_argument('--after-current-block',action='store_true')
        if command=='sync-sheet':
            q.add_argument('--sheet-id',type=int,required=True)
    return p


def status(work):
    result={'campaign_id':CAMPAIGN,'work_root':str(work),'control':control(work)}
    for name in ('campaign_manifest','controller_state','gate_receipt','startup_manifest','startup_status'):
        path=work/'preflight/gates.json' if name=='gate_receipt' else work/(name+'.json')
        result[name]=read_json(path) if path.exists() else None
    result['outbox_status_counts']={}
    for path in (work/'outbox').glob('*.json'):
        envelope=read_json(path)
        state=envelope.get('status','UNKNOWN')
        result['outbox_status_counts'][state]=result['outbox_status_counts'].get(state,0)+1
    return result


def authorize_resume(work):
    """Never reset a live controller, weights, sampler, LR or registry."""
    with exclusive_lock(work/'controller.lock'):
        path=work/'controller_state.json'
        if path.exists():
            state=read_json(path)
            if state.get('campaign_id')!=CAMPAIGN:
                raise SafetyStop('BLOCKED_IDENTITY: controller campaign')
            state['status']='READY'; state['explicit_resume_at']=timestamp()
            atomic_json(path,state)
        control(work,'RUN')


def wait_for_resource(args, manifest):
    previous={}
    def stopped(*_): control(args.work_root,'STOP')
    try:
        for sig in (signal.SIGINT,signal.SIGTERM): previous[sig]=signal.signal(sig,stopped)
        while True:
            if control(args.work_root)['action']!='RUN':
                raise SafetyStop('STOPPED_BY_OPERATOR: GPU gate not started; explicit resume required')
            inv=inventory(args.work_root,manifest['server_identity_file'])
            try:
                result=admit(inv,expected_hostname=manifest['hostname'],own_pid=os.getpid())
            except SafetyStop as error:
                if not str(error).startswith('WAIT_RESOURCE:'): raise
                result={'status':'WAIT_RESOURCE','reason':str(error)}
            atomic_json(args.work_root/'startup_status.json',{'at':timestamp(),**result})
            if result['status']=='RESOURCE_AVAILABLE': return
            print(json.dumps(result),flush=True)
            for _ in range(30):
                if control(args.work_root)['action']!='RUN': break
                time.sleep(1)
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)


def run_campaign(args):
    from .controller import Controller
    from .runtime import CampaignRuntime
    from .verify import check_gates
    check_gates(args.work_root)
    runtime=CampaignRuntime(args.work_root,args.credentials,args.device)
    # Admission is rechecked for every block and while waiting. No foreign PID
    # is signalled. STOP during resource wait remains a persistent operator stop.
    old_handlers={}
    def stop_signal(*_):
        control(args.work_root,'STOP')
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            old_handlers[sig]=signal.signal(sig,stop_signal)
        while True:
            engine=Controller(args.work_root,runtime.registry,runtime.runner,runtime.admission,
                              stage_report=runtime.stage_report)
            result=engine.run(until_operator_stop=args.until_operator_stop)
            if result['status']!='WAIT_RESOURCE':
                return result
            print(json.dumps({'status':'WAIT_RESOURCE','reason':result.get('admission'),
                              'action':'waiting without stopping existing training'}),flush=True)
            for _ in range(30):
                if control(args.work_root)['action']!='RUN':
                    break
                time.sleep(1)
    finally:
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)


def execute(args):
    args.work_root=args.work_root.resolve()
    isolated_paths(PROJECT,args.work_root)
    command=args.command
    if command=='status': return status(args.work_root)
    if command in ('pause','stop'):
        if not (args.work_root/'campaign_manifest.json').exists() and not (args.work_root/'startup_manifest.json').exists():
            raise SafetyStop('BLOCKED_IDENTITY: campaign has not been prepared')
        action='PAUSE' if command=='pause' else 'STOP_AFTER_BLOCK' if args.after_current_block else 'STOP'
        return control(args.work_root,action)
    if command=='sync-sheet':
        if not args.credentials: raise ValueError('--credentials is required')
        from .runtime import sync_sheet
        return sync_sheet(args.work_root,args.credentials,args.sheet_id)
    if command in ('preflight','start'):
        if not args.legacy_root or not args.data_root or not args.server_identity:
            raise ValueError('--legacy-root, --data-root and --server-identity are required')
        if command=='start' and (args.work_root/'control.json').exists() and control(args.work_root)['action']!='RUN':
            raise SafetyStop('Explicit resume is required for a previously stopped/paused campaign')
        if command=='start':
            if not args.credentials: raise ValueError('--credentials is required for read-only sheet gate')
            startup=dict(campaign_id=CAMPAIGN,server='s2',legacy_roots=[str(p.resolve()) for p in args.legacy_root],
                data_root=str(args.data_root.resolve()),data_catalog=str(args.data_catalog.resolve()),
                server_identity_file=str(args.server_identity.resolve()),microbatch=args.microbatch)
            path=args.work_root/'startup_manifest.json'
            if path.exists():
                old=read_json(path)
                if any(old[k]!=startup[k] for k in startup if k!='microbatch'):
                    raise SafetyStop('BLOCKED_IDENTITY: startup source binding differs')
            atomic_json(path,startup)
            if not (args.work_root/'control.json').exists(): control(args.work_root,'RUN')
        manifest=prepare(args.work_root,data_catalog=args.data_catalog,data_root=args.data_root,
            server_identity=args.server_identity,legacy_roots=args.legacy_root,microbatch=args.microbatch)
        if command=='preflight': return manifest
        if not args.credentials: raise ValueError('--credentials is required for read-only sheet gate')
        if not (args.work_root/'control.json').exists(): control(args.work_root,'RUN')
        wait_for_resource(args,manifest)
        from .verify import verify
        verify(args.work_root,all_gates=True,credentials=args.credentials,device=args.device)
        register(args.work_root,build_registry(args.microbatch))
        return run_campaign(args)
    if command=='verify':
        if not args.credentials: raise ValueError('--credentials is required for read-only sheet gate')
        from .verify import verify
        return verify(args.work_root,all_gates=args.all_gates,credentials=args.credentials,device=args.device)
    if command=='register':
        from .verify import check_gates
        manifest=check_identity(args.work_root,require_registered=False)
        check_gates(args.work_root,require_registered=False)
        return register(args.work_root,build_registry(manifest['microbatch']))
    if command=='resume':
        from .verify import check_gates
        from .preflight import reconfigure_preformal_microbatch
        if not (args.work_root/'campaign_manifest.json').exists():
            path=args.work_root/'startup_manifest.json'
            if not path.exists(): raise SafetyStop('BLOCKED_IDENTITY: no prior startup to resume')
            boot=read_json(path)
            for key,field in [('data_root','data_root'),('data_catalog','data_catalog'),('server_identity','server_identity_file')]:
                given=getattr(args,key)
                if given is not None and str(given.resolve())!=boot[field]:
                    raise SafetyStop('BLOCKED_IDENTITY: bootstrap '+key+' differs')
                setattr(args,key,Path(boot[field]))
            if args.legacy_root and [str(p.resolve()) for p in args.legacy_root]!=boot['legacy_roots']:
                raise SafetyStop('BLOCKED_IDENTITY: bootstrap legacy roots differ')
            args.legacy_root=[Path(p) for p in boot['legacy_roots']]
            args.microbatch=boot['microbatch'] if args.microbatch is None else args.microbatch
            authorize_resume(args.work_root)
            args.command='start'
            return execute(args)
        manifest=check_identity(args.work_root,require_registered=False)
        if args.microbatch is None: args.microbatch=manifest['microbatch']
        if args.microbatch!=manifest['microbatch']:
            manifest=reconfigure_preformal_microbatch(args.work_root,args.microbatch)
        if args.server_identity and str(args.server_identity.resolve())!=manifest['server_identity_file']:
            raise SafetyStop('BLOCKED_IDENTITY: resume server mapping file differs')
        if args.data_root and str(args.data_root.resolve())!=manifest['data_root']:
            raise SafetyStop('BLOCKED_IDENTITY: resume data root differs')
        if args.data_catalog and str(args.data_catalog.resolve())!=manifest['data_catalog']:
            raise SafetyStop('BLOCKED_IDENTITY: resume data catalog differs')
        if args.legacy_root and [str(p.resolve()) for p in args.legacy_root]!=manifest['legacy_roots']:
            raise SafetyStop('BLOCKED_IDENTITY: resume legacy roots differ')
        # A safe operator stop before Q00/registration must remain recoverable.
        # Registered runs can only reuse their original passed gates and recipe.
        if (args.work_root/'cases.json').exists():
            check_gates(args.work_root)
        else:
            if not args.credentials: raise ValueError('--credentials is required to finish preflight gates')
            authorize_resume(args.work_root)
            wait_for_resource(args,manifest)
            from .verify import verify
            verify(args.work_root,all_gates=True,credentials=args.credentials,device=args.device)
            register(args.work_root,build_registry(manifest['microbatch']))
        authorize_resume(args.work_root)
        return run_campaign(args)
    if command=='run': return run_campaign(args)
    raise ValueError('Unhandled command')


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        # Independent startup/controller locks avoid duplicate preflight or Q00.
        if args.command in ('start','preflight','verify','register','resume','run'):
            isolated_paths(PROJECT,args.work_root)
            with exclusive_lock(args.work_root/'startup.lock'):
                handlers={}
                try:
                    for sig in (signal.SIGINT,signal.SIGTERM):
                        handlers[sig]=signal.signal(sig,lambda *_:control(args.work_root,'STOP'))
                    result=execute(args)
                finally:
                    for sig,handler in handlers.items(): signal.signal(sig,handler)
        else:
            result=execute(args)
        print(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False),flush=True)
        return 0
    except Exception as error:
        print(json.dumps({'status':str(error).split(':',1)[0],
            'error_type':type(error).__name__,'error':str(error)},ensure_ascii=False),flush=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
