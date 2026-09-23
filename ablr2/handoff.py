"""s3-only cooperative predecessor drain. No signals, cron edits or state reset."""
from pathlib import Path
import os

from ablr2.common import ROOT,camp,read,read_json,atomic_json,immutable_json,object_sha,sha256,utcnow,RuntimePaused

# Do not pretend older block-oriented controllers understand run boundaries.
PREDECESSORS={
    'pcrepro_runner.py':dict(namespace='_pcrepro',run_command='STOP_AFTER_CURRENT_RUN',safe_command='STOP_NOW_SAFE'),
    'gfp40_runner.py':dict(namespace='_gfp40',run_command=None,safe_command='STOP_NOW_SAFE'),
    'gfb20_runner.py':dict(namespace='_gfb20',run_command=None,safe_command='STOP_NOW_SAFE'),
    'g20_runner.py':dict(namespace='_g20',run_command=None,safe_command=None),
    'l100_runner.py':dict(namespace='_l100',run_command=None,safe_command=None),
    'qg40_runner.py':dict(namespace='_qg40',run_command=None,safe_command=None)}


def _lane(server):
    if server!='s3':raise ValueError('Predecessor campaign handoff is s3 only; s4/s5 are protected')


def inventory(root=ROOT,proc=Path('/proc')):
    shared=(Path(root)/'work_dir').resolve();rows=[]
    for path in proc.glob('[0-9]*/cmdline'):
        try:
            pid=int(path.parent.name)
            if pid==os.getpid():continue
            args=[v.decode(errors='replace') for v in path.read_bytes().split(b'\0') if v]
            if '-c' in args[:3]:continue
            names=[(i,Path(a).name) for i,a in enumerate(args) if Path(a).name in PREDECESSORS or Path(a).name in ('_watchdog.sh','_run_cases.sh')]
            if not names:continue
            cwd=(path.parent/'cwd').resolve(strict=True)
            index,name=names[0];script=(cwd/args[index]).resolve();scriptroot=script.parent.parent
            if (cwd/'work_dir').resolve()!=shared and (scriptroot/'work_dir').resolve()!=shared:continue
            server=args[args.index('--server')+1] if '--server' in args else None
            stat=(path.parent/'stat').read_text().rsplit(')',1)[1].split()
            rows.append(dict(pid=pid,start_ticks=stat[19],server=server,script=name,args=args,
                cwd=str(cwd),script_root=str(scriptroot),shared_work_dir=str(shared)))
        except (OSError,ValueError,IndexError):continue
    return rows


def inspect(root=ROOT,server='s3'):
    _lane(server);rows=inventory(root);campaigns=[]
    for script,contract in PREDECESSORS.items():
        folder=Path(root)/'work_dir'/contract['namespace']/server
        state=read(folder/'state.json') or read(folder/'status.json')
        processes=[r for r in rows if r['script']==script and r['server']==server]
        if state or processes:
            campaigns.append(dict(script=script,folder=str(folder),contract=contract,state=state,
                active_run=state.get('active_run'),processes=processes,control=read(folder/'control.json')))
    return dict(server=server,campaigns=campaigns,processes=rows,
        unknown_admission_owners=[r for r in rows if r['server'] is None],
        protected_other_servers=[r for r in rows if r['server'] in ('s1','s2','s4','s5')],
        mutations_performed=False)


def request(root,server='s3',*,operator_authorized=False,boundary='RUN'):
    _lane(server)
    if operator_authorized is not True:raise PermissionError('Explicit s3 handoff activation required')
    if boundary not in ('RUN','SAFE'):raise ValueError('Only RUN or explicitly requested SAFE handoff is supported')
    folder=camp(root,server);previous=read(folder/'handoff/request.json')
    if previous:
        if previous.get('boundary')!=boundary:raise ValueError('Existing handoff request has a different explicit boundary')
        return previous
    observed=inspect(root,server)
    if observed['unknown_admission_owners']:
        raise RuntimePaused('UNKNOWN_ADMISSION_OWNER: inspect watcher routing; no broad cron/process changes')
    controls=[]
    for item in observed['campaigns']:
        state=item['state'];active=item['active_run'];busy=bool(item['processes'])
        unfinished=bool(active and not state.get('runs',{}).get(active,{}).get('complete'))
        command=item['contract']['run_command' if boundary=='RUN' else 'safe_command']
        if not command and (busy or unfinished):
            raise RuntimePaused('RUN_BOUNDARY_UNSUPPORTED: '+item['script']+
                '; original source cannot honor this request. Explicit SAFE handoff or original owner action is required')
        # A dormant supported controller is stopped too, preventing later admission.
        if command:controls.append(dict(folder=item['folder'],script=item['script'],command=command,
            previous_control=item['control'],state_before=state,active_run=active))
    value=dict(schema='ABLR2X_S3_HANDOFF_v1',server=server,boundary=boundary,requested_at_utc=utcnow(),
        operator_action=True,controls=controls,observed=observed,other_servers_touched=False,
        signals_sent=False,cron_modified=False,old_queues_rewritten=False)
    immutable_json(folder/'handoff/request.json',value)
    for item in controls:
        existing=item['previous_control'].get('command')
        if existing=='STOP_NOW_SAFE':continue # Never weaken an existing operator stop.
        atomic_json(Path(item['folder'])/'control.json',dict(command=item['command'],at_utc=utcnow(),
            reason='ABLR2X_S3_HANDOFF',operator_action=True,handoff_request_sha256=object_sha(value)))
    return value


def _preserved_run(root,run,boundary):
    if not isinstance(run,str) or Path(run).name!=run:raise ValueError('Invalid predecessor run identity')
    wd=Path(root)/'work_dir'/run
    summary=read(wd/'official/summary.json');post=read(wd/'official/postrun_status.json')
    if summary.get('complete') or post.get('official_complete'):
        return dict(run_id=run,status='ORIGINAL_RUN_COMPLETE',summary=summary or post)
    if boundary=='RUN':raise RuntimePaused('WAIT_ORIGINAL_RUN_COMPLETION: '+run)
    identity_path=wd/'last/identity.json'
    if not identity_path.is_file():
        index=read(wd/'resume/index.json');names=index.get('snapshots',[])
        if not names or Path(names[0]).name!=names[0]:raise ValueError('Predecessor has no safe original resume state')
        identity_path=wd/'resume'/names[0]/'identity.json'
    identity=read_json(identity_path);state_path=identity_path.parent/'training_state.pt'
    if identity.get('training_state_sha256')!=sha256(state_path):raise ValueError('Predecessor fullstate SHA differs')
    import torch
    state=torch.load(state_path,map_location='cpu',weights_only=False)
    if (state.get('full_state') is not True or state.get('update')!=identity.get('update')
            or any(k not in state for k in ('optimizer','scheduler','rng','sampler','model_state'))):
        raise ValueError('Predecessor safe state lacks optimizer/scheduler/RNG/sampler/model')
    return dict(run_id=run,status='PRESERVED_INCOMPLETE_NOT_A_COMPLETED_PAIR',identity=identity,
        fullstate=str(state_path),original_result_reused=False)


def verify_ready(root,server='s3'):
    _lane(server);folder=camp(root,server);request_doc=read(folder/'handoff/request.json')
    observed=inspect(root,server)
    from ablr2.resources import idle_evidence
    idle=idle_evidence()
    if any(r['server']==server or r['server'] is None for r in observed['processes']) or not idle['idle']:
        raise RuntimePaused('WAIT_PREDECESSOR_EXIT: no overlapping GPU/controller owner')
    if not request_doc:
        if observed['campaigns']:raise RuntimePaused('WAIT_EXPLICIT_S3_HANDOFF: predecessor state exists')
        return dict(complete=True,status='NO_PREDECESSOR',server=server)
    if request_doc.get('server')!=server or request_doc.get('operator_action') is not True:
        raise ValueError('Invalid s3 handoff request')
    evidence=[]
    for item in request_doc['controls']:
        actual=read(Path(item['folder'])/'control.json')
        if actual.get('command') not in (item['command'],'STOP_NOW_SAFE'):
            raise RuntimePaused('PREDECESSOR_ADMISSION_REENABLED: '+item['script'])
        if item['active_run']:evidence.append(_preserved_run(root,item['active_run'],request_doc['boundary']))
    value=dict(schema='ABLR2X_S3_HANDOFF_COMPLETE_v1',server=server,complete=True,
        request_sha256=object_sha(request_doc),verified_at_utc=utcnow(),preserved_runs=evidence,
        original_queues_and_costs_preserved=True,partial_pairs_marked_complete=False,
        other_servers_touched=False,resource_evidence=idle)
    old=read(folder/'handoff/complete.json')
    if old:return old
    immutable_json(folder/'handoff/complete.json',value)
    return value
