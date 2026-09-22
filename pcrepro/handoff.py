"""Explicit activation only: retire owned GF2 workers without broad process kills.

Unknown launchers, another server, or unverifiable checkpoint preservation block
handoff. No legacy recipe, checkpoint, metric, or Sheet is rewritten.
"""
import os
import csv
from pathlib import Path
import signal
import shlex
import subprocess
from pcrepro.common import ROOT,read,read_json,atomic_json,camp,utcnow,object_sha,sha256
from pcrepro.plan import verify_server

LEGACY={'gfp40_runner.py':'_gfp40','gfb20_runner.py':'_gfb20','l100_runner.py':'_l100',
        'g20_runner.py':'_g20','qg40_runner.py':'_qg40'}
WATCHERS={'_watchdog.sh','_run_cases.sh'}
NONTRAIN_ACTIONS={'run','start','ensure','preflight','family','mixed','calibrate','postrun','replay'}

def process_record(pid,proc=Path('/proc')):
    base=proc/str(pid)
    try:
        args=[v.decode(errors='replace') for v in (base/'cmdline').read_bytes().split(b'\0') if v]
        stat=(base/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),args=args,cwd=str((base/'cwd').resolve(strict=True)),
            ticks=stat[19],uid=base.stat().st_uid)
    except (OSError,ValueError,IndexError):return None

def inventory(root=ROOT,proc=Path('/proc')):
    root=Path(root);shared=(root/'work_dir').resolve();rows=[]
    for base in proc.glob('[0-9]*'):
        row=process_record(base.name,proc)
        if not row or row['pid']==os.getpid():continue
        args=row['args']
        if '-c' in args[:3]:continue
        scripts=[(i,Path(a).name) for i,a in enumerate(args) if Path(a).name in set(LEGACY)|{'pcrepro_runner.py','train.py'}|WATCHERS]
        if not scripts:continue
        i,name=scripts[0]
        script_path=(Path(row['cwd'])/args[i]).resolve()
        script_repo=script_path.parent.parent if script_path.parent.name=='tools' else None
        same_root=((Path(row['cwd'])/'work_dir').resolve()==shared
                   or script_repo is not None and (script_repo/'work_dir').resolve()==shared)
        if not same_root:continue
        server=None
        if '--server' in args:
            j=args.index('--server')
            if j+1<len(args):server=args[j+1]
        row.update(script=name,action=args[i+1] if i+1<len(args) else '',server=server,
            campaign=LEGACY.get(name),config=None,script_path=str(script_path),
            script_root=str(script_repo) if script_repo is not None else None,
            shared_work_dir=str(shared))
        if '--config' in args:
            j=args.index('--config')
            if j+1<len(args):row['config']=str((Path(row['cwd'])/args[j+1]).resolve())
        rows.append(row)
    return rows

def gpu_processes():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=5)
        return [int(v.strip()) for v in out.splitlines() if v.strip().isdigit()]
    except (OSError,subprocess.SubprocessError):return None

def protected_routes(root,server):
    """Never touch a repository watchdog routing to s1/s2 or another server."""
    verify_server(server)
    routes={}
    for name in ('_fh20r1/r2_local_server.txt','_fh20r1/local_server.txt','_fh12/local_server.txt'):
        path=Path(root)/'work_dir'/name
        if path.exists() or path.is_symlink():
            if not path.is_file():raise ValueError('Unverifiable legacy watchdog registration: '+str(path))
            value=path.read_text().strip()
            if value!=server:raise ValueError('Protected/other-server legacy watchdog route: '+str(path)+' -> '+value)
            routes[name]=value
    return routes


def disable_known_watchdog(root,server,origin_root=None):
    """Remove only the exact repository's tagged legacy cron entries; retain backup."""
    protected_routes(root,server)
    if origin_root is not None and (Path(origin_root)/'work_dir').resolve()!=(Path(root)/'work_dir').resolve():
        raise ValueError('Origin repository does not share the owned work_dir')
    roots={str(Path(root).resolve()),str(Path(origin_root or root).resolve())}
    try:result=subprocess.run(['crontab','-l'],capture_output=True,text=True,timeout=10)
    except FileNotFoundError:return dict(status='CRONTAB_NOT_INSTALLED',removed=0)
    if result.returncode and 'no crontab' not in result.stderr.lower():raise ValueError('Cannot inspect local cron; no takeover')
    lines=result.stdout.splitlines();removed=[];keep=[]
    for line in lines:
        try:tokens=shlex.split(line,comments=True)
        except ValueError:tokens=[]
        own=any(r+'/tools/_watchdog.sh' in tokens for r in roots)
        if own and 'PANCRAFTER-WATCHDOG' in line:removed.append(line)
        else:keep.append(line)
    if removed:
        path=camp(root,server)/'handoff/removed_watchdog_cron.json'
        old=read(path,dict(entries=[]));entries=list(dict.fromkeys(old['entries']+removed))
        atomic_json(path,dict(entries=entries,at_utc=utcnow(),scope='this repository tagged watchdog only'))
        path.chmod(0o600)
        subprocess.run(['crontab','-'],input='\n'.join(keep)+'\n',text=True,check=True,timeout=10)
    return dict(status='TAGGED_WATCHDOG_DISABLED',removed=len(removed),other_entries_preserved=True)

def signal_verified(row,sig=signal.SIGTERM):
    current=process_record(row['pid'])
    if current is None:return False
    if any(current[k]!=row[k] for k in ('args','cwd','ticks','uid')) or current['uid']!=os.getuid():
        raise ValueError('PID identity changed; refusing signal')
    os.kill(row['pid'],sig);return True

def _training_run(row):
    if row['action']!='train' or not row['config']:return None
    from pcrepro.common import read_config
    cfg=read_config(row['config']);name=Path(cfg['work_dir']).name
    field=cfg.get(row['campaign'][1:],{})
    if not field or field.get('server_id',field.get('server'))!=row['server']:raise ValueError('Old trainer server differs or is unbound')
    sensor=field.get('sensor')
    if sensor!='GF2':raise ValueError('Refusing to stop a non-GF2 legacy trainer')
    work_dir=(Path(row['cwd'])/cfg['work_dir']).resolve()
    shared=(Path(row['cwd'])/'work_dir').resolve()
    if work_dir==shared or not work_dir.is_relative_to(shared):
        raise ValueError('Legacy training directory escaped owned work_dir')
    return dict(run_id=name,work_dir=str(work_dir),config=row['config'])


def _verify_receipt(receipt,root,server,required=False):
    if not receipt:
        if required:raise ValueError('Explicit local handoff receipt is required')
        return
    shared=str((Path(root)/'work_dir').resolve())
    if receipt.get('server')!=server or receipt.get('shared_work_dir',shared)!=shared:
        raise ValueError('Handoff receipt belongs to another server or workspace')
    for job in receipt.get('jobs',[]):
        if job.get('server')!=server or job.get('script') not in set(LEGACY)|WATCHERS:
            raise ValueError('Handoff job is foreign or unrecognized')
        if job.get('shared_work_dir',shared)!=shared:
            raise ValueError('Handoff job workspace differs')
        cwd_owned=(Path(job.get('cwd',''))/'work_dir').resolve()==Path(shared)
        script=Path(job['script_path']) if job.get('script_path') else None
        script_owned=script is not None and script.parent.name=='tools' and (script.parent.parent/'work_dir').resolve()==Path(shared)
        if not cwd_owned and not script_owned:
            raise ValueError('Handoff job has no owned repository path evidence')
        training=job.get('training')
        if training:
            wd=Path(training.get('work_dir','')).resolve()
            run=training.get('run_id')
            if (job.get('action')!='train' or job.get('script') not in LEGACY
                    or not isinstance(run,str) or not run or Path(run).name!=run
                    or wd==Path(shared) or not wd.is_relative_to(shared) or wd.name!=run):
                raise ValueError('Handoff training state escaped the owned run directory')


def pending_queue_runs(root,server):
    """Read serialized original queues, not mutable imports of old registries."""
    root=Path(root);runs={}
    for namespace in ('_gfp40','_gfb20','_l100','_g20','_qg40'):
        lane=root/'work_dir'/namespace/server
        state=read(lane/'status.json');release=read(lane/'runtime_release.json')
        roots={root}
        if release.get('path'):roots.add(Path(release['path']))
        def add(payload,source):
            if isinstance(payload,list):
                for item in payload:add(item,source)
            elif isinstance(payload,dict):
                owner=payload.get('server',payload.get('server_id',server))
                if owner!=server:return
                for key in ('run_ids','unstarted_run_ids'):
                    for run in payload.get(key,[]):
                        if not isinstance(run,str) or Path(run).name!=run:raise ValueError('Invalid old queue run ID')
                        runs[run]=str(source)
                for key in ('blocks','queue'):
                    value=payload.get(key,[])
                    add(list(value.values()) if isinstance(value,dict) else value,source)
        add(state,lane/'status.json')
        for source_root in sorted(roots,key=str):
            path=source_root/'config'/namespace[1:]/f'{server}_queue.json'
            if path.is_file():add(read_json(path),path)
            from pcrepro.common import read_config
            for config_path in (source_root/'config'/namespace[1:]).glob('*.yaml'):
                cfg=read_config(config_path);field=cfg.get(namespace[1:],{})
                if field.get('server_id',field.get('server'))==server and field.get('sensor')=='GF2':
                    name=Path(cfg.get('work_dir','')).name
                    if name:runs.setdefault(name,str(config_path))
        for path in (lane/'admissions').glob('*.json'):add(read_json(path),path)
        # P40/B20 may not have materialized config queues. Their archived CSV
        # case definitions are complete, finite, and can be read without code.
        patterns={'_gfp40':'GFP40_Cases_All83.csv','_gfb20':'GFB20_Cases_All36.csv'}
        if namespace in patterns:
            for source_root in sorted(roots,key=str):
                for path in (source_root/'research_log').rglob(patterns[namespace]):
                    with path.open(encoding='utf-8-sig',newline='') as stream:
                        for row in csv.DictReader(stream):
                            if row.get('server')==server and row.get('run_id'):
                                run=row['run_id']
                                if Path(run).name!=run:raise ValueError('Invalid archived case run ID')
                                runs.setdefault(run,str(path))
    result=[]
    for run,source in sorted(runs.items()):
        wd=root/'work_dir'/run
        begun=(wd/'meta/training_start_manifest.json').exists() or (wd/'last/identity.json').exists()
        begun=begun or bool(read(wd/'meta/training_status.json')) or read(wd/'official/summary.json').get('complete')
        if not begun:result.append(dict(run_id=run,status='CANCELLED_BY_USER_DIRECTION',queue_source=source))
    return result

def begin(root,server,*,activated=False,origin_root=None):
    verify_server(server)
    if not activated:raise PermissionError('Explicit PCREPRO start/handoff activation required')
    folder=camp(root,server);receipt=read(folder/'handoff/transition.json')
    _verify_receipt(receipt,root,server)
    routes=protected_routes(root,server)
    rows=inventory(root)
    for row in rows:
        if row['script'] in WATCHERS and row['server'] is None and routes:
            row['server']=server;row['server_evidence']=dict(protected_routes=routes)
    foreign=[r for r in rows if r['script']!='pcrepro_runner.py' and (r['server']!=server or r['script'] not in set(LEGACY)|WATCHERS)]
    if foreign:raise ValueError('Unknown/other-server legacy worker: '+str([r['pid'] for r in foreign]))
    cron=disable_known_watchdog(root,server,origin_root)
    if receipt.get('status')=='COMPLETE' and not [r for r in rows if r['script'] in set(LEGACY)|WATCHERS]:return receipt
    jobs=receipt.get('jobs',[]);known={(j['pid'],j['ticks']) for j in jobs}
    for row in rows:
        if row['script'] not in set(LEGACY)|WATCHERS:continue
        if row['script'] in LEGACY and row['action'] not in NONTRAIN_ACTIONS|{'train'}:raise ValueError('Unsupported legacy action; do not guess safe termination')
        if (row['pid'],row['ticks']) not in known:
            jobs.append(dict(row,training=_training_run(row),signal_sent=False))
    value=dict(status='STOP_REQUESTED',server=server,shared_work_dir=str((Path(root)/'work_dir').resolve()),
        requested_at_utc=receipt.get('requested_at_utc',utcnow()),
        jobs=jobs,watchdog=cron,policy='STOPPED_BY_USER_DIRECTION',old_final_results_preserved=True)
    atomic_json(folder/'handoff/transition.json',value)
    # Cooperative protocol first. Controllers stop admission before child signals.
    for namespace in ('_gfp40','_gfb20'):
        old=Path(root)/'work_dir'/namespace/server
        if old.exists():atomic_json(old/'control.json',dict(command='STOP_NOW_SAFE',reason='STOPPED_BY_USER_DIRECTION',at_utc=utcnow()))
    for row in sorted(jobs,key=lambda r:(r['script'] not in WATCHERS,r['action']=='train')):
        if not row.get('signal_sent'):
            row['signal_sent']=signal_verified(row)
            atomic_json(folder/'handoff/transition.json',value)
    return value

def verify_stopped(root,server):
    """No GPU overlap; preserved partial states are recorded, never called complete."""
    folder=camp(root,server);receipt=read(folder/'handoff/transition.json')
    _verify_receipt(receipt,root,server,required=True)
    protected_routes(root,server)
    remaining=[r for r in inventory(root) if r['script']!='pcrepro_runner.py']
    gpu=gpu_processes()
    if remaining or gpu!=[]:return dict(complete=False,status='WAIT_SAFE_LEGACY_EXIT',pids=[r['pid'] for r in remaining],gpu_pids=gpu)
    evidence=[]
    for job in receipt.get('jobs',[]):
        train=job.get('training')
        if not train:continue
        wd=Path(train['work_dir']);status=read(wd/'meta/training_status.json')
        identity=read(wd/'last/identity.json')
        state=wd/'last/training_state.pt'
        if not state.is_file() or not identity or not isinstance(identity.get('update'),int):
            raise ValueError('Legacy trainer stopped without a verified full-state; review '+str(wd))
        if identity.get('training_state_sha256')!=sha256(state):
            raise ValueError('Legacy full-state SHA differs; refusing unverified handoff')
        import torch
        saved=torch.load(state,map_location='cpu',weights_only=False)
        required=('optimizer','scheduler','rng','sampler')
        from pcrepro.model import state_hash
        # G20/QG40's authenticated resume schema predates a separate tensor
        # digest. The entire state file is still SHA-verified above; compute
        # and preserve its model digest without inventing a missing field.
        model_digest=state_hash(saved['model_state']) if saved.get('model_state') else None
        identity_digest=identity.get('state_hash')
        old_digest_schema=job['script'] in {'g20_runner.py','qg40_runner.py'}
        digest_valid=(model_digest==identity_digest if identity_digest is not None
                      else old_digest_schema and model_digest is not None)
        if (identity.get('full_state') is not True or saved.get('full_state') is not True
                or any(k not in saved for k in required) or saved.get('update')!=identity['update']
                or saved.get('scheduler',{}).get('last_epoch')!=identity['update']
                or not saved.get('model_state')
                or not digest_valid):
            raise ValueError('Legacy resume state incomplete; refusing new GPU work')
        evidence.append(dict(run_id=train['run_id'],status='STOPPED_BY_USER_DIRECTION',
            actual_updates=identity['update'],original_status=status,fullstate=str(state),identity=identity,
            model_state_hash=model_digest,identity_model_hash_present=identity_digest is not None,
            evaluation_status=read(wd/'official/postrun_status.json')))
    cancelled=pending_queue_runs(root,server)
    result=dict(receipt,status='COMPLETE',complete=True,verified_at_utc=utcnow(),preserved_runs=evidence,
        cancelled_unstarted=cancelled,old_campaign_files_deleted=False)
    atomic_json(folder/'handoff/transition.json',result);return result
