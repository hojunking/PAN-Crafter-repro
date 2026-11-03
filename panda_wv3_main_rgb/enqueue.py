"""Durable s1 handoff after a verified completed ablation run, with fresh preflight."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from . import plan

LEGACY_NAME='panda-wv3-components-s1-v1'
LEGACY_CAMPAIGN='PANDA_WV3_COMPONENT_S13_20261102_v1'
CONTAINER_NAME='panda-wv3-main-rgb-s1-v1'
REQUEST_SCHEMA='WV3_MAIN_RGB_AFTER_RUN_V1'


def now():return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.handoff-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def handoff_root():return plan.campaign_root('s1')/'handoff'


def legacy_root():return plan.ROOT/'work_dir/panda_wv3_ablation'/LEGACY_CAMPAIGN/'s1/WV3'


def release_check():
    value=json.loads((plan.CONFIG/'release.json').read_text())
    bad=[p for p,digest in value['files'].items() if not (plan.ROOT/p).is_file() or sha(plan.ROOT/p)!=digest]
    if bad:raise ValueError('Pull the complete committed release: '+', '.join(bad))
    if value['source_sha256']!=plan.digest({k:v for k,v in value.items() if k!='source_sha256'}):
        raise ValueError('Release manifest digest mismatch')
    return value['source_sha256']


def inspect_container(name):
    p=subprocess.run(['docker','container','inspect',name],capture_output=True,text=True)
    return json.loads(p.stdout)[0] if p.returncode==0 else None


def verify_container(info,*,campaign,source=None,container_id=None):
    labels=info['Config'].get('Labels') or {}
    if labels.get('panda.campaign')!=campaign or labels.get('panda.server')!='s1':
        raise ValueError('Container is not the intended s1 campaign')
    if source is not None and labels.get('panda.source')!=source:
        raise ValueError('Existing main container uses another source release')
    if container_id is not None and info.get('Id')!=container_id:
        raise ValueError('Legacy container identity changed after handoff registration')
    roots=[m['Source'] for m in info.get('Mounts',[]) if m.get('Destination')=='/workspace']
    if len(roots)!=1 or Path(roots[0]).resolve()!=plan.ROOT.resolve():
        raise ValueError('Container belongs to another checkout')


def command(server,image,*,name=None,dlpan=None,gpu=True):
    p=subprocess.run(['docker','image','inspect',image],check=True,capture_output=True,text=True)
    info=json.loads(p.stdout)[0]
    env=dict(s.split('=',1) for s in info['Config'].get('Env',[]))
    env.update(PYTHONPATH='/workspace',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU',
        PANCRAFTER_DLPAN='/opt/DLPan-Toolbox',NVIDIA_DRIVER_CAPABILITIES='compute,utility')
    from edgeres48.docker_mounts import targets
    mounts=[(plan.ROOT,'/workspace',False)]+[(p,str(p),False) for p in targets(plan.ROOT,'s1')]
    if dlpan:mounts.append((Path(dlpan).resolve(strict=True),'/opt/DLPan-Toolbox',False))
    output=plan.ROOT/'work_dir/panda_wv3_main_rgb';output.mkdir(parents=True,exist_ok=True)
    mounts.append((output,'/workspace/work_dir/panda_wv3_main_rgb',True))
    cmd=['docker','run','--init','--shm-size=8g','--stop-timeout=180',
         '--user',f'{os.getuid()}:{os.getgid()}','--workdir','/workspace']
    if gpu:cmd+=['--gpus','all']
    if name:cmd+=['--detach','--name',name,'--label','panda.campaign='+plan.CAMPAIGN_ID,
                 '--label','panda.server='+server,'--label','panda.source='+release_check()]
    else:cmd+=['--rm']
    for k,v in sorted(env.items()):cmd+=['--env',k+'='+v]
    for src,dst,rw in mounts:
        if any(c in str(src)+dst for c in (',','\n','\r')):raise ValueError('Unsupported mount delimiter')
        cmd+=['--mount',f'type=bind,src={src},dst={dst}'+('' if rw else ',readonly')]
    return cmd+[info['Id']]


def preflight(args):
    base=command('s1',args.image,dlpan=args.dlpan,gpu=False)
    for action in ('check','check-sheet'):
        subprocess.run(base+['python','-m','panda_wv3_main_rgb',action,'--server','s1'],check=True)
    return base[-1]  # Immutable image ID used for the eventual launch.


def legacy_state():
    value=json.loads((legacy_root()/'campaign_state.json').read_text())
    if value.get('campaign_id')!=LEGACY_CAMPAIGN or value.get('server')!='s1':
        raise ValueError('Legacy state belongs to another campaign/server')
    return value


def capture_target(run_id=None):
    info=inspect_container(LEGACY_NAME)
    if info is None:raise ValueError('Expected legacy ablation container does not exist')
    verify_container(info,campaign=LEGACY_CAMPAIGN)
    state=legacy_state()
    if run_id is None:
        running=[r for r in state['runs'] if r['status']=='RUNNING']
        if len(running)>1:raise ValueError('Ambiguous running legacy jobs')
        if running:row=running[0]
        else:
            if state['status'] in ('ERROR','PAUSED','FINISHED_WITH_FAILURES'):
                raise ValueError('Legacy queue is not running successfully')
            completed=[r for r in state['runs'] if r['status']=='COMPLETED']
            if not completed:raise ValueError('No active or completed legacy run to bind')
            row=completed[-1]
    else:
        matches=[r for r in state['runs'] if r['run_id']==run_id]
        if len(matches)!=1:raise ValueError('Requested target run is not registered in legacy queue')
        row=matches[0]
    if row['status'] not in ('RUNNING','COMPLETED'):
        raise ValueError('Target must be running or already completed, not '+row['status'])
    run_id=row['run_id']
    if Path(run_id).name!=run_id:raise ValueError('Unsafe legacy run ID')
    cfgpath=legacy_root()/'runs'/run_id/'config.json';cfg=json.loads(cfgpath.read_text())
    if (cfg['run_id']!=run_id or cfg['campaign_id']!=LEGACY_CAMPAIGN or cfg['server']!='s1'
            or cfg['case_id']!=row['case'] or cfg['repeat']!=row['repeat']):
        raise ValueError('Target configuration differs from legacy queue')
    return {'container_id':info['Id'],'run_id':run_id,'horizon':cfg['horizon'],
            'config_sha256':cfg['config_sha256'],'config_file_sha256':sha(cfgpath)}


def target_completed(target):
    """A row being at the final update alone is not a completed/evaluated run."""
    root=legacy_root();state=legacy_state()
    matches=[r for r in state['runs'] if r['run_id']==target['run_id']]
    if len(matches)!=1:raise ValueError('Bound legacy target disappeared or duplicated')
    row=matches[0];directory=root/'runs'/target['run_id']
    if sha(directory/'config.json')!=target['config_file_sha256']:
        raise ValueError('Legacy target config changed after registration')
    if row['status'] in ('FAILED','ERROR','PAUSED'):
        raise RuntimeError('Legacy target did not finish successfully: '+row['status'])
    if row['status']!='COMPLETED':
        if state['status'] in ('ERROR','PAUSED','FINISHED_WITH_FAILURES'):
            raise RuntimeError('Legacy campaign stopped before target completion: '+state['status'])
        return False
    step=target['horizon']
    if row['step']!=step:raise ValueError('Premature legacy completed status')
    path=directory/f'evaluation_{step:07d}.json';e=json.loads(path.read_text())
    checkpoint=directory/'models'/f'step_{step:07d}.pt'
    meta=json.loads(checkpoint.with_suffix('.json').read_text())
    if (e.get('schema')!='WV3_ABLATION_EVAL_V1' or e['run_id']!=target['run_id']
            or e['config_sha256']!=target['config_sha256'] or e['step']!=step
            or e.get('same_state_verified') is not True
            or meta['step']!=step or e['checkpoint']!=meta
            or e['model_sha256']!=meta['model_sha256']
            or meta['config_sha256']!=target['config_sha256']
            or sha(checkpoint)!=meta['file_sha256']):
        raise ValueError('Legacy completed checkpoint/evaluation evidence mismatch')
    for split in ('rr','fr'):
        if [r['scene_id'] for r in e[split]['per_scene']]!=[f'WV3:{split}:{i}' for i in range(20)]:
            raise ValueError('Legacy native evaluation is incomplete')
    return True


def stop_legacy(target):
    if not target_completed(target):raise RuntimeError('Refusing to stop an unfinished target')
    info=inspect_container(LEGACY_NAME)
    if info is None:raise ValueError('Registered legacy container is missing')
    verify_container(info,campaign=LEGACY_CAMPAIGN,container_id=target['container_id'])
    subprocess.run(['docker','update','--restart=no',LEGACY_NAME],check=True)
    if info['State']['Running']:
        subprocess.run(['docker','stop','--time','180',LEGACY_NAME],check=True)
    after=inspect_container(LEGACY_NAME)
    if after is None or after['State']['Running']:
        raise RuntimeError('Legacy container did not stop')
    verify_container(after,campaign=LEGACY_CAMPAIGN,container_id=target['container_id'])
    if after['State'].get('ExitCode')!=0:
        raise RuntimeError('Legacy stop was not graceful; inspect retained logs/checkpoints before retry')
    return {'at':now(),'container':LEGACY_NAME,'target_run_id':target['run_id'],
            'target_step':target['horizon'],'action':'COMPLETED_TARGET_THEN_GRACEFUL_STOP',
            'legacy_state':legacy_state()}


def status(value):
    value=dict(value,updated_at=now(),supervisor_pid=os.getpid())
    atomic_json(handoff_root()/'status.json',value)
    return value


def execute(request):
    source=release_check()
    if request['schema']!=REQUEST_SCHEMA or request['source_sha256']!=source:
        raise ValueError('Handoff source release changed; no automatic migration')
    existing=inspect_container(CONTAINER_NAME)
    if existing is not None:
        verify_container(existing,campaign=plan.CAMPAIGN_ID,source=source)
        if existing['State']['Running']:
            return status({'status':'RUNNING','container':CONTAINER_NAME,'already_running':True})
    target=request.get('target')
    if target:
        while not target_completed(target):
            info=inspect_container(LEGACY_NAME)
            if info is None:raise RuntimeError('Legacy container disappeared before completion')
            verify_container(info,campaign=LEGACY_CAMPAIGN,container_id=target['container_id'])
            if not info['State']['Running']:raise RuntimeError('Legacy container stopped before target completion')
            status({'status':'WAITING_FOR_TARGET','target':target})
            time.sleep(1)
        # Re-check the pinned sources before any disruptive action.
        release_check()
        receipt=stop_legacy(target);atomic_json(handoff_root()/'cutover.json',receipt)
    while True:
        active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                              check=True,capture_output=True,text=True).stdout.strip()
        if not active:break
        status({'status':'WAITING_FOR_GPU','target':target,'active_pids':active.splitlines()})
        if not request['wait']:raise RuntimeError('Unrelated GPU work is active; it was not stopped. Retry with --wait.')
        time.sleep(10)
    release_check()
    if existing is not None:
        subprocess.run(['docker','start',CONTAINER_NAME],check=True)
    else:
        subprocess.run(command('s1',request['image_id'],name=CONTAINER_NAME,dlpan=request.get('dlpan'))+
            ['python','-m','panda_wv3_main_rgb','run','--server','s1','--device','cuda:0'],check=True)
    result=inspect_container(CONTAINER_NAME)
    if result is None or not result['State']['Running']:raise RuntimeError('Main container did not stay running at launch')
    verify_container(result,campaign=plan.CAMPAIGN_ID,source=source)
    return status({'status':'STARTED','container':CONTAINER_NAME,'target':target,'source_sha256':source})


def supervisor(fd):
    # The detached process inherits the parent's open-file-description lock.
    # No shell jobs, PID guesses, or detached Python import races control the GPU.
    try:
        request=json.loads((handoff_root()/'request.json').read_text())
        result=execute(request);print(json.dumps(result),flush=True)
    except BaseException as exc:
        status({'status':'ERROR','error':repr(exc)});raise
    finally:os.close(fd)


def registered_request(args,source,image_id):
    path=handoff_root()/'request.json'
    if path.exists():
        old=json.loads(path.read_text())
        if (old.get('schema')!=REQUEST_SCHEMA or old['source_sha256']!=source
                or old['image_id']!=image_id or old.get('dlpan')!=args.dlpan):
            raise ValueError('Existing handoff uses different source/image/environment')
        if args.target_run and (old.get('target') or {}).get('run_id')!=args.target_run:
            raise ValueError('Existing handoff binds another legacy run')
        if bool(old.get('target'))!=args.after_current_ablation:
            raise ValueError('Existing handoff mode differs; do not silently replace its target')
        # --wait is an operational retry preference, not a scientific identity.
        old['wait']=args.wait;atomic_json(path,old);return old
    target=capture_target(args.target_run) if args.after_current_ablation else None
    value={'schema':REQUEST_SCHEMA,'created_at':now(),'source_sha256':source,
        'image_id':image_id,'dlpan':args.dlpan,'target':target,'wait':args.wait,
        'campaign_id':plan.CAMPAIGN_ID,'server':'s1','checkout':str(plan.ROOT.resolve())}
    atomic_json(path,value);return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',choices=('s1',),required=True);p.add_argument('--image',required=True)
    p.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--after-current-ablation',action='store_true')
    p.add_argument('--target-run',help='Bind the originally active legacy run even if it has already completed')
    p.add_argument('--enqueue',action='store_true',help='Detach a durable host supervisor and return after registration')
    p.add_argument('--wait',action='store_true',help='Wait for unrelated GPU jobs; never stop them')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--supervise-fd',type=int,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.supervise_fd is not None:return supervisor(args.supervise_fd)
    if args.target_run and not args.after_current_ablation:p.error('--target-run requires --after-current-ablation')
    source=release_check()
    existing=inspect_container(CONTAINER_NAME)
    if existing is not None:
        verify_container(existing,campaign=plan.CAMPAIGN_ID,source=source)
        if existing['State']['Running'] and not args.check_only:
            print(json.dumps({'status':'ALREADY_RUNNING','container':CONTAINER_NAME}));return
    if args.check_only:
        preflight(args);print(json.dumps({'status':'CHECKED','training_modified':False}));return
    root=handoff_root();root.mkdir(parents=True,exist_ok=True)
    fd=os.open(root/'supervisor.lock',os.O_RDWR|os.O_CREAT,0o600)
    transferred=False
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'status':'ALREADY_ENQUEUED','status_file':str(root/'status.json')}));return
        image_id=preflight(args)  # Must pass before target capture or stop.
        request=registered_request(args,source,image_id)
        status({'status':'REGISTERED','target':request['target'],'source_sha256':source})
        if args.enqueue:
            env=os.environ.copy();env.update(PYTHONPATH=str(plan.ROOT),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with (root/'supervisor.log').open('ab',buffering=0) as log:
                proc=subprocess.Popen([sys.executable,'-m','panda_wv3_main_rgb.enqueue','--server','s1',
                    '--image',request['image_id'],'--supervise-fd',str(fd)],cwd=plan.ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    close_fds=True,pass_fds=(fd,))
            print(json.dumps({'status':'ENQUEUED','supervisor_pid':proc.pid,'target':request['target'],
                              'status_file':str(root/'status.json'),'log':str(root/'supervisor.log')}))
        else:
            transferred=True;return supervisor(fd)
    finally:
        if not transferred:os.close(fd)


if __name__=='__main__':main()
