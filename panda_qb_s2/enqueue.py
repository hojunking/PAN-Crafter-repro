"""Standalone s2 Docker launcher. No previous container, donor or host ML environment."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from . import registry as G

ROOT = Path(__file__).resolve().parents[1]
NAME = 'panda-qb-s2-balance-v1'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(temp,path)


def control_dir():
    return ROOT / G.OUTPUT / 'launcher_s2'


def campaign_dir():
    return ROOT / G.OUTPUT / G.CAMPAIGN_ID / 's2/QB'


def inspect(kind, name, *, missing=False):
    result=subprocess.run(['docker',kind,'inspect',name],capture_output=True,text=True)
    if result.returncode:
        if missing and ('No such' in result.stderr): return None
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)[0]


def verify_sources():
    value=read(ROOT/'config/panda_qb_s2/release.json')
    bad=[p for p,sha in value['files'].items() if not (ROOT/p).is_file()
         or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=sha]
    if bad: raise ValueError('Incomplete or modified release checkout: '+', '.join(bad))
    plain={k:v for k,v in value.items() if k!='source_sha256'}
    digest=hashlib.sha256(json.dumps(plain,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=value['source_sha256']: raise ValueError('Release manifest digest differs')
    return digest


def wait_for_gpu(wait):
    while True:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                              capture_output=True,text=True,check=True)
        pids=[s.strip() for s in result.stdout.splitlines() if s.strip()]
        if not pids: return
        if not wait: raise RuntimeError('Active GPU compute workload: '+', '.join(pids))
        print('Waiting for local GPU workloads to finish: '+', '.join(pids),flush=True)
        time.sleep(30)


def command(image, *, name=None, token=None, dlpan=None, source_sha256=None, gpu=True):
    """Build fresh mounts explicitly instead of inheriting an old server container."""
    info=inspect('image',image)
    env=dict(s.split('=',1) for s in info['Config'].get('Env',[]))
    env.update(PYTHONPATH='/workspace', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MKL_THREADING_LAYER='GNU',
               PANCRAFTER_DLPAN='/opt/DLPan-Toolbox', NVIDIA_DRIVER_CAPABILITIES='compute,utility')
    mounts=[{'Source':str(ROOT),'Destination':'/workspace','RW':False}]
    from edgeres48.docker_mounts import targets
    for path in targets(ROOT,'s2'):
        mounts.append({'Source':str(path),'Destination':str(path),'RW':False})
    if dlpan:
        path=Path(dlpan).resolve(strict=True)
        if not path.is_dir(): raise ValueError('DLPan path must be a directory')
        mounts.append({'Source':str(path),'Destination':'/opt/DLPan-Toolbox','RW':False})
    output=ROOT/G.OUTPUT; output.mkdir(parents=True,exist_ok=True)
    mounts.append({'Source':str(output),'Destination':'/workspace/'+G.OUTPUT,'RW':True})
    user=f'{os.getuid()}:{os.getgid()}'
    cmd=['docker','run','--init','--pid=host','--shm-size=8g','--user',user,'--workdir','/workspace']
    if gpu: cmd+=['--gpus','all']
    if name:
        if not token or not source_sha256: raise ValueError('Launch identity is required')
        cmd+=['--detach','--name',name,'--label','panda.campaign='+G.CAMPAIGN_ID,
              '--label','panda.launch-token='+token,'--label','panda.source='+source_sha256]
    else: cmd+=['--rm']
    for k,v in sorted(env.items()): cmd+=['--env',k+'='+v]
    for m in mounts:
        if any(c in m[k] for k in ('Source','Destination') for c in (',','\n','\r')):
            raise ValueError('Unsupported bind-mount delimiter')
        value=f"type=bind,src={m['Source']},dst={m['Destination']}"
        cmd+=['--mount',value+('' if m['RW'] else ',readonly')]
    cmd += [info['Id']]
    return cmd,dict(image=info['Id'],user=user,env=sorted(k+'='+v for k,v in env.items()),
                    mounts=mounts,working_dir='/workspace')


def training_args():
    return ['python','-m','panda_qb_s2.worker','--server','s2']


def authenticate(current, receipt):
    expected=receipt['identity']; cfg=current['Config']; labels=cfg.get('Labels') or {}
    mounts=[{k:m[k] for k in ('Source','Destination','RW')} for m in current['Mounts']]
    norm=lambda xs:sorted(xs,key=lambda m:m['Destination'])
    if (('container_id' in receipt and current.get('Id')!=receipt['container_id'])
        or current['Image']!=expected['image'] or cfg['Cmd']!=training_args()
        or cfg['User']!=expected['user'] or cfg['WorkingDir']!=expected['working_dir']
        or sorted(cfg['Env'])!=expected['env'] or norm(mounts)!=norm(expected['mounts'])
        or labels.get('panda.campaign')!=G.CAMPAIGN_ID
        or labels.get('panda.launch-token')!=receipt['token']
        or labels.get('panda.source')!=receipt['source_sha256']):
        raise ValueError('Existing container does not match this s2 launch receipt')


def launch(image, *, dlpan=None, wait=False, resume=False):
    if (ROOT/'gspread/server.txt').read_text().strip()!='s2':
        raise ValueError('Set the local server marker to s2 on the s2 server')
    if not (ROOT/'gspread/account.json').is_file():
        raise ValueError('Local Sheet credential gspread/account.json is missing')
    source_sha256=verify_sources()
    control=control_dir();control.mkdir(parents=True,exist_ok=True)
    with (control/'controller.lock').open('a+') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=read(control/'launch.json') if (control/'launch.json').exists() else None
        current=inspect('container',NAME,missing=True)
        if saved is None and current is not None and (control/'launch_intent.json').exists():
            saved=read(control/'launch_intent.json')
            if saved['source_sha256']!=source_sha256:raise ValueError('Incomplete launch used a different release')
            authenticate(current,saved)
            saved['container_id']=current['Id'];save(control/'launch.json',saved)
        if saved:
            if saved['source_sha256']!=source_sha256: raise ValueError('Started queue must retain its original code')
            if current is None: raise ValueError('Recorded container is missing; preserve output and inspect before recovery')
            authenticate(current,saved)
            if current['State']['Running']: return dict(status='ALREADY_RUNNING',container=NAME)
            state=read(campaign_dir()/'worker_status.json') if (campaign_dir()/'worker_status.json').exists() else {}
            if state.get('stage')=='COMPLETED' and current['State'].get('ExitCode')==0:
                return dict(status='ALREADY_COMPLETED',container=NAME)
            if not resume: raise ValueError('Stopped/failed queue retained; inspect logs and use --resume for the same container')
            wait_for_gpu(wait)
            subprocess.run(['docker','start',NAME],check=True)
            return dict(status='RESUMED',container=NAME)
        if current is not None: raise ValueError('Existing unowned container; do not overwrite it')
        campaign_dir().mkdir(parents=True,exist_ok=True)
        token=uuid.uuid4().hex
        cmd,identity=command(image,name=NAME,token=token,dlpan=dlpan,source_sha256=source_sha256)
        check,_=command(identity['image'],dlpan=dlpan,gpu=False)
        subprocess.run(check+['python','-m','panda_qb_s2','check','--server','s2'],check=True)
        # Real authentication/read-only Sheet preflight before spending training time.
        subprocess.run(check+['python','-m','panda_qb_s2','check-sheet','--server','s2'],check=True)
        wait_for_gpu(wait)
        profiling,_=command(identity['image'],dlpan=dlpan)
        subprocess.run(profiling+['python','-m','panda_qb_s2','cost','--server','s2'],check=True)
        wait_for_gpu(False)
        receipt=dict(campaign_id=G.CAMPAIGN_ID,server='s2',token=token,identity=identity,
                     source_sha256=source_sha256,command=cmd+training_args())
        save(control/'launch_intent.json',receipt)
        result=subprocess.run(cmd+training_args(),check=True,capture_output=True,text=True)
        receipt['container_id']=result.stdout.strip()
        save(control/'launch.json',receipt)
        return dict(status='STARTED',container=NAME,container_id=receipt['container_id'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s2',),required=True)
    parser.add_argument('--image',help='Local image tag or immutable image ID; Dockerfile.env builds the reference environment')
    parser.add_argument('--dlpan',help='Optional existing exact DLPan checkout; otherwise use the copy built into the image')
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.status:
        c=inspect('container',NAME,missing=True)
        state=campaign_dir()/'worker_status.json'
        print(json.dumps({'docker_state':None if c is None else c['State'],
                          'worker':read(state) if state.exists() else None},indent=2));return
    if not args.image and not args.resume: parser.error('--image is required for a fresh launch')
    print(json.dumps(launch(args.image,dlpan=args.dlpan,wait=args.wait,resume=args.resume),indent=2))


if __name__=='__main__':
    main()
