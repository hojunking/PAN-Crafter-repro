"""Portable remote launcher: targeted QB continuation, GPU-waiting GF2 queue."""
import argparse,json,os,subprocess,time
from pathlib import Path
from . import plan
from .release import verify
from panda_unified_campaign import enqueue as original


def name(server):return f'panda-sensor-eq40-{server}-v1'

def command(server,image,*,gpu=True,detached=False,dlpan=None):
    plan.sensor_for(server)
    info=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
    env=dict(s.split('=',1) for s in info['Config'].get('Env',[]))
    env.update(PYTHONPATH='/workspace',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU',PANCRAFTER_DLPAN='/opt/DLPan-Toolbox',
        NVIDIA_DRIVER_CAPABILITIES='compute,utility',PYTORCH_KERNEL_CACHE_PATH='/tmp')
    from edgeres48.docker_mounts import targets
    mounts=[(plan.ROOT,'/workspace',False)]+[(p,str(p),False) for p in targets(plan.ROOT,server)]
    if dlpan:mounts.append((Path(dlpan).resolve(strict=True),'/opt/DLPan-Toolbox',False))
    for package in (['panda_unified_campaign','panda_sensor_eq40'] if server=='s2' else ['panda_sensor_eq40']):
        out=plan.ROOT/'work_dir'/package;out.mkdir(parents=True,exist_ok=True)
        mounts.append((out,'/workspace/work_dir/'+package,True))
    cmd=['docker','run','--init','--pid=host','--shm-size=8g','--stop-timeout=180',
         '--user',f'{os.getuid()}:{os.getgid()}','--workdir','/workspace']
    if gpu:cmd+=['--gpus','all']
    if detached:
        cmd+=['--detach','--name',name(server),'--label','panda.campaign='+plan.CAMPAIGN_ID,
              '--label','panda.server='+server,'--label','panda.source='+verify()['source_sha256']]
    else:cmd+=['--rm']
    for k,v in sorted(env.items()):cmd+=['--env',k+'='+v]
    for src,dst,rw in mounts:
        if any(c in str(src)+dst for c in (',','\n','\r')):raise ValueError('Invalid mount path')
        cmd+=['--mount',f'type=bind,src={src},dst={dst}'+('' if rw else ',readonly')]
    return cmd+[info['Id']]


def authenticate(info,server):
    labels=info['Config'].get('Labels') or {}
    if any(labels.get(k)!=v for k,v in {'panda.campaign':plan.CAMPAIGN_ID,'panda.server':server,
                                       'panda.source':verify()['source_sha256']}.items()):
        raise ValueError('Container identity mismatch')
    paths=[m['Source'] for m in info['Mounts'] if m['Destination']=='/workspace']
    if len(paths)!=1 or Path(paths[0]).resolve()!=plan.ROOT.resolve():raise ValueError('Wrong checkout mounted')
    if info['Config']['Cmd']!=['python','-m','panda_sensor_eq40','run','--server',server]:
        raise ValueError('Unexpected container entrypoint')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',choices=('s2','s3'),required=True)
    p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--check-only',action='store_true');p.add_argument('--wait',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args()
    existing=original.inspect(name(a.server))
    if a.status:print(json.dumps(None if not existing else existing['State']));return
    verify()
    if existing:
        authenticate(existing,a.server)
        if existing['State']['Running']:print('Already running '+name(a.server));return
        if not a.resume:raise ValueError('Use --resume for this stopped campaign')
    elif a.resume:raise ValueError('No campaign container exists to resume')
    for action in ('check','check-sheet'):
        subprocess.run(command(a.server,a.image,gpu=False,dlpan=a.dlpan)+
            ['python','-m','panda_sensor_eq40',action,'--server',a.server],check=True)
    if a.check_only:return
    marker=plan.ROOT/'gspread/server.txt'
    if not marker.is_file() or marker.read_text().strip()!=a.server:
        raise ValueError('Run on the matching server; local gspread/server.txt differs')
    if a.server=='s2':
        if not (plan.donor_root('s2')/'campaign_state.json').is_file():
            raise FileNotFoundError('Original local unified QB state is required')
        prior=original.inspect(original.container_name('s2'))
        if prior and prior['State']['Running']:
            original.verify_container(prior,'s2')
            subprocess.run(['docker','kill','--signal=TERM',original.container_name('s2')],check=True)
            deadline=time.monotonic()+300
            while original.inspect(original.container_name('s2'))['State']['Running']:
                if time.monotonic()>deadline:raise RuntimeError('Checkpoint stop pending; no force kill')
                time.sleep(2)
    while True:
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if not active:break
        if not a.wait:raise RuntimeError('GPU busy; use --wait; unrelated jobs were not stopped')
        print('Waiting for GPU workload to finish',flush=True);time.sleep(30)
    if existing:subprocess.run(['docker','start',name(a.server)],check=True)
    else:subprocess.run(command(a.server,a.image,detached=True,dlpan=a.dlpan)+
        ['python','-m','panda_sensor_eq40','run','--server',a.server],check=True)
    print('Started '+name(a.server),flush=True)

if __name__=='__main__':main()
