"""Launch the server's isolated WV3 hyperparameter lane; never stop another job."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from . import plan
from .release import verify


def container_name(server):
    plan.campaign_root(server)
    return f'panda-unified-{server}-v1'


def inspect(name):
    result = subprocess.run(['docker','container','inspect',name],capture_output=True,text=True)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def verify_container(info, server):
    labels = info['Config'].get('Labels') or {}
    expected = {'panda.campaign':plan.CAMPAIGN_ID, 'panda.server':server,
                'panda.source':verify()['source_sha256']}
    if any(labels.get(k) != v for k,v in expected.items()):
        raise ValueError('Existing container belongs to another campaign/source/server')
    mounts = [m['Source'] for m in info.get('Mounts',[]) if m['Destination']=='/workspace']
    if len(mounts)!=1 or Path(mounts[0]).resolve()!=plan.ROOT.resolve():
        raise ValueError('Existing container uses another checkout')


def command(server, image, *, detached=False, gpu=True, dlpan=None):
    result = subprocess.run(['docker','image','inspect',image],check=True,capture_output=True,text=True)
    info = json.loads(result.stdout)[0]
    env = dict(s.split('=',1) for s in info['Config'].get('Env',[]))
    env.update(PYTHONPATH='/workspace', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', MKL_THREADING_LAYER='GNU',
        PANCRAFTER_DLPAN='/opt/DLPan-Toolbox', NVIDIA_DRIVER_CAPABILITIES='compute,utility',
        PYTORCH_KERNEL_CACHE_PATH='/tmp')
    from edgeres48.docker_mounts import targets
    # Resolve the sensor registered for this server, including QB original LMS.
    mounts = [(plan.ROOT,'/workspace',False)] + [(p,str(p),False) for p in targets(plan.ROOT,server)]
    if dlpan:
        mounts.append((Path(dlpan).resolve(strict=True),'/opt/DLPan-Toolbox',False))
    output = plan.ROOT/'work_dir/panda_unified_campaign'
    output.mkdir(parents=True,exist_ok=True)
    mounts.append((output,'/workspace/work_dir/panda_unified_campaign',True))
    cmd = ['docker','run','--init','--pid=host','--shm-size=8g','--stop-timeout=180',
           '--user',f'{os.getuid()}:{os.getgid()}','--workdir','/workspace']
    if gpu:
        cmd += ['--gpus','all']
    if detached:
        cmd += ['--detach','--name',container_name(server),'--label','panda.campaign='+plan.CAMPAIGN_ID,
                '--label','panda.server='+server,'--label','panda.source='+verify()['source_sha256']]
    else:
        cmd += ['--rm']
    for key,value in sorted(env.items()):
        cmd += ['--env',key+'='+value]
    for src,dst,rw in mounts:
        if any(c in str(src)+dst for c in (',','\n','\r')):
            raise ValueError('Unsupported mount delimiter')
        cmd += ['--mount',f'type=bind,src={src},dst={dst}'+('' if rw else ',readonly')]
    return cmd+[info['Id']]


def check_storage(server):
    root = plan.campaign_root(server)
    proof = json.loads((root/'preflight/verified.json').read_text())
    sizes = proof['checkpoint_bytes']
    # Conservative maximum: all controls rerun, every evaluated model retained.
    # Student snapshots retain A+T+S; the measured S1 checkpoint has only A+T.
    # Twice that observation bounds the extra same-width Student and optimizer.
    needed = sum((2 if c['stage']=='stage2' else 1)*
                 (len(c['evaluation_steps'])*sizes['evaluated'] + 2*sizes['complete_resume'])
                 for c in plan.queue(server)) + 3*2**30
    free = shutil.disk_usage(root).free
    if free < needed:
        raise RuntimeError(f'Insufficient storage: need {needed}, available {free} bytes')
    return dict(storage_available_bytes=free, storage_budget_bytes=needed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s1','s2'),required=True)
    parser.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    parser.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--wait',action='store_true',help='Wait for other GPU work without stopping it')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--status',action='store_true')
    args = parser.parse_args(argv)
    name = container_name(args.server)
    existing = inspect(name)
    if args.status:
        print(json.dumps(None if existing is None else {'name':name,'state':existing['State']}))
        return
    verify()
    if existing is not None:
        verify_container(existing,args.server)
        if existing['State']['Running']:
            print('Already running '+name)
            return
        if not args.resume:
            raise RuntimeError('Existing stopped campaign; use --resume after checking its state')
    elif args.resume:
        raise RuntimeError('No container to resume')
    base = command(args.server,args.image,gpu=False,dlpan=args.dlpan)
    for action in ('check','check-sheet'):
        subprocess.run(base+['python','-m','panda_unified_campaign',action,'--server',args.server],check=True)
    if args.check_only:
        return
    while True:
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                          '--format=csv,noheader,nounits'],text=True).strip()
        if not active:
            break
        if not args.wait:
            raise RuntimeError('Another GPU job is active; it was not stopped. Use --wait.')
        print('Waiting for GPU: '+active.replace('\n',','),flush=True)
        time.sleep(30)
    if existing is None:
        subprocess.run(command(args.server,args.image,dlpan=args.dlpan)+
            ['python','-m','panda_unified_campaign.preflight','--server',args.server],check=True)
        print(json.dumps(check_storage(args.server)),flush=True)
        verify()
        subprocess.run(command(args.server,args.image,detached=True,dlpan=args.dlpan)+
            ['python','-m','panda_unified_campaign','run','--server',args.server,'--device','cuda:0'],check=True)
    else:
        subprocess.run(['docker','start',name],check=True)
    launched = inspect(name)
    if launched is None or not launched['State']['Running']:
        raise RuntimeError('Container did not remain running; inspect Docker logs')
    print('Started '+name+'. Logs: docker logs -f '+name,flush=True)


if __name__ == '__main__':
    main()
