"""Launch the server's isolated WV3 hyperparameter lane; stop only an explicitly named old s3 container; preserve its results."""
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
    return f'panda-tab3-rgbn-{server}-v1'


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
    # Historical mount helper maps s3 to GF2. Table 3 uses the WV3 (s1) mount inventory.
    mounts = [(plan.ROOT,'/workspace',False)] + [(p,str(p),False) for p in targets(plan.ROOT,'s1')]
    if dlpan:
        mounts.append((Path(dlpan).resolve(strict=True),'/opt/DLPan-Toolbox',False))
    output = plan.ROOT/'work_dir/panda_wv3_tab3_rgbn'
    output.mkdir(parents=True,exist_ok=True)
    mounts.append((output,'/workspace/work_dir/panda_wv3_tab3_rgbn',True))
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
    # Three evaluated weights and two complete generations for each of 15 runs.
    needed = sum(len(c['evaluation_steps'])*sizes['evaluated'] + 2*sizes['complete_resume']
                 for c in plan.queue(server)) + 3*2**30
    free = shutil.disk_usage(root).free
    if free < needed:
        raise RuntimeError(f'Insufficient storage: need {needed}, available {free} bytes')
    return dict(storage_available_bytes=free, storage_budget_bytes=needed)


def validate_old_container(info, server):
    """Verify server and checkout before sending any signal; never infer from a fuzzy name."""
    if server != 's3' or info is None:
        raise ValueError('An existing s3 container must be named explicitly')
    labels=info['Config'].get('Labels') or {}
    cmd=info['Config'].get('Cmd') or []
    explicit_s3=any(cmd[i]=='--server' and cmd[i+1]=='s3' for i in range(len(cmd)-1)) or '--server=s3' in cmd
    if (labels.get('panda.server') not in (None,'s3')
            or not (labels.get('panda.server')=='s3' or explicit_s3)
            or labels.get('panda.campaign')==plan.CAMPAIGN_ID):
        raise ValueError('Refusing to stop a non-s3, ambiguous, or new-campaign container')
    mounts=[m['Source'] for m in info.get('Mounts',[]) if m['Destination']=='/workspace']
    if len(mounts)!=1 or Path(mounts[0]).resolve()!=plan.ROOT.resolve():
        raise ValueError('Old container must use this repository checkout')


def stop_old_container(name, server):
    info=inspect(name);validate_old_container(info,server)
    folder=plan.campaign_root(server)/'cutover';folder.mkdir(parents=True,exist_ok=True)
    from datetime import datetime,timezone
    # Save only public process identity, never container environment/secrets.
    record={'container':name,'container_id':info['Id'],'previous_state':info['State'],
            'at':datetime.now(timezone.utc).isoformat(),'results_deleted':False}
    if info['State']['Running']:
        subprocess.run(['docker','kill','--signal=TERM',name],check=True)
        deadline=time.monotonic()+300
        while inspect(name)['State']['Running']:
            if time.monotonic()>deadline:
                raise RuntimeError('Old runner did not stop within 300s; inspect its logs. No SIGKILL or new launch.')
            time.sleep(2)
    record['final_state']=inspect(name)['State']
    (folder/('stop_'+info['Id'][:12]+'.json')).write_text(json.dumps(record,indent=2)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s3',),required=True)
    parser.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    parser.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--wait',action='store_true',help='Wait for other GPU work without stopping it')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--stop-container',action='append',default=[],
        help='Explicit old s3 container to stop now with SIGTERM, preserving all results')
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
        subprocess.run(base+['python','-m','panda_wv3_tab3_rgbn',action,'--server',args.server],check=True)
    if args.check_only:
        return
    for old_name in args.stop_container:
        stop_old_container(old_name,args.server)
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
            ['python','-m','panda_wv3_tab3_rgbn.preflight','--server',args.server],check=True)
        print(json.dumps(check_storage(args.server)),flush=True)
        verify()
        subprocess.run(command(args.server,args.image,detached=True,dlpan=args.dlpan)+
            ['python','-m','panda_wv3_tab3_rgbn','run','--server',args.server,'--device','cuda:0'],check=True)
    else:
        subprocess.run(['docker','start',name],check=True)
    launched = inspect(name)
    if launched is None or not launched['State']['Running']:
        raise RuntimeError('Container did not remain running; inspect Docker logs')
    print('Started '+name+'. Logs: docker logs -f '+name,flush=True)


if __name__ == '__main__':
    main()
