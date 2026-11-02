"""Explicit Docker launch; wait for existing GPU work without stopping it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from . import plan


def release_check():
    value=json.loads((plan.CONFIG/'release.json').read_text())
    bad=[p for p,sha in value['files'].items() if not (plan.ROOT/p).is_file()
         or hashlib.sha256((plan.ROOT/p).read_bytes()).hexdigest()!=sha]
    if bad:raise ValueError('Pull the complete committed release: '+', '.join(bad))
    if value['source_sha256']!=plan.digest({k:v for k,v in value.items() if k!='source_sha256'}):
        raise ValueError('Release manifest digest mismatch')
    return value['source_sha256']


def command(server,image,*,name=None,dlpan=None,gpu=True):
    result=subprocess.run(['docker','image','inspect',image],check=True,capture_output=True,text=True)
    info=json.loads(result.stdout)[0]
    env=dict(s.split('=',1) for s in info['Config'].get('Env',[]))
    env.update(PYTHONPATH='/workspace',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MKL_THREADING_LAYER='GNU',
        PANCRAFTER_DLPAN='/opt/DLPan-Toolbox',NVIDIA_DRIVER_CAPABILITIES='compute,utility')
    from edgeres48.docker_mounts import targets
    # Legacy helper maps s3 to GF2. Both lanes HERE deliberately use its WV3 mapping.
    mounts=[(plan.ROOT,'/workspace',False)]+[(p,str(p),False) for p in targets(plan.ROOT,'s1')]
    if dlpan:mounts.append((Path(dlpan).resolve(strict=True),'/opt/DLPan-Toolbox',False))
    output=plan.ROOT/'work_dir/panda_wv3_ablation';output.mkdir(parents=True,exist_ok=True)
    mounts.append((output,'/workspace/work_dir/panda_wv3_ablation',True))
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


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',choices=('s1','s3'),required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--wait',action='store_true')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args(argv);release_check()
    name='panda-wv3-components-'+args.server+'-v1'
    previous=subprocess.run(['docker','container','inspect',name],capture_output=True,text=True)
    if previous.returncode==0 and not args.check_only:
        raise RuntimeError('Existing experiment container retained; use docker start -a '+name+' to resume it')
    base=command(args.server,args.image,dlpan=args.dlpan,gpu=False)
    for action in ('check','check-sheet'):
        subprocess.run(base+['python','-m','panda_wv3_ablation',action,'--server',args.server],check=True)
    if args.check_only:return
    while True:
        active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                              check=True,capture_output=True,text=True).stdout.strip()
        if not active:break
        if not args.wait:raise RuntimeError('GPU work is active. Use --wait to start after it completes.')
        print('Waiting for existing local GPU work:',active.replace('\n',','),flush=True);time.sleep(30)
    release_check()
    subprocess.run(command(args.server,args.image,name=name,dlpan=args.dlpan)+
        ['python','-m','panda_wv3_ablation','run','--server',args.server,'--device','cuda:0'],check=True)
    print('Started '+name+'. Follow with: docker logs -f '+name)


if __name__=='__main__':main()
