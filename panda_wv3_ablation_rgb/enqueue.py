"""Preflight and explicit s3-only replacement of the legacy mean8 ablation queue."""
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
    output=plan.ROOT/'work_dir/panda_wv3_ablation_rgb';output.mkdir(parents=True,exist_ok=True)
    mounts.append((output,'/workspace/work_dir/panda_wv3_ablation_rgb',True))
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


LEGACY_NAME = 'panda-wv3-components-s3-v1'
LEGACY_CAMPAIGN = 'PANDA_WV3_COMPONENT_S13_20261102_v1'
CONTAINER_NAME = 'panda-wv3-components-rgb-s3-v1'


def inspect_container(name):
    p=subprocess.run(['docker','container','inspect',name],capture_output=True,text=True)
    return json.loads(p.stdout)[0] if p.returncode==0 else None


def verify_container(info, *, campaign, source=None):
    labels=info['Config'].get('Labels') or {}
    if labels.get('panda.campaign')!=campaign or labels.get('panda.server')!='s3':
        raise ValueError('Container does not belong to the intended s3 campaign')
    if source is not None and labels.get('panda.source')!=source:
        raise ValueError('Existing RGB container uses another source release')
    roots=[m['Source'] for m in info.get('Mounts',[]) if m.get('Destination')=='/workspace']
    if len(roots)!=1 or Path(roots[0]).resolve()!=plan.ROOT.resolve():
        raise ValueError('Container belongs to another checkout')


def stop_legacy():
    info=inspect_container(LEGACY_NAME)
    if info is None:return {'action':'NO_LEGACY_CONTAINER'}
    verify_container(info,campaign=LEGACY_CAMPAIGN)
    # Disable accidental automatic restart and gracefully save the current update.
    # Keep the container, all original weights, and its published rows.
    subprocess.run(['docker','update','--restart=no',LEGACY_NAME],check=True)
    if info['State']['Running']:
        subprocess.run(['docker','stop','--time','180',LEGACY_NAME],check=True)
    after=inspect_container(LEGACY_NAME)
    if after is None or after['State']['Running']:
        raise RuntimeError('Legacy container did not stop')
    return {'action':'LEGACY_STOPPED_RETAINED','name':LEGACY_NAME,
            'exit_code':after['State'].get('ExitCode')}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',choices=('s3',),required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--wait',action='store_true')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--replace-legacy',action='store_true',
                   help='After successful preflight, stop only the verified old s3 WV3 ablation container')
    args=p.parse_args(argv);source=release_check()
    existing=inspect_container(CONTAINER_NAME)
    if existing is not None:verify_container(existing,campaign=plan.CAMPAIGN_ID,source=source)
    # Source/data/environment/Sheet checks must succeed BEFORE touching old training.
    base=command(args.server,args.image,dlpan=args.dlpan,gpu=False)
    for action in ('check','check-sheet'):
        subprocess.run(base+['python','-m','panda_wv3_ablation_rgb',action,'--server','s3'],check=True)
    if args.check_only:return
    if args.replace_legacy:
        receipt=stop_legacy()
        root=plan.campaign_root('s3');root.mkdir(parents=True,exist_ok=True)
        (root/'legacy_cutover.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if existing is not None and existing['State']['Running']:
        print('Already running '+CONTAINER_NAME);return
    while True:
        active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                              check=True,capture_output=True,text=True).stdout.strip()
        if not active:break
        if not args.wait:raise RuntimeError('Other GPU work is active; it was not stopped. Use --wait or inspect it.')
        print('Waiting for other GPU work:',active.replace('\n',','),flush=True);time.sleep(30)
    release_check()
    if existing is not None:
        subprocess.run(['docker','start',CONTAINER_NAME],check=True)
    else:
        subprocess.run(command('s3',args.image,name=CONTAINER_NAME,dlpan=args.dlpan)+
            ['python','-m','panda_wv3_ablation_rgb','run','--server','s3','--device','cuda:0'],check=True)
    print('Started '+CONTAINER_NAME+'. Follow with: docker logs -f '+CONTAINER_NAME)


if __name__=='__main__':main()
