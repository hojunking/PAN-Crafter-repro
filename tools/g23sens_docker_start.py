#!/usr/bin/env python3
"""Start the committed local s4/s5 G23 service in an existing pinned Docker image."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
LABEL='org.pancrafter.g23sens'
DEFAULT_IMAGE='hojunqueen/pancrafter-env:torch2.4.0-cu118'


def output(argv,**kwargs):
    return subprocess.check_output(argv,text=True,**kwargs).strip()


def build_mounts(root,frozen,dlpan):
    from g23sens.assets import reference_config,_path,canonical_calibration
    root=Path(root).resolve();frozen=Path(frozen).resolve();dlpan=Path(dlpan).resolve(strict=True)
    work=(root/'work_dir').resolve(strict=True)
    mounts={root:'ro',frozen:'ro',work:'rw',dlpan:'ro'}
    cfg=reference_config(root)
    paths=[]
    for key in ('train_feeder_args','val_feeder_args','test_reduced_feeder_args','test_full_feeder_args'):
        source=_path(cfg[key]['dataroot'],root)
        paths.extend((source,Path(str(source).replace('.h5','_pan.h5'))))
    paths.extend((_path(cfg['kdv']['teacher']['run'],root),
        _path(cfg['kdv']['qrecon']['asset'],root),
        root/'assets/qedge9/cue_T0_AXIS16_v1.npz',canonical_calibration(root)))
    paths.extend((root/'gspread').glob('*.json'))
    for value in paths:
        path=Path(value).resolve(strict=True)
        if root not in path.parents and work not in path.parents:
            # Exact file or package directory only; never an entire unrelated home directory.
            mounts[path]='ro'
    return [(str(p),access) for p,access in sorted(mounts.items(),key=lambda x:(len(x[0].parts),str(x[0])))]


def build_command(*,frozen,server,image_id,commit,name,mounts,dlpan,gpu='0',no_upload=False):
    from g23sens.common import verify_server
    verify_server(server)
    if not re.fullmatch(r'(?:\d+|GPU-[A-Za-z0-9-]+)',gpu):raise ValueError('Select one GPU index or UUID')
    if not re.fullmatch(r'sha256:[a-f0-9]{64}',image_id):raise ValueError('Pin the locally inspected Docker image ID')
    argv=['docker','run','--detach','--name',name,'--gpus','device='+gpu,
        '--pid=host','--user',f'{os.getuid()}:{os.getgid()}','--shm-size','8g',
        '--restart','no','--workdir',str(frozen),
        '--label',LABEL+'.server='+server,'--label',LABEL+'.commit='+commit,
        '--label',LABEL+'.runtime='+str(frozen)]
    for key,value in dict(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/g23sens-mpl',XDG_CACHE_HOME='/tmp/g23sens-cache',
        HF_HOME='/tmp/g23sens-hf',PANCRAFTER_DLPAN=str(dlpan),
        PANCRAFTER_G23SENS_GPU_UUID=gpu).items():argv+=['--env',key+'='+value]
    for path,access in mounts:
        if ',' in path:raise ValueError('Unsupported bind path comma')
        argv+=['--mount','type=bind,src='+path+',dst='+path+(',readonly' if access=='ro' else '')]
    argv+=['--entrypoint','python',image_id,str(Path(frozen)/'tools/g23sens_runner.py'),
           'start','--server',server,'--in-place','--foreground']
    if no_upload:argv.append('--no-upload')
    return argv


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s4','s5'),required=True)
    parser.add_argument('--image',default=DEFAULT_IMAGE)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--no-upload',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    from g23sens.common import camp,read,locked
    from g23sens.deployment import frozen_checkout
    image_id=output(['docker','image','inspect','--format','{{.Id}}',args.image])
    # Record the physical UUID on the host; Docker may renumber its single visible device.
    args.gpu=output(['nvidia-smi','-i',args.gpu,'--query-gpu=uuid','--format=csv,noheader,nounits'])
    commit=output(['git','rev-parse','HEAD'],cwd=ROOT)
    receipt=read(camp(ROOT,args.server)/'runtime_release.json')
    frozen=(Path(receipt['path']) if receipt else ROOT.parent/f'{ROOT.name}-runtime-g23sens-{args.server}-{commit[:12]}')
    dlpan=Path(os.environ.get('PANCRAFTER_DLPAN',str(ROOT.parent/'DLPan-Toolbox'))).resolve(strict=True)
    def command():
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        return build_command(frozen=frozen,server=args.server,image_id=image_id,commit=commit,
            name=f'pancrafter-g23sens-{args.server}-{stamp}',mounts=build_mounts(ROOT,frozen,dlpan),
            dlpan=dlpan,gpu=args.gpu,no_upload=args.no_upload)
    if args.dry_run:
        print(shlex.join(command()));return 0
    frozen=frozen_checkout(ROOT,args.server);commit=output(['git','rev-parse','HEAD'],cwd=frozen)
    with locked(camp(ROOT,args.server)/'docker_launch.lock'):
        ids=output(['docker','ps','-q','--filter','label='+LABEL+'.server='+args.server]).split()
        if ids:
            rows=json.loads(output(['docker','inspect',*ids]))
            if len(rows)!=1:raise ValueError('Multiple active local G23 owners')
            row=rows[0];labels=row['Config']['Labels']
            if row['Image']!=image_id or labels[LABEL+'.runtime']!=str(frozen) or labels[LABEL+'.commit']!=commit:
                raise ValueError('A different release/image already owns this lane')
            print(json.dumps(dict(status='ALREADY_RUNNING',container_id=ids[0])));return 0
        container=output(command())
        print(json.dumps(dict(status='START_REQUESTED',container_id=container,runtime=str(frozen),
            log_command='docker logs --tail 80 '+container,
            note='Training starts only after old current-run boundary and actual GPU preflight.')))
        return 0


if __name__=='__main__':raise SystemExit(main())
