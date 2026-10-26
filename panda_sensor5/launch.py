"""Measured cost preparation and isolated sensor Docker launch."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from . import plan

def command(args, **kwargs):
    return subprocess.run(args,check=True,cwd=plan.ROOT,**kwargs)

def launch(server):
    from .transition import predecessor_receipt,container_gate
    from directshift48.__main__ import require_idle_gpu
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!=server:
        raise ValueError('Local server marker mismatch')
    predecessor_receipt(plan.ROOT,server); container_gate(server)
    require_idle_gpu('cuda')
    sensor=plan.dataset_for_server(server)
    directory=plan.ROOT/'work_dir/panda_sensor5_launchers'/server
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'start.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        name=f'panda-sensor5-stage12-{sensor.lower()}-{server}-v1'
        old=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
        if old.returncode==0: raise ValueError('Sensor5 container already exists; inspect, never overwrite')
        source=json.loads(command(['docker','inspect',f'panda-tune30-hardonly-wv3-{server}-v1'],
                                  capture_output=True,text=True).stdout)[0]
        image=os.environ.get('PAN_SENSOR5_IMAGE',source['Image'])
        dlpan=Path(os.environ.get('PANCRAFTER_DLPAN',str(plan.ROOT.parent/'DLPan-Toolbox'))).resolve(strict=True)
        from edgeres48.docker_mounts import targets
        data=targets(plan.ROOT,'s2' if server=='s1' else 's3')
        paths=[plan.ROOT,dlpan,*data]
        if any(any(x in str(p) for x in (',','\n','\r')) for p in paths):
            raise ValueError('Unsafe Docker mount path')
        output=plan.ROOT/'work_dir/panda_sensor5'; output.mkdir(parents=True,exist_ok=True)
        common=['--user',f'{os.getuid()}:{os.getgid()}','--workdir','/workspace',
            '--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONUNBUFFERED=1',
            '--env','OMP_NUM_THREADS=4','--env','MKL_NUM_THREADS=4','--env','MKL_THREADING_LAYER=GNU',
            '--env','PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox',
            '--mount',f'type=bind,src={plan.ROOT},dst=/workspace,readonly',
            '--mount',f'type=bind,src={dlpan},dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly']
        for p in data: common.extend(['--mount',f'type=bind,src={p},dst={p},readonly'])
        # Transition evidence refers to pre-existing WV3 files, visible read-only.
        command(['docker','run','--rm',*common,image,'python','-m','panda_sensor5','check','--server',server])
        from google.oauth2 import service_account
        service_account.Credentials.from_service_account_file(str(plan.ROOT/'gspread/account.json'),
                scopes=['https://www.googleapis.com/auth/spreadsheets'])
        writable=['--mount',f'type=bind,src={output},dst=/workspace/work_dir/panda_sensor5']
        gpu=['--init','--gpus','all','--pid=host','--shm-size=8g']
        command(['docker','run','--rm',*gpu,*common,*writable,image,
                 'python','-m','panda_sensor5.cost','--server',server,'--seconds','1200'])
        require_idle_gpu('cuda')
        result=command(['docker','run','--detach','--name',name,*gpu,*common,*writable,image,
            'python','-m','panda_sensor5','run','--server',server],capture_output=True,text=True)
        from directshift48.reporting import atomic_json,utc_now
        atomic_json(directory/'launch.json',{'at':utc_now(),'server':server,'dataset':sensor,
            'container':name,'container_id':result.stdout.strip(),'image':image})
        command([sys.executable,'-m','panda_sensor5.best_publication.restart','--server',server])
        print(json.dumps({'started':name,'dataset':sensor,'stage1_runs':1,'stage2_runs':5}),flush=True)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s1','s3'),required=True)
    args=parser.parse_args(argv); launch(args.server); return 0

if __name__=='__main__':
    raise SystemExit(main())
