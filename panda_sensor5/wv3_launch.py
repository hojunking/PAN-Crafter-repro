"""Launch the control-only WV3 adapter in its exact previously used image."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from directshift48.reporting import atomic_json, utc_now
from .wv3 import adapter_pin

ROOT=Path(__file__).resolve().parents[1]


def inspect(name):
    return json.loads(subprocess.run(['docker','inspect',name],capture_output=True,text=True,check=True).stdout)[0]


def source_container(server, package):
    name=f"panda-{package.removeprefix('panda_')}-hardonly-wv3-{server}-v1"
    result=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
    if result.returncode and package=='panda_hard36':
        name=f'panda-tune30-hardonly-wv3-{server}-v1'
        result=subprocess.run(['docker','inspect',name],capture_output=True,text=True,check=True)
    elif result.returncode:
        raise ValueError('Cannot inspect original WV3 container: '+name)
    source=json.loads(result.stdout)[0]
    if source['State']['Running'] or source['State']['ExitCode']!=0:
        raise ValueError('Source WV3 container must be safely exited with code 0')
    return name,source


def command(server,package,*,name=None):
    if (ROOT/'gspread/server.txt').read_text().strip()!=server:
        raise ValueError('Local server marker mismatch')
    source_name,source=source_container(server,package)
    cmd=['docker','run','--init','--gpus','all','--pid=host','--shm-size=8g',
         '--user',source['Config']['User'],'--workdir',source['Config']['WorkingDir']]
    cmd+=['--detach','--name',name] if name else ['--rm']
    for item in source['Config']['Env']:
        cmd.extend(['--env',item])
    output=ROOT/'work_dir'/package
    output.mkdir(parents=True,exist_ok=True)
    destination='/workspace/work_dir/'+package
    own_output_seen=False
    for mount in source['Mounts']:
        if mount['Type']!='bind':
            raise ValueError('Only original explicit bind mounts may be cloned')
        if any(',' in mount[k] or '\n' in mount[k] for k in ('Source','Destination')):
            raise ValueError('Unsafe Docker bind mount delimiter')
        # A new HARD36 campaign borrows TUNE30's environment, never its output.
        if mount['Destination'].startswith('/workspace/work_dir/'):
            if mount['Destination']!=destination:
                continue
            own_output_seen=True
        value=f"type=bind,src={mount['Source']},dst={mount['Destination']}"
        if not mount['RW']:
            value+=',readonly'
        cmd.extend(['--mount',value])
    if not own_output_seen:
        cmd.extend(['--mount',f'type=bind,src={output},dst={destination}'])
    cmd.append(source['Image'])
    return cmd,source_name,source


def prepare_hard(server):
    """Existing cost and disposable exact-resume preflight; no production run."""
    from .transition import campaign_root
    from panda_hard36.runtime import code_pin
    root=campaign_root(ROOT,'panda_hard36',server)
    cost_ready=root/'cost_profiles/ready.json'
    preflight=root/'preflight/passed.json'
    if cost_ready.exists() and preflight.exists():
        if json.loads(preflight.read_text()).get('code_pin')!=code_pin():
            raise ValueError('Existing HARD36 preflight source differs')
        # In particular, a paused HARD36 run must not reopen the old wall-clock
        # budget in a profiling command merely because its sensor detour ended.
        return
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu('cuda')
    env=dict(os.environ,PAN_HOST_PYTHON=os.sys.executable,PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['bash','scripts/prepare_panda_hard36_costs.sh',server],
                   cwd=ROOT,env=env,check=True)
    cmd,_,_=command(server,'panda_hard36')
    subprocess.run(cmd+['python','-m','panda_hard36.preflight','--server',server],check=True)


def launch(server, package, mode, target_run=None):
    from directshift48.__main__ import require_idle_gpu
    require_idle_gpu('cuda')
    short=package.removeprefix('panda_')
    name=f'panda-sensor5-wv3-{short}-{mode}-{server}-{time.time_ns()}'
    cmd,source_name,source=command(server,package,name=name)
    module='panda_sensor5.wv3_continue' if package=='panda_hard36' else 'panda_sensor5.wv3'
    cmd.extend(['python','-m',module,'--server',server,
                '--package',package,'--mode',mode])
    if target_run:
        cmd.extend(['--target-run',target_run])
    result=subprocess.run(cmd,capture_output=True,text=True,check=True)
    receipt={'container':name,'container_id':result.stdout.strip(),'at':utc_now(),
        'package':package,'mode':mode,'target_run':target_run,'adapter_pin':adapter_pin(),
        'source_container':source_name,'image':source['Image']}
    directory=ROOT/'work_dir/panda_sensor5_launchers'/server
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/f'wv3_{short}_{mode}.json',receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=('s1','s3'),required=True)
    parser.add_argument('--package',choices=('panda_tune30','panda_hard36'),default='panda_tune30')
    parser.add_argument('--mode',choices=('finish-current','resume-all'),required=True)
    parser.add_argument('--target-run')
    args=parser.parse_args(argv)
    print(json.dumps(launch(args.server,args.package,args.mode,args.target_run)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
