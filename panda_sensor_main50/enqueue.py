"""Queue on s2/s3; existing GPU work finishes first. No cross-server access."""
import argparse
import os
from pathlib import Path
import subprocess
from panda_sensor_eq40 import enqueue as base
from panda_unified_campaign.enqueue import inspect
from . import plan
from .release import verify

def name(server):return f'panda-sensor-main50-{server}-v1'

def command(server,image,*,gpu=False,detached=False,dlpan=None):
    cmd=base.command(server,image,gpu=gpu,dlpan=dlpan)
    img=cmd.pop()
    # Read-only original runs; only the new campaign output is writable.
    for i,item in enumerate(cmd):
        if item=='--mount' and '/workspace/work_dir/' in cmd[i+1] and not cmd[i+1].endswith(',readonly'):
            cmd[i+1]+=',readonly'
    output=plan.ROOT/'work_dir/panda_sensor_main50';output.mkdir(parents=True,exist_ok=True)
    cmd+=['--mount',f'type=bind,src={output},dst=/workspace/work_dir/panda_sensor_main50']
    if detached:
        cmd.remove('--rm')
        cmd+=['--detach','--name',name(server),'--label','panda.server='+server,
              '--label','panda.campaign='+plan.CAMPAIGN_ID,'--label','panda.source='+verify()['source_sha256']]
    return cmd+[img,'python','-m','panda_sensor_main50']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',choices=('s2','s3'),required=True)
    p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--dlpan',default=os.environ.get('PANCRAFTER_DLPAN'))
    p.add_argument('--check-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();release=verify();old=inspect(name(a.server))
    if old:
        labels=old['Config'].get('Labels') or {}
        expected={'panda.server':a.server,'panda.campaign':plan.CAMPAIGN_ID,'panda.source':release['source_sha256']}
        mounts=[m['Source'] for m in old['Mounts'] if m['Destination']=='/workspace']
        if (any(labels.get(k)!=v for k,v in expected.items()) or len(mounts)!=1
            or Path(mounts[0]).resolve()!=plan.ROOT.resolve()
            or old['Config']['Cmd']!=['python','-m','panda_sensor_main50','run','--server',a.server]):
            raise ValueError('Existing container identity mismatch')
        if old['State']['Running']:print('Already queued/running '+name(a.server));return
        if not a.resume:raise ValueError('Stopped queue exists; inspect logs then use --resume')
    for action in ('check','check-sheet'):
        subprocess.run(command(a.server,a.image,dlpan=a.dlpan)+[action,'--server',a.server],check=True)
    if a.check_only:return
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:
        raise ValueError('Launch only on the matching physical server')
    if old:subprocess.run(['docker','start',name(a.server)],check=True)
    else:subprocess.run(command(a.server,a.image,gpu=True,detached=True,dlpan=a.dlpan)+['run','--server',a.server],check=True)
    print('Queued '+name(a.server)+'; full50K, no metric early stop',flush=True)

if __name__=='__main__':main()
