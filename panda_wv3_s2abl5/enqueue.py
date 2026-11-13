"""Durable after-main queue; never interrupt the current main run."""
import argparse
import json
from pathlib import Path
import subprocess
from panda_unified_campaign import enqueue as base
from . import plan
from .release import verify

NAME='panda-wv3-s2abl5-s1-v1'
def command(image,*,gpu=False,detached=False):
    cmd=base.command('s1',image,gpu=gpu)
    output=plan.ROOT/'work_dir/panda_wv3_s2abl5';output.mkdir(parents=True,exist_ok=True)
    img=cmd.pop()
    cmd+=['--mount',f'type=bind,src={output},dst=/workspace/work_dir/panda_wv3_s2abl5']
    if detached:
        cmd.remove('--rm');cmd+=['--detach','--name',NAME,'--label','panda.server=s1',
            '--label','panda.campaign='+plan.CAMPAIGN_ID,'--label','panda.source='+verify()['source_sha256']]
    return cmd+[img,'python','-m','panda_wv3_s2abl5']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--check-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();release=verify();old=base.inspect(NAME)
    if old:
        labels=old['Config'].get('Labels') or {}
        expected={'panda.server':'s1','panda.campaign':plan.CAMPAIGN_ID,'panda.source':release['source_sha256']}
        mounts=[m['Source'] for m in old['Mounts'] if m['Destination']=='/workspace']
        if (any(labels.get(k)!=v for k,v in expected.items()) or len(mounts)!=1
            or Path(mounts[0]).resolve()!=plan.ROOT.resolve()
            or old['Config']['Cmd']!=['python','-m','panda_wv3_s2abl5','run','--server','s1']):
            raise ValueError('Existing container identity mismatch')
        if old['State']['Running']:print('Already queued/running '+NAME);return
        if not a.resume:raise ValueError('Stopped queue exists; use --resume after checking its logs')
    for action in ('check','check-sheet'):
        subprocess.run(command(a.image)+[action,'--server','s1'],check=True)
    if a.check_only:return
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!='s1':raise ValueError('s1 only; no donor transfer')
    if old:subprocess.run(['docker','start',NAME],check=True)
    else:subprocess.run(command(a.image,gpu=True,detached=True)+['run','--server','s1'],check=True)
    print('Queued '+NAME+'; main must finish before preflight/training',flush=True)

if __name__=='__main__':main()
