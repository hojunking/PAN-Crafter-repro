"""Resume an authenticated stopped campaign using the read-only IO compatibility overlay."""
import argparse
import importlib
from pathlib import Path
import subprocess
from panda_unified_campaign.enqueue import inspect
from .release import identity

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',choices=('panda_wv3_s2abl5','panda_sensor_main50'),required=True)
    p.add_argument('--server',choices=('s1','s2','s3'),required=True)
    p.add_argument('--image',default='hojunqueen/pancrafter-env:torch2.4.0-cu118')
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    plan=importlib.import_module(a.package+'.plan');launch=importlib.import_module(a.package+'.enqueue')
    source=importlib.import_module(a.package+'.release').verify();plan.sensor_for(a.server)
    oldname=launch.NAME if a.package=='panda_wv3_s2abl5' else launch.name(a.server)
    name=oldname+'-readguard';old=inspect(oldname);current=inspect(name)
    command=['python','-m','panda_donor_read_guard','--package',a.package,'--server',a.server]
    expected={'panda.server':a.server,'panda.campaign':plan.CAMPAIGN_ID,'panda.source':source['source_sha256']}
    for info,is_overlay in ((old,False),(current,True)):
        if not info:continue
        labels=info['Config'].get('Labels') or {};mounts=[m['Source'] for m in info['Mounts'] if m['Destination']=='/workspace']
        if (any(labels.get(k)!=v for k,v in expected.items()) or len(mounts)!=1
            or Path(mounts[0]).resolve()!=plan.ROOT.resolve()
            or info['Config']['Cmd']!=(command if is_overlay else ['python','-m',a.package,'run','--server',a.server])):
            raise ValueError('Container identity/checkout mismatch')
        if is_overlay and labels.get('panda.read_guard')!=identity()['sha256']:
            raise ValueError('Existing read-guard container has different IO source')
        if info['State']['Running']:
            if is_overlay:print('Already running '+name);return
            raise ValueError('Original campaign still running; no concurrent writer or forced stop')
    def base(gpu=False):
        return (launch.command(a.image,gpu=gpu) if a.package=='panda_wv3_s2abl5'
                else launch.command(a.server,a.image,gpu=gpu))
    for action in ('check','check-sheet'):
        subprocess.run(base()+[action,'--server',a.server],check=True)
    if a.check_only:return
    if (plan.ROOT/'gspread/server.txt').read_text().strip()!=a.server:
        raise ValueError('Wrong physical server')
    if current:subprocess.run(['docker','start',name],check=True)
    else:
        cmd=base(gpu=True)[:-3];image=cmd.pop();cmd.remove('--rm')
        cmd+=['--detach','--name',name]
        for k,v in {**expected,'panda.read_guard':identity()['sha256']}.items():cmd+=['--label',k+'='+v]
        subprocess.run(cmd+[image]+command,check=True)
    print('Started '+name+' using existing run state, optimizer and evidence',flush=True)

if __name__=='__main__':main()
